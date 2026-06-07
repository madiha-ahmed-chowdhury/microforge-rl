#!/usr/bin/env python3
"""
guest agent — runs inside the Firecracker microVM as a persistent server.

Communication protocol (over vsock port 52):
  HOST → GUEST : JSON line: {"code": "<python source>", "timeout": 10,
                              "memory_limit_mb": 64, "cpu_limit_sec": 5}
  GUEST → HOST : JSON line: {"exit_code": 0, "stdout": "...", "stderr": "...",
                              "wall_time_ms": 123, "cpu_user_ms": 45,
                              "cpu_sys_ms": 12, "mem_peak_kb": 8192,
                              "timed_out": false, "oom_killed": false}

  memory_limit_mb and cpu_limit_sec are optional (default 0 = no limit).
  oom_killed is True when exit_code == 137 (ulimit -v exceeded).
"""

import socket
import json
import subprocess
import time
import resource
import os
import sys
import tempfile
import threading

LISTEN_PORT  = 52
AF_VSOCK     = 40
VMADDR_CID_ANY = 0xFFFFFFFF


def build_exec_command(code_path: str, memory_limit_mb: int, cpu_limit_sec: int) -> list:
    """
    Build the command to execute the Python code file.
    Applies ulimit constraints if limits are specified.
    Returns a list suitable for subprocess.Popen.
    """
    ulimit_parts = []

    # Stack size limit — always applied for safety
    ulimit_parts.append("ulimit -s 65536")

    if memory_limit_mb > 0:
        # ulimit -v limits virtual memory in KB
        ulimit_parts.append(f"ulimit -v {memory_limit_mb * 1024}")

    if cpu_limit_sec > 0:
        # ulimit -t limits CPU time in seconds
        ulimit_parts.append(f"ulimit -t {cpu_limit_sec}")

    if memory_limit_mb > 0 or cpu_limit_sec > 0:
        ulimit_str = " && ".join(ulimit_parts)
        cmd = ["bash", "-c", f"{ulimit_str} && python3 {code_path}"]
    else:
        ulimit_parts_str = " && ".join(ulimit_parts)
        cmd = ["bash", "-c", f"{ulimit_parts_str} && python3 {code_path}"]

    return cmd


def measure_process(code_str: str, timeout_sec: int,
                    memory_limit_mb: int = 0, cpu_limit_sec: int = 0) -> dict:
    """
    Write code to a temp file, run it with resource limits, measure usage.
    Returns a dict with timing, memory, exit code, stdout, stderr, oom_killed.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                     delete=False, dir='/tmp') as f:
        f.write(code_str)
        code_path = f.name

    result = {
        "exit_code":    -1,
        "stdout":       "",
        "stderr":       "",
        "wall_time_ms": 0,
        "cpu_user_ms":  0,
        "cpu_sys_ms":   0,
        "mem_peak_kb":  0,
        "timed_out":    False,
        "oom_killed":   False,
    }

    try:
        cmd        = build_exec_command(code_path, memory_limit_mb, cpu_limit_sec)
        wall_start = time.monotonic()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
            result["exit_code"] = proc.returncode
            result["stdout"]    = stdout.decode("utf-8", errors="replace")[:10485760]
            result["stderr"]    = stderr.decode("utf-8", errors="replace")[:1048576]

            stderr_text = result["stderr"]

            # exit code 137 = SIGKILL from OOM killer or ulimit -v (hard kill)
            if proc.returncode == 137:
                result["oom_killed"] = True
                result["stderr"] += "\n[OOM: memory limit exceeded]"

            # exit code 1 + MemoryError = ulimit -v caused ENOMEM → Python exception
            elif proc.returncode == 1 and "MemoryError" in stderr_text:
                result["oom_killed"] = True
                result["stderr"] += "\n[OOM: MemoryError - memory limit exceeded]"

            # exit code 152 = SIGXCPU from ulimit -t (CPU time exceeded)
            if proc.returncode == 152:
                result["stderr"] += "\n[CPU time limit exceeded]"

        except subprocess.TimeoutExpired:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            result["timed_out"] = True
            result["exit_code"] = -9
            result["stderr"]    = "TIMEOUT"

        wall_end = time.monotonic()
        result["wall_time_ms"] = int((wall_end - wall_start) * 1000)

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result["cpu_user_ms"] = int(usage.ru_utime * 1000)
        result["cpu_sys_ms"]  = int(usage.ru_stime * 1000)
        result["mem_peak_kb"] = usage.ru_maxrss

    finally:
        try:
            os.unlink(code_path)
        except OSError:
            pass

    return result


def handle_connection(conn):
    """Handle one host→guest request over an accepted vsock connection."""
    try:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk

        if not buf.strip():
            return

        request         = json.loads(buf.decode("utf-8"))
        code            = request.get("code", "")
        timeout         = int(request.get("timeout", 10))
        memory_limit_mb = int(request.get("memory_limit_mb", 0))
        cpu_limit_sec   = int(request.get("cpu_limit_sec", 0))

        print(f"[agent] Received {len(code)} chars, timeout={timeout}s "
              f"mem_limit={memory_limit_mb}MB cpu_limit={cpu_limit_sec}s",
              flush=True)

        metrics = measure_process(code, timeout, memory_limit_mb, cpu_limit_sec)

        print(f"[agent] Done: exit={metrics['exit_code']} "
              f"wall={metrics['wall_time_ms']}ms "
              f"mem={metrics['mem_peak_kb']}KB "
              f"oom={metrics['oom_killed']}",
              flush=True)

        response = json.dumps(metrics) + "\n"
        conn.sendall(response.encode("utf-8"))

    except Exception as e:
        err = json.dumps({"exit_code": -1, "stderr": str(e),
                          "timed_out": False, "oom_killed": False,
                          "wall_time_ms": 0, "cpu_user_ms": 0,
                          "cpu_sys_ms": 0, "mem_peak_kb": 0,
                          "stdout": ""}) + "\n"
        try:
            conn.sendall(err.encode("utf-8"))
        except Exception:
            pass
    finally:
        conn.close()


def main():
    print(f"[agent] Starting vsock server on port {LISTEN_PORT}", flush=True)

    server = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((VMADDR_CID_ANY, LISTEN_PORT))
    server.listen(5)

    print(f"[agent] Listening...", flush=True)

    while True:
        conn, addr = server.accept()
        print(f"[agent] Connection from CID={addr[0]}", flush=True)
        t = threading.Thread(target=handle_connection, args=(conn,))
        t.daemon = True
        t.start()


if __name__ == "__main__":
    main()
