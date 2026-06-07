"""
vm.py — Firecracker microVM lifecycle management.

Handles booting, stopping, vsock communication, and cgroups resource control.
"""

import dataclasses
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

WORK_DIR           = Path(__file__).parent.resolve()
FIRECRACKER_BIN    = str(WORK_DIR / "firecracker")
VSOCK_PORT         = 52
AGENT_BOOT_TIMEOUT = 180   # seconds


# ── VM State ──────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class VMState:
    vm_id: str
    proc: subprocess.Popen
    vsock_sock: str          # Unix socket path for vsock proxy
    api_sock: str            # Firecracker API socket path
    tmp_dir: str             # Temp dir holding per-VM config + sockets
    cgroup_path: Optional[Path] = None  # dedicated cgroup for this VM
    last_execution: Optional[dict] = None
    created_at: float = dataclasses.field(default_factory=time.monotonic)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def start_vm(vm_id: str) -> VMState:
    """Boot a new Firecracker VM. Returns a VMState when the process is running."""
    tmp_dir    = tempfile.mkdtemp(prefix=f"fc-{vm_id}-")
    vsock_sock = os.path.join(tmp_dir, "vsock.sock")
    api_sock   = os.path.join(tmp_dir, "api.sock")

    config = {
        "boot-source": {
            "kernel_image_path": str(WORK_DIR / "vmlinux"),
            "boot_args": "console=ttyS0 reboot=k panic=1 pci=off nomodules ro",
        },
        "drives": [{
            "drive_id": "rootfs",
            "path_on_host": str(WORK_DIR / "rootfs.ext4"),
            "is_root_device": True,
            "is_read_only": False,
        }],
        "machine-config": {"vcpu_count": 1, "mem_size_mib": 256},
        "vsock": {"guest_cid": 3, "uds_path": vsock_sock},
    }
    config_path = os.path.join(tmp_dir, "vm_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    cmd = [
        FIRECRACKER_BIN,
        "--api-sock", api_sock,
        "--config-file", config_path,
        "--log-path", os.path.join(tmp_dir, "firecracker.log"),
        "--level", "Info",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    cgroup_path = _setup_vm_cgroup(vm_id, proc.pid)
    return VMState(vm_id=vm_id, proc=proc, vsock_sock=vsock_sock,
                   api_sock=api_sock, tmp_dir=tmp_dir, cgroup_path=cgroup_path)


def stop_vm(vm: VMState) -> None:
    """Terminate the VM process and clean up its temp directory and cgroup."""
    vm.proc.terminate()
    try:
        vm.proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        vm.proc.kill()
    _teardown_vm_cgroup(vm.cgroup_path)
    shutil.rmtree(vm.tmp_dir, ignore_errors=True)


# ── vsock communication ───────────────────────────────────────────────────────

def _vsock_connect(vsock_sock: str, timeout_sec: float) -> socket.socket:
    """Open a vsock connection to the guest agent."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_sec)
    sock.connect(vsock_sock)
    sock.sendall(f"CONNECT {VSOCK_PORT}\n".encode())
    resp = b""
    while b"\n" not in resp:
        chunk = sock.recv(32)
        if not chunk:
            break
        resp += chunk
    if not resp.startswith(b"OK "):
        sock.close()
        raise OSError(f"vsock handshake failed: {resp!r}")
    return sock


def wait_for_agent(vm: VMState, timeout_sec: int = AGENT_BOOT_TIMEOUT) -> bool:
    """Poll until the guest agent responds to a ping. Returns True if ready."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            s = _vsock_connect(vm.vsock_sock, 2)
            s.sendall(json.dumps({"code": "print('ping')", "timeout": 3}).encode() + b"\n")
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(1024)
                if not chunk:
                    break
                buf += chunk
            s.close()
            if json.loads(buf.decode()).get("exit_code") == 0:
                return True
        except (ConnectionRefusedError, OSError, json.JSONDecodeError, socket.timeout):
            time.sleep(1)
    return False


def send_code(vm: VMState, code: str, timeout_ms: int,
              memory_limit_mb: int = 0, cpu_limit_sec: int = 0) -> dict:
    """
    Send code to the guest VM for execution.
    Returns execution result dict: exit_code, wall_time_ms, cpu_user_ms,
    cpu_sys_ms, mem_peak_kb, stdout, stderr, timed_out, oom_killed.

    memory_limit_mb: guest ulimit -v in MB (0 = no limit)
    cpu_limit_sec:   guest ulimit -t in seconds (0 = no limit)
    """
    sock = _vsock_connect(vm.vsock_sock, timeout_ms / 1000 + 15)
    sock.settimeout(timeout_ms / 1000 + 15)
    payload = {"code": code, "timeout": timeout_ms // 1000}
    if memory_limit_mb > 0:
        payload["memory_limit_mb"] = memory_limit_mb
    if cpu_limit_sec > 0:
        payload["cpu_limit_sec"] = cpu_limit_sec
    sock.sendall((json.dumps(payload) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    sock.close()
    return json.loads(buf.decode())


# ── Resource control ──────────────────────────────────────────────────────────

def _setup_vm_cgroup(vm_id: str, pid: int) -> Optional[Path]:
    """
    Create a dedicated cgroup for the VM and move the process into it.
    Returns the cgroup path, or None if setup failed (cgroups won't be enforced).

    The VM inherits whatever cgroup the server runs in (e.g. a Chromium VS Code
    scope). We can't write cpu.max/memory.max there. Instead we create our own
    leaf cgroup under the user's delegated slice and move the VM process there.
    """
    uid = os.getuid()
    user_service = Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service")
    if not user_service.exists():
        print(f"[vm] WARNING: user service cgroup not found at {user_service}, resource limits disabled")
        return None

    fc_root  = user_service / "firecracker.slice"
    vm_cg    = fc_root / f"fc-{vm_id}"

    try:
        fc_root.mkdir(exist_ok=True)
        vm_cg.mkdir(exist_ok=True)
    except PermissionError as e:
        print(f"[vm] WARNING: cannot create cgroup dirs: {e}, resource limits disabled")
        return None

    # Enable cpu + memory controllers at each level (ignore if already set)
    for ctrl_file in (user_service / "cgroup.subtree_control",
                      fc_root / "cgroup.subtree_control"):
        try:
            ctrl_file.write_text("+cpu +memory\n")
        except (PermissionError, OSError):
            pass  # may already be enabled or parent doesn't support it

    # Move the VM process into the leaf cgroup
    try:
        (vm_cg / "cgroup.procs").write_text(str(pid))
    except PermissionError as e:
        print(f"[vm] WARNING: cannot move pid {pid} into cgroup: {e}, resource limits disabled")
        try:
            vm_cg.rmdir()
        except OSError:
            pass
        return None

    return vm_cg


def _teardown_vm_cgroup(cgroup_path: Optional[Path]) -> None:
    """Remove the VM's cgroup directory after the process has exited."""
    if cgroup_path is None:
        return
    try:
        cgroup_path.rmdir()
    except OSError:
        pass
    try:
        cgroup_path.parent.rmdir()  # fc root — only succeeds when empty
    except OSError:
        pass


def apply_cgroups(vm: VMState, cpu_millicores: int, memory_limit_mb: int) -> None:
    """
    Apply CPU and memory limits to the VM's dedicated cgroup (cgroups v2).
    No-ops gracefully if no cgroup was set up (logs a warning instead of crashing).
    """
    base = vm.cgroup_path
    if base is None:
        print(f"[vm] WARNING: no cgroup for VM {vm.vm_id}, skipping resource limits")
        return

    period_us = 100_000
    quota_us  = int(cpu_millicores / 1000 * period_us)
    (base / "cpu.max").write_text(f"{quota_us} {period_us}\n")
    (base / "memory.max").write_text(f"{memory_limit_mb * 1024 * 1024}\n")
