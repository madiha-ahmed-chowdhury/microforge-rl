#!/usr/bin/env python3
"""
server.py — FastAPI REST server for Firecracker RL data collection and inference.

Endpoints:
  POST /run/one        — run a single random instance (boots VM, calls LLM, saves results)
  POST /run/specific   — run a specific instance by idx
  POST /run/all        — run the full dataset in background
  GET  /status         — collection progress + live VMs

  POST   /vm/create            — boot a VM for inference
  DELETE /vm/{vm_id}           — shut it down
  POST   /vm/{vm_id}/execute   — run code in VM
  PATCH  /vm/{vm_id}/resources — apply cgroups v2 resource limits
  GET    /vm/{vm_id}/metrics   — last execution result
  GET    /vm/{vm_id}/status    — VM alive/uptime

Run:
    venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
"""

import dataclasses
import datetime
import json
import os
import random
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from actions import (
    ACTION_CONFIGS, OUTPUT_CODE_PAIRS, OUTPUT_TRANSITIONS,
    dataset_size, load_all, load_one, process_instance,
)
from vm import VMState, apply_cgroups, send_code, start_vm, stop_vm, wait_for_agent  # used by /vm/* inference endpoints

# ── Collection mode ───────────────────────────────────────────────────────────
# True  → fresh VM per action config (isolated, matches collecting_dataset.py)
# False → one shared VM, cgroups mutated between runs (faster, less isolated)
FRESH_VM_PER_ACTION = True

# ── How many instances to process by default ─────────────────────────────────
#    Change this freely. Set to None to always run the full dataset.
DEFAULT_N = 100

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request models
# ─────────────────────────────────────────────────────────────────────────────

class RunOneRequest(BaseModel):
    dataset: str = "mbpp"
    idx: Optional[int] = None   # None → random

class RunAllRequest(BaseModel):
    dataset: str = "mbpp"
    n: Optional[int] = None     # None → DEFAULT_N; 0 → full dataset
    offset: int = 0

class ExecuteRequest(BaseModel):
    code: str
    timeout_ms: int = 10_000

class ResourcePatch(BaseModel):
    cpu_millicores: int
    memory_limit_mb: int

# ─────────────────────────────────────────────────────────────────────────────
# VM registry
# ─────────────────────────────────────────────────────────────────────────────

vms: dict[str, VMState] = {}

# ─────────────────────────────────────────────────────────────────────────────
# Collection state (background run/all)
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class CollectionState:
    running: bool = False
    dataset: str = ""
    total: int = 0
    done: int = 0
    transitions_written: int = 0
    error: str = ""
    started_at: str = ""
    finished_at: str = ""

_collect      = CollectionState()
_collect_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Background collection loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_all(dataset: str, n: Optional[int], offset: int):
    with _collect_lock:
        _collect.running = True
        _collect.dataset = dataset
        _collect.error = ""
        _collect.done = 0
        _collect.transitions_written = 0
        _collect.started_at = datetime.datetime.utcnow().isoformat() + "Z"
        _collect.finished_at = ""

    try:
        instances = load_all(dataset, n, offset)
        with _collect_lock:
            _collect.total = len(instances)
        print(f"[server/run-all] {len(instances)} instances from {dataset}")
        print(f"[server/run-all] fresh_vm_per_action={FRESH_VM_PER_ACTION}")

        written = 0
        for i, instance in enumerate(instances):
            print(f"[server/run-all] {i+1}/{len(instances)} | {instance['task_id']}")
            try:
                result = process_instance(instance, fresh_vm_per_action=FRESH_VM_PER_ACTION)
            except Exception as e:
                print(f"[server/run-all] error on {instance['task_id']}: {e} — skipping")
                with _collect_lock:
                    _collect.done += 1
                continue

            with open(OUTPUT_TRANSITIONS, "a") as f:
                for t in result["transitions"]:
                    f.write(json.dumps(t) + "\n")
            with open(OUTPUT_CODE_PAIRS, "a") as f:
                f.write(json.dumps(result["code_pair"]) + "\n")

            written += len(result["transitions"])
            with _collect_lock:
                _collect.transitions_written = written
                _collect.done = i + 1

    except Exception as e:
        print(f"[server/run-all] FATAL: {e}")
        with _collect_lock:
            _collect.error = str(e)
    finally:
        with _collect_lock:
            _collect.running = False
            _collect.finished_at = datetime.datetime.utcnow().isoformat() + "Z"
        print(f"[server/run-all] Done. {_collect.transitions_written} transitions written.")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Firecracker RL Server")

# ── /run/* ────────────────────────────────────────────────────────────────────

@app.post("/run/one")
def run_one(req: RunOneRequest):
    """
    Run a single instance through LLM + VM synchronously.
    idx=null → random. Returns transitions + code pair immediately.
    Saves to rl_transitions.jsonl and rl_code_pairs.jsonl.
    """
    if req.dataset not in ("mbpp", "humaneval"):
        raise HTTPException(400, "dataset must be 'mbpp' or 'humaneval'")

    idx = req.idx if req.idx is not None else random.randint(0, dataset_size(req.dataset) - 1)
    instance = load_one(req.dataset, idx)

    try:
        result = process_instance(instance, fresh_vm_per_action=FRESH_VM_PER_ACTION)
    except Exception as e:
        raise HTTPException(500, str(e))

    with open(OUTPUT_TRANSITIONS, "a") as f:
        for t in result["transitions"]:
            f.write(json.dumps(t) + "\n")
    with open(OUTPUT_CODE_PAIRS, "a") as f:
        f.write(json.dumps(result["code_pair"]) + "\n")

    return {
        "idx":         idx,
        "task_id":     instance["task_id"],
        "dataset":     instance["dataset"],
        "prompt":      instance["prompt"],
        "code_pair":   result["code_pair"],
        "transitions": result["transitions"],
    }


@app.post("/run/specific")
def run_specific(req: RunOneRequest):
    """
    Run a specific instance by dataset + idx.
    idx is required here.
    """
    if req.idx is None:
        raise HTTPException(400, "idx is required for /run/specific")
    return run_one(req)


@app.post("/run/all")
def run_all(req: RunAllRequest):
    """
    Run the full (or partial) dataset in the background.
    n=null → DEFAULT_N. n=0 → entire dataset. offset → start index.
    Poll /status for progress.
    """
    with _collect_lock:
        if _collect.running:
            raise HTTPException(409, "Collection already running")
    if req.dataset not in ("mbpp", "humaneval"):
        raise HTTPException(400, "dataset must be 'mbpp' or 'humaneval'")

    n = req.n if req.n is not None else DEFAULT_N
    n = None if n == 0 else n

    t = threading.Thread(target=_run_all, args=(req.dataset, n, req.offset), daemon=True)
    t.start()
    return {
        "status":         "started",
        "dataset":        req.dataset,
        "n":              n or "all",
        "offset":         req.offset,
        "action_configs": len(ACTION_CONFIGS),
        "output":         OUTPUT_TRANSITIONS,
    }


@app.get("/status")
def status():
    """Collection progress + list of currently live VMs."""
    with _collect_lock:
        return {
            "collection": {
                "running":             _collect.running,
                "dataset":             _collect.dataset,
                "instances_total":     _collect.total,
                "instances_done":      _collect.done,
                "transitions_written": _collect.transitions_written,
                "error":               _collect.error or None,
                "started_at":          _collect.started_at or None,
                "finished_at":         _collect.finished_at or None,
            },
            "live_vms": list(vms.keys()),
        }

# ── /vm/* (inference phase) ───────────────────────────────────────────────────

@app.post("/vm/create")
def vm_create():
    """Boot a new VM for inference. Returns vm_id."""
    vm_id = uuid.uuid4().hex[:8]
    vm = start_vm(vm_id)
    vms[vm_id] = vm
    time.sleep(3)
    if not wait_for_agent(vm):
        stop_vm(vm)
        vms.pop(vm_id, None)
        raise HTTPException(500, "Guest agent did not start")
    return {"vm_id": vm_id, "status": "ready"}


@app.delete("/vm/{vm_id}")
def vm_delete(vm_id: str):
    vm = vms.pop(vm_id, None)
    if vm is None:
        raise HTTPException(404, f"VM {vm_id!r} not found")
    stop_vm(vm)
    return {"vm_id": vm_id, "status": "stopped"}


@app.post("/vm/{vm_id}/execute")
def vm_execute(vm_id: str, req: ExecuteRequest):
    """Send code to a VM and return execution metrics."""
    vm = vms.get(vm_id)
    if vm is None:
        raise HTTPException(404, f"VM {vm_id!r} not found")
    try:
        result = send_code(vm, req.code, req.timeout_ms)
    except Exception as e:
        raise HTTPException(500, str(e))
    vm.last_execution = result
    return result


@app.patch("/vm/{vm_id}/resources")
def vm_patch_resources(vm_id: str, req: ResourcePatch):
    """Apply CPU/memory limits via cgroups v2 (inference phase)."""
    vm = vms.get(vm_id)
    if vm is None:
        raise HTTPException(404, f"VM {vm_id!r} not found")
    try:
        apply_cgroups(vm, req.cpu_millicores, req.memory_limit_mb)
    except Exception as e:
        raise HTTPException(500, f"cgroups error: {e}")
    return {"vm_id": vm_id, "cpu_millicores": req.cpu_millicores,
            "memory_limit_mb": req.memory_limit_mb, "status": "applied"}


@app.get("/vm/{vm_id}/metrics")
def vm_metrics(vm_id: str):
    """Return the last execution result for this VM."""
    vm = vms.get(vm_id)
    if vm is None:
        raise HTTPException(404, f"VM {vm_id!r} not found")
    if vm.last_execution is None:
        raise HTTPException(404, "No execution yet for this VM")
    return vm.last_execution


@app.get("/vm/{vm_id}/status")
def vm_status(vm_id: str):
    vm = vms.get(vm_id)
    if vm is None:
        raise HTTPException(404, f"VM {vm_id!r} not found")
    return {
        "vm_id":      vm_id,
        "alive":      vm.proc.poll() is None,
        "pid":        vm.proc.pid,
        "uptime_sec": round(time.monotonic() - vm.created_at, 1),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
