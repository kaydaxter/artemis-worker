"""
RunPod Serverless handler for Artemis-31B-v1i (Q8_0 GGUF) via llama.cpp.

Speaks the contract vaelico-web/app expects (app/src/routes/chat.js):
  input  = { "messages":[...], "sampling_params":{...}, "stream":bool }
  output (non-stream) = { "text":str, "usage":{prompt_tokens,completion_tokens} }
  output (stream)     = raw string pieces (one yield per token chunk)

Boot (model download + llama-server) runs in a BACKGROUND THREAD so the worker
becomes healthy immediately instead of crash-looping. If boot fails, the first
job returns the traceback via /status — RunPod worker logs aren't reachable with
an API key (runpod-python#400), so this is how we surface the real error.
"""
import os
import sys
import time
import json
import shutil
import subprocess
import threading
import traceback

import requests
import runpod

LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
BASE = f"http://127.0.0.1:{LLAMA_PORT}"
MODEL_REPO = os.environ.get("MODEL_REPO", "BeaverAI/Artemis-31B-v1i-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "Artemis-31B-v1i-Q8_0.gguf")
CTX = os.environ.get("CTX_SIZE", "32768")
NGL = os.environ.get("GPU_LAYERS", "999")
PARALLEL = os.environ.get("PARALLEL", "4")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def download_model():
    from huggingface_hub import hf_hub_download
    log(f"[boot] downloading {MODEL_REPO}/{MODEL_FILE}")
    # No local_dir: keep a single copy in the HF cache (avoids two 30GB copies
    # that would blow the container disk).
    p = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, token=os.environ.get("HF_TOKEN"))
    log(f"[boot] model at {p}")
    return p


def start_server(model_path):
    binp = shutil.which("llama-server") or "/app/llama-server"
    cmd = [binp, "-m", model_path, "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
           "-ngl", NGL, "-c", CTX, "--parallel", PARALLEL, "--cont-batching"]
    log("[boot] launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    for _ in range(1200):
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                log("[boot] llama-server healthy")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        time.sleep(1)
    raise RuntimeError("llama-server never became healthy in time")


BOOT_ERROR = None
READY = threading.Event()


def _boot():
    global BOOT_ERROR
    try:
        start_server(download_model())
    except Exception:
        BOOT_ERROR = traceback.format_exc()
        log("[boot] FAILED:\n" + BOOT_ERROR)
    finally:
        READY.set()


# Boot in the background so serverless.start() runs immediately (worker healthy)
# and a boot failure is reported on the first job rather than crash-looping.
threading.Thread(target=_boot, daemon=True).start()


def _payload(inp):
    sp = inp.get("sampling_params", {}) or {}
    p = {
        "messages": inp.get("messages", []),
        "max_tokens": sp.get("max_tokens", 1024),
        "temperature": sp.get("temperature", 0.8),
        "repeat_penalty": sp.get("repeat_penalty", 1.1),
    }
    for k in ("top_p", "seed", "stop"):
        if sp.get(k) is not None:
            p[k] = sp[k]
    return p


def handler(job):
    READY.wait()
    if BOOT_ERROR:
        yield {"error": "worker boot failed", "traceback": BOOT_ERROR[-3500:]}
        return
    inp = job.get("input", {}) or {}
    payload = _payload(inp)
    try:
        if inp.get("stream"):
            payload["stream"] = True
            with requests.post(f"{BASE}/v1/chat/completions", json=payload, stream=True, timeout=900) as r:
                r.raise_for_status()
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        piece = json.loads(data)["choices"][0]["delta"].get("content", "")
                    except Exception:
                        continue
                    if piece:
                        yield piece
        else:
            r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=900)
            r.raise_for_status()
            obj = r.json()
            usage = obj.get("usage", {}) or {}
            yield {
                "text": obj["choices"][0]["message"]["content"],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
            }
    except Exception:
        log("[handler] error:\n" + traceback.format_exc())
        yield {"error": "generation failed on the worker"}


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
