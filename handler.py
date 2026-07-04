"""
RunPod Serverless handler for Artemis-31B-v1i (Q8_0 GGUF) via llama.cpp.

Speaks the EXACT contract vaelico-web/app expects (see app/src/routes/chat.js):
  input  = { "messages": [...], "sampling_params": {...}, "stream": bool }
  output (non-stream) = { "text": str, "usage": {"prompt_tokens","completion_tokens"} }
  output (stream)     = a sequence of raw string pieces (one yield per token chunk)

The gateway calls /runsync (stream:false) -> aggregated list -> unwrap()[0].text,
and /run + /stream (stream:true) -> each yielded string is item.output.

Defensive extras (learned the hard way): print tracebacks and sleep before dying
so RunPod's log collector captures the cause of a fast crash.
"""
import os
import sys
import time
import json
import subprocess
import traceback

import requests
import runpod

LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
BASE = f"http://127.0.0.1:{LLAMA_PORT}"
MODEL_REPO = os.environ.get("MODEL_REPO", "BeaverAI/Artemis-31B-v1i-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "Artemis-31B-v1i-Q8_0.gguf")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
CTX = os.environ.get("CTX_SIZE", "32768")
NGL = os.environ.get("GPU_LAYERS", "999")
PARALLEL = os.environ.get("PARALLEL", "4")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def download_model():
    from huggingface_hub import hf_hub_download
    os.makedirs(MODEL_DIR, exist_ok=True)
    log(f"[boot] downloading {MODEL_REPO}/{MODEL_FILE} -> {MODEL_DIR}")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=MODEL_DIR,
        token=os.environ.get("HF_TOKEN"),
    )
    log(f"[boot] model ready at {path}")
    return path


def start_server(model_path):
    cmd = [
        "llama-server", "-m", model_path,
        "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
        "-ngl", NGL, "-c", CTX,
        "--parallel", PARALLEL, "--cont-batching",
        "--flash-attn",  # Gemma-4 sliding-window: cheap KV, keep flash attn on
    ]
    log("[boot] launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    for _ in range(900):  # up to ~15 min for a cold weight load
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                log("[boot] llama-server healthy")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode})")
        time.sleep(1)
    raise RuntimeError("llama-server never became healthy")


# --- boot once at module load (before RunPod starts handing us jobs) ---
try:
    _model_path = download_model()
    _server = start_server(_model_path)
except Exception:
    log("[boot] FATAL:")
    log(traceback.format_exc())
    time.sleep(60)  # let the log collector catch it before the worker dies
    raise


def _payload(inp):
    sp = inp.get("sampling_params", {}) or {}
    p = {
        "messages": inp.get("messages", []),
        "max_tokens": sp.get("max_tokens", 1024),
        "temperature": sp.get("temperature", 0.8),
        # TheDrummer's recommended default for Artemis; harmless if overridden.
        "repeat_penalty": sp.get("repeat_penalty", 1.1),
    }
    for k in ("top_p", "seed", "stop"):
        if sp.get(k) is not None:
            p[k] = sp[k]
    return p


def handler(job):
    inp = job.get("input", {}) or {}
    payload = _payload(inp)
    try:
        if inp.get("stream"):
            payload["stream"] = True
            with requests.post(f"{BASE}/v1/chat/completions", json=payload,
                               stream=True, timeout=900) as r:
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
                        yield piece  # gateway reads this as item.output (string)
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
        log("[handler] generation error:")
        log(traceback.format_exc())
        yield {"error": "generation failed on the worker"}


# return_aggregate_stream: /runsync returns the list of yields, so a single
# non-stream yield becomes output=[{text,usage}] -> gateway's unwrap()[0]. Match.
runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
