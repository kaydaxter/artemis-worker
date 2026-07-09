"""Worker RunPod serverless: Artemis-31B-v1i (Gemma 4) via vLLM 0.24. v4.

Base: el patron probado de qwen35-worker-vllm v3 (boot bloqueante, logs con
flush, fatal() que duerme 60s para que el log se capture, higiene: cero
contenido de usuario en logs). Cambios v4 para Artemis:

- SAMPLERS: passthrough del kit que vLLM implementa (top_p, top_k, min_p,
  presence/frequency/repetition_penalty, seed, stop, logit_bias) aceptando
  ademas el dialecto llama.cpp del gateway (repeat_penalty ->
  repetition_penalty; top_k 0 = desactivado se omite). Los exoticos de
  llama.cpp (DRY/XTC/mirostat) se descartan aqui: vLLM no los tiene.
- PREFILL: si el ultimo mensaje es del assistant ("Start Reply With" de
  SillyTavern), se manda add_generation_prompt=false +
  continue_final_message=true para que vLLM CONTINUE ese turno en vez de
  abrir uno nuevo (llama-server lo hacia solo; vLLM lo pide explicito).

Contrato con la app (app/src/routes/chat.js), sin cambios:
input {messages|prompt, sampling_params, stream} ->
non-stream: {text, usage} · stream: piezas de texto crudas.
"""
import json
import os
import shlex
import subprocess
import sys
import time
import traceback

print("[worker] handler.py v4 (vLLM) arrancando (python OK)", flush=True)


def fatal(msg, exc=True):
    print(f"[worker][FATAL] {msg}", flush=True)
    if exc:
        traceback.print_exc()
    print("[worker] durmiendo 60s para que el log se capture...", flush=True)
    time.sleep(60)
    sys.exit(1)


try:
    import urllib.request
    import urllib.error
    import runpod
    print("[worker] imports OK (runpod %s)" % getattr(runpod, "__version__", "?"), flush=True)
except Exception:
    fatal("fallo importando dependencias")

MODEL = os.environ.get("MODEL_NAME", "TigerKay/Artemis-31B-v1i-fp8")
PORT = "8000"
BASE = f"http://127.0.0.1:{PORT}"

cmd = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL,
    "--served-model-name", "artemis",
    "--host", "127.0.0.1", "--port", PORT,
    "--dtype", os.environ.get("DTYPE", "bfloat16"),
    "--max-model-len", os.environ.get("MAX_MODEL_LEN", "65536"),
    # 0.95: con 0.93 el KV en 48GB queda por debajo de lo que exige una
    # peticion de 64k y vLLM 0.24 no arranca (medido en el pod L40S).
    "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"),
]
extra = os.environ.get("VLLM_EXTRA_ARGS", "").strip()
if extra:
    try:
        cmd += shlex.split(extra)
    except Exception:
        fatal(f"VLLM_EXTRA_ARGS no parsea: {extra!r}")

print("[worker] lanzando vLLM (modelo: %s)" % MODEL, flush=True)
t0 = time.time()
try:
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
except Exception:
    fatal("Popen de vllm fallo")

while True:
    rc = proc.poll()
    if rc is not None:
        fatal(f"vllm murio en el arranque con exit code {rc} "
              f"(a los {time.time()-t0:.0f}s) — su error debe estar arriba", exc=False)
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    el = int(time.time() - t0)
    if el and el % 60 < 2:
        print(f"[worker] esperando a vLLM... {el}s", flush=True)
    time.sleep(2)
print(f"[worker] vLLM LISTO en {time.time()-t0:.1f}s", flush=True)


def _req(path, body, timeout=900):
    return urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")


# Claves numericas float que vLLM acepta tal cual en /v1/chat/completions.
_FLOAT_KEYS = ("top_p", "min_p", "presence_penalty", "frequency_penalty",
               "repetition_penalty")


def _build_body(inp):
    sp = inp.get("sampling_params", {}) or {}
    body = {
        "model": "artemis",
        "max_tokens": int(sp.get("max_tokens", 256)),
        "temperature": float(sp.get("temperature", 0.7)),
    }
    for k in _FLOAT_KEYS:
        if sp.get(k) is not None:
            body[k] = float(sp[k])
    # Dialecto llama.cpp del gateway: repeat_penalty -> repetition_penalty.
    if sp.get("repeat_penalty") is not None and "repetition_penalty" not in body:
        body["repetition_penalty"] = float(sp["repeat_penalty"])
    # top_k: llama.cpp usa 0 = off; vLLM exige >=1 (su off es -1) -> se omite.
    if sp.get("top_k") is not None and int(sp["top_k"]) >= 1:
        body["top_k"] = int(sp["top_k"])
    if sp.get("seed") is not None:
        body["seed"] = int(sp["seed"])
    if sp.get("stop") is not None:
        body["stop"] = sp["stop"]
    if sp.get("logit_bias") is not None:
        body["logit_bias"] = sp["logit_bias"]

    if "messages" in inp:
        msgs = inp["messages"]
        body["messages"] = msgs
        # Prefill de assistant (Start Reply With): continuar el turno final.
        if msgs and (msgs[-1] or {}).get("role") == "assistant":
            body["add_generation_prompt"] = False
            body["continue_final_message"] = True
        return "/v1/chat/completions", body, True
    if "prompt" in inp:
        return "/v1/completions", {**body, "prompt": inp["prompt"]}, False
    return None, None, None


def _iter_sse(path, body, chat):
    body = {**body, "stream": True}
    with urllib.request.urlopen(_req(path, body), timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                ch = json.loads(payload)["choices"][0]
                delta = (ch.get("delta", {}).get("content") if chat
                         else ch.get("text")) or ""
            except Exception:
                continue
            if delta:
                yield delta


def handler(job):
    """Generador (RunPod detecta streaming por is_generator(handler)).
    stream=True -> yield de deltas de texto.
    stream=False -> yield unico con el resultado completo (output = [dict])."""
    inp = (job or {}).get("input", {}) or {}
    path, body, chat = _build_body(inp)
    if path is None:
        yield {"error": "input necesita 'messages' o 'prompt'"}
        return
    try:
        if inp.get("stream"):
            for delta in _iter_sse(path, body, chat):
                yield delta
        else:
            with urllib.request.urlopen(_req(path, body), timeout=900) as r:
                out = json.loads(r.read().decode())
            text = (out["choices"][0]["message"]["content"] if chat
                    else out["choices"][0]["text"])
            yield {"text": text, "usage": out.get("usage")}
    except urllib.error.HTTPError as e:
        yield {"error": f"HTTP {e.code}", "body": e.read().decode()[:800]}
    except Exception as e:
        yield {"error": f"{type(e).__name__}: {e}"}


try:
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
except Exception:
    fatal("runpod.serverless.start fallo")
