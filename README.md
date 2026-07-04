# Artemis-31B serverless worker (llama.cpp / GGUF)

Serves `BeaverAI/Artemis-31B-v1i-GGUF` (Q8_0) on **RunPod Serverless** and speaks
the exact job contract `vaelico-web/app` expects — so the gateway needs **no
changes**, only a `models` row pointing at this endpoint.

## Why this stack
- Artemis ships **only as GGUF** → llama.cpp, not vLLM.
- llama.cpp is where TheDrummer's testers validate Artemis (matches §3.6).
- Sidesteps the Gemma-4 fp8-dynamic gibberish bug in vLLM (#39049).

## The contract (already implemented in handler.py)
- **input**: `{ "messages": [...], "sampling_params": { max_tokens, temperature, top_p?, seed?, stop? }, "stream": bool }`
- **non-stream** → `{ "text": "...", "usage": { prompt_tokens, completion_tokens } }`
- **stream** → one raw string piece per yield (gateway reads `item.output`)

## Sizing
| Quant | File | GPU |
|---|---|---|
| **Q8_0** (this) | 30.4 GB | **48 GB** (L40S / RTX 6000 Ada) |
| Q6_K | 23.5 GB | 32–48 GB |
| Q4_K_M | 17.4 GB | 24 GB (cheaper, set `MODEL_FILE`) |

Gemma-4 sliding-window ⇒ tiny KV, so `CTX_SIZE=32768` (or higher) is cheap.

## Deploy

### 1. Build the image (your GitHub Actions → GHCR)
Put this folder in the build repo (e.g. `kaydaxter/qwen35-worker-vllm` or a new
`artemis-worker`), let Actions build & push to GHCR, then reference the **digest**
(RunPod caches `:latest`, so pin the immutable `@sha256:...`).

### 2. Create the RunPod Serverless endpoint
- Image: the GHCR digest from step 1.
- GPU: **48 GB** (L40S / RTX 6000 Ada / A6000).
- Volume: none (multi-DC, like the gemma worker).
- Env vars: `MODEL_FILE` (default Q8_0), `CTX_SIZE`, `PARALLEL`.
- **Secret**: `HF_TOKEN` (only if the repo is private).
- Workers: `min=0` (scale-to-zero) or `min=1` (hot, no cold start).
- `idleTimeout` ~120s, `executionTimeout` 900s.

> **Cold start**: first request downloads 30 GB → minutes. For testers, either set
> `min=1` (a hot worker) or accept a slow first message. To kill it entirely, bake
> the GGUF into the image later (bigger image, faster boot).

### 3. Register the model in the app's D1
`artemis-31b` is already seeded (`app/seed.sql`) with `free_allowed=0`. Point it at
the new endpoint:
```sql
UPDATE models SET endpoint_id = '<runpod_endpoint_id>', enabled = 1 WHERE id = 'artemis-31b';
```
Testers connect via the app (SillyTavern → `https://api.vaelico.ai/v1`, key `vk_...`,
model `artemis-31b`). The RunPod key never leaves the Worker.

## Not done here (needs your creds)
- Build/push (your GitHub) · RunPod endpoint create (your key, can be scripted) ·
  app deploy to Cloudflare (your `wrangler login`).
