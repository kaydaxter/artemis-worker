# Worker RunPod serverless de Artemis sobre vLLM 0.24 oficial (fix del bug de
# sliding-window KV de gemma-4 que colgaba >=2 peticiones largas en 0.19).
# Sustituye a la imagen llama.cpp: el modelo ya no es GGUF sino el checkpoint
# dequantizado a safetensors (HF privado -> requiere secret HF_TOKEN).
# En Ada el FP8 es nativo; en Ampere vLLM cae a kernels Marlin automaticamente.
#
# Config por env del template (sin rebuild): MODEL_NAME, DTYPE, MAX_MODEL_LEN,
# GPU_MEMORY_UTILIZATION, VLLM_EXTRA_ARGS.
FROM vllm/vllm-openai:v0.24.0

ENTRYPOINT []

# logs del SDK de RunPod sin contenido de jobs (higiene: prompts de usuario
# fuera de los logs de la consola). Sobreescribible por env del template.
ENV RUNPOD_DEBUG_LEVEL=WARN

RUN pip install --no-cache-dir runpod \
 && python3 -c "import runpod; print('[build] runpod OK')"

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
