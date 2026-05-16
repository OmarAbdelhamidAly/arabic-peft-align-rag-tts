FROM vllm/vllm-openai:v0.6.3

# Copy custom configuration if needed
COPY app/ /app/

EXPOSE 8080

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--model", "${MODEL_ID}", \
     "--max-model-len", "${MAX_MODEL_LEN}", \
     "--max-num-seqs", "${MAX_NUM_SEQS}", \
     "--dtype", "bfloat16", \
     "--disable-log-requests"]
