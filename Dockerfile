FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

COPY pyproject.toml ./
COPY src ./src
# Editable install keeps PROJECT_ROOT pointing at /app, where prompts/ and config/ live.
RUN pip install --no-cache-dir -e .

COPY prompts ./prompts
COPY config ./config
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "ragapp.api:app", "--host", "0.0.0.0", "--port", "8000"]
