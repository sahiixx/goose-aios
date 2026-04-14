FROM python:3.12-slim

WORKDIR /app

# System deps for playwright browsers (optional, for web browsing tool)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY agent.py server.py start.bat ./
COPY static/ static/
COPY config/ config/
COPY tools/ tools/
COPY core/ core/

# Create runtime directories
RUN mkdir -p conversations knowledge memory/episodes .external

# Default env vars
ENV AIOS_API_KEY="" \
    OLLAMA_HOST="http://host.docker.internal:11434"

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8765/api/health || exit 1

CMD ["python", "server.py"]
