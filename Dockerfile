FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download both models at build time so the container starts without network access.
COPY belanjawan/ belanjawan/
RUN python -c "from belanjawan.index import embed_query; embed_query('warm up')" \
 && python -c "from belanjawan.retrieve import get_ranker; get_ranker()"

COPY index/ index/
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/v1/health')"
CMD ["sh", "-c", "uvicorn belanjawan.api:app --host 0.0.0.0 --port ${PORT}"]
