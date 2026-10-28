# The air-defence demo server with Typical's recorded answers (record.py): no model, no GPU.
# Build context: airdef/ (wrangler.jsonc image_build_context).
FROM python:3.12-slim
RUN pip install --no-cache-dir "fastapi==0.115.*" "uvicorn==0.32.*" "numpy==2.*" "pydantic==2.*"
WORKDIR /app/airdef
COPY *.py ./
COPY web ./web
COPY brand ./brand
COPY recorded ./recorded
ENV AIRDEF_MODEL=recorded PYTHONUNBUFFERED=1
EXPOSE 8790
CMD ["uvicorn", "--app-dir", "/app/airdef", "server:app", "--host", "0.0.0.0", "--port", "8790"]
