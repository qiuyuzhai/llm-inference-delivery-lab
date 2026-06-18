FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY gateway ./gateway

EXPOSE 8080
CMD ["uvicorn", "gateway.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
