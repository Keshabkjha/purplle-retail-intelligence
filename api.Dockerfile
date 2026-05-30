FROM python:3.10-slim

WORKDIR /workspace

COPY api-requirements.txt .
RUN pip install --no-cache-dir -r api-requirements.txt

COPY . .

# Cloud Run sets the PORT environment variable
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
