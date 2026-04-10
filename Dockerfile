FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy core logic
COPY models.py .
COPY client.py .
COPY openenv.yaml .
COPY server/ ./server/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Expose the API port
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]