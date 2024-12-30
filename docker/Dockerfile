# Start with a lightweight Python image
FROM python:3.9-slim

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    libpq-dev \
    && apt-get clean

# Install MLflow and its dependencies
RUN pip install --no-cache-dir \
    mlflow \
    psycopg2-binary \
    boto3

# Expose the port that MLflow will run on
EXPOSE 5000

# Default command to run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
