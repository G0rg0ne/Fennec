# Use NVIDIA CUDA-enabled base image with Python 3.9
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install Python 3.9 and necessary tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    gcc \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set a working directory inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Expose the default MLflow port
EXPOSE 5000
