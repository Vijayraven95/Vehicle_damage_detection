# Use official python image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for Detectron2, PyTorch, and OpenCV
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libpq-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire current directory into the container
COPY . /app

# Install PyTorch and Torchvision (for CUDA 11.8)
RUN pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install Detectron2 from GitHub
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@ebe8b45437f86395352ab13402ba45b75b4d1ddb'

# Install remaining Python dependencies manually
RUN pip install fastapi==0.115.0 \
    GitPython==3.1.43 \
    matplotlib==3.7.5 \
    opencv-python==4.10.0.84 \
    pandas==2.0.3 \
    pillow==10.4.0 \
    python-multipart==0.0.12 \
    streamlit==1.39.0 \
    tensorboard==2.14.0 \
    tensorboard-data-server==0.7.2 \
    uvicorn==0.31.0

# Expose the port for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Start both FastAPI and Streamlit when the container launches
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.port 8501 --server.enableCORS false"]
