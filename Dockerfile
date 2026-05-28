FROM nvidia/cuda:12.4.0-base-ubuntu22.04

WORKDIR /app

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# System deps for Python and image libs
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
       python3-pip python3-dev build-essential \
       libgl1 libglib2.0-0 \
       tesseract-ocr tesseract-ocr-tha tesseract-ocr-chi-sim tesseract-ocr-chi-tra \
    && rm -rf /var/lib/apt/lists/*

# Ensure CUDA compat libs are visible
RUN ldconfig /usr/local/cuda-12.4/compat/

# Install CUDA-enabled PyTorch first to avoid CPU wheels pulled by downstream deps
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Then install the rest
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    READER_LANGS=ch_sim,en

CMD ["python3", "-u", "handler.py"]
