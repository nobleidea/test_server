# base image with cuda 12.1
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y wget curl git vim sudo cmake build-essential \
    libssl-dev libffi-dev python3-dev python3-venv python3-pip libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip3 install packaging psutil pexpect ipywidgets jupyterlab ipykernel \
    librosa soundfile

# upgrade pip
RUN pip3 install --upgrade pip


# install remaining dependencies from PyPI
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# install torch with cuda support
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu121


# copy files
COPY download_weights.py schemas.py handler.py test_input.json /

# ARG for HF token - only available during build
ARG HF_TOKEN

# download the weights from hugging face
# Use the token during download, then unset it
RUN if [ -n "$HF_TOKEN" ]; then \
        export HF_TOKEN=$HF_TOKEN && \
        python3 /download_weights.py && \
        unset HF_TOKEN; \
    else \
        echo "Warning: No HF_TOKEN provided. Attempting download without authentication..." && \
        python3 /download_weights.py; \
    fi

# run the handler
CMD python3 -u /handler.py