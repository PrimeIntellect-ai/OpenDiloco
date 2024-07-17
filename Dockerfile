FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
LABEL maintainer="prime intellect"
LABEL repository="open_diloco"

# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Set CUDA_HOME and update PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:/usr/local/cuda/bin

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
  build-essential \
  curl \
  wget \
  git \
  vim \
  htop \
  nvtop \
  iperf \
  tmux \
  openssh-server \   
  git-lfs \   
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Git LFS
RUN git-lfs install

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN echo "export PATH=\"/opt/conda/bin:/root/.cargo/bin:\$PATH\"" >> /root/.bashrc

# Install Python dependencies (The gradual copies help with caching)
WORKDIR open_diloco
RUN pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu
RUN pip install flash-attn>=2.5.8
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY requirements-dev.txt requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
RUN pip install .
RUN rm -rf ~/.cache/pip
