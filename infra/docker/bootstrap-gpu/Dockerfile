FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      cmake \
      curl \
      git \
      jq \
      ninja-build \
      openmpi-bin \
      libopenmpi-dev \
      pciutils \
      pkg-config \
      python3 \
      python3-pip \
      python3-venv \
      lsb-release \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY infra/docker/bootstrap-gpu/entrypoint.sh /usr/local/bin/pyc-bootstrap-entrypoint
RUN chmod +x /usr/local/bin/pyc-bootstrap-entrypoint

ENTRYPOINT ["/usr/local/bin/pyc-bootstrap-entrypoint"]
CMD ["bash", "infra/bootstrap_gpu_box.sh"]
