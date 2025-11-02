FROM node:20-bullseye

# 開発ツール
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl git vim iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# GitHub Copilot CLI
RUN npm install -g @githubnext/github-copilot-cli && \
    ln -s /usr/local/bin/github-copilot-cli /usr/local/bin/copilot

# DNS（通信安定化）
RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf

WORKDIR /workspace
CMD ["/bin/bash"]
