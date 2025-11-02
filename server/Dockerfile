# ===== Node + Python 統合イメージ =====
FROM node:20-bullseye

# Install OS utilities + Python
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl git vim iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install Copilot CLI
RUN npm install -g @githubnext/github-copilot-cli \
    && ln -s /usr/local/bin/github-copilot-cli /usr/local/bin/copilot

# Set working directory
WORKDIR /app

# Copy Python requirements and install
COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy entire server code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command: start FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
