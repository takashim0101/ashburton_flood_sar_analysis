from fastapi import FastAPI, Request
import requests
import json

app = FastAPI(title="Ollama FastAPI Gateway")

OLLAMA_HOST = "http://host.docker.internal:11434"  # WindowsでのDocker連携用

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    model = data.get("model", "llama3:8b")

    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Ollama FastAPI Gateway running!"}
