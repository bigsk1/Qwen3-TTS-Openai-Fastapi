# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS OpenAI-Compatible FastAPI Server.

A high-performance TTS API server providing OpenAI-compatible endpoints
for the Qwen3-TTS model.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8880"))
WORKERS = int(os.getenv("WORKERS", "1"))

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Get the directory containing static files
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization."""
    
    # Print startup banner
    boundary = "░" * 24
    startup_msg = f"""
{boundary}

    ╔═╗┬ ┬┌─┐┌┐┌╔═╗  ╔╦╗╔╦╗╔═╗
    ║═╬╡│││├┤ │││╚═╗───║  ║ ╚═╗
    ╚═╝└┴┘└─┘┘└┘╚═╝   ╩  ╩ ╚═╝
    
    OpenAI-Compatible TTS API

{boundary}
"""
    logger.info(startup_msg)
    logger.info(f"Server starting on http://{HOST}:{PORT}")
    logger.info(f"API Documentation: http://{HOST}:{PORT}/docs")
    logger.info(f"Web Interface: http://{HOST}:{PORT}/")
    logger.info(boundary)
    
    # Pre-load the TTS model (optional, can be lazy loaded)
    try:
        from .routers.openai_compatible import get_tts_model
        logger.info("Pre-loading TTS model...")
        await get_tts_model()
        logger.info("TTS model loaded successfully!")
    except Exception as e:
        logger.info(f"Note: Model will be loaded on first request. ({e})")
    
    yield
    
    # Cleanup
    logger.info("Server shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Qwen3-TTS API",
    description="""
## Qwen3-TTS OpenAI-Compatible API

A high-performance text-to-speech API server powered by Qwen3-TTS, 
providing full compatibility with OpenAI's TTS API specification.

### Features
- 🎯 OpenAI API compatible endpoints
- 🌍 Multi-language support (10+ languages)
- 🎨 Multiple voice options
- 📊 Multiple audio formats (MP3, Opus, AAC, FLAC, WAV, PCM)
- ⚡ GPU-accelerated inference
- 🔧 Text normalization and sanitization

### Quick Start
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="qwen3-tts",
    voice="Vivian",
    input="Hello! This is Qwen3-TTS speaking."
)
response.stream_to_file("output.mp3")
```
""",
    version="0.1.0",
    lifespan=lifespan,
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from .routers.openai_compatible import router as openai_router
app.include_router(openai_router, prefix="/v1")

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    # Return a simple HTML page if index.html doesn't exist
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Qwen3-TTS API</title>
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background: #1a1a2e; 
            color: #eee; 
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
        }
        pre { color: #00ff88; }
        a { color: #00aaff; }
        h1 { color: #fff; }
    </style>
</head>
<body>
    <pre>
    ╔═╗┬ ┬┌─┐┌┐┌╔═╗  ╔╦╗╔╦╗╔═╗
    ║═╬╡│││├┤ │││╚═╗───║  ║ ╚═╗
    ╚═╝└┴┘└─┘┘└┘╚═╝   ╩  ╩ ╚═╝
    </pre>
    <h1>Qwen3-TTS OpenAI-Compatible API</h1>
    <p>Welcome to the Qwen3-TTS API server!</p>
    <ul>
        <li><a href="/docs">API Documentation (Swagger UI)</a></li>
        <li><a href="/redoc">API Documentation (ReDoc)</a></li>
        <li><a href="/v1/models">List Models</a></li>
        <li><a href="/v1/voices">List Voices</a></li>
    </ul>
</body>
</html>
"""


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Run the server using uvicorn."""
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        reload=False,
    )


if __name__ == "__main__":
    main()
