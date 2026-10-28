"""
===============================================================================
Sanskrit Pronunciation Assessment — Web Application Server (FastAPI)
===============================================================================
Course  : CS30033 — Speech & Video Processing
Project : Sanskrit Pronunciation Assessment Engine
Target  : Render.com Cloud Web Service & Local Development
===============================================================================
"""

import os
import io
import sys
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure utf-8 output for logs
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from speech_engine import evaluate_audio_pair, SAMPLE_RATE

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
REF_DIR = os.path.join(BASE_DIR, "reference_audio")
DEFAULT_REF_PATH = os.path.join(REF_DIR, "reference_krsna.mp3")

app = FastAPI(
    title="Sanskrit Pronunciation Assessment API",
    description="Acoustic analysis using 7-step MFCC, DTW alignment, and LPC formant analysis",
    version="1.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the single-page application."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Sanskrit Pronunciation Assessment Service Active</h1>")


@app.get("/health")
async def health_check():
    """Health check endpoint for Render."""
    return {
        "status": "healthy",
        "service": "Sanskrit Pronunciation Assessment Engine",
        "sample_rate_hz": SAMPLE_RATE,
        "default_reference_available": os.path.exists(DEFAULT_REF_PATH)
    }


@app.get("/api/reference-audio")
async def get_reference_audio():
    """Stream the authentic Sanskrit reference audio ('kṛṣṇā') for browser playback."""
    if not os.path.exists(DEFAULT_REF_PATH):
        raise HTTPException(status_code=404, detail="Default reference audio not found.")
    return FileResponse(
        DEFAULT_REF_PATH,
        media_type="audio/mpeg",
        filename="reference_krsna.mp3"
    )


@app.post("/api/assess")
async def assess_pronunciation(
    test_file: UploadFile = File(..., description="Student or learner audio recording/file"),
    ref_file: Optional[UploadFile] = File(None, description="Optional custom reference/teacher audio file")
):
    """
    Evaluate pronunciation acoustic similarity between reference and test audio.
    Supports audio files (.wav, .mp3, .ogg, .flac, .m4a) and browser mic recordings.
    """
    try:
        # 1. Read test audio bytes
        test_bytes = await test_file.read()
        if not test_bytes or len(test_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Test audio file is empty or corrupted.")

        test_label = test_file.filename or "Learner Voice"

        # 2. Determine reference audio
        if ref_file and ref_file.filename:
            ref_bytes = await ref_file.read()
            if not ref_bytes or len(ref_bytes) < 1000:
                raise HTTPException(status_code=400, detail="Custom reference audio is empty or corrupted.")
            ref_source = ref_bytes
            ref_label = ref_file.filename
        else:
            if not os.path.exists(DEFAULT_REF_PATH):
                raise HTTPException(status_code=500, detail="Default Sanskrit reference audio is missing on server.")
            ref_source = DEFAULT_REF_PATH
            ref_label = "Authentic Sanskrit (krsna)"

        # 3. Run evaluation pipeline
        result = evaluate_audio_pair(
            ref_bytes_or_path=ref_source,
            test_bytes_or_path=test_bytes,
            ref_label=ref_label,
            test_label=test_label
        )

        return JSONResponse({
            "status": "success",
            "ref_label": ref_label,
            "test_label": test_label,
            "scores": result["scores"],
            "plots": result["plots"],
            "metadata": result["metadata"]
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Acoustic evaluation failed: {str(e)}")


if __name__ == "__main__":
    # Render binds dynamically to the $PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    print(f"[*] Starting Sanskrit Pronunciation Assessment Server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
