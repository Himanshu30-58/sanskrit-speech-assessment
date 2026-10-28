# Sanskrit Pronunciation Assessment Engine (संस्कृत उच्चारण मूल्याङ्कन)

[![Render Deploy](https://img.shields.io/badge/Render-Deploy%20Live-46E3B7?logo=render&logoColor=white)](https://render.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Acoustics](https://img.shields.io/badge/DSP-MFCC%20%C2%B7%20DTW%20%C2%B7%20LPC-6366F1)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

> **CS30033 — Speech & Video Processing Course Project**  
> **KIIT Deemed to be University, Bhubaneswar**  
> Academic Year: Autumn 2026 | B.Tech 5th Semester

A complete cloud-ready acoustic engineering web platform for automated assessment of spoken Sanskrit pronunciation against authentic master recitations. Utilizes a rigorous **7-step MFCC extraction pipeline**, **Cepstral Mean Normalization (CMN)**, **Dynamic Time Warping (DTW)** sequence alignment, and **Linear Predictive Coding (LPC)** formant tracking.

---

## 🌟 Key Features

1. **🎙️ Live In-Browser Microphone Recording**:
   - Record spoken audio directly from your PC or mobile microphone.
   - Built-in pure 16-bit PCM WAV encoder running in JavaScript (zero codecs or browser extensions required).
   - Real-time recording countdown, timer, and instant playback preview.

2. **📁 Multi-Format Audio File Upload**:
   - Upload audio recordings in `.wav`, `.mp3`, `.m4a`, `.ogg`, or `.flac` formats.
   - Automatic Voice Activity Detection (VAD) silence trimming and 16 kHz mono normalization.

3. **⚖️ Custom Pair Comparison (Teacher vs Student)**:
   - Upload a custom reference audio (Guru/Teacher) and evaluate student audio against it.
   - Side-by-side interactive audio players.

4. **📊 5-Point Acoustic & Perceptual Scoring**:
   - **MFCC Similarity (35%)**: Dynamic Time Warping alignment of phonetic feature contours.
   - **Spectral Envelope Match (20%)**: Log-power spectral correlation reflecting vocal tract shape.
   - **Duration & Tempo Match (15%)**: Human speech tempo tolerance windowing.
   - **Mel Energy Distribution (20%)**: Auditory filterbank energy distribution.
   - **Temporal Dynamics (10%)**: Delta ($\Delta$) velocity and acceleration transition matching.

5. **📈 Comprehensive High-Resolution Visual Plots**:
   - **Time-Domain Waveforms**: Reference vs Learner amplitude envelope.
   - **Spectrograms**: Wideband (formants, 5ms window) vs Narrowband (pitch harmonics, 40ms window).
   - **MFCC Heatmaps**: 13 cepstral coefficients ($c_0$ to $c_{12}$) across frames.
   - **DTW Alignment Matrix**: Accumulated cost surface with optimal dynamic programming path.
   - **LPC Spectral Envelope ($p=16$)**: Vocal tract resonance formant peaks ($F_1, F_2, F_3$).
   - **Assessment Summary**: Horizontal metric bar chart and radar graph.
   - **Fullscreen Modal**: Click any visualization to zoom in high-definition.

---

## 📂 Repository Structure

```
sanskrit_speech_web_app/
├── app.py                      # FastAPI web server and API endpoints
├── speech_engine.py            # DSP core: MFCC (7 steps), DTW, LPC, CMN, plotting
├── requirements.txt            # Python package dependencies
├── render.yaml                 # Render Blueprint for automated cloud deployment
├── Procfile                    # Render / Heroku process runner command
├── .gitignore                  # Git ignore rules for clean repo
├── README.md                   # Project documentation
├── reference_audio/
│   └── reference_krsna.mp3     # Authentic Sanskrit recitation reference ("कृष्ण")
├── static/
│   └── index.html              # Modern, responsive single-page frontend (Tailwind CSS)
└── templates/
    └── index.html              # Template mirror
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.10 or 3.11 installed.

### 2. Clone and Setup Environment
```bash
git clone https://github.com/YOUR_USERNAME/sanskrit-speech-assessment.git
cd sanskrit-speech-assessment

# (Optional) Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Server
```bash
python app.py
# Or using uvicorn directly:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at: **`http://localhost:8000`**

---

## 🌐 How to Deploy to Render (Step-by-Step)

### Step 1: Push to GitHub
1. Create a **new public repository** on GitHub (e.g., `sanskrit-speech-assessment`).
2. Run the following commands in this directory:
```bash
git init
git add .
git commit -m "Initial commit: Sanskrit Pronunciation Assessment Engine"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sanskrit-speech-assessment.git
git push -u origin main
```

### Step 2: Deploy on Render.com
1. Go to [Render.com](https://render.com) and sign in (or sign up with your GitHub account).
2. Click **"New +"** in the top right corner and select **"Web Service"**.
3. Under **"Connect a repository"**, select your `sanskrit-speech-assessment` repository.
4. Fill in the deployment details (Render automatically detects most of them from `render.yaml`):
   - **Name**: `sanskrit-speech-assessment` (or any name you prefer)
   - **Language**: `Python`
   - **Branch**: `main`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: **Free**
5. Click **"Deploy Web Service"**.
6. Render will build the environment and provide you with a live URL (e.g., `https://sanskrit-speech-assessment.onrender.com`).

---

## 🔬 Scientific & Engineering Methodology

### 1. The 7-Step MFCC Extraction Pipeline
1. **Pre-emphasis**: $y[n] = x[n] - \alpha x[n-1]$ with $\alpha = 0.97$ to boost high-frequency vowel formants.
2. **Framing & Windowing**: 25 ms Hamming window with 10 ms frame shift (60% overlap) ensuring quasi-stationary spectral estimation.
3. **FFT & Power Spectrum**: 512-point Discrete Fourier Transform computing the periodogram $P(k) = \frac{1}{N}|X(k)|^2$.
4. **Mel-Scale Triangular Filterbank**: 26 overlapping triangular filters spaced according to the human auditory Mel scale:
   $$M(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
5. **Log Filter Energies**: Homomorphic deconvolution separating glottal excitation from vocal tract spectral envelope.
6. **Discrete Cosine Transform (DCT-II)**: Orthogonal decorrelation yielding 13 static MFCC coefficients ($c_0$ to $c_{12}$).
7. **Dynamic Derivatives**: First ($\Delta$) and second ($\Delta\Delta$) order regression derivatives yielding 39-dimensional acoustic feature vectors.

### 2. Cepstral Mean Normalization (CMN)
Microphone sensitivity and acoustic channel variation introduce additive offsets in the cepstral domain:
$$\hat{c}_m[n] = c_m[n] - \mu_m, \quad m = 1, \dots, 12$$
Excluding energy coefficient $c_0$ ensures volume-invariant phonetic comparison.

### 3. Dynamic Time Warping (DTW)
Accounts for non-linear temporal variations (speakers elongating or compressing syllables):
$$D(i, j) = d(i, j) + \min \left\{ D(i-1, j), D(i, j-1), D(i-1, j-1) \right\}$$

### 4. Linear Predictive Coding (LPC) Formants
A 16th-order all-pole filter model $H(z) = \frac{G}{1 - \sum_{k=1}^p a_k z^{-k}}$ solved via the Levinson-Durbin recursion reveals the vocal tract resonance formants ($F_1, F_2, F_3$) critical for authentic Sanskrit retroflex consonants like *ṣ* (ष) and vocalic *ṛ* (ऋ).

---

## 👥 Contributors & Academic Acknowledgments
- **Institution**: School of Computer Engineering, KIIT Deemed to be University, Bhubaneswar
- **Course**: CS30033 — Speech & Video Processing
- **Faculty & Evaluators**: Department of Computer Science & Engineering
