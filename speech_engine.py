"""
===============================================================================
Sanskrit Speech Analysis Engine — MFCC, DTW, LPC & Acoustic Scoring
===============================================================================
Course  : CS30033 — Speech & Video Processing
Project : Sanskrit Pronunciation Assessment Engine
Engine  : Core DSP Module for Web and API Deployment
===============================================================================
"""

import os
import io
import base64
import numpy as np
import scipy.signal as signal
from scipy.fftpack import dct
import soundfile as sf
import librosa

# Matplotlib headless backend for cloud server deployment (Render / Linux / Docker)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Standard DSP Constants (Aligned with CS30033 Syllabus)
SAMPLE_RATE = 16000     # 16 kHz standard telephony & ASR sampling rate
FRAME_SIZE_MS = 25.0    # 25 ms frame length (quasi-stationary speech assumption)
FRAME_SHIFT_MS = 10.0   # 10 ms frame shift (50-60% overlap)
PRE_EMPHASIS = 0.97     # High-frequency tilt compensation factor
N_FFT = 512             # FFT points (power of 2)
N_FILTERS = 26          # Mel filterbank channels (Davis & Mermelstein standard)
N_MFCC = 13             # Number of cepstral coefficients (c0 to c12)
LPC_ORDER = 16          # LPC all-pole order p = fs/1000 + 4 to 5 = 16


def load_and_preprocess_audio(audio_source, fs=SAMPLE_RATE):
    """
    Load any audio (file path or BytesIO buffer), resample to 16 kHz mono,
    apply Voice Activity Detection (VAD) silence trimming, and normalize.
    """
    if isinstance(audio_source, bytes):
        audio_source = io.BytesIO(audio_source)
    
    # Load via librosa (handles wav, mp3, ogg, flac, etc.)
    audio, sr = librosa.load(audio_source, sr=fs, mono=True)

    # Trim leading and trailing silence (VAD)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)

    # Fallback if audio was too quiet or over-trimmed (< 0.2s)
    if len(audio_trimmed) < int(0.2 * fs):
        audio_trimmed = audio

    # Amplitude normalization to [-0.95, +0.95]
    max_val = np.max(np.abs(audio_trimmed))
    if max_val > 0:
        audio_trimmed = audio_trimmed / max_val * 0.95

    return audio_trimmed


def pre_emphasis(signal_in, alpha=PRE_EMPHASIS):
    """
    Step 1: Pre-emphasis filtering.
    Compensates for the -6 dB/octave glottal volume velocity roll-off.
    Transfer function: H(z) = 1 - alpha * z^(-1)
    """
    return np.append(signal_in[0], signal_in[1:] - alpha * signal_in[:-1])


def frame_signal(sig, fs=SAMPLE_RATE, frame_size_ms=FRAME_SIZE_MS, frame_shift_ms=FRAME_SHIFT_MS):
    """
    Step 2a: Framing.
    Divides speech into overlapping frames where speech is quasi-stationary.
    """
    frame_length = int(round(frame_size_ms * fs / 1000.0))
    frame_step = int(round(frame_shift_ms * fs / 1000.0))
    signal_length = len(sig)

    if signal_length <= frame_length:
        num_frames = 1
        pad_signal = np.pad(sig, (0, frame_length - signal_length), mode='constant')
    else:
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
        pad_signal_length = (num_frames - 1) * frame_step + frame_length
        pad_signal = np.pad(sig, (0, pad_signal_length - signal_length), mode='constant')

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) +
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )
    return pad_signal[indices.astype(np.int32, copy=False)], frame_length, frame_step


def apply_hamming_window(frames, frame_length):
    """
    Step 2b: Hamming Windowing.
    w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
    Minimizes spectral leakage and sidelobe artifacts at frame boundaries.
    """
    window = np.hamming(frame_length)
    return frames * window


def compute_fft_power_spectrum(frames, n_fft=N_FFT):
    """
    Step 3: Fast Fourier Transform & Power Spectrum.
    Computes periodogram estimate of power spectral density: P = |FFT|^2 / N.
    """
    mag_spectrum = np.abs(np.fft.rfft(frames, n_fft))
    power_spectrum = (1.0 / n_fft) * (mag_spectrum ** 2)
    return mag_spectrum, power_spectrum


def hz_to_mel(hz):
    """Convert frequency from Hertz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    """Convert frequency from Mel scale to Hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def get_mel_filterbank(num_filters=N_FILTERS, n_fft=N_FFT, fs=SAMPLE_RATE, low_freq=0, high_freq=None):
    """
    Step 4: Mel-Scale Triangular Filterbank Construction.
    Models human auditory perception (dense below 1 kHz, logarithmic above).
    """
    if high_freq is None:
        high_freq = fs / 2.0

    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)

    fbank = np.zeros((num_filters, int(n_fft // 2 + 1)))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (f_m_plus - f_m)

    return fbank


def apply_mel_filterbank(power_spectrum, fbank):
    """
    Step 5: Apply filterbank and take logarithm (Log-Mel energies).
    Homomorphic deconvolution separating excitation from vocal tract response.
    """
    filter_energies = np.dot(power_spectrum, fbank.T)
    filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)
    return np.log(filter_energies)


def compute_dct_mfcc(log_energies, num_ceps=N_MFCC):
    """
    Step 6: Discrete Cosine Transform (DCT-II).
    Decorrelates filterbank energies and compresses spectral envelope into 13 coefficients.
    """
    mfcc = dct(log_energies, type=2, axis=1, norm='ortho')
    return mfcc[:, :num_ceps]


def compute_delta(features, N=2):
    """
    Step 7: Delta Dynamic Feature Estimation (Velocity).
    d_t = sum_{n=1}^N n * (c_{t+n} - c_{t-n}) / (2 * sum_{n=1}^N n^2)
    """
    num_frames, num_feats = features.shape
    deltas = np.zeros_like(features)
    denom = 2 * sum([i ** 2 for i in range(1, N + 1)])

    padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
    for t in range(num_frames):
        delta_sum = np.zeros(num_feats)
        for n in range(1, N + 1):
            delta_sum += n * (padded[t + N + n] - padded[t + N - n])
        deltas[t] = delta_sum / denom
    return deltas


def compute_mfcc_pipeline(audio_signal, fs=SAMPLE_RATE):
    """
    Executes the full 7-step MFCC extraction pipeline.
    """
    emphasized = pre_emphasis(audio_signal)
    frames, frame_len, frame_step = frame_signal(emphasized, fs)
    windowed = apply_hamming_window(frames, frame_len)
    mag_spec, power_spec = compute_fft_power_spectrum(windowed, N_FFT)
    fbank = get_mel_filterbank(N_FILTERS, N_FFT, fs)
    log_mel = apply_mel_filterbank(power_spec, fbank)
    mfcc = compute_dct_mfcc(log_mel, N_MFCC)
    delta = compute_delta(mfcc)
    delta2 = compute_delta(delta)
    features_39d = np.hstack([mfcc, delta, delta2])

    return {
        'emphasized': emphasized,
        'frames': frames,
        'windowed': windowed,
        'power_spectrum': power_spec,
        'fbank': fbank,
        'log_energies': log_mel,
        'mfcc': mfcc,
        'delta': delta,
        'delta2': delta2,
        'features_39d': features_39d,
        'fs': fs
    }


def apply_cmn(mfcc_features):
    """
    Cepstral Mean Normalization (CMN) excluding c0 energy.
    Removes linear channel distortion and microphone frequency bias.
    """
    c = mfcc_features[:, 1:]
    return c - np.mean(c, axis=0)


def compute_dtw(seq_ref, seq_test):
    """
    Dynamic Time Warping (DTW) with dynamic programming alignment.
    Aligns variable-speed phonetic sequences.
    """
    n, m = len(seq_ref), len(seq_test)
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i, j] = np.linalg.norm(seq_ref[i] - seq_test[j])

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = cost[i - 1, j - 1]
            dtw_matrix[i, j] = c + min(
                dtw_matrix[i - 1, j],      # Insertion
                dtw_matrix[i, j - 1],      # Deletion
                dtw_matrix[i - 1, j - 1]   # Match
            )

    # Backtracking optimal warping path
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        steps = [
            dtw_matrix[i - 1, j - 1],
            dtw_matrix[i - 1, j],
            dtw_matrix[i, j - 1]
        ]
        best_step = np.argmin(steps)
        if best_step == 0:
            i, j = i - 1, j - 1
        elif best_step == 1:
            i = i - 1
        else:
            j = j - 1
        path.append((i - 1, j - 1))

    path.reverse()
    normalized_distance = dtw_matrix[n, m] / len(path)
    return normalized_distance, cost, path


def compute_lpc(audio, order=LPC_ORDER, fs=SAMPLE_RATE):
    """
    Linear Predictive Coding (LPC) via Levinson-Durbin recursion.
    Estimates vocal tract all-pole filter coefficients.
    """
    frame_len = int(0.025 * fs)
    mid = len(audio) // 2
    seg = audio[max(0, mid - frame_len // 2): mid + frame_len // 2]
    if len(seg) < frame_len:
        seg = np.pad(seg, (0, frame_len - len(seg)), mode='constant')

    seg = seg * np.hamming(len(seg))
    r = np.correlate(seg, seg, mode='full')
    r = r[len(seg) - 1: len(seg) + order]

    a = np.zeros(order + 1)
    a[0] = 1.0
    k = np.zeros(order)
    alpha = r[0]

    for i in range(1, order + 1):
        if alpha <= 0:
            break
        sum_val = sum(a[j] * r[i - j] for j in range(1, i))
        ki = (r[i] - sum_val) / alpha
        if np.abs(ki) >= 1.0:
            ki = np.sign(ki) * 0.999
        k[i - 1] = ki
        a[i] = ki
        for j in range(1, (i // 2) + 1):
            tmp1 = a[j] - ki * a[i - j]
            tmp2 = a[i - j] - ki * a[j]
            a[j] = tmp1
            a[i - j] = tmp2
        alpha *= (1.0 - ki ** 2)

    return a, alpha


def lpc_to_spectrum(a, n_pts=512, fs=SAMPLE_RATE):
    """Compute LPC frequency response spectrum in dB."""
    w, h = signal.freqz(1.0, a, worN=n_pts, fs=fs)
    spec_db = 20 * np.log10(np.abs(h) + 1e-10)
    return w, spec_db


def compute_similarity_scores(ref_results, test_results, dtw_dist):
    """
    Acoustic & Perceptual Pronunciation Similarity Assessment Breakdown.
    """
    scores = {}

    # 1. Phonetic MFCC Similarity (via DTW distance on CMN-normalized features)
    max_expected_dist = 30.0
    mfcc_sim = max(0.0, min(100.0, 100.0 * (1.0 - dtw_dist / max_expected_dist)))
    scores['MFCC Similarity'] = round(float(mfcc_sim), 1)

    # 2. Spectral Envelope Similarity (Log-power spectral correlation)
    ref_log_spec = np.log(np.mean(ref_results['power_spectrum'], axis=0) + 1e-8)
    test_log_spec = np.log(np.mean(test_results['power_spectrum'], axis=0) + 1e-8)
    min_len = min(len(ref_log_spec), len(test_log_spec))
    spec_corr = np.corrcoef(ref_log_spec[:min_len], test_log_spec[:min_len])[0, 1]
    scores['Spectral Similarity'] = round(float(max(0.0, min(100.0, ((spec_corr + 1) / 2.0) * 100.0))), 1)

    # 3. Duration & Tempo Match (soft human speech tolerance window)
    ref_dur = ref_results['mfcc'].shape[0]
    test_dur = test_results['mfcc'].shape[0]
    dur_ratio = min(ref_dur, test_dur) / max(ref_dur, test_dur)
    dur_score = min(100.0, dur_ratio * 125.0)
    scores['Duration Match'] = round(float(max(0.0, dur_score)), 1)

    # 4. Mel Energy Distribution Similarity
    ref_log_mel = np.mean(ref_results['log_energies'], axis=0)
    test_log_mel = np.mean(test_results['log_energies'], axis=0)
    mel_corr = np.corrcoef(ref_log_mel, test_log_mel)[0, 1]
    scores['Mel Energy Similarity'] = round(float(max(0.0, min(100.0, ((mel_corr + 1) / 2.0) * 100.0))), 1)

    # 5. Temporal Dynamic Transition Similarity (Delta MFCC)
    ref_delta = compute_delta(ref_results['mfcc'])
    test_delta = compute_delta(test_results['mfcc'])
    ref_delta_mean = np.mean(np.abs(ref_delta))
    test_delta_mean = np.mean(np.abs(test_delta))
    delta_ratio = min(ref_delta_mean, test_delta_mean) / (max(ref_delta_mean, test_delta_mean) + 1e-10)
    scores['Temporal Dynamics'] = round(float(max(0.0, min(100.0, delta_ratio * 100.0))), 1)

    # Weighted Overall Score
    weights = {
        'MFCC Similarity': 0.35,
        'Spectral Similarity': 0.20,
        'Duration Match': 0.15,
        'Mel Energy Similarity': 0.20,
        'Temporal Dynamics': 0.10,
    }
    overall = sum(scores[k] * weights[k] for k in weights)
    scores['Overall'] = round(float(overall), 1)

    return scores


def _fig_to_base64(fig):
    """Converts a matplotlib Figure into a Base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_visual_plots(ref_audio, test_audio, ref_results, test_results,
                          dtw_dist, cost_matrix, path, scores,
                          ref_label="Reference", test_label="Your Speech"):
    """
    Generates all 6 comprehensive scientific comparison plots and returns Base64 PNGs.
    """
    fs = SAMPLE_RATE
    plots = {}

    # Plot 1: Waveforms Comparison
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.5))
    t_ref = np.arange(len(ref_audio)) / fs
    t_test = np.arange(len(test_audio)) / fs
    max_t = max(t_ref[-1], t_test[-1])

    axes[0].plot(t_ref, ref_audio, color='#2563EB', linewidth=0.8)
    axes[0].set_title(f'Reference Signal: {ref_label} ({len(ref_audio)/fs:.2f}s)', fontweight='bold', fontsize=11)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim(0, max_t)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].plot(t_test, test_audio, color='#DC2626', linewidth=0.8)
    axes[1].set_title(f'Learner Signal: {test_label} ({len(test_audio)/fs:.2f}s)', fontweight='bold', fontsize=11)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlim(0, max_t)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('Acoustic Waveform Time-Domain Alignment', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plots['waveform'] = _fig_to_base64(fig)

    # Plot 2: Spectrogram Comparison (Wideband vs Narrowband)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, (audio, label) in enumerate([(ref_audio, ref_label), (test_audio, test_label)]):
        # Wideband spectrogram (5ms window -> formants & time resolution)
        n_wb = int(0.005 * fs)
        f_wb, t_wb, s_wb = signal.spectrogram(audio, fs, nperseg=n_wb, noverlap=n_wb // 2)
        im0 = axes[0, col].pcolormesh(t_wb, f_wb, 10 * np.log10(s_wb + 1e-10), shading='gouraud', cmap='inferno')
        axes[0, col].set_title(f'Wideband (Formants) — {label}', fontweight='bold', fontsize=10)
        axes[0, col].set_ylabel('Frequency (Hz)')
        axes[0, col].set_ylim(0, 4500)

        # Narrowband spectrogram (40ms window -> pitch harmonics)
        n_nb = int(0.040 * fs)
        f_nb, t_nb, s_nb = signal.spectrogram(audio, fs, nperseg=n_nb, noverlap=n_nb * 3 // 4)
        im1 = axes[1, col].pcolormesh(t_nb, f_nb, 10 * np.log10(s_nb + 1e-10), shading='gouraud', cmap='inferno')
        axes[1, col].set_title(f'Narrowband (Harmonics) — {label}', fontweight='bold', fontsize=10)
        axes[1, col].set_ylabel('Frequency (Hz)')
        axes[1, col].set_xlabel('Time (seconds)')
        axes[1, col].set_ylim(0, 4500)

    plt.suptitle('Spectrogram Comparison — Wideband (Formants) & Narrowband (Harmonics)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plots['spectrogram'] = _fig_to_base64(fig)

    # Plot 3: MFCC Heatmap
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5))
    im1 = axes[0].imshow(ref_results['mfcc'].T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f'Reference 13-MFCC Coefficients ({ref_label})', fontweight='bold', fontsize=11)
    axes[0].set_ylabel('MFCC Index (c0-c12)')
    plt.colorbar(im1, ax=axes[0], label='Cepstral Amp')

    im2 = axes[1].imshow(test_results['mfcc'].T, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(f'Learner 13-MFCC Coefficients ({test_label})', fontweight='bold', fontsize=11)
    axes[1].set_ylabel('MFCC Index (c0-c12)')
    axes[1].set_xlabel('Frame Index (10ms step)')
    plt.colorbar(im2, ax=axes[1], label='Cepstral Amp')

    plt.suptitle('Mel-Frequency Cepstral Coefficients (MFCC) Heatmap', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plots['mfcc'] = _fig_to_base64(fig)

    # Plot 4: DTW Alignment Matrix
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cax = ax.imshow(cost_matrix, origin='lower', cmap='YlOrRd', aspect='auto')
    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    ax.plot(path_x, path_y, color='deepskyblue', linewidth=2.8, label=f'Optimal Path (Dist: {dtw_dist:.2f})')
    ax.set_xlabel(f'Learner Frames ({test_label})', fontsize=11)
    ax.set_ylabel(f'Reference Frames ({ref_label})', fontsize=11)
    ax.set_title('Dynamic Time Warping (DTW) Cost Matrix & Alignment Path', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    plt.colorbar(cax, ax=ax, label='Euclidean Frame Cost')
    plt.tight_layout()
    plots['dtw'] = _fig_to_base64(fig)

    # Plot 5: LPC Formant Resonances
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ref_lpc, _ = compute_lpc(ref_audio, LPC_ORDER, fs)
    test_lpc, _ = compute_lpc(test_audio, LPC_ORDER, fs)
    w_ref, spec_ref = lpc_to_spectrum(ref_lpc, 512, fs)
    w_test, spec_test = lpc_to_spectrum(test_lpc, 512, fs)

    ax.plot(w_ref, spec_ref, color='#2563EB', linewidth=2.2, label=f'Reference LPC Formants ({ref_label})')
    ax.plot(w_test, spec_test, color='#DC2626', linewidth=2.0, linestyle='--', label=f'Learner LPC Formants ({test_label})')
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Magnitude (dB)', fontsize=11)
    ax.set_title('LPC Spectral Envelope (p=16) — Vocal Tract Resonances & Formants', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 4500)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plots['lpc'] = _fig_to_base64(fig)

    # Plot 6: Overall Score Summary (Horizontal Bar + Radar)
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1])

    ax1 = fig.add_subplot(gs[0])
    cats = [k for k in scores if k != 'Overall']
    vals = [scores[k] for k in cats]
    colors = ['#10B981' if v >= 80 else '#F59E0B' if v >= 65 else '#EF4444' for v in vals]
    bars = ax1.barh(cats, vals, color=colors, height=0.55, edgecolor='black', linewidth=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Score (%)', fontsize=11)
    ax1.set_title('Metric Score Breakdown', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', linestyle=':', alpha=0.7)

    for bar, val in zip(bars, vals):
        ax1.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontweight='bold', fontsize=10)

    # Radar Chart
    ax2 = fig.add_subplot(gs[1], polar=True)
    n_vars = len(cats)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]
    vals_radar = vals + vals[:1]

    ax2.plot(angles, vals_radar, color='#4F46E5', linewidth=2)
    ax2.fill(angles, vals_radar, color='#818CF8', alpha=0.35)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(cats, fontsize=8, fontweight='semibold')
    ax2.set_ylim(0, 100)
    ax2.set_title(f"Overall: {scores['Overall']:.1f}%", fontsize=13, fontweight='bold', pad=15)

    plt.suptitle('Sanskrit Pronunciation Acoustic Assessment Report', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plots['report'] = _fig_to_base64(fig)

    return plots


def evaluate_audio_pair(ref_bytes_or_path, test_bytes_or_path,
                        ref_label="Reference", test_label="Learner"):
    """
    Complete end-to-end evaluation pipeline returning score metrics,
    DSP metadata, and generated base64 visual comparison charts.
    """
    # 1. Preprocess both audio signals
    ref_audio = load_and_preprocess_audio(ref_bytes_or_path)
    test_audio = load_and_preprocess_audio(test_bytes_or_path)

    # 2. Extract 7-step MFCCs and derivatives
    ref_results = compute_mfcc_pipeline(ref_audio)
    test_results = compute_mfcc_pipeline(test_audio)

    # 3. Apply Cepstral Mean Normalization (CMN) & Dynamic Time Warping (DTW)
    ref_cmn = apply_cmn(ref_results['mfcc'])
    test_cmn = apply_cmn(test_results['mfcc'])
    dtw_dist, cost_matrix, path = compute_dtw(ref_cmn, test_cmn)

    # 4. Compute acoustic similarity metrics
    scores = compute_similarity_scores(ref_results, test_results, dtw_dist)

    # 5. Generate high-resolution visual plots
    plots = generate_visual_plots(
        ref_audio, test_audio, ref_results, test_results,
        dtw_dist, cost_matrix, path, scores,
        ref_label=ref_label, test_label=test_label
    )

    metadata = {
        "ref_duration_sec": round(len(ref_audio) / SAMPLE_RATE, 2),
        "test_duration_sec": round(len(test_audio) / SAMPLE_RATE, 2),
        "ref_frames": int(ref_results['mfcc'].shape[0]),
        "test_frames": int(test_results['mfcc'].shape[0]),
        "dtw_distance": round(float(dtw_dist), 4),
        "sample_rate_hz": SAMPLE_RATE,
        "lpc_order": LPC_ORDER
    }

    return {
        "scores": scores,
        "plots": plots,
        "metadata": metadata
    }
