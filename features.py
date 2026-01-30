import numpy as np
import torch
import torchaudio
import soundfile as sf

# Target sample rate for all audio
TARGET_SR = 16_000
N_MFCC = 13

# Predefined MFCC transform (torchaudio only used here, not for loading)
_mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=TARGET_SR,
    n_mfcc=N_MFCC,
    melkwargs={
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 40,
        "center": True,
        "power": 2.0,
    },
)


def _load_and_prepare(audio_path: str) -> torch.Tensor:
    """
    Load audio using soundfile (NOT torchaudio.load, so no torchcodec),
    convert to mono, and resample to TARGET_SR using torchaudio.
    Returns a tensor of shape (1, T).
    """
    # sf.read gives numpy array and sample rate
    samples, sr = sf.read(audio_path, always_2d=False)

    # If stereo/multi-channel -> average to mono
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    # Convert to torch tensor with shape (1, T)
    waveform = torch.from_numpy(samples).float().unsqueeze(0)

    # Resample if needed
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    return waveform


def extract_features(audio_path: str, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract MFCC + simple energy features from an audio file.

    Returns:
        1D numpy array of features, or None if extraction fails.
    """
    try:
        waveform = _load_and_prepare(audio_path)  # (1, T)

        # MFCCs: (1, n_mfcc, time) -> (n_mfcc, time)
        mfcc = _mfcc_transform(waveform).squeeze(0)

        mfcc_mean = mfcc.mean(dim=1)   # (n_mfcc,)
        mfcc_std = mfcc.std(dim=1)     # (n_mfcc,)

        # Energy features
        rms = torch.sqrt(torch.mean(waveform ** 2))         # scalar
        amp_std = waveform.std()                            # scalar

        # Stack everything
        feature_tensor = torch.hstack([
            mfcc_mean,
            mfcc_std,
            torch.tensor([rms.item(), amp_std.item()], dtype=torch.float32),
        ])

        return feature_tensor.detach().cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"[extract_features] Error processing {audio_path}: {e}")
        return None
