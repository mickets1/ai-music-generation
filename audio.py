import io
import os
import numpy as np
from PIL import Image
from scipy.io import wavfile
import torch
import torchaudio
from pydub import AudioSegment
import hashlib

# Cache dictionary for mel inverse transformation
mel_inv_cache = {}

def hash_spectrogram(Sxx: np.ndarray) -> str:
    """Generate a unique hash for the spectrogram."""
    return hashlib.md5(Sxx.tobytes()).hexdigest()

def spectrogram_from_image(image: Image.Image, max_volume: float = 50, power_for_image: float = 0.25) -> np.ndarray:
    """Compute a spectrogram magnitude array from a spectrogram image."""
    data = np.array(image).astype(np.float32)
    data = data[::-1, :, 0]  # Invert and extract the first channel
    data = 255 - data  # Invert colors
    data = data * max_volume / 255
    data = np.power(data, 1 / power_for_image)
    return data

def wav_bytes_from_spectrogram_image(image: Image.Image) -> io.BytesIO:
    """Reconstruct a WAV audio clip from a spectrogram image."""
    max_volume = 50
    power_for_image = 0.25
    Sxx = spectrogram_from_image(image, max_volume=max_volume, power_for_image=power_for_image)

    sample_rate = 44100
    clip_duration_ms = 40000

    bins_per_image = image.height
    n_mels = image.height

    window_duration_ms = 100
    padded_duration_ms = 400
    step_size_ms = 10

    # Adjusting for larger image size
    num_samples = int(image.width / float(bins_per_image) * clip_duration_ms) * sample_rate
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)

    waveform = waveform_from_spectrogram(
        Sxx=Sxx,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        num_samples=num_samples,
        sample_rate=sample_rate,
        mel_scale=True,
        n_mels=n_mels,
        num_griffin_lim_iters=32,
    )

    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, waveform.astype(np.int16))
    wav_bytes.seek(0)

    return wav_bytes

def waveform_from_spectrogram(Sxx: np.ndarray, n_fft: int, hop_length: int, win_length: int,
                              num_samples: int, sample_rate: int, mel_scale: bool = True,
                              n_mels: int = 512, num_griffin_lim_iters: int = 32) -> np.ndarray:
    """Reconstruct a waveform from a spectrogram."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Sxx_torch = torch.from_numpy(Sxx).to(device)
    
    spectrogram_hash = hash_spectrogram(Sxx)
    if mel_scale and spectrogram_hash in mel_inv_cache:
        Sxx_torch = mel_inv_cache[spectrogram_hash]
    else:
        if mel_scale:
            mel_inv_scaler = torchaudio.transforms.InverseMelScale(
                n_mels=n_mels,
                sample_rate=sample_rate,
                f_min=0,
                f_max=10000,
                n_stft=n_fft // 2 + 1,
                norm=None,
                mel_scale="htk",
            ).to(device)

            Sxx_torch = mel_inv_scaler(Sxx_torch)
            mel_inv_cache[spectrogram_hash] = Sxx_torch

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,
        n_iter=num_griffin_lim_iters,
    ).to(device)

    waveform = griffin_lim(Sxx_torch).cpu().numpy()

    return waveform

def convert_audio_image(image_path: str, output_dir: str, crop_width: int = None) -> str:
    image = Image.open(image_path)
    print(f"Image height: {image.height} Image width: {image.width}")
    if crop_width is not None and crop_width < image.width:
        image = image.crop((0, 0, crop_width, image.height))

    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.splitext(os.path.basename(image_path))[0] + ".wav"

    wav_bytes = wav_bytes_from_spectrogram_image(image)

    output_file_mp3 = os.path.join(output_dir, os.path.splitext(filename)[0] + ".mp3")
    audio_segment = AudioSegment.from_wav(wav_bytes)
    audio_segment.export(output_file_mp3, format="mp3")
    print("Converted to MP3:", output_file_mp3)

    return output_file_mp3

def process_folder(input_folder: str, output_folder: str):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            convert_audio_image(image_path, output_folder)

# Example usage:
if __name__ == "__main__":
    INPUT_FOLDER = "INPUT_FOLDER"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
