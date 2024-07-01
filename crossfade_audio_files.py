import numpy as np
import librosa
import soundfile as sf

# Function to find optimal shift using cross-correlation
def find_optimal_shift(y1, y2):
    # Compute cross-correlation
    corr = np.correlate(y1, y2, mode='full')
    shift = np.argmax(corr) - len(y2) + 1
    return shift

# Function to blend two audio signals with beat alignment using cross-correlation
def blend_audio_with_beat_alignment(y1, y2, sr):
    # Find optimal shift
    shift_amount = find_optimal_shift(y1, y2)

    # Align y2 with y1
    if shift_amount > 0:
        y2_aligned = np.concatenate((np.zeros(shift_amount), y2))
        y1_aligned = y1[:len(y2_aligned)]
    else:
        y1_aligned = np.concatenate((np.zeros(-shift_amount), y1))
        y2_aligned = y2[:len(y1_aligned)]

    # Blend aligned segments
    blend_length = int(sr * 0.3)  # 0.1 second overlap
    blend = np.concatenate((y1_aligned[:-blend_length], y1_aligned[-blend_length:] + y2_aligned[:blend_length], y2_aligned[blend_length:]))

    return blend

# Load the audio files
y1, sr1 = librosa.load('1.wav', sr=None)
y2, sr2 = librosa.load('2.wav', sr=None)

# Resample if sample rates are different
if sr1 != sr2:
    y2 = librosa.resample(y2, sr2, sr1)
    sr2 = sr1

# Apply blending with beat alignment using cross-correlation
blended_audio = blend_audio_with_beat_alignment(y1, y2, sr1)

# Save the blended audio using soundfile
sf.write('OUTPUT_FOLDER/crossfaded_audio.wav', blended_audio, sr1)