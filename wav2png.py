import math
import os

import librosa as librosa
import numpy as np
import skimage.io

"""
Source and target datasets will be structured as follows:
wav-dataset / spectrograms
    speakers
        audio files (.wav) / spectrograms (.png)
"""

input_dir = r"LibriSpeech/wav-dataset"
output_dir = r"LibriSpeech/spectrograms"


# adapted from https://stackoverflow.com/questions/56719138/how-can-i-save-a-librosa-spectrogram-plot-as-a-specific-sized-image/57204349#57204349
def scale_minmax(x, min_target=0.0, max_target=1.0):
    x_std = (x - x.min()) / (x.max() - x.min())
    x_scaled = x_std * (max_target - min_target) + min_target
    return x_scaled


def spectrogram_image(x, fs, out_file, hop_length, n_bins):
    mels = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=n_bins, n_fft=hop_length * 2, hop_length=hop_length)
    mels = np.log(mels + 1e-9)
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    skimage.io.imsave(out_file, img)


def save_spectrogram(input_file, output_file):
    x, fs = librosa.load(input_file, sr=None, mono=True)
    audio_len = x.shape[0]  # == 80k
    time_steps = 224  # number of time-steps. Width of image
    n_bins = 224  # number of bins in spectrogram. Height of image
    hop_length = math.ceil(audio_len / time_steps)  # == 358
    spectrogram_image(x=x, fs=fs, out_file=output_file, hop_length=hop_length, n_bins=n_bins)


# Check output folder
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Generate dataset samples from the raw data
for speaker in os.listdir(input_dir):
    speaker_path = os.path.join(input_dir, speaker)
    i = 1
    if not os.path.isdir(speaker_path):
        continue
    if not os.path.isdir(os.path.join(output_dir, speaker)):  # Create output folders if they don't exist
        os.mkdir(os.path.join(output_dir, speaker))
    print(f"Processing files from {speaker}.")
    for file in os.listdir(speaker_path):
        file_path = os.path.join(speaker_path, file)
        if not file_path.endswith(".wav"):
            continue
        output_path = os.path.join(output_dir, speaker, file.split(".")[0] + ".png")
        save_spectrogram(file_path, output_path)
        i += 1
    print(f"{i} spectrograms generated.")
print("Done!")
