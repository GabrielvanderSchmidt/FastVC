import os
from pydub import AudioSegment

"""
Original dataset (input_dir) is structured as follows:
train-clean-100
    speakers
        chapters
            audio files (.flac)

My desired structure in the sampled dataset is:
wav-dataset
    speakers
        audio files (.wav)
"""

input_dir = r"LibriSpeech/train-clean-100"
output_dir = r"LibriSpeech/wav-dataset"

# Check output folder
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Generate dataset samples from the raw data
window_size = 5 * 1000
stride = 4 * 1000
for speaker in os.listdir(input_dir):
    speaker_path = os.path.join(input_dir, speaker)
    if not os.path.isdir(speaker_path):
        continue
    if not os.path.isdir(os.path.join(output_dir, speaker)):  # Create output folders if they don't exist
        os.mkdir(os.path.join(output_dir, speaker))
    print(f"Processing files from {speaker}.")
    i = 1
    for chapter in os.listdir(speaker_path):
        chapter_path = os.path.join(speaker_path, chapter)
        if not os.path.isdir(chapter_path):
            continue
        for file in os.listdir(chapter_path):
            file_path = os.path.join(chapter_path, file)
            if not file_path.endswith(".flac"):
                continue
            original = AudioSegment.from_file(file=file_path, format="flac")
            frame_start = 0
            frame_end = window_size
            while frame_end < len(original):
                frame = original[frame_start:frame_end]
                frame.export(out_f=os.path.join(output_dir, speaker, f"{speaker}-{i:05}.wav"), format="wav")
                i += 1
                frame_start += stride
                frame_end += stride
    print(f"{i} samples obtained.")
print("Done!")