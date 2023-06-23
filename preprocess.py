from multiprocessing import Pool
from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np
import argparse
import os

from torchaudio.transforms import Resample
from transformers import WhisperProcessor
from torchaudio import load
from tqdm import tqdm


def parse_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model-size", "-s", type=str, required=True)
    argument_parser.add_argument("--input-dir", "-i", type=str, required=True)
    argument_parser.add_argument("--output-dir", "-o", type=str, required=True)
    argument_parser.add_argument("--transcripts-path", "-t", type=str, required=True)
    argument_parser.add_argument("--metadata-output-path", "-m", type=str, required=True)

    return argument_parser.parse_args()


def load_audio(audio_path, sampling_rate=16000):
    """Loads audio from a path
    Args:
        audio_path (Str): Path to the audio file
        sampling_rate (int, optional): Audio sampling rate. Defaults to 16000.
    Returns:
        np.array: Audio signal array.
    """
    signal, signal_sr = load(audio_path, normalize=True)

    if sampling_rate != signal_sr:
        return Resample(signal_sr, sampling_rate)(signal)[0]

    return signal[0]

    
def main(path, args):
    filename = path.split("/")[-1].replace(".flac", "")

    audio = load_audio(path, sampling_rate=16000)
    audio_features = processor(audio=audio, sampling_rate=16000)["input_features"][0]
    np.save(os.path.join(args.output_dir, f"{filename}.npy"), audio_features)


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.transcripts_path, sep="\t", names=["id", "transcript"])
    df["filepath"] = df["id"].apply(lambda x: os.path.join(args.input_dir, x.split("_")[0], x.split("_")[1], f"{x}.flac"))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df.to_csv(args.metadata_output_path, index=None)

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{args.model_size}")
    paths = df["filepath"].values.tolist()
    main_= partial(main, args=args)

    with Pool(processes=os.cpu_count() - 1) as pool:
        progress_bar = tqdm(pool.imap_unordered(main_, paths), \
            desc="Extracting features...", total=len(paths), \
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for _ in progress_bar:
            pass