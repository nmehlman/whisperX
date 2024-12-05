import os
from typing import List
import whisperx
from whisperx.prosody_features.utils import generate_char_frame_sequence
import gc
import json
import tqdm
import argparse
from whisperx.transcribe import load_model
from whisperx.alignment import load_align_model, align_for_prosody_features
from whisperx.audio import load_audio

MODEL_DIR = "/project/shrikann_35/nmehlman/vpc/models"

SPLITS = ("libri_dev_enrolls",  "libri_dev_trials_f",  "libri_dev_trials_m",  "libri_test_enrolls",  "libri_test_trials_f",  "libri_test_trials_m")

ROOT = "/project/shrikann_35/nmehlman/vpc/dev_test_orig/"

SAVE_DIR = "/project/shrikann_35/nmehlman/vpc/original_char_feats"

def get_aligned_chars(
    whisper_model,
    alignment_model,
    alignmet_model_metadata,
    audio_file: str,
    device: str = "cpu",
) -> List[dict]:

    batch_size = 16  # reduce if low on GPU mem

    audio = load_audio(audio_file)
    result = whisper_model.transcribe(audio, batch_size=batch_size, language="en")
    
    result = align_for_prosody_features(
        result["segments"],
        alignment_model,
        alignmet_model_metadata,
        audio,
        device,
        return_char_alignments=True,
    )

    chars = result["char_segments"]
    
    return chars
   


if __name__ == "__main__":

    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Feature extraction script with alignment."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for model inference. Default: 'cuda'.",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="Type of compute format to use. Default: 'float16'.",
    )

    args = parser.parse_args()
    root = ROOT
    device = args.device
    compute_type = args.compute_type

    # Pre-load models
    whisper_model = load_model("large-v2", device, compute_type=compute_type)
    alignment_model, alignmet_model_metadata = load_align_model(
        language_code="en", device=device
    )

    bad_files = []

    for split in SPLITS:

        save_dir = os.path.join(SAVE_DIR, split)
        os.mkdir(save_dir)

        # Get list of split files
        split_wav_path = os.path.join(root, split, 'wav.scp')
        split_paths = [line.split(' ')[1].replace('data/', root) for line in open(split_wav_path).readlines()]  

        for full_path in tqdm.tqdm(
            split_paths, desc=f"extracting features for {split}"
        ):  # For each audio file

            save_name = full_path.split('/')[-1].rstrip().replace(".wav", ".json")

            save_path = os.path.join(save_dir, save_name)

            # Perform alignment and generate char sequence feature
            try: 
                aligned_chars = get_aligned_chars(
                    whisper_model=whisper_model,
                    alignment_model=alignment_model,
                    alignmet_model_metadata=alignmet_model_metadata,
                    audio_file=full_path,
                    device=device,
                )
            except Exception as e:
                breakpoint()
                print("ERROR: failed to align file")
                bad_files.append(full_path)
                continue

            # Handels error cases
            if aligned_chars is None or aligned_chars == []:
                print("ERROR: failed to align file")
                bad_files.append(full_path)
                continue

            char_seq = generate_char_frame_sequence(aligned_chars)

            if char_seq is None:
                print("ERROR: failed to generate char sequence")
                bad_files.append(full_path)
                continue

            # Save
            with open(save_path, "w") as save_file:
                json.dump(char_seq, save_file)

    print("BAD FILES:")
    for file in bad_files:
        print(file)
