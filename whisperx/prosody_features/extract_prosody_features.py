import os
import torch
from typing import List
from whisperx.prosody_features.utils import generate_char_frame_sequence
import json
import tqdm
import argparse
from whisperx.transcribe import load_model
from whisperx.alignment import load_align_model, align_for_prosody_features
from whisperx.audio import load_audio

def get_aligned_chars(
    whisper_model,
    alignment_model,
    alignmet_model_metadata,
    audio_file: str,
    device: str = "cpu",
) -> List[dict]:
    """Perform transcription and alignment for a given audio file."""
    batch_size = 4  # Adjust if running out of memory

    audio = load_audio(audio_file)
    trans_result = whisper_model.transcribe(audio, batch_size=batch_size, language="en")

    try:
        align_result = align_for_prosody_features(
            trans_result["segments"],
            alignment_model,
            alignmet_model_metadata,
            audio,
            device,
            return_char_alignments=True,
        )
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return []

    return align_result["char_segments"]


def process_files(all_audio_files, args):
    """Main function executed for processing files."""
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    whisper_model = load_model("large-v2", device=device, compute_type=args.compute_type, language='en') 
    alignment_model, alignmet_model_metadata = load_align_model(language_code="en", device=device)

    bad_files = []
    bad_file_log = os.path.join(args.save_root, 'bad_files.json')

    pbar = tqdm.tqdm(total=len(all_audio_files), desc="Processing Progress", position=0, leave=True)

    for audio_file_path, save_path in all_audio_files:
        
        if os.path.exists(save_path) and args.skip_existing:
            pass
        
        else:
            aligned_chars = get_aligned_chars(
                whisper_model=whisper_model,
                alignment_model=alignment_model,
                alignmet_model_metadata=alignmet_model_metadata,
                audio_file=audio_file_path,
                device=device,
            )

            if not aligned_chars:
                print(f"ERROR: failed to align file {audio_file_path}")
                bad_files.append(audio_file_path)
                with open(bad_file_log, "w") as save_file:
                    json.dump(bad_files, save_file)
                continue

            char_seq = generate_char_frame_sequence(aligned_chars)

            if char_seq is None:
                print(f"ERROR: failed to generate char sequence for {audio_file_path}")
                bad_files.append(audio_file_path)
                with open(bad_file_log, "w") as save_file:
                    json.dump(bad_files, save_file)
                continue

            with open(save_path, "w") as save_file:
                json.dump(char_seq, save_file)

        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
   
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Feature extraction script with alignment.")
    parser.add_argument("--data-root", type=str, help="Root data directory.")
    parser.add_argument("--save-root", type=str, help="Root directory for saving prosody features.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for model inference.")
    parser.add_argument("--compute-type", type=str, default="float32", help="Compute format type.")
    parser.add_argument("--file-type", type=str, default="wav", help="Type of audio file.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip processing of existing files.")
    args = parser.parse_args()

    # Locate audio files
    all_audio_files = []
    for dirpath, _, filenames in os.walk(args.data_root):
        rel_path = os.path.relpath(dirpath, args.data_root)
        save_dir_path = os.path.join(args.save_root, rel_path)
        os.makedirs(save_dir_path, exist_ok=True)

        for file in filenames:
            if file.endswith(args.file_type):
                audio_file_path = os.path.join(dirpath, file)
                save_path = os.path.join(save_dir_path, file.replace(args.file_type, "json"))
                all_audio_files.append((audio_file_path, save_path))

    print(f"Found {len(all_audio_files)} audio files for processing.")

    # Process files
    process_files(all_audio_files, args)