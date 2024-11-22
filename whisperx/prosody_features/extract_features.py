import os
from typing import List
import whisperx
from whisperx.prosody_features.utils import generate_char_frame_sequence
import gc 
import json
import tqdm

MODEL_DIR = "/project/shrikann_35/nmehlman/vpc/models"

def get_aligned_chars(
    whisper_model, 
    alignment_model,
    alignmet_model_metadata, 
    audio_file: str, 
    device: str = 'cpu') -> List[dict]:

    batch_size = 16 # reduce if low on GPU mem

    audio = whisperx.load_audio(audio_file)
    result = whisper_model.transcribe(audio, batch_size=batch_size, language='en')
    result = whisperx.align_for_prosody_features(result["segments"], alignment_model, alignmet_model_metadata, audio, device, return_char_alignments=True)

    try:
        return result["segments"][0]["chars"]
    except IndexError:
        print('ERROR no speech detected')
        return None

if __name__ == "__main__":

    root = "/project/shrikann_35/nmehlman/vpc"
    device = "cuda"
    
    # Pre-load models
    whisper_model = whisperx.load_model("large-v2", device)
    alignment_model, alignmet_model_metadata = whisperx.load_align_model(language_code='en', device=device)

    for dirpath, dirnames, filenames in os.walk(root):
        
        if 'wav' in dirpath: 

            save_dir = dirpath.replace('wav', 'char_feats')

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            
            for file in tqdm.tqdm(
                os.listdir(dirpath),
                desc=f'extracting features for {dirpath}'
            ): # For each audio file
                
                full_path = os.path.join(dirpath, file)
                save_path = os.path.join(save_dir, file.replace('.wav', '.json'))
                
                # Skip previously generated files
                if os.path.exists(save_path):
                    continue

                # Perform alignment and generate char sequence feature
                aligned_chars = get_aligned_chars(
                    whisper_model=whisper_model,
                    alignment_model=alignment_model,
                    alignmet_model_metadata=alignmet_model_metadata,
                    audio_file=full_path, 
                    device=device)
                
                # Handels case where no audio is detected
                if aligned_chars is None:
                    continue
                
                char_seq = generate_char_frame_sequence(aligned_chars)
                
                # Save
                with open(save_path, "w") as save_file:
                    json.dump(char_seq, save_file)


