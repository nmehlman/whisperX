from torch.utils.data import Dataset, DataLoader, random_split
import torch
from typing import List, Tuple, Dict, Union
import os
import json
from whisperx.prosody_features.tokenizer import CharLevelTokenizer

MAX_SAMPLE_LENGTH = 1000


class LibriSpeechDataset(Dataset):
    """
    Dataset for LibriSpeech with character-level features.

    Args:
        root_path (str): Path to the root directory containing data.
        tokenizer (CharLevelTokenizer): Tokenizer for encoding character sequences.
        split (str): Dataset split to use. Must be one of VALID_SPLITS.

    Raises:
        AssertionError: If the specified split is not valid.
    """

    def __init__(
        self,
        root_path: str,
        tokenizer: CharLevelTokenizer,
        split: str = "train",
    ):
        self.root_path = root_path
        self.split = split
        self.tokenizer = tokenizer

        splits_path = os.path.join(root_path, "splits.json")
        splits = json.load(open(splits_path))
        assert self.split in splits, f"Split {self.split} not found in splits.json"

        # Load data paths and speaker labels
        self.samples = splits[self.split]

        # Renumber speakers to ensure they are sequential
        self._renumber_speakers()

    def _build_system_data_paths(self, system: str) -> Tuple[List[str], List[int]]:
        """
        Build data paths and speaker labels for a specific system.

        Args:
            system (str): VC system identifier.

        Returns:
            Tuple[List[str], List[int]]: File paths and corresponding speaker labels.
        """
        sys_data_dir = os.path.join(
            self.root_path, system, "data", f"{self.split}_{system}"
        )
        utt_to_speak_path = os.path.join(sys_data_dir, "utt2spk")
        feats_dir = os.path.join(sys_data_dir, "char_feats")

        # Map utterance IDs to speaker IDs
        utt_to_speak = {
            line.split()[0]: int(line.split()[1])
            for line in open(utt_to_speak_path).readlines()
        }

        paths, speakers = [], []
        for feat_file in os.listdir(feats_dir):  # For each feature file
            full_file_path = os.path.join(feats_dir, feat_file)
            utt_id = feat_file.replace(".json", "")
            speaker = utt_to_speak[utt_id]
            paths.append(full_file_path)
            speakers.append(speaker)

        return paths, speakers

    def _renumber_speakers(self):
        """
        Renumber speakers to ensure IDs are sequential and compute the total number of unique speakers.
        """
        unique_speakers = sorted(
            list(set([sample["speaker"] for sample in self.samples]))
        )
        self.speaker_id_map = {
            old_id: i for i, old_id in enumerate(unique_speakers)
        }  # Map old IDs to new ones

        self.num_speakers = len(unique_speakers)  # Total number of unique speakers

        print(f"Found {self.num_speakers} total speakers")

    def total_speakers(self) -> int:
        """
        Get the total number of unique speakers.

        Returns:
            int: Total unique speakers in the dataset.
        """
        return self.num_speakers

    def __len__(self) -> int:
        """
        Get the total number of data samples.

        Returns:
            int: Total number of data samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a data sample and its corresponding speaker ID.

        Args:
            index (int): Index of the data sample.

        Returns:
            Tuple[torch.Tensor, int]: Tokenized character sequence and speaker ID.
        """
        sample = self.samples[index]

        path = sample["path"]
        speaker_raw = sample["speaker"]
        speaker_id = self.speaker_id_map[speaker_raw]

        # Load character sequence and tokenize
        char_seq = json.load(open(path))
        tokens = self.tokenizer.encode(char_seq)

        if len(tokens) > MAX_SAMPLE_LENGTH:
            print(
                f"WARNING: truncating token sequence (exceeds max length {MAX_SAMPLE_LENGTH})"
            )
            tokens = tokens[:MAX_SAMPLE_LENGTH]

        return tokens, speaker_id


if __name__ == "__main__":

    import numpy as np

    tokenizer = CharLevelTokenizer()
    dataset = LibriSpeechDataset(
        root_path="/project/shrikann_35/nmehlman/psid_data/librispeech",
        tokenizer=tokenizer,
    )

    idx = np.random.randint(len(dataset))
    tokens, speaker_id = dataset[idx]
    print(f"Sample {idx} - Speaker ID: {speaker_id}")
    print(f"Tokens: {tokens}")
    print(f"Tokens shape: {tokens.shape}")
