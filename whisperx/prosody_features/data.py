from torch.utils.data import Dataset, DataLoader, random_split
import torch
from typing import List, Tuple, Dict, Union
import os
import json
from whisperx.prosody_features.tokenizer import CharLevelTokenizer

VC_SYSTEMS = ("B3", "B4", "B5", "T10-2", "T12-5", "T25-1", "T8-5")
VALID_SPLITS = (
    "libri_dev_enrolls",
    "libri_test_enrolls",
    "libri_dev_trials_f",
    "libri_test_trials_f",
    "libri_dev_trials_m",
    "libri_test_trials_m",
)

MAX_SAMPLE_LENGTH = 1000

class VPCDataset(Dataset):
    """
    Dataset for Voice Conversion (VC) systems with character-level features.

    Args:
        root_path (str): Path to the root directory containing data.
        tokenizer (CharLevelTokenizer): Tokenizer for encoding character sequences.
        system (str): Specific VC system to use or "all" for all systems. Defaults to "all".
        split (str): Dataset split to use. Must be one of VALID_SPLITS.

    Raises:
        AssertionError: If the specified split is not valid.
    """

    def __init__(
        self,
        root_path: str,
        tokenizer: CharLevelTokenizer,
        split: str = "train-clean-360",
        return_id: bool = False
    ):
        self.root_path = root_path
        self.split = split
        self.tokenizer = tokenizer
        self.return_id = return_id

        # Validate the split
        assert split in VALID_SPLITS, f"Invalid split. Must be one of {VALID_SPLITS}"

        self.paths = self._build_system_data_paths()

    def _build_system_data_paths(self) -> Tuple[List[str], List[int]]:
        """
        Build data paths and speaker labels for a specific system.

        Args:
            system (str): VC system identifier.

        Returns:
            Tuple[List[str], List[int]]: File paths and corresponding speaker labels.
        """
        data_dir = os.path.join(
            self.root_path, self.split
        )

        paths = []
        for feat_file in os.listdir(data_dir):  # For each feature file
            full_file_path = os.path.join(data_dir, feat_file)
            paths.append(full_file_path)

        return paths

    def __len__(self) -> int:
        """
        Get the total number of data samples.

        Returns:
            int: Total number of data samples.
        """
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a data sample and its corresponding speaker ID.

        Args:
            index (int): Index of the data sample.

        Returns:
            Tuple[torch.Tensor, int]: Tokenized character sequence and speaker ID.
        """
        path = self.paths[index]

        id = path.split('/')[-1].replace('.json', '')

        # Load character sequence and tokenize
        char_seq = json.load(open(path))
        tokens = self.tokenizer.encode(char_seq)

        if len(tokens) > MAX_SAMPLE_LENGTH:
            print(f'WARNING: truncating token sequence (exceeds max length {MAX_SAMPLE_LENGTH})')
            tokens = tokens[:MAX_SAMPLE_LENGTH]

        return tokens, id
    

def collate_fn(
    batch: List[Tuple[torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to pad sequences to the same length for batching.

    Args:
        batch (List[Tuple[torch.Tensor, int]]): A batch of data samples, where each sample is a tuple of
                                                (sequence tensor, speaker ID).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Padded sequences (torch.Tensor) of shape (batch_size, max_seq_len).
            - Speaker IDs (torch.Tensor) of shape (batch_size).
    """
    # Separate sequences and speaker IDs
    sequences, speaker_ids = zip(*batch)

    # Find the length of the longest sequence in the batch
    max_seq_len = max(seq.size(0) for seq in sequences)

    # Initialize a tensor for padded sequences with zeros
    padded_sequences = torch.zeros(len(sequences), max_seq_len, dtype=torch.long)

    # Copy each sequence into the padded tensor
    for i, seq in enumerate(sequences):
        padded_sequences[i, : seq.size(0)] = seq  # Copy the sequence up to its length

    # Convert speaker IDs to a tensor
    if isinstance(speaker_ids[0], str):
        speaker_ids = [id for id in speaker_ids]
    else:
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)

    return padded_sequences, speaker_ids


def get_dataloaders(
    root_path: str,
    tokenizer: CharLevelTokenizer,
    return_id: bool = False,
    train_frac: float = 1.0,
    batch_size: int = 16,
    num_workers: int = 1,
    shuffle: bool = True,
    **dataloader_kwargs,
) -> Union[DataLoader, Dict[str, DataLoader]]:
    """
    Create DataLoaders for training and validation.

    Args:
        root_path (str): Path to the dataset root.
        tokenizer (CharLevelTokenizer): Tokenizer for encoding character sequences.
        system (str): VC system to use or "all".
        train_frac (float): Fraction of data for training. Defaults to 1.0.
        batch_size (int): Batch size for DataLoader. Defaults to 16.
        num_workers (int): Number of workers for DataLoader. Defaults to 1.
        shuffle (bool): Whether to shuffle the data. Defaults to True.
        **dataloader_kwargs: Additional arguments for DataLoader.

    Returns:
        Union[DataLoader, Dict[str, DataLoader]]: A dict with "train" and (possibly) "val" DataLoaders.
    """
    full_dataset = VPCDataset(
        root_path=root_path, tokenizer=tokenizer, return_id=return_id
    )

    if train_frac < 1.0:  # Create a validation split
        train_size = int(train_frac * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Build dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=32,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

        return {"train": train_dataloader, "val": val_dataloader}
    else:  # Train on the full dataset
        train_dataloader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )
        return {"train": train_dataloader}

if __name__ == "__main__":
    
    tokenizer = CharLevelTokenizer()
    dataset = VPCDataset(root_path="/project/shrikann_35/nmehlman/vpc", tokenizer=tokenizer)

    for sample, _ in dataset:
        print(len(sample))