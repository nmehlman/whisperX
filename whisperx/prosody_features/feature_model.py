import math
import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    """
    Applies positional encoding to input embeddings, adding temporal information
    for sequence modeling tasks.

    Args:
        d_model (int): Dimension of the embedding space.
        dropout (float): Dropout rate applied to the positional encoding. Defaults to 0.1.
        max_len (int): Maximum length of sequences supported. Defaults to 5000.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        position = torch.arange(max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape ``[batch_size, seq_len, embedding_dim]``.

        Returns:
            Tensor: Output tensor of shape ``[batch_size, seq_len, embedding_dim]``.
        """
        _, seq_len, _ = x.size()
        
        # Add positional encodings
        x = x + self.pe[:seq_len].unsqueeze(0)  # Shape: [1, seq_len, d_model]
        return self.dropout(x)


class ProsodyFeatureModel(nn.Module):
    """
    A model for extracting features from prosodic input using embeddings, positional
    encoding, and a Transformer encoder.

    Args:
        num_tokens (int): Number of unique tokens in the vocabulary.
        embedding_dim (int): Dimension of the embedding space. Defaults to 128.
        num_layers (int): Number of layers in the Transformer encoder. Defaults to 2.
        dropout (float): Dropout rate applied to embeddings and encoder. Defaults to 0.0.
    """
    def __init__(self, num_tokens: int, embedding_dim: int = 128, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_dim)
        
        # Positional encoding layer
        self.pos_encoding = PositionalEncoding(d_model=embedding_dim, dropout=dropout)
        
        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dropout=dropout),
            num_layers=num_layers
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Processes input tokens through embedding, positional encoding, and Transformer encoder.

        Args:
            x (Tensor): Input tensor of shape ``[batch_size, seq_len]``, where each entry
                        is a token index.

        Returns:
            Tensor: Encoded tensor of shape ``[batch_size, embedding_dim]`` representing
                    the mean-pooled sequence features.
        """
        # Embed tokens and apply positional encoding
        embeds = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        embeds_pe = self.pos_encoding(embeds)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # Encode sequences using Transformer encoder
        z = self.encoder(embeds_pe)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # Mean-pool along the sequence dimension
        z_mean = z.mean(dim=1)  # Shape: [batch_size, embedding_dim]
        return z_mean
