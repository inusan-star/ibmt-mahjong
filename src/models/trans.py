import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    Vanilla Transformer
    """

    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        num_classes: int,
        emb_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.input_projection = nn.Linear(input_dim, emb_dim)
        self.input_norm = nn.LayerNorm(emb_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, emb_dim))
        self.embedding_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(emb_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.action_head = nn.Linear(emb_dim, 1)

        self.apply(self._init_weights)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def _init_weights(
        self,
        module,
        bias_init_val=0.0,
    ):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.constant_(module.bias, bias_init_val)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass
        """
        _, seq_len, input_dim = x.shape

        assert seq_len == self.seq_len, f"Expected sequence length {self.seq_len}, got {seq_len}"
        assert input_dim == self.input_dim, f"Expected input dimension {self.input_dim}, got {input_dim}"

        x = self.input_projection(x)
        x = x + self.pos_embedding
        x = self.input_norm(x)
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.final_norm(x)
        x = self.dropout_layer(x)

        action = self.action_head(x).squeeze(-1)

        assert action.shape[1] == self.num_classes, f"Expected classes {self.num_classes}, got {action.shape[1]}"

        return {"action": action}
