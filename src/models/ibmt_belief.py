import torch
import torch.nn as nn

import src.config as config


class IBMTBelief(nn.Module):
    """
    Intent-Belief Multi-task Transformer for Belief Prediction
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
        self.num_heads = num_heads

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

        # MST: Buffer for the 4 static tile-topology matrices
        self.register_buffer("relation_matrices", self._generate_relation_matrices(seq_len))
        # MST: Trainable coefficients to scale each relation per head
        self.alphas = nn.Parameter(torch.full((num_heads, 4), 0.1))

        # IBMT_belief: Belief head for 3 players' shanten number prediction
        self.belief_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 3 * config.NUM_SHANTEN_CLASSES)
        )

        self.apply(self._init_weights)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def _generate_relation_matrices(self, seq_len: int) -> torch.Tensor:
        """
        MST: Build R_ident, R_adj, R_gap, and R_suit for mahjong tile space.
        """
        matrices = torch.zeros(4, seq_len, seq_len)

        if seq_len != 34:
            return matrices

        # Index ranges: 0-8:m, 9-17:p, 18-26:s, 27-33:z
        suits = [(0, 8), (9, 17), (18, 26)]

        for i in range(34):
            for j in range(34):
                # R_ident: Identity relationship
                if i == j:
                    matrices[0, i, j] = 1.0

                for start, end in suits:
                    if start <= i <= end and start <= j <= end:
                        dist = abs(i - j)

                        # R_adj: Adjacency relationship
                        if dist == 1:
                            matrices[1, i, j] = 1.0

                        # R_gap: Gap relationship
                        elif dist == 2:
                            matrices[2, i, j] = 1.0

                        # R_suit: Suit relationship
                        matrices[3, i, j] = 1.0

                # R_suit: Suit relationship
                if 27 <= i <= 33 and 27 <= j <= 33:
                    matrices[3, i, j] = 1.0

        return matrices

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
        batch_size, seq_len, input_dim = x.shape

        assert seq_len == self.seq_len, f"Expected sequence length {self.seq_len}, got {seq_len}"
        assert input_dim == self.input_dim, f"Expected input dimension {self.input_dim}, got {input_dim}"

        x = self.input_projection(x)
        x = x + self.pos_embedding
        x = self.input_norm(x)
        x = self.embedding_dropout(x)

        # MST: Weighted sum of relations
        bias_mst = torch.sum(self.alphas.view(self.num_heads, 4, 1, 1) * self.relation_matrices.unsqueeze(0), dim=1)
        # MST: Robust shape adjustment for MultiheadAttention
        mst_mask = bias_mst.repeat(batch_size, 1, 1)

        x = self.transformer_encoder(x, mask=mst_mask)
        x = self.final_norm(x)

        x_avg = x.mean(dim=1)

        # IBMT_belief: Belief prediction
        belief = self.belief_head(x_avg)

        x_out = self.dropout_layer(x)

        action = self.action_head(x_out).squeeze(-1)

        assert action.shape[1] == self.num_classes, f"Expected classes {self.num_classes}, got {action.shape[1]}"

        return {"action": action, "belief": belief}
