import torch
import math
import torch.nn.functional as F

from torch.nn.modules import TransformerEncoderLayer
from torch.nn.modules import TransformerEncoder as Enc

from torch import nn
from torch.nn.modules.normalization import LayerNorm


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):

    def __init__(self,
                 vocab: int,
                 d_model: int,
                 n_head: int,
                 n_layers: int,
                 dim_ff: int,
                 dropout: float,
                 pad_id: int):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab, d_model)
        self.position_embedding = PositionalEncoding(d_model, dropout)

        enc_layer = TransformerEncoderLayer(d_model, n_head, dim_ff)
        enc_norm = LayerNorm(d_model)
        self.encoder = Enc(enc_layer, n_layers, enc_norm)

    def forward(self, src):
        '''
        src: input token index sequence: N * S
        '''
        if len(src.size()) == 1:
            src = src.reshape(1, -1)
        seq_mask = (src == self.pad_id)

        x = self.embedding(src) * math.sqrt(self.d_model)  # (N * S * E)
        x = self.position_embedding(x)
        x = x.transpose(1, 0)  # (S * N * E)

        # S * S
        attn_mask = torch.full((x.size()[0], x.size()[0]), -float('Inf'), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)  # for forward attention

        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=seq_mask).transpose(1, 0)  # (N * S * E)
        return x


class Classifier(nn.Module):
    def __init__(self, d_model, n_class, d_ff=512, dropout=0.5):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, n_class)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class SeqTagger(nn.Module):
    def __init__(self,
                 vocab: int,
                 d_model: int,
                 n_head: int,
                 n_layers: int,
                 dim_ff: int,
                 n_class: int,
                 dropout: float,
                 pad_id: int):
        super().__init__()

        self.Enc = TransformerEncoder(vocab=vocab,
                                      d_model=d_model,
                                      n_head=n_head,
                                      n_layers=n_layers,
                                      dim_ff=dim_ff,
                                      dropout=dropout,
                                      pad_id=pad_id)

        self.classifier = Classifier(d_model=d_model,
                                     n_class=n_class,
                                     dropout=dropout)

    def forward(self, src_id: torch.LongTensor):
        x = self.Enc(src_id)
        logits = self.classifier(x)
        return logits
