"""DistilBERT bi- and cross-encoder (`distilbert-base-uncased`): address pair -> match logit.

- Bi-encoder: each address alone ([CLS] A [SEP]), masked mean pooling, head on [|u-v|; u*v] (as the siamese BiLSTM).
- Cross-encoder: both addresses in one row ([CLS] A [SEP] B [SEP]), masked mean + max pooling over all tokens,
  ESIM's classifier head (MLP 200, tanh, dropout 0.3).

Attribute names are the state_dict keys of the saved checkpoints. Training: training.py.
"""
from functools import lru_cache

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from siamese.model import PairDataset

MODEL_NAME = "distilbert-base-uncased"
MAX_TOKENS = 64                  # bi-encoder: word pieces kept per address (before [CLS] / [SEP])
BATCH_SIZE = 32
MLP_HIDDEN, DROPOUT = 200, 0.3   # cross-encoder head, as in ESIM


@lru_cache(maxsize=None)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


# ---------- batches ----------

def encode_batch(strings, vocab, max_len=MAX_TOKENS):
    """Bi-encoder: strings -> (ids (B, T) padded with PAD, attention mask (B, T)); one address per row."""
    tok = get_tokenizer()
    unk_id, pad_id = tok.unk_token_id, tok.pad_token_id
    strings = [["[CLS]"] + tok.tokenize(s)[:max_len] + ["[SEP]"] for s in strings]
    batch_len = max([len(s) for s in strings])
    ret = []
    for tokens in strings:
        res = [vocab.get(token, unk_id) for token in tokens]
        res += [pad_id] * (batch_len - len(res))
        ret.append(res)
    lengths = [len(s) for s in strings]
    max_length = max(lengths)
    lengths_mask = (torch.arange(max_length)[None, :] < torch.tensor(lengths, dtype=torch.long).unsqueeze(-1)).long()
    return torch.tensor(ret, dtype=torch.long), lengths_mask


def make_bi_collate(vocab):
    """(address_a, address_b, label) samples -> ids_a, mask_a, ids_b, mask_b, y."""
    def collate(samples):
        addr_a, addr_b, labels = zip(*samples)
        ids_a, mask_a = encode_batch(list(addr_a), vocab)
        ids_b, mask_b = encode_batch(list(addr_b), vocab)
        y = torch.tensor(labels, dtype=torch.float32)
        return ids_a, mask_a, ids_b, mask_b, y
    return collate


def make_cross_collate():
    """(address_a, address_b, label) samples -> ids, mask, y; one row per pair: [CLS] A [SEP] B [SEP]."""
    tok = get_tokenizer()

    def collate(samples):
        addr_a, addr_b, labels = zip(*samples)
        lengths = torch.tensor([len(x) for x in tok(addr_a, addr_b)["input_ids"]])
        ids = tok(addr_a, addr_b, padding=True, return_tensors="pt")
        mask = (torch.arange(lengths.max())[None, :] < lengths[:, None]).long()
        assert torch.equal(mask, ids["attention_mask"])
        y = torch.tensor(labels, dtype=torch.float32)
        return ids["input_ids"], mask, y
    return collate


def make_loaders(kind, splits):
    """kind: "bi" or "cross"; splits: {"train": (df, labels), ...} -> {name: DataLoader}; only train shuffles."""
    collate = make_bi_collate(get_tokenizer().get_vocab()) if kind == "bi" else make_cross_collate()
    return {name: DataLoader(PairDataset(df, labels), batch_size=BATCH_SIZE, shuffle=(name == "train"),
                             collate_fn=collate)
            for name, (df, labels) in splits.items()}


# ---------- layers ----------

class MeanPooling(nn.Module):
    def forward(self, h, mask):
        # masked mean: zero the pad positions, divide by the true length
        return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)   # (N, T, 768) -> (N, 768)


class MeanMaxPooling(nn.Module):
    def forward(self, h, mask):
        m = mask.unsqueeze(-1).bool()                                  # (B, T, 1), True = real token
        mean_pool = (h * m).sum(1) / m.sum(1)                          # pads add 0 and don't count
        max_pool = h.masked_fill(~m, float("-inf")).max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=1)


class BertEncoder(nn.Module):
    """DistilBERT + pooling. pooling="mean" (bi-encoder) or "mean_max" (cross-encoder)."""
    def __init__(self, pooling="mean"):
        super().__init__()
        self.pool = MeanPooling() if pooling == "mean" else MeanMaxPooling()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.output_dim = self.bert.config.hidden_size * (1 if pooling == "mean" else 2)

    def forward(self, ids, mask):
        bert_res = self.bert(input_ids=ids, attention_mask=mask)
        return self.pool(bert_res.last_hidden_state, mask)


class ClassifierHead(nn.Module):
    """Pooled vector -> MLP (MLP_HIDDEN, tanh, dropout) -> 1 logit."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, MLP_HIDDEN)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(MLP_HIDDEN, 1)

    def forward(self, vectors):
        return self.linear2(self.dropout(self.tanh(self.linear1(vectors)))).squeeze(-1)


# ---------- models ----------

class BiEncoder(nn.Module):
    """Pair of address ids -> one match logit. ONE encoder, called on both sides (weight sharing)."""
    def __init__(self):
        super().__init__()
        self.encoder = BertEncoder("mean")
        self.head = nn.Linear(2 * self.encoder.output_dim, 1)

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        a_out = self.encoder(ids_a, mask_a)                  # (B, T_a) -> (B, 768)
        b_out = self.encoder(ids_b, mask_b)
        combined = torch.cat([(a_out - b_out).abs(), a_out * b_out], dim=1)   # order-free: score(a, b) == score(b, a)
        return self.head(combined).squeeze(-1)               # logits (B,); BCEWithLogitsLoss applies the sigmoid


class CrossEncoder(nn.Module):
    """Joined pair row -> one match logit."""
    def __init__(self):
        super().__init__()
        self.encoder = BertEncoder("mean_max")
        self.head = ClassifierHead(self.encoder.output_dim)

    def forward(self, ids, mask):
        return self.head(self.encoder(ids, mask))


MODELS = {"bi": BiEncoder, "cross": CrossEncoder}
