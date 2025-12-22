import torch
import torch.nn as nn
import pickle
import re

DEVICE = torch.device("cpu")

MAX_LEN = 30

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        _, (h, c) = self.encoder(src)

        tgt = self.embedding(tgt)
        out, _ = self.decoder(tgt, (h, c))
        return self.fc(out)

# Load vocab
with open("model/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

with open("model/idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

vocab_size = len(word2idx)

# Load model
encoder = Encoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
model = Seq2SeqAttention(encoder, decoder).to(DEVICE)

state = torch.load("model/seq2seq_attention.pt", map_location=DEVICE)
model.load_state_dict(state)
model.eval()


def encode(sentence):
    tokens = tokenize(sentence)
    seq = [word2idx["<SOS>"]] + \
          [word2idx.get(w, word2idx["<UNK>"]) for w in tokens] + \
          [word2idx["<EOS>"]]

    seq = seq[:MAX_LEN]
    seq += [word2idx["<PAD>"]] * (MAX_LEN - len(seq))
    return torch.tensor(seq).unsqueeze(0)

def decode(indices):
    words = []
    for idx in indices:
        word = idx2word.get(idx, "")
        if word in ["<EOS>", "<PAD>"]:
            break
        if word not in ["<SOS>"]:
            words.append(word)
    return " ".join(words)

def generate_answer(question):
    src = encode(question)

    with torch.no_grad():
        embedded = model.embedding(src)
        _, (h, c) = model.encoder(embedded)

        decoder_input = torch.tensor([[word2idx["<SOS>"]]])
        outputs = []

        for _ in range(MAX_LEN):
            dec_embed = model.embedding(decoder_input)
            out, (h, c) = model.decoder(dec_embed, (h, c))
            logits = model.fc(out.squeeze(1))
            next_token = logits.argmax(dim=1).item()

            if next_token == word2idx["<EOS>"]:
                break

            outputs.append(next_token)
            decoder_input = torch.tensor([[next_token]])

    return decode(outputs)
