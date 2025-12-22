import torch
import torch.nn as nn
import re
import pickle

# ===============================
# Device
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Load tokenizer
# ===============================
with open("model/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

with open("model/idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

PAD_IDX = word2idx["<pad>"]
SOS_IDX = word2idx["<sos>"]
EOS_IDX = word2idx["<eos>"]

VOCAB_SIZE = len(word2idx)
EMBED_DIM = 128
HIDDEN_DIM = 128
MAX_LEN = 40

# ===============================
# Tokenization
# ===============================
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def encode(text):
    tokens = tokenize(text)
    ids = [word2idx.get(w, PAD_IDX) for w in tokens][:MAX_LEN]
    return torch.tensor(ids).unsqueeze(0).to(DEVICE)

# ===============================
# Attention
# ===============================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)

# ===============================
# Encoder (BIDIRECTIONAL)
# ===============================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            EMBED_DIM,
            HIDDEN_DIM,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)

        # Combine forward + backward
        hidden = hidden[0] + hidden[1]
        cell = cell[0] + cell[1]

        return outputs, hidden, cell

# ===============================
# Decoder
# ===============================
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.attention = Attention(HIDDEN_DIM)
        self.lstm = nn.LSTM(EMBED_DIM + HIDDEN_DIM * 2, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, token, hidden, cell, encoder_outputs):
        token = token.unsqueeze(1)
        embedded = self.embedding(token)

        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(
            lstm_input, (hidden.unsqueeze(0), cell.unsqueeze(0))
        )

        prediction = self.fc(output.squeeze(1))
        return prediction, hidden.squeeze(0), cell.squeeze(0)

# ===============================
# Seq2Seq
# ===============================
class Seq2SeqAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

# ===============================
# Load model
# ===============================
model = Seq2SeqAttention().to(DEVICE)
model.load_state_dict(
    torch.load("model/seq2seq_attention.pt", map_location=DEVICE)
)
model.eval()

# ===============================
# Rewrite function
# ===============================
@torch.no_grad()
def rewrite_text(legal_text: str) -> str:
    if not legal_text or len(legal_text.strip()) < 20:
        return legal_text

    try:
        src = encode(legal_text)
        encoder_outputs, hidden, cell = model.encoder(src)

        token = torch.tensor([SOS_IDX], device=DEVICE)
        result = []

        for _ in range(MAX_LEN):
            output, hidden, cell = model.decoder(
                token, hidden, cell, encoder_outputs
            )

            pred = output.argmax(1).item()
            if pred == EOS_IDX:
                break

            result.append(idx2word.get(pred, ""))
            token = torch.tensor([pred], device=DEVICE)

        rewritten = rewrite_text(factual_answer)

        if len(rewritten.split()) < 8:
            return factual_answer

        return rewritten


    except Exception as e:
        print("Rewriter error:", e)
        return legal_text
