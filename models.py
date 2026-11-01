import torch
import torch.nn as nn
import torch.nn.functional as F


class ElmanRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hx = nn.Linear(hidden_size, hidden_size)
        self.W_yh = nn.Linear(hidden_size, vocab_size)

    def forward(self,x,prev_hidden):
        seq_length, batch_size = x.size()
        outputs = []

        for t in range(seq_length):
            x_t = self.embedding(x[t])

            prev_hidden = torch.tanh(self.W_hx(x_t) + self.W_hh(prev_hidden))

            y_t = self.W_yh(prev_hidden)

            outputs.append(y_t)

        outputs = torch.stack(outputs)

        return outputs, prev_hidden


    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size, device=device)



class JordanRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.W_hy = nn.Linear(vocab_size, hidden_size)
        self.W_hx = nn.Linear(hidden_size, hidden_size)
        self.W_yh = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, prev_output):
        seq_length, batch_size = x.size()
        outputs = []

        for t in range(seq_length):
            x_t = self.embedding(x[t])

            hidden = torch.tanh(self.W_hx(x_t) + self.W_hy(prev_output))

            y_t = self.W_yh(hidden)

            prev_output = F.softmax(y_t, dim=-1)

            outputs.append(y_t)

        outputs = torch.stack(outputs)

        return outputs, prev_output

    def init_output(self, batch_size,device):
        return torch.zeros(batch_size, self.vocab_size, device=device)
