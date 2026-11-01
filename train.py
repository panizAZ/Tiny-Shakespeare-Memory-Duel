import torch
import torch.nn as nn
import math
import time
from torch.utils.data import Dataset, DataLoader, random_split

from preprocess import preprocess
from models import ElmanRNN, JordanRNN


class CharDataset(Dataset):
    def __init__(self, X,Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train_model(
        rnn_model="Jordan",
        path="data/tiny-shakespeare.txt",
        seq_length=50,
        data_fraction=0.25,
        batch_size=64,   #64 sequence
        hidden_size=256,
        num_epoch=10,
        lr = 0.0005,
        device=None
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load and Preprocess
    X, Y, vocab_size, char_to_ix, ix_to_char = preprocess(path, seq_length, data_fraction)

    dataset = CharDataset(X, Y)
    train_size = int(len(dataset)*0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2) Build Model
    if rnn_model.lower() == "elman":
        model = ElmanRNN(vocab_size, hidden_size)
    elif rnn_model.lower() == "jordan":
        model = JordanRNN(vocab_size, hidden_size)
    else:
        raise ValueError("RNN type must be Elman or Jordan")

    model.to(device)

    # 3) Load and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    # perplexity
    train_ppls = []
    val_ppls = []

    # Accuracy
    train_accs = []
    val_accs = []

    # 4) Training loop
    start_time = time.time()
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0

        train_total_correct = 0
        train_total_chars = 0
        for x_batch,y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)


            #(batch_size, seq_length) -> (seq_length, batch_size)
            # That's how our RNN works
            x_batch = x_batch.transpose(0,1)
            y_batch = y_batch.transpose(0,1)

            # For last batch and mismatch error
            current_batch_size = x_batch.size(1)

            # because torch accumulates gradients
            optimizer.zero_grad()

            # assumption: batches are independent
            if rnn_model.lower() == "elman":
                prev_hidden = model.init_hidden(current_batch_size,device)
                outputs,_ = model(x_batch, prev_hidden)
            elif rnn_model.lower() == "jordan":
                prev_output = model.init_output(current_batch_size,device)
                outputs,_ = model(x_batch, prev_output)

            # reshape for loss:
            # outputs: (seq_len, batch_size, vocab_size)
            # -> (seq_len * batch_size, vocab_size)
            outputs = outputs.reshape(-1, vocab_size)
            # targets: (seq_len, batch_size) -> (seq_len * batch_size)
            y_batch = y_batch.reshape(-1)

            preds = torch.argmax(outputs, dim=1)

            train_total_correct += (preds == y_batch).sum().item()
            train_total_chars += y_batch.numel()


            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # loss.item -> only need the numerical value of loss
            total_loss += loss.item()

        # Validation loop
        model.eval()

        val_loss_total = 0
        val_total_correct = 0
        val_total_chars = 0

        with torch.no_grad():

            for x_batch, y_batch in val_loader:

                x_batch = x_batch.transpose(0, 1).to(device)
                y_batch = y_batch.transpose(0, 1).to(device)

                current_batch_size = x_batch.size(1)

                if rnn_model.lower() == "elman":
                    state = model.init_hidden(current_batch_size, device)
                else:
                    state = model.init_output(current_batch_size, device)

                outputs, _ = model(x_batch, state)

                outputs = outputs.reshape(-1, vocab_size)
                targets = y_batch.reshape(-1)

                preds = torch.argmax(outputs, dim=1)


                val_total_correct += (preds == targets).sum().item()
                val_total_chars += targets.numel()

                loss = criterion(outputs, targets)

                val_loss_total += loss.item()

        # -----Train data-------
        avg_train_loss = total_loss / len(train_loader)
        # on unseen data
        #train_ppl = torch.exp(torch.tensor(avg_loss))
        train_ppl = math.exp(avg_train_loss)
        train_losses.append(avg_train_loss)
        train_ppls.append(train_ppl)

        train_accuracy = train_total_correct/train_total_chars
        train_accs.append(train_accuracy)

        # -----Validation data------
        avg_val_loss = val_loss_total / len(val_loader)
        val_ppl = math.exp(avg_val_loss)

        val_losses.append(avg_val_loss)
        val_ppls.append(val_ppl)

        val_accuracy = val_total_correct / val_total_chars
        val_accs.append(val_accuracy)

        # Compare vaidation and train
        print(f"Epoch [{epoch + 1}/{num_epoch}]")
        print(
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train PPL: {train_ppl:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val PPL: {val_ppl:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    end_time = time.time()

    total_time = end_time - start_time

    print(f"Training Time: {total_time:.2f} seconds")

    # 5) Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'char_to_ix': char_to_ix,
        'ix_to_char': ix_to_char,
        'hidden_size': hidden_size,
        'rnn_model': rnn_model,
        'train_losses': train_losses,
        'train_ppls': train_ppls,
        'train_accuracies': train_accs,
        'val_losses': val_losses,
        'val_ppls': val_ppls,
        'val_accuracies': val_accs,
        'train_time': total_time
    }, f'{rnn_model.lower()}_model.pth')

    print(f"Model saved as {rnn_model.lower()}_model.pth")

if __name__ == "__main__":
    train_model()

