import torch
import torch.nn.functional as F

from models import ElmanRNN,JordanRNN


def generate_text(
        model_path,
        text_seed="to be",
        length=200,
        temperature=1.0,
        device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1)Load Checkpoint
    check_point = torch.load(model_path, map_location=device)

    vocab_size=check_point['vocab_size']
    char_to_ix = check_point['char_to_ix']
    ix_to_char = check_point['ix_to_char']
    hidden_size = check_point['hidden_size']
    rnn_model = check_point['rnn_model']

    # 2)Rebuild model
    if rnn_model.lower() == "elman":
        model = ElmanRNN(vocab_size,hidden_size)

    else:
        model = JordanRNN(vocab_size,hidden_size)

    model.load_state_dict(check_point['model_state_dict'])
    model.to(device)
    model.eval()

    # 3)initial state
    if rnn_model.lower() == "elman":
        state = model.init_hidden(batch_size=1, device=device)
    else:
        state = model.init_output(batch_size=1, device=device)

    # 4)convert seed text to indices
    # error handling if char is unknown
    seed_index = [char_to_ix[char] for char in text_seed if char in char_to_ix]
    generated = text_seed


    # 5)warm up with seed -> to start from a hidden state with context
    for ix in seed_index[:-1]:
        # for test and prediction :  seq_length = 1 & batch_size = 1
        x = torch.tensor([[ix]], dtype=torch.long).to(device)
        outputs, state= model(x,state)

    # We want to start prediction from the last character
    if len(text_seed) != 0:
        current_index = seed_index[-1]
    else:
        raise ValueError("text seed is empty")

    # 6)generate character
    for l in range(length):
        # for test and prediction :  seq_length = 1 & batch_size = 1
        x = torch.tensor([[current_index]], dtype=torch.long).to(device)
        outputs, state= model(x,state)

        # (1, 1, vocab_size) -> (vocab_size) : raw, unnormalized scores for each char
        logits = outputs.squeeze()

        probs = F.softmax(logits/temperature, dim=0)

        next_ix = torch.multinomial(probs, 1).item()
        #next_ix = torch.argmax(probs, dim=0).item()

        next_char = ix_to_char[next_ix]
        generated += next_char

        current_index = next_ix

    return generated


if __name__ == "__main__":

    text = generate_text(
        model_path="jordan_model.pth",
        text_seed="To be",
        length=500,
        temperature=1.2
    )
    print(text)

