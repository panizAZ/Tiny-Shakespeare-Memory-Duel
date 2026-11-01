import numpy as np

def load_text(path, data_fraction=0.1):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Use only a fraction of dataset
    cut = int(len(text)*data_fraction)
    text = text[:cut]

    return text


def vocab_index(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    ix_to_char={i:ch for i,ch in enumerate(chars)}
    char_to_ix={ch:i for i,ch in enumerate(chars)}

    return ix_to_char, char_to_ix, chars, vocab_size


def encode_text(text, char_to_ix):
    return np.array([char_to_ix[char] for char in text])


def create_sequence(encoded_text, seq_length = 5):
    X = []
    Y = []

    for i in range(len(encoded_text) - seq_length):
        x_seq = encoded_text[i:i+seq_length]
        y_seq = encoded_text[i+1:i+seq_length+1]

        X.append(x_seq)
        Y.append(y_seq)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def preprocess(path,seq_length=5,data_fraction=0.5):
    text = load_text(path, data_fraction)
    ix_to_char, char_to_ix, chars, vocab_size = vocab_index(text)
    encoded_text = encode_text(text, char_to_ix)
    X,Y = create_sequence(encoded_text, seq_length)

    return X,Y,vocab_size,char_to_ix, ix_to_char

