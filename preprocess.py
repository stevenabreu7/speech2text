import torch
import numpy as np
import pickle as pkl

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

trainy = np.load('data/dev_transcripts_raw.npy', encoding='bytes')

# create lists of all characters and words in the data
syms = []
vocab = []
for idx, label in enumerate(trainy):
    if idx % 100 == 0:
        print(idx, end='\r')
    for word in label:
        if word not in vocab:
            vocab.append(word)
            for ch in word:
                if ch not in syms:
                    syms.append(ch)
print()

# save words
words = []
for idx, label in enumerate(trainy):
    if idx % 100 == 0:
        print(idx, end='\r')
    lbl = []
    for word in label:
        lbl.append(vocab.index(word))
    words.append(np.array(lbl))
print()
words = np.array(words)


# save letters
letters = []
for idx, label in enumerate(trainy):
    # d = {' ': 31, '<sos>': 32, '<eos>': 33}
    if idx % 100 == 0:
        print(idx, end='\r')
    lbl = [32]
    for word in label:
        for letter in word:
            lbl.append(syms.index(letter))
        lbl.append(31)
    lbl.append(33)
    letters.append(np.array(lbl))
print()
letters = np.array(letters)

# save the data as words and letters
np.save('data/dev_transcripts_w.npy', words)
np.save('data/dev_transcripts_l.npy', letters)

# save the dictionaries to lookup words and letters
d_syms = {i : x for i, x in enumerate(syms)}
d_vocab = {i : x for i, x in enumerate(vocab)}
save_obj(d_syms, 'data/dev_syms')
save_obj(d_vocab, 'data/dev_vocab')