import numpy as np

devy = np.load('data/dev_transcripts_raw.npy', encoding='bytes')
trainy = np.load('data/train_transcripts_raw.npy', encoding='bytes')

# create list of all characters
chars = []
for idx, label in enumerate(trainy):
    for word in label:
        for ch in word:
            if ch not in chars:
                chars.append(ch)

for idx, label in enumerate(devy):
    for word in label:
        for ch in word:
            if ch not in chars:
                chars.append(ch)

# add the following special characters
chars.append(' ')
chars.append('<sos>')
chars.append('<eos>')

# log
print('All characters:', chars)
print('Number of characters:', len(chars))

# create dictionary for quick mapping
char_to_num = {ch : i for i, ch in enumerate(chars)}

# reverse map
num_to_char = {i : ch for i, ch in enumerate(chars)}

# char to num
print('char to num', char_to_num)

# num to char
print('num to char', num_to_char)

# convert data to letters
trainy_letters = []
for idx, label in enumerate(trainy):
    line = []
    line.append(char_to_num['<sos>'])
    for word in label:
        for letter in word:
            line.append(char_to_num[letter])
        line.append(char_to_num[' '])
    line.append(char_to_num['<eos>'])
    trainy_letters.append(line)

# convert data to letters
devy_letters = []
for idx, label in enumerate(devy):
    line = []
    line.append(char_to_num['<sos>'])
    for word in label:
        for letter in word:
            line.append(char_to_num[letter])
        line.append(char_to_num[' '])
    line.append(char_to_num['<eos>'])
    devy_letters.append(line)

# convert to numpy
devy_letters = np.array(devy_letters)
trainy_letters = np.array(trainy_letters)

# save the data as letters
np.save('data/dev_transcripts_l.npy', devy_letters)
np.save('data/train_transcripts_l.npy', trainy_letters)
