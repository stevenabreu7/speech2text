
TODO:
  implement custom packed sequences
  implement loss function properly
  teacher forcing

Dimensions:
  BS:  BATCH_SIZE
    batch size.
  AUF: AUDIO_FEATURES
    number of audio features in the input.
  HFL: HIDDEN_FEATURES_LISTENER
    number of hidden units in the listener.
  AUL: AUDIO_LENGTH
    length of the input audio sequence batch.
  RAL: REDUCED_AUDIO_LENGTH
    length of the input audio sequence after the pBLSTM.
  CS:  CONTEXT_SIZE
    context size for the listener output and speller output.
  LAL: LABEL_LENGTH
    length of the text sequence that is our target.
  VOC: VOCAB_SIZE
    number of different words/characters in the vocabulary.

Symbols:
  SOS = 32
  EOS = 33