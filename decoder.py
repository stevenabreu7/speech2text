import pickle as pkl

ts = pkl.load(open('data/train_syms.pkl', 'rb'))
ds = pkl.load(open('data/dev_syms.pkl', 'rb'))

def decode_train(s):
    res = ''
    for ch in s:
        if ch == 32:
            res += '<sos>'
        elif ch == 31:
            res += '-'
        elif ch == 33:
            res += '<eos>'
        else:
            res += chr(ts[ch])
    return res

# # logging
# print(decode_train(y[0].numpy()))
# p = torch.argmax(pred[0], dim=0)
# print(decode_train(p.numpy()))