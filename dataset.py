import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import wordpunct_tokenize
from collections import Counter
from hparams import *

class MyDataset(Dataset):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3
    SOD = 4
    vocab_counter = 5

    def __init__(self, source_file, vocab = None):
        self.vocab = vocab
        dataset, self.vocab = self.init_dataset(source_file)
        
        inputs = list()
        targets = list()
        for entry in dataset:
            entry = self.to_num(entry)
            for i in range(2, len(entry)):
                if(i > MAX_CONTEXT):
                    offset = i - MAX_CONTEXT
                else:
                    offset = 0
            inputs.append(entry[offset:i])
            targets.append(entry[i])
        self.inputs = inputs
        self.targets = targets
        
    def init_dataset(self, source_file):
        c = Counter()
        dataset = []
        with open(source_file, 'r') as f:
            for dialog in f:
                dialog = [wordpunct_tokenize(u.strip(" ").lower()) for u in dialog.strip("__eou__\n").split("__eou__")]
                dataset.append(dialog)
                for utt in dialog:
                    c.update(utt)
        
        if self.vocab:
            vocab = self.vocab
        else:
            vocab = dict()
            vocab['num_to_word'] = dict()
            vocab['word_to_num'] = dict()
            vocab['num_to_word'][self.PAD] = 'pad'
            vocab['num_to_word'][self.SOS] = 'sos'
            vocab['num_to_word'][self.EOS] = 'eos'
            vocab['num_to_word'][self.UNK] = 'unk'
            vocab['num_to_word'][self.SOD] = 'sod'
            vocab['word_to_num']['pad'] = 0
            vocab['word_to_num']['sos'] = 1
            vocab['word_to_num']['eos'] = 2
            vocab['word_to_num']['unk'] = 3
            vocab['word_to_num']['sod'] = 4


            for k,_ in c.most_common(VOCAB_SIZE-self.vocab_counter):
                vocab['word_to_num'][k] = self.vocab_counter
                vocab['num_to_word'][self.vocab_counter] = k
                self.vocab_counter += 1
            
        return dataset, vocab
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
    def to_num(self, dialog):
        new_dialog = list()
        new_dialog.append([self.SOS, self.SOD, self.EOS])
        for i, utt in enumerate(dialog):
            new_utt = list()
            new_utt.append(self.SOS)
            for j, word in enumerate(utt):
                if j == MAX_UTT - 2:
                    break;
                if word in self.vocab['word_to_num']:
                    new_utt.append(self.vocab['word_to_num'][word])
                else:
                    new_utt.append(self.UNK)
            new_utt.append(self.EOS)
            new_dialog.append(new_utt)
        return new_dialog
    
    def to_text(self, utt):
        text = []
        for num in utt:
            if num == self.PAD:
                break
            text.append(self.vocab["num_to_word"][num])
        return text
    
    def padded(self, utt, max_len = MAX_UTT):
        return utt + [self.PAD] * (max_len - len(utt))

    def collector(self, samples):
        X = list()
        Y = list()
        for context, target in samples:
            X.append(context)
            Y.append(target)
        max_context = max([len(c) for c in X])
        for i in range(len(X)):
            X[i] += [[]] * (max_context - len(X[i]))
        Xlen = list()
        for context in X:
            Xlen += [len(utt) for utt in context]

        max_utt = max(Xlen)
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] = self.padded(X[i][j], max_utt)

        Ylen = [len(utt) for utt in Y]
        max_target = max(Ylen)

        for i in range(len(Y)):
            Y[i] = self.padded(Y[i], max_target)

        return torch.tensor(X), torch.tensor(Xlen), torch.tensor(Y), torch.tensor(Ylen)
