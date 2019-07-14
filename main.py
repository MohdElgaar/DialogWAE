import torch
from model import DialogWAE
from os import makedirs

makedirs('checkpoints', exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DialogWAE('train/dialogues_train.txt',
                  'test/dialogues_test.txt',
                  None,
                  32,
                  device).to(device)
# model = DialogWAE('train/dialogues_train.txt',
#                   'test/dialogues_test.txt',
#                   'glove.twitter.27B.200d.txt',
#                   32,
#                   device).to(device)
# model = torch.load('checkpoints/model.pkl').to(device)
model.fit(1000, 1)
