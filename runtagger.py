# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>
# should take torch model, word_to_idx dict, char_to_idx dict, tag_to_idx dict

import os
import math
import sys
import datetime
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNNBiLSTMTagger(nn.Module):
    def __init__(self, D_in, D_in_char, H, H_char, char_size, vocab_size, tagset_size):
        super(CharCNNBiLSTMTagger, self).__init__()
        self.H = H
        self.word_embeddings = nn.Embedding(vocab_size + 1, D_in, padding_idx=vocab_size)
        self.char_embeddings = nn.Embedding(char_size + 1, D_in_char, padding_idx=char_size)
        # self.conv1d = nn.Conv1d(D_in_char, H_char, kernel_size=3)
        self.lstm = nn.LSTM(D_in, H, num_layers=1, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(H*2, tagset_size)

    def forward(self, x):
        embeds = self.word_embeddings(x)
        lstm_out, _ = self.lstm(embeds.view(len(x), 1, -1))
        hidden2tag_out = self.hidden2tag(lstm_out.view(len(x), -1))
        tag_scores = F.log_softmax(hidden2tag_out)
        return tag_scores

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    # use torch library to load model_file
    with open(model_file, "rb") as f:
        data = pickle.load(f)
        model_state_dict = data[0]
        char_to_idx = data[1]
        word_to_idx = data[2]
        tag_to_idx = data[3]
        idx_to_tag = data[4]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CharCNNBiLSTMTagger(300, 10, 50, 5, len(char_to_idx), len(word_to_idx), len(tag_to_idx))
    model.to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    with open(test_file, 'r') as f:
        test_sents = f.readlines()

    with open(out_file, 'w+') as f:
        with torch.no_grad():
            for test_sent in test_sents: # tag sentence and write output to out_file
                splitted = test_sent.split()
                word_idxs = []
                for word in splitted:
                    if word in word_to_idx:
                        word_idxs.append(word_to_idx[word])
                    else:
                        word_idxs.append(len(word_to_idx))
                x = torch.tensor(word_idxs, dtype=torch.long)
                x = x.to(device)
                y_pred = model(x)

                pred_tags = torch.argmax(y_pred, dim=1)

                tagged_sent = ""
                for i in range(len(splitted)):
                    tagged_sent = tagged_sent + splitted[i] + "/" + idx_to_tag[pred_tags[i].item()] + " "
                tagged_sent += '\n'
                f.write(tagged_sent)

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)

