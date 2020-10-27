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
    def __init__(self, word_embed_dim, char_embed_dim, char_hidden_dim, lstm_hidden,
                 char_size, vocab_size, tagset_size):
        super(CharCNNBiLSTMTagger, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size + 1, word_embed_dim, padding_idx=vocab_size)
        self.char_embeddings = nn.Embedding(char_size + 1, char_embed_dim, padding_idx=char_size)

        self.conv1d = nn.Conv1d(char_embed_dim, char_hidden_dim, kernel_size=3)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.lstm = nn.LSTM(word_embed_dim + char_hidden_dim,
                            lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(lstm_hidden*2, tagset_size)

    def forward(self, x1, x2):
        word_embeds = self.word_embeddings(x1)
        char_embeds = self.char_embeddings(x2).transpose(1,2)
        try:
            conv1d_out = self.conv1d(char_embeds)
        except Exception as e:
            print(char_embeds)
            print(char_embeds.shape)
            print(char_embeds.size)
            exit()
        pooled = self.maxpool(conv1d_out)
        pooled = pooled.view(len(x2), -1)
        # print(word_embeds.shape)
        # print(pooled.shape)
        combined_embeds = torch.cat((word_embeds, pooled), 1)
        lstm_out, _ = self.lstm(combined_embeds.view(len(x1), 1, -1))
        # lstm_out, _ = self.lstm(word_embeds.view(len(x1), 1, -1))
        hidden2tag_out = self.hidden2tag(lstm_out.view(len(x1), -1))
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
    model = CharCNNBiLSTMTagger(100, 10, 10, 30, len(char_to_idx), len(word_to_idx), len(tag_to_idx))
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
                char_idxs = []
                max_word_len = 3
                for word in splitted:
                    if (len(word)) > max_word_len:
                        max_word_len = len(word)
                    if word in word_to_idx:
                        word_idxs.append(word_to_idx[word])
                    else:
                        word_idxs.append(len(word_to_idx)) # unk
                for word in splitted:
                    word_char_idxs = []
                    for char in word:
                        if char in char_to_idx:
                            word_char_idxs.append(char_to_idx[char])
                        else:
                            word_char_idxs.append(len(char_to_idx)) # unk
                    while len(word_char_idxs) < max_word_len: # pad
                        word_char_idxs.append(len(char_to_idx) + 1)
                    char_idxs.append(word_char_idxs)

                x1 = torch.tensor(word_idxs, dtype=torch.long)
                x1 = x1.to(device)
                x2 = torch.tensor(char_idxs, dtype=torch.long)
                x2 = x2.to(device)
                y_pred = model(x1, x2)

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

