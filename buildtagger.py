# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>
# buildtagger should pickle the torch model, the char_to_idx dict,
# word_to_idx dict, tag_to_idx dict and idx_to_tag dict

import os
import math
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

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

def prepareDicts(train_file):
    word_to_idx = {}
    char_to_idx = {}
    tag_to_idx = {}
    idx_to_tag = {}
    training_data = []
    with open(train_file, "r") as f:
        training_sents = f.readlines()
        for sent in training_sents:
            splitted = sent.split()
            char_word_tag = []
            for i, split in enumerate(splitted):
                separator_index = split.rfind('/')
                word = split[:separator_index]
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)

                tag = split[separator_index + 1:]
                if tag not in tag_to_idx:
                    idx = len(tag_to_idx)
                    tag_to_idx[tag] = idx
                    idx_to_tag[idx] = tag

                char_vector = []
                for char in word:
                    if char not in char_to_idx:
                        char_to_idx[char] = len(char_to_idx)
                    char_vector.append(char_to_idx[char])

                char_word_tag.append((char_vector, word_to_idx[word], tag_to_idx[tag]))

            training_data.append(char_word_tag)

    return char_to_idx, word_to_idx, tag_to_idx, idx_to_tag, training_data

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    char_to_idx, word_to_idx, tag_to_idx, idx_to_tag, training_data = prepareDicts(train_file)
    print("number of chars: ", len(char_to_idx))
    print("size of vocab: ", len(word_to_idx))
    print("number of tags: ", len(tag_to_idx))
    print("number of sentences: ", len(training_data))

    model = CharCNNBiLSTMTagger(300, 10, 50, 5, len(char_to_idx), len(word_to_idx), len(tag_to_idx))
    model.to(device)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for i in range(10):
        accumulated_loss = 0
        for sent in training_data:
            word_idxs = []
            tag_idxs = []
            for triple in sent:
                word_idxs.append(triple[1])
                tag_idxs.append(triple[2])
            x = torch.tensor(word_idxs, dtype=torch.long)
            y = torch.tensor(tag_idxs, dtype=torch.long)
            x = x.to(device)
            y = y.to(device)

            model.zero_grad()
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            # print(loss.item())
            accumulated_loss += loss.item()
        print(accumulated_loss / len(training_data))
        print("epoch ", i + 1)

    with open(model_file, "wb") as f:
        pickle.dump((model.state_dict(), char_to_idx, word_to_idx, tag_to_idx, idx_to_tag), f)

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)


