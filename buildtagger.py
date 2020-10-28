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
import numpy as np

wordembed_dim = 256
char_embed_dim = 128
cnn_filters = 16
lstm_hidden = 32
time_limit = datetime.timedelta(minutes=1)

class CharCNNBiLSTMTagger(nn.Module):
    def __init__(self, word_embed_dim, char_embed_dim, char_hidden_dim, lstm_hidden,
                 char_size, vocab_size, tagset_size):
        super(CharCNNBiLSTMTagger, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size + 2, word_embed_dim, padding_idx=vocab_size + 1)
        self.char_embeddings = nn.Embedding(char_size + 2, char_embed_dim, padding_idx=char_size + 1)

        self.conv1d = nn.Conv1d(char_embed_dim, char_hidden_dim, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.lstm = nn.LSTM(word_embed_dim + char_hidden_dim,
                            lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(lstm_hidden*2, tagset_size)
        self.dropout = nn.Dropout(0.1)

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
        conv1d_out = self.relu(conv1d_out)
        pooled = self.maxpool(conv1d_out)
        pooled = pooled.view(len(x2), -1)
        # print(word_embeds.shape)
        # print(pooled.shape)
        combined_embeds = torch.cat((word_embeds, pooled), 1)
        lstm_out, _ = self.lstm(combined_embeds.view(len(x1), 1, -1))
        # lstm_out, _ = self.lstm(word_embeds.view(len(x1), 1, -1))
        lstm_out = self.dropout(lstm_out)
        hidden2tag_out = self.hidden2tag(lstm_out.view(len(x1), -1))
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
    time_exceeded = False

    char_to_idx, word_to_idx, tag_to_idx, idx_to_tag, training_data = prepareDicts(train_file)
    print("number of chars: ", len(char_to_idx))
    print("size of vocab: ", len(word_to_idx))
    print("number of tags: ", len(tag_to_idx))
    print("number of sentences: ", len(training_data))

    model = CharCNNBiLSTMTagger(wordembed_dim, char_embed_dim, cnn_filters, lstm_hidden, len(char_to_idx), len(word_to_idx), len(tag_to_idx))
    model.to(device)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    indices = np.arange(len(training_data))
    for epoch in range(5):
        accumulated_loss = 0
        np.random.shuffle(np.arange(len(training_data)))
        for sample_id in indices:
            sent = training_data[sample_id]
            word_idxs = []
            tag_idxs = []
            char_idxs = []
            for triple in sent:
                char_idxs.append(triple[0])
                word_idxs.append(triple[1])
                tag_idxs.append(triple[2])
            # padding
            max_word_len = 3
            for i, chars in enumerate(char_idxs):
                if len(chars) > max_word_len:
                    max_word_len = len(chars)
            for i, chars in enumerate(char_idxs):
                while len(chars) < max_word_len:
                    chars.append(len(char_to_idx))

            x1 = torch.tensor(word_idxs, dtype=torch.long)
            x2 = torch.tensor(char_idxs, dtype=torch.long)
            y = torch.tensor(tag_idxs, dtype=torch.long)
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            model.zero_grad()
            y_pred = model(x1, x2)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            # print(loss.item())
            accumulated_loss += loss.item()

            if datetime.datetime.now() - start_time > time_limit:
                print("time exceeded")
                time_exceeded = True
                break
        if time_exceeded:
            break
        print(accumulated_loss / len(training_data))
        print("epoch ", epoch + 1)

    with open(model_file, "wb") as f:
        pickle.dump((model.state_dict(), char_to_idx, word_to_idx, tag_to_idx, idx_to_tag,
                     (wordembed_dim, char_embed_dim, cnn_filters, lstm_hidden)), f)

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)


