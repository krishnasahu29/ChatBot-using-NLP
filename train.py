import json
# import numpy as np
from main import tokenize, stemming, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import __NeuralNet__

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)  # w is itself a array, if append, it will create array of array, therefore extend used
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stemming(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

# hyper-params
batch_size = 8
learning_rate = 0.001
num_epochs = 1000
input_size = len(all_words)
hidden_size = 8
output_size = len(tags)

class __Dataset__(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = __Dataset__()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = __NeuralNet__(input_size, hidden_size, output_size)
# model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        # words = words.to(device)
        # labels = labels.to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'Final Loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = 'data.pth'
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')
