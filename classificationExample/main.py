
import torch

import matplotlib.pyplot as plt

#################
#hyperparameters#
#################
num_classes = 2
input_size = 10
epochs = 500
batch_size = 32
num_samples = 1200
LR = 0.01
###################

# import the regression class
from model import Classifier
# instantiate it
model = Classifier(input_size, 32, num_classes)
model_optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# define the loss function
LossCriteria = torch.nn.CrossEntropyLoss()

# importing a function that generates data and a dataset class.
from dataGenerator import generator, Sample_Dataloader

# generate the data
x, y = generator(input_size, num_samples, num_classes)

# cutoffs for train, validation, test
cutoff1 = int(.60 * len(x))
cutoff2 = int(.80 * len(x))

# train, validation, test split
train_x, train_y = x[:cutoff1], y[:cutoff1]
val_x, val_y = x[cutoff1:cutoff2], y[cutoff1:cutoff2]
test_x, test_y = x[cutoff2:], y[cutoff2:]

# create datasets
train_dataset = Sample_Dataloader([train_x, train_y])
val_dataset = Sample_Dataloader([val_x, val_y])
test_dataset = Sample_Dataloader([test_x, test_y])

# create dataloader objects
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# list that holds loss over-time
TrainlossOverTime = []
valLossOverTime = []

# training
for i in range(epochs):

    trainEpochLoss = 0
    valEpochLoss = 0

    trainSteps = 1
    valSteps = 1

    correct = 0
    total = 0
    for x, y in trainloader:
        # get the prediction
        pred = model(x)
        # get the loss
        loss = LossCriteria(pred, y).mean()

        # zero out gradients
        model.zero_grad()
        # calc the gradients
        loss.backward()
        # take a step
        model_optimizer.step()

        pred_ = torch.argmax(pred, dim=1)

        correct = torch.sum(pred_ == y)
        total = len(pred_)

        trainEpochLoss += correct / total
        trainSteps += 1

        correct = 0
        total = 0

        for x_val, y_val in valloader:

            pred = model(x_val)

            loss = LossCriteria(pred, y_val).mean()

            pred_ = torch.argmax(pred, dim=1)

            correct = torch.sum(pred_ == y_val)
            total = len(pred_)

            valEpochLoss += correct / total
            valSteps += 1

    # append the losses.
    TrainlossOverTime.append(trainEpochLoss / trainSteps)
    valLossOverTime.append(valEpochLoss / valSteps)

print(TrainlossOverTime)
print(valLossOverTime)
