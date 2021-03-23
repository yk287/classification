import numpy as np
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.linear_model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )


    def forward(self, x):

        output = self.linear_model(x)

        # get rid of the 2nd dimension
        output = output.squeeze(1)

        return output