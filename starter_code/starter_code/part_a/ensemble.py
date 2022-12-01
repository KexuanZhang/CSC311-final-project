import random

from starter_code.starter_code.utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

from matplotlib import pyplot as plt

from neural_network import *


def bootstrap_sample(train_data):
    """
    Create 3 bootstrap samples from the training data.
    """
    samples = []
    zero_samples = []
    for i in range(0, 3):
        sample = np.zeros(train_data.shape)
        for j in range(0, len(train_data)):
            # student = train_data[j]
            # sample_point = random.choice(student, size=len(student), )
            sample_point = random.randint(0, len(train_data)-1)
            sample[j] = train_data[sample_point]

        zero_sample = sample.copy()
        zero_sample[np.isnan(sample)] = 0

        samples.append(torch.FloatTensor(sample))
        zero_samples.append(torch.FloatTensor(zero_sample))
    return samples, zero_samples


def bagging_evaluate(model1, model2, model3, zero_train_data, valid_data):
    """ Evaluate the valid_data on three models trained on three samples
    created from bagging.
    """
    # Tell PyTorch you are evaluating the model.
    model1.eval()
    model2.eval()
    model3.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs1 = Variable(zero_train_data[u]).unsqueeze(0)
        output1 = model1(inputs1)

        inputs2 = Variable(zero_train_data[u]).unsqueeze(0)
        output2 = model2(inputs2)

        inputs3 = Variable(zero_train_data[u]).unsqueeze(0)
        output3 = model3(inputs3)

        guess_sum = output1[0][valid_data["question_id"][i]].item() \
            + output2[0][valid_data["question_id"][i]].item() \
            + output3[0][valid_data["question_id"][i]].item()

        guess = (guess_sum / 3) >= 0.5

        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    # create the 3 datasets
    samples, zero_samples = bootstrap_sample(train_matrix)
    sample1 = samples[0]
    sample2 = samples[1]
    sample3 = samples[2]

    zero_sample1 = zero_samples[0]
    zero_sample2 = zero_samples[1]
    zero_sample3 = zero_samples[2]
    print('hello')

    # train the models
    k = 10
    lr = 0.01
    lamb = 0.01
    num_epoch = 300

    model1 = AutoEncoder(sample1.shape[1], 10)
    train(model1, lr, lamb, sample1, zero_sample1,
          valid_data, num_epoch)

    model2 = AutoEncoder(sample2.shape[1], 10)
    train(model2, lr, lamb, sample2, zero_sample2,
          valid_data, num_epoch)

    model3 = AutoEncoder(sample3.shape[1], 10)
    train(model3, lr, lamb, sample3, zero_sample3,
          valid_data, num_epoch)

    # compute accuracy ?
    acc = bagging_evaluate(model1, model2, model3, zero_train_matrix, valid_data)
    print(acc)


if __name__ == "__main__":
    main()

