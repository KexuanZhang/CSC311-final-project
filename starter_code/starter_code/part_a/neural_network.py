from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

from matplotlib import pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = torch.sigmoid(self.h(torch.sigmoid(self.g(inputs))))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses = []
    validation_accs = []
    validation_loss = []
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) \
                   + lamb * 0.5 * (model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc, valid_loss = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        train_losses.append(train_loss)
        validation_accs.append(valid_acc)
        validation_loss.append(valid_loss)

    return train_losses, validation_accs, validation_loss
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    valid_loss = 0.

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1

        valid_loss \
            += ((output[0][valid_data["question_id"][i]] - valid_data["is_correct"][i]) ** 2.).item()
    return correct / float(total), valid_loss


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = [10, 50, 100, 200, 500]
    # model = AutoEncoder(train_matrix.shape[1], k)

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 200
    lamb = 0.001

    # Choosing K
    """
    # Q4c
    accuracies = []
    for k_i in k:
        print(f"Training for k = {k_i}")
        model = AutoEncoder(train_matrix.shape[1], k_i)
        train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
        accuracies.append(evaluate(model, zero_train_matrix, valid_data)[0])

    plt.plot(k, accuracies, marker='o')
    plt.show()
    """

    """
    # Q4d
    # k* = 10, plotting training loss & validation accuracy as function
    # of num_epoch
    model = AutoEncoder(train_matrix.shape[1], 10)
    train_loss, valid_acc, valid_loss = \
        train(model, lr, 0, train_matrix, zero_train_matrix, valid_data, num_epoch)

    fig, ax = plt.subplots()
    ax.plot(range(num_epoch), train_loss, color='red', marker='o', label='train loss')
    ax.set_xlabel("Number of epoch")
    ax.set_ylabel("Train loss")
    ax2 = ax.twinx()
    ax2.plot(range(num_epoch), valid_loss, color='blue', marker='o', label='valid loss')
    ax2.set_ylabel("Validation loss")
    plt.title("Train & Validation losses vs number of epoch")
    plt.legend()
    plt.show()

    test_acc = evaluate(model, zero_train_matrix, test_data)[0]
    print(f"Test accuracy is {test_acc:.4f}")
    """

    """
    Q4e
    # k* = 10, choosing lambda
    lambs = [0.001, 0.01, 0.1, 1]
    accuracies = []
    for lamb in lambs:
        print(f"Training for lambda = {lamb}")
        model = AutoEncoder(train_matrix.shape[1], 10)
        train(model, lr, lamb, train_matrix, zero_train_matrix,
              valid_data, num_epoch)
        accuracies.append(evaluate(model, zero_train_matrix, valid_data)[0])

    plt.plot(['0.001', '0.01', '0.1', '1'], accuracies, marker='o')
    plt.xlabel("Lambda")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation accuracy for different lambdas")
    plt.show()
    """

    # Final model & parameters
    # k* = 10, lambda* = 0.001
    model = AutoEncoder(train_matrix.shape[1], 10)
    train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
    final_validation_acc = evaluate(model, zero_train_matrix, valid_data)[0]
    final_test_acc = evaluate(model, zero_train_matrix, test_data)[0]

    print(f"Final validation accuracy is {final_validation_acc:.4f}")
    print(f"Final test accuracy is {final_test_acc:.4f}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
