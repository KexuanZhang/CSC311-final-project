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
    # zero_train_matrix[np.isnan(train_matrix)] = 0

    # Change default Nan assignment to 0.5
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

def subject_confidence(train_data, confi_index):
    """
    Return the subject confidence of the training data

    :param inputs: user vector
    :return: confidence dictionary
    """
    question_mat = train_data.T
    num_ques = np.shape(question_mat)[0]
    sub_data = load_meta_question()
    meta_ques = sub_data["question_id"]
    meta_sub = sub_data["subject_id"]

    meta_map = dict()

    # initialize question-subjects map
    for q in range(num_ques):
        meta_map[meta_ques[q]] = meta_sub[q]

    # initialize subject correct count map
    count_map = dict()
    # there are 387 subjects by reading dataset
    num_sub = 388
    for s in range(num_sub):
        count_map[s] = [0, 0]

    for q in range(num_ques):
        curr_count = np.count_nonzero(question_mat[q] == 1)
        total_count = np.count_nonzero(question_mat[q] == 1) + np.count_nonzero(question_mat[q] == 0)
        for subject in meta_map[q]:
            count_map[subject][0] += curr_count
            count_map[subject][1] += total_count

    confidence_map = dict()
    for s in range(num_sub):
        if count_map[s][1] != 0:
            confidence_map[s] = count_map[s][0] / count_map[s][1]
        else:
            confidence_map[s] = count_map[0][0] / count_map[0][1]

    #normalize confidence map:
    min_acc = min(confidence_map.values())
    max_acc = max(confidence_map.values())
    dist = max_acc - min_acc
    for s in range(num_sub):
        confidence_map[s] = confi_index * (((confidence_map[s] - min_acc) / dist) - 0.5)

    return confidence_map, meta_map


def load_meta_question():
    # A helper function to load the csv file.
    path = os.path.join("../data", "question_meta.csv")
    # path = os.path.join("/data", "train_data.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "question_id": [],
        "subject_id": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                clean_row_1 = row[1].strip('[').strip(']').split(',')
                int_lst = [int(x) for x in clean_row_1]
                data["subject_id"].append(int_lst)
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


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

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            #updated
            # output = model(inputs)

            output = model(inputs)

            # print(output)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) \
                   + lamb * 0.5 * (model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
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
    confidence_map, meta_map = subject_confidence(train_data, 0.1)

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        question = valid_data["question_id"][i]
        subjects = meta_map[question]
        confidence = 0
        for s in subjects:
            confidence += confidence_map[s]

        guess = output[0][valid_data["question_id"][i]].item() + confidence >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1

    return correct / float(total)


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
    num_epoch = 150
    lamb = 0.01

    # Choosing K

    accuracies = []
    for k_i in k:
        print(f"Training for k = {k_i}")
        model = AutoEncoder(train_matrix.shape[1], k_i)
        train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
        accuracies.append(evaluate(model, zero_train_matrix, valid_data))

    plt.plot(k, accuracies, marker='o')
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
