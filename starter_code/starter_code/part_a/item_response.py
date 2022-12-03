from starter_code.starter_code.utils import *

import numpy as np
import matplotlib.pyplot as plt

from starter_code.starter_code.utils import load_train_csv, load_train_sparse, load_valid_csv, load_public_test_csv


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    users = data["user_id"]
    questions = data["question_id"]
    corrects = data["is_correct"]
    for i in range(len(users)):
        user = users[i]
        question = questions[i]
        correct = corrects[i]
        log_lklihood += correct * (theta[user] - beta[question]) - np.log((1 + np.exp(theta[user] - beta[question])))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    gd_theta = np.zeros_like(theta)
    gd_beta = np.zeros_like(beta)
    users = data["user_id"]
    questions = data["question_id"]
    corrects = data["is_correct"]
    for i in range(len(users)):
        user = users[i]
        question = questions[i]
        correct = corrects[i]

        gd_theta[user] += correct - sigmoid((theta[user] - beta[question]))
        gd_beta[question] += - correct + sigmoid((theta[user] - beta[question]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta + lr * gd_theta, beta + lr * gd_beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(542)
    beta = np.random.rand(1774)

    val_acc_lst = []
    train_neglld_lst = []
    valid_neglld_lst = []


    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        train_neglld_lst.append(neg_lld)

        valid_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        valid_neglld_lst.append(valid_neg_lld)

        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_neglld_lst, valid_neglld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print(train_data["question_id"])

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    iterations = 100
    lr = 0.001
    theta, beta, val_acc_lst, train_neglld_lst, valid_neglld_lst = irt(train_data, val_data, lr, iterations)
    iter_axis = list(range(iterations))

    # plt.title('training negative loglikelihood')
    # plt.plot(iter_axis, train_neglld_lst)
    # plt.savefig('train_lld.png')

    # plt.title('validation negative loglikelihood')
    # plt.plot(iter_axis, valid_neglld_lst)
    # plt.savefig('valid_lld.png')
    #
    print("the accuracy for validation data:", evaluate(data=val_data, theta=theta, beta=beta))
    print("the accuracy for test data:", evaluate(data=test_data, theta=theta, beta=beta))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j_1, j_2, j_3 = 12, 34, 56
    ordered_t = np.sort(theta)
    plt.plot(ordered_t, sigmoid((ordered_t - beta[j_1])))
    plt.plot(ordered_t, sigmoid((ordered_t - beta[j_2])))
    plt.plot(ordered_t, sigmoid((ordered_t - beta[j_3])))
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
