from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    imputer = KNNImputer(n_neighbors=k)
    mat = imputer.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy by item: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    """
    k = [1, 6, 11, 16, 21, 26]
    accuracy = []
    by_item_accuracy = []
    for i in k:
        accuracy.append(knn_impute_by_user(sparse_matrix, val_data, i))
        by_item_accuracy.append(knn_impute_by_item(sparse_matrix, val_data, i))
    plt.plot(k, accuracy, marker='o')
    plt.title("Validation accuracy of KNN by user")
    plt.show()
    plt.plot(k, by_item_accuracy, marker='o')
    plt.title("Validation accuracy of KNN by item")
    plt.show()

    print("Highest Validation Accuracy by user is when k = 11.")
    print("Highest Validation Accuracy by item is when k = 21.")
    """
    user_test = knn_impute_by_user(sparse_matrix, test_data, 11)
    item_test = knn_impute_by_item(sparse_matrix, test_data, 21)
    print(f"Test accuracy by user when k = 11 is {user_test:.5f}.")
    print(f"Test accuracy by item when k = 21 is {item_test:.5f}.")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
