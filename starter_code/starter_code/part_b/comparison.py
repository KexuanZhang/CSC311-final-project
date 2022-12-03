from IPython.display import display

from part_a.knn import *
from part_a.item_response import *
from part_b.extended_neural_net import *
from part_b import extended_neural_net
from part_a import item_response
from part_a import neural_network

import pandas as pd

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    models = ["Extended Model", "KNN", "IRT", "Neural Net"]
    validation_accuracies = []
    test_accuracies = []
    # extended model
    lr = 0.01
    num_epoch = 200
    lamb = 0.001
    # k1 = 50
    # k2 = 10

    model_enn = extended_neural_net.AutoEncoder(train_matrix.shape[1])
    extended_neural_net.train(model_enn, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
    validation_accuracies.append(extended_neural_net.evaluate(model_enn, zero_train_matrix, valid_data))
    test_accuracies.append(extended_neural_net.evaluate(model_enn, zero_train_matrix, test_data))

    # KNN
    validation_accuracies.append(knn_impute_by_item(train_matrix, valid_data, 21))
    test_accuracies.append(knn_impute_by_item(train_matrix, test_data, 21))


    # IRT
    train_csv = load_train_csv("../data")
    theta, beta, val_acc_lst, train_neglld_lst, valid_neglld_lst = irt(train_csv, valid_data,
                                                                       0.001, 100)

    validation_accuracies.append(item_response.evaluate(valid_data, theta, beta))
    test_accuracies.append(item_response.evaluate(test_data, theta, beta))


    #NN
    model = neural_network.AutoEncoder(train_matrix.shape[1], 10)
    neural_network.train(model, lr, 0.001, train_matrix, zero_train_matrix,
          valid_data, 120)
    validation_accuracies.append(neural_network.evaluate(model, zero_train_matrix, valid_data)[0])
    test_accuracies.append(neural_network.evaluate(model, zero_train_matrix, test_data)[0])

    df = pd.DataFrame({"validation accs": validation_accuracies, "test accs":test_accuracies}, index=models)
    display(df)
    ax = df.plot.bar(rot=0, ylim=(0.65, 0.73))
    plt.show()



if __name__ == '__main__':
    main()
