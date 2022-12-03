from neural_network import *
from ensemble import *

def early_stop(limit):
    return


def train_early_stop(model, lr, lamb, train_data, zero_train_data, valid_data,
          num_epoch, wait_limit):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :param wait_limit: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    limit = 0
    highest_acc = 1
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

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        if valid_acc <= (highest_acc - 0.01):
            limit += 1
        if valid_acc > highest_acc or highest_acc == 1:
            highest_acc = valid_acc

        if limit > wait_limit:
            return


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

    # KNN
    #k = 11
    #nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    #mat = nbrs.fit_transform(sample2.numpy())

    # Autoencoders
    k = 10
    lr = 0.01
    lamb = 0.001
    num_epoch = 300

    model1 = AutoEncoder(sample1.shape[1], 10)
    train_early_stop(model1, lr, lamb, sample1, zero_sample1,
          valid_data, num_epoch, wait_limit=10)

    model2 = AutoEncoder(sample2.shape[1], 10)
    train_early_stop(model2, lr, lamb, sample2, zero_sample2,
           valid_data, num_epoch, wait_limit=10)
    #
    model3 = AutoEncoder(sample3.shape[1], 10)
    train_early_stop(model3, lr, lamb, sample3, zero_sample3,
          valid_data, num_epoch, wait_limit=10)

    # train_early_stop(model1, lr, lamb, train_matrix, zero_train_matrix,
    #       valid_data, num_epoch, wait_limit=10)
    # compute accuracy
    acc = bagging_evaluate(model1, model2, model3, zero_train_matrix, valid_data)
    #acc = bagging_evaluate2(model1, mat, model3, zero_train_matrix, valid_data)
    print(acc)




if __name__ == "__main__":
    main()
