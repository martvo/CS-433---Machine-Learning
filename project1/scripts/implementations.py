from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
import numpy as np


def least_squares(y, tx):
    """ Least squares using the normal equation """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return loss, w


def compute_loss(y, tx, w):
    return 1/(2*tx.shape[0]) * sum((y-tx.dot(w))**2)


def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Least squares using Gradient Descent"""

    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # Compute loss, grad and error with the least squares method
        loss, grad, error = least_squares(y, tx, w)

        # Update w
        w = w - grad*gamma

        # store w and loss
        ws.append(w)
        losses.append(loss)

        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size):
    """Least squares using Stochastic Gradient Descent"""

    ws = [initial_w]
    losses = []
    w = initial_w
    current_y = y
    current_tx = tx
    for n_iter in range(max_iters):
        # Choose subset to use for this iteration, use np.random.permutation
        # Get a permuted list of the length of the length of tx as this list can be used to choose the same
        # samples from both tx and y. Which is important to get the right result!

        permuted_list = np.random.permutation(tx.shape[0])
        current_tx = tx[permuted_list[:batch_size]]
        current_y = y[permuted_list[:batch_size]]

        # Compute loss, grad and error with the least squares method
        loss, grad, error = least_squares(current_y, current_tx, w)

        # Update w
        w = w - grad * gamma

        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses[-1], ws[-1]


if __name__ == "__main__":
    y_train, x_train, ids_train = load_csv_data("../data/train.csv")
    y_test, x_test, ids_test = load_csv_data("../data/test.csv")

    # Vi må behandle dataen vi har før vi bruker den
    # Visualiser data i en notebook, enklere der.. Kan ikke gjøre det nå pga. Anaconda er fucka..