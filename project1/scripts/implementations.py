from proj1_helpers import *
import numpy as np


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Least squares using Gradient Descent"""

    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # Compute loss, grad and error with the least squares method
        grad, error = compute_gradient(y, tx, w)
        loss = calculate_mse(error)
        # Update w
        w = w - grad*gamma

        # store w and loss
        ws.append(w)
        losses.append(loss)

        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters): 
        for by, btx in batch_iter(y,tx, batch_size):
            grad, error = compute_gradient(y, tx, w)
            loss = calculate_mse(error)
            # ***************************************************
            # INSERT YOUR CODE HERE
            # TODO: update w by gradient
            # ***************************************************
            w = w - grad*gamma
            # store w and loss
            ws.append(w)
            losses.append(loss)
    return losses[-1], ws[-1]

def least_squares(y, tx):
    """ Least squares using the normal equation """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    
    return loss, w

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y,tx,w)
    return loss, w



if __name__ == "__main__":
    y_train, x_train, ids_train = load_csv_data("../data/train.csv")
    y_test, x_test, ids_test = load_csv_data("../data/test.csv")
    
    
    # Vi må behandle dataen vi har før vi bruker den
    # Visualiser data i en notebook, enklere der.. Kan ikke gjøre det nå pga. Anaconda er fucka..
    