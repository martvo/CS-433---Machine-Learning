from proj1_helpers import *
import numpy as np


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute loss, grad and error with the least squares method
        grad, error = compute_gradient(y, tx, w)
        loss = calculate_mse(error)
        
        # update w by gradient
        w = w - grad*gamma

        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses[-1], ws[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    ws = [initial_w]
    losses = []
    batch_size = 1
    w = initial_w
    
    for n_iter in range(max_iters): 
        for by, btx in batch_iter(y,tx, batch_size):
            grad, error = compute_gradient(by, btx, w)
            loss = calculate_mse(error)

            # update w by gradient
            w = w - grad*gamma
            
            # store w and loss
            ws.append(w)
            losses.append(loss)
            
    return losses[-1], ws[-1]

def least_squares(y, tx):
    """Least squares regression using normal equations."""
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    
    return loss, w

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    
    return loss, w


def logistic_regression(y, x, inital_w, max_iters, gamma):
    """Logistic regression using stochastic gradient descent."""
    threshold = 1e-8
    batch_size = 1
    loss = []
    weight_list = [inital_w]
    w = inital_w
    
    for i in range(max_iters):
        for by, btx in batch_iter(y, x, batch_size):
            new_loss = logistic_regression_loss(by, btx, w)
            w -= gamma * logistic_regression_gradient(by, btx, w)
            loss.append(new_loss)
            weight_list.append(w)
            if len(loss) > 1 and np.abs(loss[-1] - loss[-2]) < threshold:
                break
                
    return loss[-1], weight_list[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using stochastic gradient descent."""
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for by, btx in batch_iter(y, tx, batch_size=1, num_batches=1):
            loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
            w = w - gradient*gamma
            ws.append(w)
            losses.append(loss)
            
    return losses[-1], ws[-1]


if __name__ == "__main__":
    y_train, x_train, ids_train = load_csv_data("../data/train.csv")
    y_test, x_test, ids_test = load_csv_data("../data/test.csv")
