from proj1_helpers import *
import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset."""
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def calculate_mse(e):
    """Calculate the mean square error for vector e."""
    N = len(e)

    loss = 1/(2*N) * np.sum(e**2, axis=0)
    return loss

def compute_loss(y, tx, w):
    """Calculate the mean square error."""
    return 1/(2*tx.shape[0]) * sum((y-tx.dot(w))**2)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def split_data(x, y, ratio, myseed=1):
    """Split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, y_tr, x_te, y_te


def sigmoid(x):
    """Apply the sigmoid function."""
    x[x > 700] = 700
    return 1.0 / (1.0 + np.exp(-x))


def logistic_regression_loss(y, x, w):
    """Calculate the cost."""
    # Have to calculate it in this way because of overflow in the np.exp() function
    loss = 0
    for i in range(len(y)):
        z_logistic = x[i].dot(w)

        if np.max(z_logistic) > 700:
            loss += z_logistic
        else:
            loss += np.log(1 + np.exp(z_logistic))

        loss -= y[i] * (z_logistic)
    return loss


def logistic_regression_gradient(y, x, w):
    """Compute the gradient."""
    z_logistic = sigmoid(x.dot(w))
    grad = x.T.dot(z_logistic - y)
    return grad


def penalized_logistic_regression(y, tx, w, lambda_):
    """Return the loss and gradient."""
    loss = logistic_regression_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = logistic_regression_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

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
