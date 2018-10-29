from proj1_helpers import *
import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset."""
    data_size = len(y)
    np.random.seed(1)

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
    return 1/2 * np.mean(e**2)

def compute_loss(y, tx, w):
    """Calculate the mean square error."""
    e = (y - tx.dot(w))
    return e.T.dot(e) / len(y)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -(tx.T.dot(err) / len(y))
    return grad, err

def sigmoid(x):
    """Apply the sigmoid function."""
    x[x > 600] = 600
    return 1.0 / (1.0 + np.exp(-x))


def logistic_regression_loss(y, x, w):
    """Calculate the cost."""
    # Have to calculate it in this way because of overflow in the np.exp() function
    loss = 0
    for i in range(len(y)):
        
        z_logistic = x[i].dot(w)
        sum_z = 0

        if np.max(z_logistic) > 600:
            sum_z += z_logistic
        else:
            sum_z += np.log(1 + np.exp(z_logistic))

        loss += (sum_z - (y[i] * (z_logistic)))
        
        """
        
        add = y[i]*(np.log(sigmoid(x[i].dot(w))))
        add1 = (1-y[i])*(np.log(1-sigmoid(x[i].dot(w))))
        s += add +add1
        """
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
    """Least squares using Gradient Descent"""

    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # Compute loss, grad and error with the least squares method
        grad, error = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        # Update w
        w = w - (grad*gamma)

        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses[-1], ws[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    batch_size = 1
    w = initial_w
    for n_iter in range(max_iters): 
        for by, btx in batch_iter(y,tx, batch_size):
            grad, error = compute_gradient(by, btx, w)
            loss = compute_loss(by, btx, w)
            
            w = w - (grad*gamma)
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Iteration " + str(n_iter) + " loss=" + str(loss))
            #if (len(losses) > 1 and (abs(losses[-1] - losses[-2]) < 0.0001)):
            #    break
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
    loss = compute_loss(y, tx, w)
    return loss, w


def logistic_regression(y, x, inital_w, max_iters, gamma):
    """Logistic regression using GD"""
    threshold = 1e-5
    batch_size = 1
    loss = []
    weight_list = [inital_w]
    w = inital_w
    for i in range(max_iters):
        new_loss = logistic_regression_loss(y, x, w)
        # new_loss = sum(sum(np.logaddexp(0, btx.dot(w)) - y*(btx.dot(w))))
        w -= (gamma * logistic_regression_gradient(y, x, w))
        loss.append(new_loss)
        weight_list.append(w)
        # print("Loss in iteration " + str(i + 1) + ": " + str(new_loss))
        if len(loss) > 1 and np.abs(loss[-1] - loss[-2]) < threshold:
            break
    return loss[-1], weight_list[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """regularized logistic regression using gradient descent."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - (gradient*gamma)
        ws.append(w)
        losses.append(loss)        
    return losses[-1], ws[-1]


if __name__ == "__main__":
    y_train, x_train, ids_train = load_csv_data("../data/train.csv")
    y_test, x_test, ids_test = load_csv_data("../data/test.csv")
