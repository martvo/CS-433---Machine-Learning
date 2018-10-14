def least_squares(y, tx):
    """calculate the least squares solution."""
    N = tx.shape[0]
    w = np.linalg.solve(np.transpose(tx).dot(tx),np.transpose(tx).dot(y))
    mse = 1/(2*N)*sum((y-tx.dot(w))**2)
    
    return (mse, w)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        loss =compute_loss(y,tx,w)
        w = w - grad*gamma
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws