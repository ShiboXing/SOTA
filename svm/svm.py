import numpy as np

# def compute_loss(X, yhat, y, w, lamb=0.1) -> float:
#     hinge_loss = np.mean(np.max(0, 1 - np.dot(y, yhat)))
#     reg_term = lamb * np.dot(w, w)
#     loss = reg_term + hinge_loss

#     dw = np.dot(-y, X) + 2 * w
#     db = y

#     return loss, dw, db


def compute_loss(X, yhat, y, w, lamb=0.1) -> float:
    hinge_loss = max(0, 1 - y * yhat)
    reg_term = lamb * w * w
    loss = hinge_loss + reg_term

    dw, db = 0.0, 0.0
    if hinge_loss > 0:
        dw = -y * X + 2 * w * lamb
        db = y

    return loss, dw, db


# def train(iter, learning_rate):
#     for i in iter:
#         dw = None
#         db = None
#         w -= learning_rate * dw
#         b -= learning_rate * db

#     return w, b

# def predict(X, w, b) -> np.ndarray:
#     y_hat = np.dot(X, w) + b

#     return y_hat


def predict(X, w, b):
    y_hat = X * w - b

    return y_hat


def compute_error(y, y_hat):
    err = (np.sum(np.abs(y_hat - y))) / y.shape[0]
    formatted_err = "{:.6f}".format(err)
    return formatted_err
