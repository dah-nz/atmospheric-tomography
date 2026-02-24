import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(314)

def embed_gs(x, y):

    r = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(x, y)

    a = np.concatenate([r, t, x, y])
    n = a.shape[0]

    W1 = rng.normal(n)
    W2 = rng.normal(n)
    W3 = rng.uniform(size=n)

    b1a = rng.normal(n)
    b1b = rng.normal(n)
    b1c = rng.normal(n)
    b2a = rng.normal(n)
    b2b = rng.normal(n)
    b2c = rng.normal(n)
    b3 = rng.uniform(size=n)

    return np.concatenate([
        a,
        np.sin(W1*a+b1a),
        np.sin(W1*a+b1b),
        np.sin(W1*a+b1c),
        np.cos(W2*a+b2a),
        np.cos(W2*a+b2b),
        np.cos(W2*a+b2c),
        np.sign(a) * np.log(W3*np.abs(a)+b3)
    ])

x = rng.uniform(-1, 1, size=3)
y = rng.uniform(-1, 1, size=3)

e1 = embed_gs(x, y)
e2 = embed_gs(x, y)

e1 = np.reshape(e1, newshape=(3, -1))
e2 = np.reshape(e2, newshape=(3, -1))

fig, axs = plt.subplots(2, 3)
axs[0, 0].imshow(e1)
axs[0, 1].imshow(e2)
axs[0, 2].imshow(e1.T @ e2)

X = rng.normal(size=(3, 33))

def embed_meas(X):

    X = np.ravel(X)
    n = X.shape[0]

    W1 = rng.normal(n)
    W2 = rng.normal(n)
    W3 = rng.uniform(size=n)

    b1a = rng.normal(n)
    b1b = rng.normal(n)
    b1c = rng.normal(n)
    b2a = rng.normal(n)
    b2b = rng.normal(n)
    b2c = rng.normal(n)
    b3 = rng.uniform(size=n)

    return np.concatenate([
        X,
        np.sin(W1*X+b1a),
        np.sin(W1*X+b1b),
        np.sin(W1*X+b1c),
        np.cos(W2*X+b2a),
        np.cos(W2*X+b2b),
        np.cos(W2*X+b2c),
        np.sign(X) * np.log(W3*np.abs(X)+b3)
    ])

eX = embed_meas(X)
print(eX.shape)
eX = np.reshape(eX, (33, -1))

axs[1, 1].imshow(eX)

plt.show()