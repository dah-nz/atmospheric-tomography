import numpy as np
import joblib

def load_asterism_data(mag, n=None, limit=0):
    data = []
    # Stars 1, 2, 3
    for star in range(3):
        filename = f'data/satellites/data_{mag}_{star+1}.pak'
        with open(filename, 'rb') as f:
            print(f'Opening file "{filename}"')
            data.append(joblib.load(f))

    invalids = []

    for i in range(len(data[1])):
        a = data[0][i]
        b = data[1][i]
        c = data[2][i]
        if a > b or b > c or a > c:
            invalids.append(i)
            # print(f"[{i}]: {a},{b},{c}")
    for k in range(3):
        data[k] = np.delete(data[k], invalids)

    data = np.reshape(np.concatenate(data, axis=0), (3, -1)).T

    if limit > 0:
        data = data[data[:, 2] <= limit, :]

    if n is not None:
        data = sample(data, n)
        
    return data

def sample(data, n):
    l = data.shape[0]
    indices = np.random.choice(l, size=n)
    return data[indices, :] 

if __name__ == "__main__":
    data = load_asterism_data(14, n=100, limit=60)
    print(data.shape)