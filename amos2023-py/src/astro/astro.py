import numpy as np

def point_on_sphere(n=1):
    rng = np.random.default_rng()
    vec = rng.normal(size=(3, n))
    vec = vec / np.linalg.norm(vec, axis=0)
    return vec

def get_ra(vec, non_negative_ra=False):
    ra = np.arctan2(vec[0], vec[1]) * (180 / np.pi)
    if non_negative_ra:
        ra_neg = ra[ra < 0] + 360
        ra[ra < 0] = ra_neg
    return ra

def get_dec(vec):
    return np.arctan2(np.sqrt(vec[0]**2+vec[1]**2), vec[2]) * (180 / np.pi) - 90

def to_ra_dec(vec, non_negative_ra=False):
    return np.array([get_ra(vec, non_negative_ra), get_dec(vec)])

def main():
    import matplotlib.pyplot as plt

    uniform_points = point_on_sphere(10000)

    x = uniform_points[0, :]
    y = uniform_points[1, :]
    z = uniform_points[2, :]

    ra = [get_ra(v) for v in uniform_points.T]
    dec = [get_dec(v) for v in uniform_points.T]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, marker='.')
    ax.set_aspect("equal")

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(ra, dec, marker='.')

    plt.show()

if __name__ == "__main__":
    main()