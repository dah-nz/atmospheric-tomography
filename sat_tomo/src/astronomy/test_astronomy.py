import joblib
import matplotlib.pyplot as plt
import numpy as np

from astronomy import asterisms

def run_tests():
    test_asterism_load_data()

    # ONLY UNCOMMENT THIS IS YOU ARE HAPPY TO HAVE SATELLITE MAPS UPDATED
    # test_asterism_load_data(False)

def test_asterism_load_data(ql=True):
    data = asterisms.load_data(14, limit=120, quick_load=ql)

    plt.figure()
    for d in data:
        x = [d[k][0] *  np.sin(d[k][1]) for k in range(3)]
        y = [d[k][0] * -np.cos(d[k][1]) for k in range(3)]
        plt.plot(x, y, marker='.', linestyle='', alpha=0.3)

    print(f"{len(data)}")

    plt.figure()

    angles = []
    for d in data:
        for k in range(3):
            angles.append(d[k][1])

    plt.hist(angles)
