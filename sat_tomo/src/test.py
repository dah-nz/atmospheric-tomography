import joblib
import matplotlib.pyplot as plt
import numpy as np

import astronomy.test_astronomy
import tomography.test_tomography

if __name__ == "__main__":
    # astronomy.test_astronomy.run_tests()

    tomography.test_tomography.run_tests()

    plt.show()