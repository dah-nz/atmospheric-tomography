import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

import pickle
with open('data/ivcnz/df.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    data = pickle.load(f)

print(ttest_ind(data['nn rmse'], data['tomo rmse']))