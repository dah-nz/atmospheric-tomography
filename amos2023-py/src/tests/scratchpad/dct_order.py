"""    
Order is defined on a snaking method method

y x 0 | 1 | 2 | 3
0 | 0   1   5   6   
1 | 2   4   7  13
2 | 3   8  12
3 | 9  11
"""

# 0 = (0, 0)
# 1 = (1, 0)
# 2 = (0, 1)
# 3 = (0, 2)
# 4 = (1, 1)
# 5 = (2, 0)

import numpy as np

def zig_zag_index(k, n):
    # upper side of interval
    if k >= n * (n + 1) // 2:
        i, j = zig_zag_index(n * n - 1 - k, n)
        return n - 1 - i, n - 1 - j
    # lower side of interval
    i = int((np.sqrt(1 + 8 * k) - 1) / 2)
    j = k - i * (i + 1) // 2
    return (j, i - j) if i & 1 else (i - j, j)
def zig_zag_value(i, j, n):
    # upper side of interval
    if i + j >= n:
        return n * n - 1 - zig_zag_value(n - 1 - i, n - 1 - j, n)
    # lower side of interval
    k = (i + j) * (i + j + 1) // 2
    return k + i if (i + j) & 1 else k + j

n = 8
M = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(n):
        M[i, j] = zig_zag_value(i, j, n)

print(M)

M = np.zeros((n, n), dtype=int)
for k in range(n * n):
    M[zig_zag_index(k, n)] = k

print(M)