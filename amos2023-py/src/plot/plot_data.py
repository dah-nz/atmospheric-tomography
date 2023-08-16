import matplotlib.pyplot as plt
import numpy as np
from plot import plot

data = []
keys = []

with open("./results/results_residuals_16_SNR_0.data", "r") as f:
    raw_data = f.readlines()

keys = raw_data[0].strip().split(',')
keys = [k.strip() for k in keys]

for l in raw_data[1:]:
    data.append(l.strip().split(','))

print(keys)

# b = Baseline - performance given a flat guess (all coefficients = 0)
# r = Reconstruction
# i = Ideal (best possible reconstruction given the number of coefficients and a perfect wavefront sensor)
# a = Actual (Actual Kolmogorov phase screen results)
# Some metrics have to be compared to another phase screen
# ra = Reconstruction vs Actual
# ri = Reconstruction vs Ideal
# ia = Ideal vs Actual
# ba = Baseline vs Ideal
# all = an aggregate across all target positions
rmse = {'ba': {}, 'ra': {}, 'ri': {}, 'ia': {}}
fwhm = {'ra': {}, 'ia': {}}
max_intensity = {'ra': {}, 'ia': {}, 'r': {}, 'a': {}, 'i': {}}
mssim = {'ra-ia': {}, 'ra': {}, 'ri': {}, 'ia': {}}

for d in [rmse, fwhm, max_intensity, mssim]:
    for k in list(d.keys()):
        d[k + "_all"] = {}

bf = set()
target_pos = set()

"""
Add data point to dictionary
inputs:
    dct     Dictionary
    p       Data point to add
    t       Type ['r', 'a', 'i' etc]
    b       Basis function
    ng      Number of guide stars

"""
def add_data(dct, p, t, b, ng, tg):
    try:
        dct[t][(b, ng, tg)].append(p)
    except KeyError:
        dct[t][(b, ng, tg)] = [p]

    try:
        dct[t + '_all'][(b, ng)].append(p)
    except KeyError:
        dct[t + '_all'][(b, ng)] = [p]

for d in data:
    b = d[keys.index('Basis Function')]
    bf.add(b)
    ng = int(d[keys.index('N Guide Stars')])
    tg = float(d[keys.index('Target Gamma')])
    target_pos.add(tg)

    for i, t in [('RMSE B:A', 'ba'), ('RMSE R:A', 'ra'), ('RMSE I:R', 'ri'), ('RMSE I:A', 'ia')]:
        add_data(rmse, float(d[keys.index(i)]), t, b, ng, tg)

    for i, t in [('FWHM R:A', 'ra'), ('FWHM I:A', 'ia')]:
        add_data(fwhm, float(d[keys.index(i)]), t, b, ng, tg)

    for i, t in [("Max Int. R:A", 'ra'), ("Max Int. I:A", 'ia'), ("Max Int. R", 'r'), ("Max Int. A", 'a'), ("Max Int. I", 'i')]:
        add_data(max_intensity, float(d[keys.index(i)]), t, b, ng, tg)
    
    for i, t in [('MSSIM R:A-I:A', 'ra-ia'), ('MSSIM R:A', 'ra'), ('MSSIM R:I', 'ri'), ('MSSIM I:A', 'ia')]:
        add_data(mssim, float(d[keys.index(i)]), t, b, ng, tg)

bf = sorted(list(bf))
target_pos = sorted(list(target_pos))
colours = ['r', 'g', 'b', 'orange', 'purple']

biggest = lambda a: max([max(b) for b in a])
least = lambda a: min([min(b) for b in a])

# PLOT RMSE of Reconstruction
for k in ['ba', 'ra', 'ri', 'ia']:
    fig, axs = plt.subplots(1,3)
    lim_min = 0.0
    lim_max = 0.0
    for b, c in zip(bf, colours):
        r0 = rmse[k]
        y = [r0[(b, 3, tg)] for tg in target_pos]
        lim_max = max(lim_max, biggest(y))
        lim_min = min(lim_min, least(y))
        plot.plot_error(axs[0], target_pos, y, c)
        axs[0].set_xlabel('Target Angle (arcseconds)')
        axs[0].set_ylabel('RMSE (radians)')
        axs[0].set_title('(a)')
        y = [r0[(b, 6, tg)] for tg in target_pos]
        lim_max = max(lim_max, biggest(y))
        lim_min = min(lim_min, least(y))
        plot.plot_error(axs[1], target_pos, y, c)
        axs[1].set_xlabel('Target Angle (arcseconds)')
        axs[1].set_ylabel('RMSE (radians)')
        axs[1].set_title('(b)')
        y = [r0[(b, 5, tg)] for tg in target_pos]
        lim_max = max(lim_max, biggest(y))
        lim_min = min(lim_min, least(y))
        plot.plot_error(axs[2], target_pos, y, c)
        axs[2].set_xlabel('Target Angle (arcseconds)')
        axs[2].set_ylabel('RMSE (radians)')
        axs[2].set_title('(c)')

    axs[0].set_ylim(0, 6)
    axs[1].set_ylim(0, 6)
    axs[2].set_ylim(0, 6)
    # axs[0].set_ylim(lim_min+0.1, lim_max-0.1)
    # axs[1].set_ylim(lim_min+0.1, lim_max-0.1)
    # axs[2].set_ylim(lim_min+0.1, lim_max-0.1)

    for c, n in zip(colours, ["DCT", 'DST', 'Wavelets (Haar)', 'Wavelets (db2)', 'Zernike']):
        axs[2].plot([], c=c, label=n)
    axs[2].legend()

plt.show()