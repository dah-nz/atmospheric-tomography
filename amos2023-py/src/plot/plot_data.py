import matplotlib.pyplot as plt
import numpy as np
from plot import plot

data = []
keys = []

with open("./results/good/results_psf_16_SNR_0.data", "r") as f:
    raw_data = f.readlines()

keys = raw_data[0].strip().split(',')
keys = [k.strip() for k in keys]

for l in raw_data[1:]:
    data.append(l.strip().split(','))

print(keys)

# r = Reconstruction
# i = Ideal (best possible reconstruction given the number of coefficients and a perfect wavefront sensor)
# a = Actual (Actual Kolmogorov phase screen results)
# all = an aggregate across all target positions
rmse = {'r': {}, 'i': {}, 'a': {}, 'r_all': {}, 'i_all': {}, 'a_all': {}}
fwhm = {'r': {}, 'i': {}, 'a': {}, 'r_all': {}, 'i_all': {}, 'a_all': {}}
max_intensity = {'r': {}, 'i': {}, 'a': {}, 'r_all': {}, 'i_all': {}, 'a_all': {}}

# For MSSIM, each screen has to be compared to something else
# ra = Reconstruction vs Actual
# ri = Reconstruction vs Ideal
# ia = Ideal vs Actual
mssim = {'ra': {}, 'ri': {}, 'ia': {}, 'ra_all': {}, 'ri_all': {}, 'ia_all': {}}

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
    # tt = float(d[keys.index('Target Theta')]) # Ignore theta at this point - assume there is no affect on results
    target_pos.add(tg)

    add_data(rmse, float(d[keys.index('RMSE')]), 'r', b, ng, tg)
    add_data(rmse, float(d[keys.index('RMSE Ideal')]), 'i', b, ng, tg)
    add_data(rmse, float(d[keys.index('RMSE Actual-Ideal')]), 'a', b, ng, tg)

    add_data(fwhm, float(d[keys.index('FWHM Actual')]), 'a', b, ng, tg)
    add_data(fwhm, float(d[keys.index('FWHM Recons')]), 'r', b, ng, tg)
    add_data(fwhm, float(d[keys.index('FWHM Ideal')]), 'i', b, ng, tg)

    add_data(max_intensity, float(d[keys.index('Max Intensity Actual')]), 'a', b, ng, tg)
    add_data(max_intensity, float(d[keys.index('Max Intensity Recons')]), 'r', b, ng, tg)
    add_data(max_intensity, float(d[keys.index('Max Intensity Ideal')]), 'i', b, ng, tg)

    add_data(mssim, float(d[keys.index('MSSIM Recon-Actual')]), 'ra', b, ng, tg)
    add_data(mssim, float(d[keys.index('MSSIM Recon-Ideal')]), 'ri', b, ng, tg)
    add_data(mssim, float(d[keys.index('MSSIM Ideal-Actual')]), 'ia', b, ng, tg)

bf = sorted(list(bf))
target_pos = sorted(list(target_pos))
colours = ['r', 'g', 'b', 'orange', 'purple']

biggest = lambda a: max([max(b) for b in a])
least = lambda a: min([min(b) for b in a])

# PLOT RMSE of Reconstruction
fig, axs = plt.subplots(1,3)
lim_min = 0.0
lim_max = 0.0
for b, c in zip(bf, colours):
    r0 = rmse['r']
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

axs[0].set_ylim(lim_min+0.1, lim_max-0.1)
axs[1].set_ylim(lim_min+0.1, lim_max-0.1)
axs[2].set_ylim(lim_min+0.1, lim_max-0.1)

for c, n in zip(colours, ["DCT", 'DST', 'Wavelets (Haar)', 'Wavelets (db2)', 'Zernike']):
    axs[2].plot([], c=c, label=n)
axs[2].legend()

# PLOT DIFFERENCE BETWEEN RMSE of Ideal and Reconstruction
fig, axs = plt.subplots(1,3)
lim_min = 0.0
lim_max = 0.0

for b, c in zip(bf, colours):
    r0 = rmse['r']
    r1 = rmse['i']
    y = [np.array(r0[(b, 3, tg)]) - np.array(r1[(b, 3, tg)]) for tg in target_pos]
    lim_max = max(lim_max, biggest(y))
    lim_min = min(lim_min, least(y))
    plot.plot_error(axs[0], target_pos, y, c)
    axs[0].set_xlabel('Target Angle (arcseconds)')
    axs[0].set_ylabel('RMSE (radians)')
    axs[0].set_title('(a)')
    y = [np.array(r0[(b, 6, tg)]) - np.array(r1[(b, 6, tg)]) for tg in target_pos]
    lim_max = max(lim_max, biggest(y))
    lim_min = min(lim_min, least(y))
    plot.plot_error(axs[1], target_pos, y, c)
    axs[1].set_xlabel('Target Angle (arcseconds)')
    axs[1].set_ylabel('RMSE (radians)')
    axs[1].set_title('(b)')
    y = [np.array(r0[(b, 5, tg)]) - np.array(r1[(b, 5, tg)]) for tg in target_pos]
    lim_max = max(lim_max, biggest(y))
    lim_min = min(lim_min, least(y))
    plot.plot_error(axs[2], target_pos, y, c)
    axs[2].set_xlabel('Target Angle (arcseconds)')
    axs[2].set_ylabel('RMSE (radians)')
    axs[2].set_title('(c)')

axs[0].set_ylim(lim_min-0.1, lim_max+0.1)
axs[1].set_ylim(lim_min-0.1, lim_max+0.1)
axs[2].set_ylim(lim_min-0.1, lim_max+0.1)

for c, n in zip(colours, ["DCT", 'DST', 'Wavelets (Haar)', 'Wavelets (db2)', 'Zernike']):
    axs[2].plot([], c=c, label=n)
axs[2].legend()



plt.show()