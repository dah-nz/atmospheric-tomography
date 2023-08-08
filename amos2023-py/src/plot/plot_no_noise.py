import matplotlib.pyplot as plt
import numpy as np

data = []
keys = []

with open("./results/results_psf_16_sigma_0.data", "r") as f:
    raw_data = f.readlines()

keys = raw_data[0].strip().split(',')
keys = [k.strip() for k in keys]

for l in raw_data[1:]:
    data.append(l.strip().split(','))

print(keys)

mse = {}

fwhm = {}


bf = set()
target_pos = set()

for d in data:
    n = int(d[keys.index('N Coeffs')])
    b = d[keys.index('Basis Function')]
    bf.add(b)
    ng = int(d[keys.index('N Guide Stars')])
    gtheta = float(d[keys.index('Guide Star Angle')])
    tg = float(d[keys.index('Target Gamma')])
    tt = float(d[keys.index('Target Theta')])
    target_pos.add((tt, tg))
    err = float(d[keys.index('RMSE')])
    fwhm_actual = float(d[keys.index('FWHM Actual')])
    fwhm_recons = float(d[keys.index('FWHM Recons')])

    try:
        mse[(b, ng, (tt, tg))].append(err)
    except KeyError:
        mse[(b, ng, (tt, tg))] = [err]
        
    # wavelength = 600e-9 # m
    telescope_diameter = 0.6 # m
    Q = 8 # pixels per resolution unit
    effective_focal_length = telescope_diameter * 6.5 # (f/6.5 system)
    # um = 1e6 # um/m
    # resolution = (wavelength * effective_focal_length / Q) * um
    resolution = effective_focal_length / Q
    try:
        fwhm[(b, ng, (tt, tg))].append((fwhm_actual - fwhm_recons))
    except KeyError:
        fwhm[(b, ng, (tt, tg))] = [(fwhm_actual - fwhm_recons)]

bf = sorted(list(bf))
bf.insert(0, bf.pop())
bf.insert(-1, bf.pop())
target_pos = sorted(list(target_pos))

numg = [5, 3, 6]
nameg = ["Natural", "3 star, 7.5 arcseconds", "6 star, 15 arcseconds"]
c = ['#D7191C', '#2CBB36', '#2C7BB6']

def set_box_color(bp, color, alpha=1.0):
    plt.setp(bp['boxes'], color=color, alpha=alpha)
    plt.setp(bp['whiskers'], color=color, alpha=alpha)
    plt.setp(bp['caps'], color=color, alpha=alpha)
    plt.setp(bp['medians'], color=color, alpha=alpha)

ticks = bf

for tp in target_pos:
    # PLOT RMSE RESULTS
    figr, axsr = plt.subplots(1, 1)
    figr.set_size_inches(7, 6)
    figr.suptitle("RMSE for target at\ntheta = {} radians, gamma = {} arcseconds".format(tp[0], tp[1]))


    data_all = [np.array([[mse[(bas, n, tp)]] for n in numg]).ravel() for bas in bf]
    bp = axsr.boxplot(data_all, positions=np.array(range(len(data_all))) - (3/9) + 0.5, sym='', widths=0.27)
    aggc = '#000000'
    set_box_color(bp, aggc)
    axsr.plot([], c=aggc, label="Aggregate")

    for i in range(len(numg)):
        data_n = [y for y in [mse[(bas, numg[i], tp)] for bas in bf]]

        bn = axsr.boxplot(data_n, positions=np.array(range(len(data_n))) + (((2*i)-0.5)/9) + 0.5, sym='', widths=0.18)
        set_box_color(bn, c[i])
        axsr.plot([], c=c[i], label=nameg[i])

    axsr.vlines(range(1, len(ticks)), ymin=0, ymax=4, colors='k', alpha=0.2, linewidth=0.5)
    axsr.set_xticks(np.array(range(0, len(ticks))) + 0.5, ticks)
    axsr.set_xlim(0, len(ticks))
    axsr.set_ylim(0, 4)
    axsr.set_ylabel("RMSE (radians)")
    axsr.set_xlabel("Basis Functions")
    axsr.legend()
    figr.tight_layout()
    figr.savefig('results/img/rmse_{}_{}.png'.format(tp[0],tp[1]))

    # PLOT FWHM RESULTS
    figf, axsf = plt.subplots(1, 1)
    figf.set_size_inches(7, 6)
    figf.suptitle("FWHM for target at\ntheta = {} radians, gamma = {} arcseconds".format(tp[0], tp[1]))

    data_all = [np.array([[fwhm[(bas, n, tp)]] for n in numg]).ravel() for bas in bf]
    bp = axsf.boxplot(data_all, positions=np.array(range(len(data_all))) - (3/9) + 0.5, sym='', widths=0.27)
    aggc = '#000000'
    set_box_color(bp, aggc)
    axsf.plot([], c=aggc, label="Aggregate")

    for i in range(len(numg)):
        data_n = [y for y in [fwhm[(bas, numg[i], tp)] for bas in bf]]

        bn = axsf.boxplot(data_n, positions=np.array(range(len(data_n))) + (((2*i)-0.5)/9) + 0.5, sym='', widths=0.18)
        set_box_color(bn, c[i])
        axsf.plot([], c=c[i], label=nameg[i])

    axsf.vlines(range(1, len(ticks)), ymin=-3, ymax=3, colors='k', alpha=0.2, linewidth=0.5)
    axsf.set_xticks(np.array(range(0, len(ticks))) + 0.5, ticks)
    axsf.set_xlim(0, len(ticks))
    axsf.set_ylim(-3, 3)
    axsf.set_ylabel("Difference from actual (pixels)")
    axsf.set_xlabel("Basis Functions")
    axsf.legend()
    figf.tight_layout()
    figf.savefig('results/img/fwhm_{}_{}.png'.format(tp[0],tp[1]))

plt.show()