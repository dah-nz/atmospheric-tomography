import matplotlib.pyplot as plt
import numpy as np

data = []
keys = []

with open("./results/results_residuals_16_SNR_0.data", "r") as f:
    raw_data = f.readlines()

keys = raw_data[0].strip().split(',')
keys = [k.strip() for k in keys]

for l in raw_data[1:]:
    data.append(l.strip().split(','))

print(keys)

rmse_baseline = {}
rmse_ideal = {}
rmse_recon = {}
fwhm = {}

mssim = {}
strehl = {}

bf = set()
target_pos = set()

for d in data:
    n = int(d[keys.index('N Coeffs')])
    b = d[keys.index('Basis Function')]
    bf.add(b)
    ng = int(d[keys.index('N Guide Stars')])
    gtheta = float(d[keys.index('Guide Star Angle')])
    tg = float(d[keys.index('Target Gamma')])
    target_pos.add(tg)
    err_baseline = float(d[keys.index('RMSE B:A')])
    err_ideal = float(d[keys.index('RMSE I:A')])
    err_recon = float(d[keys.index('RMSE R:A')])
    fi = float(d[keys.index('FWHM I:A')])
    fr = float(d[keys.index('FWHM R:A')])
    intr = float(d[keys.index('Max Int. R:A')])
    inti = float(d[keys.index('Max Int. I:A')])
    sim = float(d[keys.index('MSSIM R:A-I:A')])

    try:
        rmse_baseline[tg].append(err_baseline)
    except KeyError:
        rmse_baseline[tg] = [err_baseline]
    
    try:
        rmse_ideal[(b, tg)].append(err_ideal)
    except KeyError:
        rmse_ideal[(b, tg)] = [err_ideal]

    try:
        rmse_recon[(b, ng, tg)].append(err_recon)
    except KeyError:
        rmse_recon[(b, ng, tg)] = [err_recon]
        
    wavelength = 600e-9         # m
    telescope_diameter = 0.6    # m
    Q = 8                       # pixels per resolution unit
    effective_focal_length = telescope_diameter * 6.5 # (f/6.5 system)
    um = 1e6                    # um/m
    resolution = (wavelength * effective_focal_length / Q) * um
    # resolution = effective_focal_length / Q
    try:
        fwhm[(b, ng, tg)].append((fr - fi) * resolution)
    except KeyError:
        fwhm[(b, ng, tg)] = [(fr - fi) * resolution]

    try:
        strehl[(b, ng, tg)].append(intr/inti)
    except KeyError:
        strehl[(b, ng, tg)] = [intr/inti]

    try:
        mssim[(b, ng, tg)].append(sim)
    except KeyError:
        mssim[(b, ng, tg)] = [sim]

bf = sorted(list(bf))
bf.insert(0, bf.pop())
# bf.insert(-1, bf.pop())
bf.reverse()
target_pos = sorted(list(target_pos))

numg = [3, 6, 5]
nameg = ['Symmetric (3 star, 30")', 'Symmetric (6 star, 60")', "Asymmetric"]
c = ['#3bb1c5', '#162f44', '#1340a5']

def set_box_color(bp, color, alpha=1.0):
    for pc in bp['bodies']:
        pc.set_facecolor('white')
        pc.set_edgecolor(color)
        # pc.set_alpha(0.5)
    plt.setp(bp['cmeans'], color=color, alpha=alpha)
    plt.setp(bp['cbars'], color=color, alpha=0.8)
    plt.setp(bp['cmins'], color=color, alpha=0.3)
    plt.setp(bp['cmaxes'], color=color, alpha=0.3)

ticks = bf

def plot_bwplots(ax, data, target, xlabel="Basis Functions", ylabel="RMSE (radians)"):
    ymn = 0
    ymx = 1

    for i in range(len(numg)):
        data_n = [y for y in [data[(bas, numg[i], target)] for bas in bf]]
        ymn = min(ymn, np.amin(data_n) - 0.2)
        ymx = max(ymx, np.amax(data_n) + 0.2)

        bn = ax.violinplot(data_n, vert=False, showmeans=True, positions=np.array(range(len(data_n))) + (i-1)*0.3 + 0.5, widths=0.25)
        set_box_color(bn, c[i])
        ax.plot([], c=c[i], label=nameg[i])

    # ax.vlines(range(1, len(ticks)), ymin=ymn, ymax=ymx, colors='k', alpha=0.2, linewidth=0.5)
    ax.set_yticks(np.array(range(0, len(ticks))) + 0.5, ticks)
    ax.set_ylim(0, len(ticks))
    ax.set_xlim(ymn, ymx)
    ax.set_xlabel(ylabel)
    ax.set_ylabel(xlabel)
    ax.grid(True, axis='x', alpha=0.1)
    ax.legend()

for tp in target_pos:
    # PLOT RMSE RESULTS
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(9, 6)
    fig.suptitle("RMSE for target at {} arcseconds".format(tp))

    plot_bwplots(ax, rmse_recon, tp)

    baseline = np.mean(rmse_baseline[tp])
    plt.plot([baseline, baseline], [0, 5], 'k', alpha=0.2)
    ideal = [np.mean(rmse_ideal[(b, tp)]) for b in bf]
    for i, ideal_mean in enumerate(ideal):
        plt.plot([ideal_mean, ideal_mean], [i, i+1], 'k', alpha=0.2)

    ax.plot([0, 0], [0, 0], 'k', alpha=0.2)
    fig.tight_layout()
    fig.savefig('results/img/' + 'rmse_{}.png'.format(tp))
    # plot_bwplots(rmse_baseline, tp, title="RMSE Reconstruction difference to baseline for target at {} arcseconds".format(tp), save="rmse_baseline_{}.png".format(tp))
    # plot_bwplots(rmse_ideal, tp, title="RMSE Reconstruction difference to ideal for target at {} arcseconds".format(tp), save="rmse_ideal_{}.png".format(tp))

    # PLOT FWHM RESULTS
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(9, 6)
    fig.suptitle("FWHM for target at {} arcseconds".format(tp))

    plot_bwplots(ax, fwhm, tp, ylabel="Difference from actual (um)")
    ax.plot([0, 0], [0, 5], 'k', alpha=0.2)
    fig.tight_layout()
    fig.savefig('results/img/' + 'fwhm_{}.png'.format(tp))

    # PLOT STREHL RESULTS
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(9, 6)
    fig.suptitle("Strehl for target at {} arcseconds".format(tp))

    plot_bwplots(ax, strehl, tp, ylabel="Strehl Ratio")
    ax.plot([1, 1], [0, 5], 'k', alpha=0.2)
    fig.tight_layout()
    fig.savefig('results/img/' + 'strehl_{}.png'.format(tp))

    # PLOT MSSIM RESULTS
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(9, 6)
    fig.suptitle("MSSIM for target at {} arcseconds".format(tp))

    plot_bwplots(ax, mssim, tp, ylabel="MSSIM")
    ax.plot([1, 1], [0, 5], 'k', alpha=0.2)
    fig.tight_layout()
    fig.savefig('results/img/' + 'mssim_{}.png'.format(tp))


plt.show()