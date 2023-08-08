import matplotlib.pyplot as plt
import numpy as np

data = []
keys = []

with open("./results/results_64.data", "r") as f:
    raw_data = f.readlines()

keys = raw_data[0].strip().split(',')
for k in keys:
    print(k.strip())
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
        
    try:
        fwhm[(b, ng, (tt, tg))].append(fwhm_actual / fwhm_recons)
    except KeyError:
        fwhm[(b, ng, (tt, tg))] = [fwhm_actual / fwhm_recons]

bf = sorted(list(bf))
target_pos = sorted(list(target_pos))

for tp in target_pos:
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(12, 5)
    fig.suptitle("Mean Squared Error for target at {}, {}".format(tp[0], tp[1]))

    numg = [1, 3, 6]
    nameg = ["Natural", "3 star, 7.5 arcseconds", "6 star, 15 arcseconds"]
    clr = ['#D7191C', '#2CBB36', '#2C7BB6']

    for i, (n, nm, c) in enumerate(zip(numg, nameg, clr)):

        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)

        data_n = [y for y in [mse[(bas, n, tp)] for bas in bf]]

        ticks = bf

        bp = axs[i].boxplot(data_n, positions=np.array(range(len(data_n)))*2.0-0.6, sym='', widths=0.5)
        set_box_color(bp, c) # colors are from http://colorbrewer2.org/

        axs[i].set_xticks(range(0, len(ticks) * 2, 2), ticks)
        axs[i].set_xlim(-2, len(ticks)*2)
        axs[i].set_title(nm)
        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45)

        # draw temporary red and blue lines and use them to create a legend
        axs[i].plot([], c=c, label=nm)
    
    for i in range(len(numg)):
        axs[i].legend()
    fig.tight_layout()
    fig.savefig('results/img/boxcompare_mse_{}.png'.format(n))

# for n in [3, 6, 12]:
#     mse[(64, "wavelets sym2", n, 30)] = []
#     mse[(64, "wavelets sym2", n, 60)] = []
#     mse[(64, "wavelets sym2", n, 100)] = []

#     fig, axs = plt.subplots(1, 3)
#     fig.set_size_inches(12, 5)
#     # fig.suptitle("Mean Structural Similarity for {} guidestars".format(n))

#     def set_box_color(bp, color):
#         plt.setp(bp['boxes'], color=color)
#         plt.setp(bp['whiskers'], color=color)
#         plt.setp(bp['caps'], color=color)
#         plt.setp(bp['medians'], color=color)

#     for i, a in enumerate([30, 60, 100]):
#         data_a = [y for y in [mssim[(25, bas, n, a)] for bas in bf]]
#         data_b = [y for y in [mssim[(36, bas, n, a)] for bas in bf]]
#         data_c = [y for y in [mssim[(64, bas, n, a)] for bas in bf]]

#         ticks = bf

#         bpl = axs[i].boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.6, sym='', widths=0.5)
#         bpc = axs[i].boxplot(data_b, positions=np.array(range(len(data_b)))*2.0, sym='', widths=0.5)
#         bpr = axs[i].boxplot(data_c, positions=np.array(range(len(data_c)))*2.0+0.6, sym='', widths=0.5)
#         set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
#         set_box_color(bpc, '#2CBB36')
#         set_box_color(bpr, '#2C7BB6')
#         axs[i].set_xticks(range(0, len(ticks) * 2, 2), ticks)
#         axs[i].set_xlim(-2, len(ticks)*2)
#         axs[i].set_title("{} arcseconds".format(a))
#         plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45)

#         # draw temporary red and blue lines and use them to create a legend
#         axs[i].plot([], c='#D7191C', label='25')
#         axs[i].plot([], c='#2CBB36', label='36')
#         axs[i].plot([], c='#2C7BB6', label='64')
#         axs[i].legend()

#     fig.tight_layout()
#     fig.savefig('results/img/boxcompare_mssim_{}.png'.format(n))


plt.show()