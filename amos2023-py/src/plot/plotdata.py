import matplotlib.pyplot as plt
import numpy as np

data = []
keys = []

with open("./results/results_basis.data", "r") as f:
    raw_data = f.readlines()

keys = raw_data[0].strip().split(',')

for l in raw_data[1:]:
    data.append(l.strip().split(','))

mse = {}
mssim = {}

#['N Coeffs', 'Basis Function', 'Layer 1 Seed', 'Layer 2 Seed', 'N Guide Stars', 'Guide Star Angle', 'Target Theta', 'Target Gamma', 'MSSIM Mean', 'MSSIM Stdev', 'MSE Mean', 'MSE Stdev', 'Time Mean', 'Time Stdev']
keys.index('N Coeffs')

bf = set()

for d in data:
    n = int(d[keys.index('N Coeffs')])
    b = d[keys.index('Basis Function')]
    bf.add(b)
    ng = int(d[keys.index('N Guide Stars')])
    gtheta = int(d[keys.index('Guide Star Angle')])
    tg = float(d[keys.index('Target Gamma')])
    sim = float(d[keys.index('MSSIM Mean')])
    err = float(d[keys.index('MSE Mean')])

    if tg == 0:
        try:
            mse[(n, b, ng, gtheta)].append(err)
        except KeyError:
            mse[(n, b, ng, gtheta)] = [err]
        
        try:
            mssim[(n, b, ng, gtheta)].append(sim)
        except KeyError:
            mssim[(n, b, ng, gtheta)] = [sim]

bf = sorted(list(bf))

for n in [3, 6, 12]:
    mse[(64, "wavelets sym2", n, 30)] = []
    mse[(64, "wavelets sym2", n, 60)] = []
    mse[(64, "wavelets sym2", n, 100)] = []
    mse[(64, "DCT", n, 100)] = []
    mse[(64, "DST", n, 100)] = []

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(12, 5)
    # fig.suptitle("Mean Squared Error for {} guidestars".format(n))

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    for i, a in enumerate([30, 60, 100]):
        data_a = [y for y in [mse[(25, bas, n, a)] for bas in bf]]
        data_b = [y for y in [mse[(36, bas, n, a)] for bas in bf]]
        data_c = [y for y in [mse[(64, bas, n, a)] for bas in bf]]

        ticks = bf

        bpl = axs[i].boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.6, sym='', widths=0.5)
        bpc = axs[i].boxplot(data_b, positions=np.array(range(len(data_b)))*2.0, sym='', widths=0.5)
        bpr = axs[i].boxplot(data_c, positions=np.array(range(len(data_c)))*2.0+0.6, sym='', widths=0.5)
        set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
        set_box_color(bpc, '#2CBB36')
        set_box_color(bpr, '#2C7BB6')
        axs[i].set_xticks(range(0, len(ticks) * 2, 2), ticks)
        axs[i].set_xlim(-2, len(ticks)*2)
        axs[i].set_title("{} arcseconds".format(a))
        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45)

        # draw temporary red and blue lines and use them to create a legend
        axs[i].plot([], c='#D7191C', label='25')
        axs[i].plot([], c='#2CBB36', label='36')
        axs[i].plot([], c='#2C7BB6', label='64')
        axs[i].legend()

    fig.tight_layout()
    fig.savefig('results/img/boxcompare_mse_{}.png'.format(n))

for n in [3, 6, 12]:
    mse[(64, "wavelets sym2", n, 30)] = []
    mse[(64, "wavelets sym2", n, 60)] = []
    mse[(64, "wavelets sym2", n, 100)] = []

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(12, 5)
    # fig.suptitle("Mean Structural Similarity for {} guidestars".format(n))

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    for i, a in enumerate([30, 60, 100]):
        data_a = [y for y in [mssim[(25, bas, n, a)] for bas in bf]]
        data_b = [y for y in [mssim[(36, bas, n, a)] for bas in bf]]
        data_c = [y for y in [mssim[(64, bas, n, a)] for bas in bf]]

        ticks = bf

        bpl = axs[i].boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.6, sym='', widths=0.5)
        bpc = axs[i].boxplot(data_b, positions=np.array(range(len(data_b)))*2.0, sym='', widths=0.5)
        bpr = axs[i].boxplot(data_c, positions=np.array(range(len(data_c)))*2.0+0.6, sym='', widths=0.5)
        set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
        set_box_color(bpc, '#2CBB36')
        set_box_color(bpr, '#2C7BB6')
        axs[i].set_xticks(range(0, len(ticks) * 2, 2), ticks)
        axs[i].set_xlim(-2, len(ticks)*2)
        axs[i].set_title("{} arcseconds".format(a))
        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45)

        # draw temporary red and blue lines and use them to create a legend
        axs[i].plot([], c='#D7191C', label='25')
        axs[i].plot([], c='#2CBB36', label='36')
        axs[i].plot([], c='#2C7BB6', label='64')
        axs[i].legend()

    fig.tight_layout()
    fig.savefig('results/img/boxcompare_mssim_{}.png'.format(n))

# fig, axs = plt.subplots(1, 3)
# fig.suptitle("Mean Squared Error, 12 Guide Stars, 100 arcsecond radius")

# x = 
# axs[0].set_title("25 Coefficients")
# labels = ["{} [{}]".format(b, n) for b in bf for n in [25, 36, 64]]
# axs[0].boxplot(x, labels=labels)

# x = [y for y in [mse[(36, bas, 12, 100)] for bas in bf]]
# axs[1].set_title("36 Coefficients")
# axs[1].boxplot(x, labels=bf)

# x = [y for y in [mse[(64, bas, 12, 100)] for bas in bf]]
# axs[2].set_title("64 Coefficients")
# axs[2].boxplot(x, labels=bf)

# fig, axs = plt.subplots(1, 3)
# fig.suptitle("Mean Squared Error, 6 Guide Stars, 60 arcsecond radius")

# x = [y for y in [mse[(25, bas, 6, 60)] for bas in bf]]
# axs[0].set_title("25 Coefficients")
# axs[0].boxplot(x, labels=bf)

# x = [y for y in [mse[(36, bas, 6, 60)] for bas in bf]]
# axs[1].set_title("36 Coefficients")
# axs[1].boxplot(x, labels=bf)

# x = [y for y in [mse[(64, bas, 6, 60)] for bas in bf]]
# axs[2].set_title("64 Coefficients")
# axs[2].boxplot(x, labels=bf)

# fig, axs = plt.subplots(1, 3)
# fig.suptitle("Mean Squared Error, 3 Guide Stars, 30 arcsecond radius")

# x = [y for y in [mse[(25, bas, 3, 30)] for bas in bf]]
# axs[0].set_title("25 Coefficients")
# axs[0].boxplot(x, labels=bf)

# x = [y for y in [mse[(36, bas, 3, 30)] for bas in bf]]
# axs[1].set_title("36 Coefficients")
# axs[1].boxplot(x, labels=bf)

# x = [y for y in [mse[(64, bas, 3, 30)] for bas in bf]]
# axs[2].set_title("64 Coefficients")
# axs[2].boxplot(x, labels=bf)

# for k, v in mse.items():
#     print("{}, {}, {}".format(k, np.average(v), np.std(v)))

plt.show()