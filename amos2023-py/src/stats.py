import numpy as np
from fitter import Fitter
import joblib
import matplotlib.pyplot as plt
import scipy

# extend to get more best distributions
class FitterN(Fitter):
    def get_bestN(self, Nbest=5, method="sumsquare_error"):
        """Return best fitted distribution and its parameters

        a dictionary with one key (the distribution name) and its parameters

        """
        # self.df should be sorted, so then us take the first one as the best
        distros = {}
        for i in range(Nbest):
            print(self.df_errors.sort_values(method).iloc[i])
            name = self.df_errors.sort_values(method).iloc[i].name
            try:
                params = self.fitted_param[name]
                distribution = getattr(scipy.stats, name)
                param_names = (distribution.shapes + ", loc, scale").split(", ") if distribution.shapes else ["loc", "scale"]

                param_dict = {}
                for d_key, d_val in zip(param_names, params):
                    param_dict[d_key] = d_val
                distros[name] = param_dict
            except:
                pass

        return distros


def load_star_data(mag, n_stars=3, n=None):
    data = {}

    for star in range(1, n_stars + 1):
        filename = f'data/satellites/data_{mag}_{star}.pak'
        with open(filename, 'rb') as f:
            print(f'Opening file "{filename}"')
            if n is None:
                data[star] = np.array(joblib.load(f))
            else:
                data[star] = np.array(np.random.choice(joblib.load(f), size=n))
    
    return data

def build_models(data, distributions=None, n=500, timeout=60, Nbest=10, method="sumsquare_error"):

    sample = np.random.choice(data, size=n)
    f = FitterN(sample, distributions=distributions, timeout=timeout)
    f.fit()
    models = f.get_bestN(Nbest=Nbest, method=method)

    return models

def build_distribution_func(name, params):
    distribution = getattr(scipy.stats, name)
    def dis(p):
        return lambda n: distribution.rvs(size=n, **p)
    return dis(params)

if __name__ == "__main__":
    initial_distributions = ['betaprime', 'f', 'lognorm', 'ncf', 'gumbel_r', 'johnsonsu', 'nct', 'burr', 'fisk', 'mielke', 'invgamma', 'burr12', 'jf_skew_t', 'johnsonsb', 'genhyperbolic', 'exponorm', 'alpha', 'moyal', 'norminvgauss']

    # 1) Find distribution for closest stars
    data = load_star_data(mag=14, n_stars=3)

    invalids = []
    for i in range(len(data[1])):
        a = data[1][i]
        b = data[2][i]
        c = data[3][i]
        if a > b or b > c or a > c:
            invalids.append(i)
            print(f"[{i}]: {a},{b},{c}")
    for k in [1, 2, 3]:
        data[k] = np.delete(data[k], invalids)

    n = 500
    # 2) Direct Sampling
    direct = list(zip(data[1], data[2], data[3]))
    print(len(direct))
    direct_sample = np.asarray([direct[i] for i in np.random.choice(len(direct), size=n)])
    print(f"Direct sample size: {np.shape(direct_sample)}")

    # 3) Rejection sampling
    # Build models for each distance
    rej_model = {}
    for key in [1, 2, 3]:
        models = build_models(data[key], distributions=initial_distributions, n=n, timeout=120, Nbest=5, method="ks_statistic")
        model_list = list(models.keys())
        models = build_models(data[key], distributions=model_list, n=10*n, timeout=120, Nbest=1, method="ks_statistic")
        rej_model[key] = [getattr(scipy.stats, name)(**params) for name, params in models.items()][0]

    # Take a sample, rejecting samples that do not match the criteria
    rej_sample = []
    while len(rej_sample) < n:
        a = rej_model[1].rvs(1)
        b = rej_model[2].rvs(1)
        c = rej_model[3].rvs(1)
        if a < b and b < c:
            rej_sample.append((a, b, c))
    rej_sample = np.reshape(np.asarray(rej_sample), (n, 3))
    print(f"Rejection sample size: {np.shape(rej_sample)}")

    # 4) Difference modelling
    # build models, where the second and third closest are based on differences from the previous closest
    dif_model = {}
    dif_data = {1: data[1], 2: data[2] - data[1], 3: data[3] - data[2]}
    for key in [1, 2, 3]:
        models = build_models(data[key], distributions=initial_distributions, n=n, timeout=120, Nbest=5, method="ks_statistic")
        model_list = list(models.keys())
        models = build_models(data[key], distributions=model_list, n=10*n, timeout=120, Nbest=1, method="ks_statistic")
        dif_model[key] = [getattr(scipy.stats, name)(**params) for name, params in models.items()][0]

    # take a sample, adding second and third to the results
    a = dif_model[1].rvs(n)
    b = a + dif_model[2].rvs(n)
    c = b + dif_model[3].rvs(n)
    dif_sample = np.asarray(list(zip(a, b, c)))
    print(f"Difference sample size: {np.shape(dif_sample)}")

    # 5) Sum modelling
    # Build model using sums of distances from the satellite
    sum_data = np.sum([data[1], data[2], data[3]], axis=1)
    models = build_models(sum_data, distributions=initial_distributions, n=n, timeout=120, Nbest=5, method="ks_statistic")
    model_list = list(models.keys())
    models = build_models(sum_data, distributions=model_list, n=10*n, timeout=120, Nbest=1, method="ks_statistic")
    sum_model = [getattr(scipy.stats, name)(**params) for name, params in models.items()][0]

    # take samples, splitting into three
    s = sum_model.rvs(n)
    print(np.shape(s))
    def split(v):
        a = rej_model[1].rvs(1)
        while a > v / 3:
            a = rej_model[1].rvs(1)
        b = np.random.uniform(a, (v - a)/2)
        c = v - a - b
        return tuple(sorted([a, b, c]))
    sum_sample = np.reshape(np.asarray(list(map(split, s))), (n, 3))
    print(f"Sum of squares sample size: {np.shape(sum_sample)}")

    # 5b) Sum modelling (squares)
    # Build model using sum of squares of distances from the satellite
    ss_data = np.sum([data[1] ** 2, data[2] ** 2, data[3] ** 2], axis=1)
    models = build_models(ss_data, distributions=initial_distributions, n=n, timeout=120, Nbest=5, method="ks_statistic")
    model_list = list(models.keys())
    models = build_models(ss_data, distributions=model_list, n=10*n, timeout=120, Nbest=1, method="ks_statistic")
    ss_model = [getattr(scipy.stats, name)(**params) for name, params in models.items()][0]

    # take samples, splitting into three, then taking the square root
    s = ss_model.rvs(n)
    print(np.shape(s))
    def split_root(v):
        a = rej_model[1].rvs(1) ** 2
        while a > v/3:
            a = rej_model[1].rvs(1) ** 2
        b = np.random.uniform(a, (v - a)/2)
        c = v - a - b
        return tuple(sorted([a ** 0.5, b ** 0.5, c ** 0.5]))
    ss_sample = np.reshape(np.asarray(list(map(split_root, s))), (n, 3))
    print(f"Sum of squares sample size: {np.shape(ss_sample)}")

    fig, axs = plt.subplots(3, 5)

    xmin = 0
    xmax = 1250

    for n, d in enumerate([direct_sample, rej_sample, dif_sample, sum_sample, ss_sample]):
        axs[0, n].hist(d[:, 0], 100, density=True)
        axs[1, n].hist(d[:, 1], 100, density=True)
        axs[2, n].hist(d[:, 2], 100, density=True)
        axs[0, n].set_xlim([xmin, xmax])
        axs[1, n].set_xlim([xmin, xmax])
        axs[2, n].set_xlim([xmin, xmax])

    axs[0, 0].set_title("Raw Data")
    axs[0, 1].set_title("Rejection Sampling")
    axs[0, 2].set_title("Modelling Differences")
    axs[0, 3].set_title("Modelling Sum")
    axs[0, 4].set_title("Modelling Sum of Squares")

    axs[0, 0].set_ylabel("Closest star")
    axs[1, 0].set_ylabel("2nd closest star")
    axs[2, 0].set_ylabel("3rd closest star")

    print("> > Comparison between raw data and rejection sampling")
    r = scipy.stats.kstest(direct_sample[:, 0], rej_sample[:, 0])
    print(f"> Closest: KS Test p-value = {r.pvalue}")
    r = scipy.stats.kstest(direct_sample[:, 1], rej_sample[:, 1])
    print(f"> Second closest: KS Test p-value = {r.pvalue}")
    r = scipy.stats.kstest(direct_sample[:, 2], rej_sample[:, 2])
    print(f"> Third closest: KS Test p-value = {r.pvalue}")

    print("> > Comparison between raw data and difference sampling")
    r = scipy.stats.kstest(direct_sample[:, 0], dif_sample[:, 0])
    print(f"> Closest: KS Test p-value = {r.pvalue}")
    r = scipy.stats.kstest(direct_sample[:, 1], dif_sample[:, 1])
    print(f"> Second closest: KS Test p-value = {r.pvalue}")
    r = scipy.stats.kstest(direct_sample[:, 2], dif_sample[:, 2])
    print(f"> Third closest: KS Test p-value = {r.pvalue}")

    print("> > Comparison between raw data and sum sampling")
    r = scipy.stats.kstest(direct_sample[:, 0], sum_sample[:, 0])
    print(f"> Closest: KS Test p-value = {r.pvalue}")
    r = scipy.stats.kstest(direct_sample[:, 1], sum_sample[:, 1])
    print(f"> Second closest: KS Test p-value = {r.pvalue}")
    r = scipy.stats.kstest(direct_sample[:, 2], sum_sample[:, 2])
    print(f"> Third closest: KS Test p-value = {r.pvalue}")

    print("> > Comparison between raw data and sum of squares sampling")
    r = scipy.stats.kstest(direct_sample[:, 0], ss_sample[:, 0])
    print(f"> Closest: KS Test p-value = {r.pvalue}")
    r = scipy.stats.kstest(direct_sample[:, 1], ss_sample[:, 1])
    print(f"> Second closest: KS Test p-value = {r.pvalue}")
    r = scipy.stats.kstest(direct_sample[:, 2], ss_sample[:, 2])
    print(f"> Third closest: KS Test p-value = {r.pvalue}")

    
    plt.show()

