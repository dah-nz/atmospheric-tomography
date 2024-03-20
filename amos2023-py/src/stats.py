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
    data = load_star_data(mag=14, n_stars=1)

    for k, v in data.items():
        print(f"{k}: {len(v)}")
    quit()

    models = build_models(data[1], distributions=initial_distributions, n=5000, timeout=120, Nbest=10)

    model_list = list(models.keys())
    models = build_models(data[1], distributions=model_list, n=50000, timeout=120, Nbest=1, method="ks_statistic")
    print(models)
    for k, v in models.items():
        print(f'> {k} <> {v}')

    model = [getattr(scipy.stats, name)(**params) for name, params in models.items()]
    m = model[0]

    n = 1000

    measured_data = np.random.choice(data[1], size=n)
    model_data = m.rvs(n)

    fig, axs = plt.subplots(1, 2)
    x = np.linspace(m.ppf(0.01), m.ppf(0.99), 100)

    axs[0].hist(measured_data, 100, density=True)
    axs[0].plot(x, m.pdf(x), alpha=0.5)
    axs[0].set_title(f"Measured Data")

    axs[1].hist(model_data, 100, density=True)
    axs[1].plot(x, m.pdf(x), alpha=0.5)
    axs[1].set_title(f"Model Output")

    print("Comparison between closest model, and closest data")
    print(scipy.stats.kstest(model_data, measured_data))

    # 2) Find distribution for next closest stars given a closest star
    for k, v in data.items():
        data[k] = np.sort(v)

    conditional = {}
    conditional[2] = {}
    conditional[3] = {}

    for i in range(2, 4):

        p = np.arange(data[i - 1][0], data[i - 1][-1], step=1)

        dist = [getattr(scipy.stats, name) for name, _ in models.items()]
        d = dist[0]

        print(len(p))
        n = 5000

        for value in p:
            data_subset = data[i][data[1] >= value]
            params = {}
            if len(data_subset) > 0:
                data_subset = np.random.choice(data_subset, size=n)
                kde = d.fit(data_subset)
                param_names = (d.shapes + ", loc, scale").split(", ") if d.shapes else ["loc", "scale"]
                for d_key, d_val in zip(param_names, kde):
                    params[d_key] = d_val
            
            conditional[i][value] = params
            print(value)

    # 3) Compare distribution generated for model to the actual distribution of stars
            
    def get_three_stars(n):
        first = m.rvs(n)
        fkeys = np.arange(data[1][0], data[1][-1], step=1)
        skeys = np.arange(data[2][0], data[2][-1], step=1)
        stars = []
        for f in first:
            fa = fkeys[fkeys >= f]
            while len(fa) == 0:
                f = m.rvs()
                fa = fkeys[fkeys >= f]
            fk = np.amin(fa)
            s = conditional[2][fk]
            try:
                s = d(**s).rvs()
            except TypeError:
                print(s)
                quit()
            sa = skeys[skeys >= s]
            while len(sa) == 0:
                s = m.rvs()
                sa = skeys[skeys >= s]
            sk = np.amin(sa)
            t = conditional[3][sk]
            try: 
                t = d(**t).rvs()
            except TypeError:
                print(t)
                quit()
            stars += [f, s, t]
        # stars = np.reshape(stars, (1, -1))
        return stars
        
    n = 5000
    modelled = get_three_stars(n)
    consolid = np.concatenate([data[1], data[2], data[3]])

    fig, axs = plt.subplots(1, 3)

    con_models = build_models(consolid, distributions=initial_distributions, n=n, timeout=120, Nbest=1)
    con_models = [getattr(scipy.stats, name)(**params) for name, params in con_models.items()]
    c = con_models[0]
    x = np.linspace(c.ppf(0.01), c.ppf(0.99), 100)
    cm_data = c.rvs(n)
    cd_data = np.random.choice(consolid, size=n)
    sm_data = np.random.choice(modelled, size=n)

    axs[0].hist(cd_data, 100, density=True)
    axs[0].plot(x, c.pdf(x), alpha=0.5)
    axs[0].set_title(f"Measured Data")

    axs[1].hist(cm_data, 100, density=True)
    axs[1].plot(x, c.pdf(x), alpha=0.5)
    axs[1].set_title(f"Model Output")

    axs[2].hist(sm_data, 100, density=True)
    axs[2].plot(x, c.pdf(x), alpha=0.5)
    axs[2].set_title(f"Model Output from KDE")

    print("Comparison between kde model, and measured data")
    print(scipy.stats.kstest(sm_data, cd_data))

    plt.show()

