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


def load_star_data(mag, n_stars=3):
    data = {}

    for star in range(1, n_stars + 1):
        filename = f'data/satellites/data_{mag}_{star}.pak'
        with open(filename, 'rb') as f:
            print(f'Opening file "{filename}"')
            data[star] = joblib.load(f)
    
    return data

def build_models(data, distributions=None, n=500, timeout=60, Nbest=10):
    models = {}
    consol = []

    for k, v in data.items():
        sample = np.random.choice(v, size=n)
        f = FitterN(sample, distributions=distributions, timeout=timeout)
        f.fit()
        s = f.get_bestN(Nbest=Nbest)

        models[k] = s
        consol += list(sample)
    
    f = FitterN(sample, distributions=distributions, timeout=timeout)
    f.fit()
    s = f.get_bestN(Nbest=Nbest)

    models['consol'] = s

    return models

def build_distribution_func(name, params):
    distribution = getattr(scipy.stats, name)
    def dis(p):
        return lambda n: distribution.rvs(size=n, **p)
    return dis(params)

def load_star_pos():

    mag = 14
    star = 1
    n = 2000
    test_dist = ['betaprime', 'f', 'lognorm', 'ncf']
    timeout = 60

    consolidated = []

    model_params = []

    for star in range(1, 4):
        filename = f"data/satellites/data_{mag}_{star}.pak"
        with open(filename, 'rb') as f:
            print(f"Opening file '{filename}'")
            data = joblib.load(f)
            consolidated += data
        sample = np.random.choice(data, size=n)
        # sample = data

        # f = Fitter(sample, distributions=test_dist, timeout=timeout)
        f = Fitter(sample, timeout=timeout)
        f.fit()
        s = f.summary(Nbest=10)

        print(s)

        m = f.get_best()
        model_params.append(m)

    sample = np.random.choice(consolidated, size=3*n)
    # sample = data
    # f = Fitter(sample, distributions=test_dist, timeout=timeout)
    f = Fitter(sample, timeout=timeout)
    f.fit()
    s = f.summary()

    print(s)

    m = f.get_best()
    model_params.append(m)
    models = []

    for m in model_params:
        name = list(m.keys())[0]
        model = m[name]
        print(name)
        print(model)
        if name == 'betaprime':
            def f(m1):
                return lambda n: scipy.stats.betaprime.rvs(a=m1['a'], b=m1['b'], loc=m1['loc'], scale=m1['scale'], size=n)
            mf = f(model)
        elif name == 'f':
            def f(m2):
                return lambda n: scipy.stats.f.rvs(dfn=m2['dfn'], dfd=m2['dfd'], loc=m2['loc'], scale=m2['scale'], size=n)
            mf = f(model)
        elif name == 'lognorm':
            def f(m3):
                return lambda n: scipy.stats.lognorm.rvs(s=m3['s'], loc=m3['loc'], scale=m3['scale'], size=n)
            mf = f(model)
        elif name == 'ncf':
            def f(m4):
                return lambda n: scipy.stats.ncf.rvs(dfn=m4['dfn'], dfd=m4['dfd'], nc=m4['nc'], loc=m4['loc'], scale=m4['scale'], size=n)
            mf = f(model)
        else:
            return
        models.append(mf)

    data = []
    fig, axs = plt.subplots(2, len(models) - 1)
    for i in range(len(models) - 1):
        d = models[i](n)
        axs[0, i].hist(d, 200)
        data = np.concatenate((data, d))
        axs[0, i].set_title(f"Plot for model {i+1}")

    d0 = models[-1](3*n)
    axs[1, 1].hist(d0, 200)
    axs[1, 1].set_title("Consolidated Model")
    axs[1, 2].hist(data, 200)
    axs[1, 2].set_title("Individual Models")
    axs[1, 0].hist(sample, 200)
    axs[1, 0].set_title("Actual Data")

    print("Comparison between single model, and multiple models")
    print(scipy.stats.kstest(d0, data))

    print("Comparison between actual data and multiple models")
    print(scipy.stats.kstest(sample, data))

    print("Comparison between actual data and single model")
    print(scipy.stats.kstest(sample, d0))

    plt.show()


if __name__ == "__main__":
    data = load_star_data(mag=6)
    models = build_models(data, n=500, timeout=60, Nbest=10)
    model_list = set()
    for _, v in models.items():
        model_list.update(v.keys())
    print(list(model_list))
    models = build_models(data, distributions=list(model_list), n=5000, timeout=60, Nbest=3)
    print(models)
    for k, v in models.items():
        print(f'> {k} <> {v}')