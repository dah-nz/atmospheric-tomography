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
            # data[star] = joblib.load(f)
            data[star] = np.random.choice(joblib.load(f), size = 1000)
    
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

if __name__ == "__main__":
    initial_distributions = ['betaprime', 'f', 'lognorm', 'ncf', 'gumbel_r', 'johnsonsu', 'nct', 'burr', 'fisk', 'mielke', 'invgamma', 'burr12', 'jf_skew_t', 'johnsonsb', 'genhyperbolic', 'exponorm', 'alpha', 'moyal', 'norminvgauss']

    data = load_star_data(mag=14)
    models = build_models(data, distributions=initial_distributions, n=500, timeout=120, Nbest=10)
    model_list = set()
    for _, v in models.items():
        model_list.update(v.keys())
    print(list(model_list))
    models = build_models(data, distributions=list(model_list), n=5000, timeout=120, Nbest=1)
    print(models)
    for k, v in models.items():
        print(f'> {k} <> {v}')
        print(v.items())
    model_rsv = [build_distribution_func(k1, v1) for v in models.values() for k1, v1 in v.items()]

    consolidated = []
    n = 500
    fig, axs = plt.subplots(2, len(model_rsv) - 1)
    for i in range(len(model_rsv) - 1):
        d = model_rsv[i](n)
        axs[0, i].hist(d, 200)
        consolidated = np.concatenate((consolidated, d))
        axs[0, i].set_title(f"Plot for model {i+1}")

    d0 = model_rsv[-1](3*n)
    d1 = data[1] + data[2] + data[3]
    sample = np.random.choice(d1, size=3*n)
    axs[1, 1].hist(d0, 20)
    axs[1, 1].set_title("Consolidated Model")
    axs[1, 2].hist(consolidated, 20)
    axs[1, 2].set_title("Individual Models")
    axs[1, 0].hist(sample, 20)
    axs[1, 0].set_title("Actual Data")

    print("\n SINGLE STAR MODELS")
    print("Comparison between closest model, and closest data")
    print(scipy.stats.kstest(model_rsv[0](n), np.random.choice(data[1], size=n)))

    print("Comparison between middle model, and middle data")
    print(scipy.stats.kstest(model_rsv[1](n), np.random.choice(data[2], size=n)))

    print("Comparison between farthest model, and farthest data")
    print(scipy.stats.kstest(model_rsv[2](n), np.random.choice(data[3], size=n)))

    
    print("\nDATA COMPARISONS")
    print("Comparison between single model, and multiple models")
    print(scipy.stats.kstest(d0, consolidated))

    print("Comparison between actual data and multiple models")
    print(scipy.stats.kstest(sample, consolidated))

    print("Comparison between actual data and single model")
    print(scipy.stats.kstest(sample, d0))

    print("Comparison between actual data and itself (for sanity checking)")
    print(scipy.stats.kstest(np.random.choice(d1, size=n), np.random.choice(d1, size=n)))

    closest = []
    middle = []
    farthest = []

    generated = np.concatenate([model_rsv[0](n), model_rsv[1](n), model_rsv[2](n)])
    generated = np.reshape(generated, (-1, 3))
    sorted = np.sort(generated, axis=-1)
    closest = sorted[:, 0]
    middle = sorted[:, 1]
    farthest = sorted[:, 2]

    print("\nSINGLE MODELS")
    print("Comparison between closest model, and generated closest")
    print(scipy.stats.kstest(model_rsv[0](n), closest))

    print("Comparison between middle model and generated middle")
    print(scipy.stats.kstest(model_rsv[1](n), middle))

    print("Comparison between farthest model and generated farthest")
    print(scipy.stats.kstest(model_rsv[2](n), farthest))

    generated = model_rsv[-1](3*n)
    generated = np.reshape(generated, (-1, 3))
    sorted = np.sort(generated)
    closest = sorted[:, 0]
    middle = sorted[:, 1]
    farthest = sorted[:, 2]

    print("\nCONSOLIDATED MODELS:")
    
    print("Comparison between closest model, and consolidated generated closest")
    print(scipy.stats.kstest(model_rsv[0](n), closest))

    print("Comparison between middle model and consolidated generated middle")
    print(scipy.stats.kstest(model_rsv[1](n), middle))

    print("Comparison between farthest model and consolidated generated farthest")
    print(scipy.stats.kstest(model_rsv[2](n), farthest))

    print("Comparison between generated model and generated model (for sanity checking)")
    print(scipy.stats.kstest(model_rsv[-1](n), model_rsv[-1](n)))


    plt.show()