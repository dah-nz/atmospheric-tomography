"""
Calculate r0 for a set of three phase screens at different altitudes.
I want to have two modes:
Equal turbulence, and a realistic approach where the ground layer has the most distortion, but stil lresults in the same total r0 across all three layers.

"""
import numpy as np
from tomography import phase_screen, beacon
from scipy import stats
from util import circle
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.diagnostic import anderson_statistic

D = 4

def run_tests():
    # test_r0()
    test_r0_distribution()

def variance(D, r0):
    return 1.0299*(D/r0)**(5/3)

def test_r0():
    
    N = 1000
    N_r0 = 100

    ap_pix = 100

    alts = [480, 5100, 11240]

    data = {
        "Test": [],
        "Type": [],
        "Layer": [],
        "Value": [],
    }
    target = beacon.Beacon(0,0,0, aperture_mask=circle(ap_pix))
    mask = target.aperture_mask

    rng = np.random.default_rng()

    r0_index = 0

    # for r0 in [[0.1202, 0.7949, 0.7798], [0.6766, 0.6766, 0.6766]]:
    for r0 in rng.normal(0.4, 0.1, (N_r0, 3)):
        
        r0 = np.array(r0)
        r0[r0 < 0] = 0
        r0[r0 > 1.0] = 1.0

        r0 = list(r0)
        # if type(r0) is not list:
        #     r0 = [r0*(len(alts)**(3/5)) for _ in alts]

        r0_layers = np.array(r0)  # in meters

        for l in range(3):
            data["Test"].append(r0_index)
            data["Type"].append("Expected r_0")
            data["Layer"].append(f"{l+1}")
            data["Value"].append(r0[l])

            data["Test"].append(r0_index)
            data["Type"].append("Expected variance")
            data["Layer"].append(f"{l+1}")
            data["Value"].append(variance(D, r0[l]))

        r0_total = (np.sum(r0_layers**(-5/3)))**(-3/5)

        data["Test"].append(r0_index)
        data["Type"].append("Expected r_0")
        data["Layer"].append("Total")
        data["Value"].append(r0_total)

        data["Test"].append(r0_index)
        data["Type"].append("Expected variance")
        data["Layer"].append(f"Total")
        data["Value"].append(variance(D, r0_total))

        # for l in range(3):
        #     print(f"l{l+1},{r0[l]*100:.2f}")
        # print(f"  ,{r0_total*100:.2f}")

        gen = phase_screen.Atmosphere(n=N,
                                layer_altitudes=alts,
                                r0=r0,
                                D=D,
                                aperture_size=ap_pix)
        
        gen2 = phase_screen.Atmosphere(n=N,
                                layer_altitudes=[0],
                                r0=[r0_total],
                                D=D,
                                aperture_size=ap_pix)
        
        def covar(x, y, mask):
            xi = x - np.mean(x, where=mask>0) * mask
            yi = y - np.mean(y, where=mask>0) * mask
            p = np.reshape(np.ravel(xi) * np.ravel(yi), shape=xi.shape)
            return np.sum(p, where=mask>0)/np.sum(mask)

        for atm in gen2:
            cutouts = [target.cutout(D, a, l, d, remove_piston=True) for a, l, d in zip([0], atm, gen2.metapupil_diameters)]

            r0_out = phase_screen.calculate_r0(cutouts[0], D, mask=mask)

            data["Test"].append(r0_index)
            data["Type"].append("Single r_0")
            data["Layer"].append(f"Total")
            data["Value"].append(r0_out)

            data["Test"].append(r0_index)
            data["Type"].append("Single variance")
            data["Layer"].append(f"Total")
            data["Value"].append(np.var(cutouts[0], where=mask > 0))

        
        for atm in gen:
            # n_atm = []
            # for scr, er0 in zip(atm, r0):
            #     n_atm.append(np.sqrt(np.var(scr)/(1.0299*(D/er0)**(5/3))) * scr)

            cutouts = [target.cutout(D, a, l, d, remove_piston=True) for a, l, d in zip(alts, atm, gen.metapupil_diameters)]

            var_sum = 0
            cutout_sum = None

            for l in range(3):
                co = cutouts[l]

                if cutout_sum is None:
                    cutout_sum = co
                else:
                    cutout_sum = cutout_sum + co

                r0_out = phase_screen.calculate_r0(co, D, mask=mask)

                data["Test"].append(r0_index)
                data["Type"].append("Actual r_0")
                data["Layer"].append(f"{l+1}")
                data["Value"].append(r0_out)

                data["Test"].append(r0_index)
                data["Type"].append("Actual variance")
                data["Layer"].append(f"{l+1}")
                data["Value"].append(np.var(co, where=mask > 0))

                # plt.figure()
                # plt.imshow(co)

                # if l > 0:
                #     cv = covar(co, cutouts[l-1], mask)
                # else:
                #     cv = covar(co, cutouts[-1], mask)

                var_sum += np.var(co, where=mask > 0)
                # print(f"layer {l+1},{r0[l]*100:.2f},{r0_out*100:.2f},{var_sum:.2f},{np.var(co, where=mask > 0):.2f},{variance(D, r0[l]):.2f},{np.mean(co, where=mask>0)}")

        
            cutout = target.total_cutout(D, alts, atm, gen.metapupil_diameters)

            r0_out = phase_screen.calculate_r0(cutout, D, mask=mask)
            r0_out2 = phase_screen.calculate_r0(cutout_sum, D, mask=mask)

            print(f"combine,{r0_total*100:.2f},{r0_out*100:.2f},{r0_out2*100:.2f},{var_sum:.2f},{np.var(cutout, where=mask > 0):.2f},{np.var(cutout_sum, where=mask > 0):.2f},{variance(D, r0_total):.2f},{np.mean(cutout, where=mask>0)}")

            data["Test"].append(r0_index)
            data["Type"].append("Actual r_0")
            data["Layer"].append("Total")
            data["Value"].append(r0_out)

            data["Test"].append(r0_index)
            data["Type"].append("Actual variance")
            data["Layer"].append("Total")
            data["Value"].append(np.var(cutout, where=mask > 0))
        
        r0_index += 1

    # for k, v in data.items():
    #     print(k, len(v))

    df = pd.DataFrame(data)
    df.to_excel("data/tests/r0.xlsx")
    
def test_r0_distribution():
    global D

    df = pd.read_excel("data/tests/r0.xlsx")
    N_bootstrap = 1000
    distributions = ["norm"]
    # distributions = ["lognorm", "gamma", "expon", "weibull_min", "norm"]
    # distributions = ["lognorm", "gamma", "weibull_min", "norm"]

    r0_to_var = lambda r0: 1.0299 * (D/r0) ** (5/3)
    var_to_r0 = lambda var: D * (1.0299 / var) ** (3/5)

    mean_diffs = []
    single_diffs = []

    for i in df["Test"].unique():
        df_t = df[(df["Test"] == i) & (df["Layer"] == "Total")]
        expected_r0 = df_t.loc[df_t["Type"] == f"Expected r_0", "Value"].iloc[0]
        expected_var = df_t.loc[df_t["Type"] == f"Expected variance", "Value"].iloc[0]

        actual_r0 = df_t.loc[df_t["Type"] == f"Actual r_0", "Value"]
        # single_r0 = df_t.loc[df_t["Type"] == f"Single r_0", "Value"]

        actual_var = df_t.loc[df_t["Type"] == f"Actual variance", "Value"]
        # single_var = df_t.loc[df_t["Type"] == f"Single variance", "Value"]

        l1_r0 = df.loc[(df["Test"] == i) & (df["Layer"] == "1") & (df["Type"] == "Actual r_0"), "Value"]
        l2_r0 = df.loc[(df["Test"] == i) & (df["Layer"] == "2") & (df["Type"] == "Actual r_0"), "Value"]
        l3_r0 = df.loc[(df["Test"] == i) & (df["Layer"] == "3") & (df["Type"] == "Actual r_0"), "Value"]

        l1_var = df.loc[(df["Test"] == i) & (df["Layer"] == "1") & (df["Type"] == "Actual variance"), "Value"]
        l2_var = df.loc[(df["Test"] == i) & (df["Layer"] == "2") & (df["Type"] == "Actual variance"), "Value"]
        l3_var = df.loc[(df["Test"] == i) & (df["Layer"] == "3") & (df["Type"] == "Actual variance"), "Value"]

        layer_r0 = np.reshape(np.concat([l1_r0, l2_r0, l3_r0]), shape=(-1, 3))
        layer_r0 = np.sum(layer_r0**(-5/3), axis=1) ** (-3/5)

        layer_var = np.reshape(np.concat([l1_var, l2_var, l3_var]), shape=(-1, 3))
        layer_var = np.sum(layer_var, axis=1)

        mean_r0 = np.mean(actual_r0)
        mean_layer_r0 = np.mean(layer_r0)
        # single_mean_r0 = np.mean(single_r0)

        mean_var = np.mean(actual_var)
        mean_layer_var = np.mean(layer_var)
        # single_mean_var = np.mean(single_var)
        mean_to_transform_r0 = var_to_r0(np.mean(actual_var))
        transform_to_mean_r0 = np.mean(var_to_r0(actual_var))
        transformed_r0 = np.mean(var_to_r0(actual_var) ** (-5/3)) ** (-3/5)

        print(mean_r0, expected_r0, mean_layer_r0)
        print(mean_var, expected_var, mean_layer_var)
        print(mean_to_transform_r0, transform_to_mean_r0, transformed_r0)

        # mean_diffs.append(transform(expected_r0) - transform(mean))
        # single_diffs.append(transform(expected_r0) - transform(single_mean))

    return

    for dist in distributions:
        print(dist)
        exec(f"params_{dist} = stats.{dist}.fit(single_diffs)")

    plt.figure()
    plt.title(f"Difference of Means (single layer)")
    single_mean = np.mean(single_diffs)

    x = np.linspace(min(single_diffs), max(single_diffs), 200)
    plt.hist(single_diffs, bins=min(len(single_diffs), 50), density=True, alpha=0.5)
    plt.xlabel("$\\Delta r_0$ (m)")
    plt.ylabel("n")

    height = 0
    for dist in distributions:
        st = eval(f"stats.{dist}.pdf(x, *params_{dist})")
        plt.plot(x, st, label=dist)
        height = max(height, max(st))

    plt.plot([single_mean, single_mean], [0,height], 'k', alpha=0.3, label="Mean difference")
    plt.plot([0, 0], [0,height], 'r', alpha=0.3, label=f"Expected difference")

    plt.legend()

    for dist in distributions:
        print(dist)
        exec(f"params_{dist} = stats.{dist}.fit(mean_diffs)")

    plt.figure()
    plt.title(f"Difference of Means (Multi-layer)")
    mean = np.mean(mean_diffs)

    x = np.linspace(min(mean_diffs), max(mean_diffs), 200)
    plt.hist(mean_diffs, bins=min(len(mean_diffs), 50), density=True, alpha=0.5)
    plt.xlabel("$\\Delta r_0$ (m)")
    plt.ylabel("n")

    height = 0
    for dist in distributions:
        st = eval(f"stats.{dist}.pdf(x, *params_{dist})")
        plt.plot(x, st, label=dist)
        height = max(height, max(st))

    plt.plot([mean, mean], [0,height], 'k', alpha=0.3, label="Mean difference")
    plt.plot([0, 0], [0,height], 'r', alpha=0.3, label=f"Expected difference")

    plt.legend()

    for dist in distributions:
        plt.figure()
        eval(f"stats.probplot(mean_diffs, dist='{dist}', sparams=params_{dist}, plot=plt)")
        plt.title(f"Probability Plot: {dist}")

    means = []
    for _ in range(N_bootstrap):
        sample = np.random.choice(mean_diffs, size=len(mean_diffs), replace=True)
        means.append(np.mean(sample))
    lower, upper = np.percentile(means, [2.5, 97.5])

    print("Differences of sample means to expected r0 (multi-layer)")
    print(f"Bootstrap upper: {upper}, lower:{lower}")
    print(f"Calculated mean: {mean}")

    means = []
    for _ in range(N_bootstrap):
        sample = np.random.choice(single_diffs, size=len(single_diffs), replace=True)
        means.append(np.mean(sample))
    lower, upper = np.percentile(means, [2.5, 97.5])

    print("Differences of sample means to expected r0 (single layer)")
    print(f"Bootstrap upper: {upper}, lower:{lower}")
    print(f"Calculated mean: {single_mean}")

    plt.show()

    # for k, v in data.items():
    #     # t_stat, p_value = stats.ttest_1samp(v, k)
    #     print(f"{k*100:.2f},{np.mean(v)*100:.2f},{np.mean(v)/k:.3f}")
    print("TEST")