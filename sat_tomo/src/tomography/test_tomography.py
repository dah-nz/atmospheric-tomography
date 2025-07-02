import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from functools import lru_cache
import pandas as pd
import time
import pickle

from tomography import tomography as tomo
from tomography import beacon, phase_screen, projection, wavefront, asterisms, strehl
from basis import zernike
from astronomy.asterisms import load_data, sample

from tqdm import tqdm

from itertools import combinations

import util

def run_tests():
    # test_layer_shapes()

    # test_strehl()
    # test_beacons()
    # test_atmospheres()
    # test_leastsquares()
    display_leastsquares_results()
    # test_asterisms()
    # projection.test_projection()

    # C = zernike.covar(21)
    # D = 1.8
    # r0 = 0.07
    # C = C * (D / r0) ** (5/3)
    # plt.figure()
    # plt.imshow(C)

    plt.show()

def test_asterisms():
    from astronomy.asterisms import load_data

    angles = np.geomspace(1, 121, 31) - 1
    N_TESTS = 64

    for angle in angles:
        asterisms = load_data(14, limit=angle, quick_load=True, use_cache=False)
        print(f"Angle: {angle}; N: {len(asterisms)}")

def test_atmospheres():
    rng = np.random.default_rng()

    N_TESTS = 100
    alts = [1000, 5000, 10000]
    
    r0s = np.linspace(0.05, 0.15, 11, endpoint=True)

    D = 1.8
    ap = 64

    mask = util.circle(ap)
    target = beacon.Beacon(0, 0, aperture_mask=mask)

    for r0 in r0s:
        atms = phase_screen.Atmosphere(N_TESTS, alts, r0, D, ap, rng=rng)

        variance = {
        f"l{i}": [] for i in range(len(alts))
        }
        variance["target"] = []

        for atm in atms:
            target_screen = target.total_cutout(D, alts, atm, atms.metapupil_diameters)
            variance['target'].append(np.var(target_screen))

            vars = tomo.get_layer_variance_for_asterism([target], D, alts, atm, atms.metapupil_diameters)
            for i in range(len(alts)):
                variance[f"l{i}"].append(vars[i])

        var = np.mean(variance['target'])
        calculated_r0 = D / ((var / 1.03) ** (3/5))
        print(f"Actual r0: {calculated_r0:.4f}m, intended r0: {r0}m [var: {var:.4f}]")

        for i in range(len(alts)):
            var = np.mean(variance[f'l{i}'])
            calculated_r0 = D / ((var / 1.03) ** (3/5))
            print(f"Layer {i} r0: {calculated_r0:.4f}m, intended r0: {atms.layer_r0[i]:.4f}m [var: {var:.4f}]")

def test_leastsquares():
    rng = np.random.default_rng()

    alts = [1000, 5000, 10000]
    r0_total = 0.25

    D = 1.8
    N = 21
    N_TESTS = 10
    N_GS = 3

    ap_sizes = [64, 128, 256]
    angles = np.linspace(1, 121, 7) - 1
    # angles = [1, 5, 15, 30, 50, 75]
    nagle = 30 # asterism angle

    # wfs = wavefront.SHWFS(D, N, noise=False)
    wfs = None

    bss = zernike.Zernike(N)

    _perturb_str = " (Perturbed)"

    tgt_fit = "Fitting"
    tgt_ol = "Open Loop"
    tgt_glao = "GLAO"
    tgt_ao = "Adaptive Optics"
    tgt_p_glao = tgt_glao + _perturb_str
    tgt_p_ao = tgt_ao + _perturb_str

    max_perturb = 0.5 # arcseconds
    n_perturbs = 3 # number of perturbations in each direction
    x = np.linspace(-max_perturb, max_perturb, n_perturbs)
    y = np.linspace(-max_perturb, max_perturb, n_perturbs)
    perturbations = np.stack(np.meshgrid(x, y)).T.reshape(-1, 2)
    perturbations = [tuple(p) for p in perturbations]

    ap_data = []
    sv_data = []

    for ap in ap_sizes:
        glao = tomo.GLAO(bss, D, N, alts, ap)
        ao = tomo.AO(bss, D, N, alts, ap)

        print(f"Building atmospheres for {N_TESTS} tests, with aperture size: {ap}")
        atms = phase_screen.Atmosphere(N_TESTS, alts, r0_total, D, ap, rng=rng)
    
        tests = {
            # "LSQ [TA⁺]":         tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="Ainv",  proj_grid=True, display=False),
            # "LSQ [TA⁺] Pix":     tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="Ainv",  proj_grid=False, display=False),
            # "LSQ [(AT⁺)⁺]":          tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="Tinv",  proj_grid=True, display=False),
            "LSQ [(AT⁺)⁺] Pix":      tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="Tinv",  proj_grid=False, display=False),
            # "LSQ [(AᵀA+εK)⁺Aᵀ]":     tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="K",     proj_grid=True),
            "LSQ [(AᵀA+εK)⁺Aᵀ] Pix": tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="K",     proj_grid=False),
            # "LSQ [(TᵀT+εK)⁺Tᵀ] Pix": tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="K2",     proj_grid=False),
            # "LSQ [f(A,T)]":      tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="dotAT", proj_grid=True),
            # "LSQ [f(A,T)] Pix":      tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="dotAT", proj_grid=False),
            # "LSQ [f(A,K)]":      tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="dotAK", proj_grid=True),
            # "LSQ [f(A,K)] Pix":      tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="dotAK", proj_grid=False),
            # "LSQ [f(T,K)]":      tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="dotTK", proj_grid=True),
            # "LSQ [f(T,K)] Pix":      tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="dotTK", proj_grid=False),
            # "LSQ [all, mean]":   tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=True, start_angle=30, end_angle=45, angle_mix="ave"),
            # "LSQ [all, max]":    tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=True, start_angle=30, end_angle=45, angle_mix="max"),
            # "LSQ [all, min]":    tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=True, start_angle=30, end_angle=45, angle_mix="min"),
            # "LSQ [all, median]": tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=True, start_angle=30, end_angle=45, angle_mix="med"),
            # "LSQ 22\" Pix":   tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=False, start_angle=18, end_angle=22, angle_mix="ave"),
            # "LSQ 45\" Pix":   tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=False, start_angle=18, end_angle=45, angle_mix="ave"),
            # "LSQ [all, max] Pix":    tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=False, start_angle=18, end_angle=22, angle_mix="max"),
            # "LSQ [all, min] Pix":    tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=False, start_angle=18, end_angle=22, angle_mix="min"),
            # "LSQ [all, median] Pix": tomo.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=False, start_angle=18, end_angle=22, angle_mix="med"),
        }

        variance = {
            f"l{i}": [] for i in range(len(alts))
        }
        variance["target"] = []

        mask = util.circle(ap)

        tstart = time.time()
        t0 = tstart

        for nagle in angles:
            
            fixed_asts = asterisms.generate_fixed_asterisms(N_TESTS, nagle, mask=mask)
            
            data = load_data(14, limit=nagle, quick_load=True, use_cache=False, asterism_size=5)
            var_asts = []
            for guidestars in data:
                for ast in combinations(guidestars, 3):
                    var_asts.append([beacon.Beacon(t, g, magnitude=m, aperture_mask=mask) for g, t, m in ast])
            # asts = asterisms.generate_variable_asterisms(N_TESTS, limiting_angle=nagle, zero_rate=0.0, fixed_rate=0.0, mask=mask)
            if len(var_asts) < N_TESTS:
                for n in range(N_TESTS - len(var_asts)):
                    theta = rng.uniform(0, np.pi * 2, size=N_GS) # 
                    gamma = rng.uniform(0.3*nagle, nagle, size=N_GS)
                    beacons = [beacon.Beacon(t, g, magnitude=6, aperture_mask=mask) for t, g in zip(theta, gamma)]
                    var_asts.append(beacons)
            else:
                var_asts = sample(np.array(var_asts), N_TESTS)

            target = beacon.Beacon(0, 0, magnitude=6, aperture_mask=mask)

            for j, atm in enumerate(atms):

                for p in perturbations:
                    if N_TESTS > 100:
                        update_outputs = (j + 1) % 100 == 0
                    else:
                        update_outputs = j % 100 == 0

                    beacons = fixed_asts[j]
                    actual_coeffs = target.cutout_coeffs(bss, D, alts, atm, atms.metapupil_diameters)
                    target_screen = target.total_cutout(D, alts, atm, atms.metapupil_diameters)

                    # Run target fitting
                    fitting = bss.compose(ap, actual_coeffs)
                    target_screen = target_screen * util.circle(ap)

                    fitting_rmse = util.rmse(target_screen, fitting)

                    ol_recon_rmse = util.rmse(target_screen, np.zeros_like(target_screen))
                    ol_coeff_rmse = util.rmse(actual_coeffs, np.zeros_like(actual_coeffs))

                    variance['target'].append(np.var(target_screen))
                    # beacons = var_asts[j]
                    dt = time.time() - t0
                    t0 = time.time()

                    vars = tomo.get_layer_variance_for_asterism(beacons, D, alts, atm, atms.metapupil_diameters)
                    for i in range(len(alts)):
                        variance[f"l{i}"].append(vars[i])

                    if update_outputs:
                        print(f"Running test {j+1}/{N_TESTS}, angle={nagle:6}\", perturbation={p} (t: {t0-tstart:.4} [dt: {dt:.4}])")

                    perturbed_beacons = asterisms.perturb_asterism(beacons, p)
                    perturbed_coeffs = tomo.get_coeffs_for_asterism(perturbed_beacons, bss, D, atm, alts, atms.metapupil_diameters, wfs=wfs)

                    for perturbation_adjusted in [False, True]:
                    
                        test_beacons = perturbed_beacons if perturbation_adjusted else beacons

                        # Run GLAO
                        glao_coeffs = glao.run(test_beacons, target, perturbed_coeffs)
                        glao_recon = bss.compose(ap, glao_coeffs)

                        # Run Adaptive Optics
                        ao_coeffs = ao.run(test_beacons, target, perturbed_coeffs)
                        ao_recon = bss.compose(ap, ao_coeffs)
                    
                        glao_recon_rmse = util.rmse(target_screen, glao_recon)
                        glao_coeff_rmse = util.rmse(actual_coeffs, glao_coeffs)
                    
                        ao_recon_rmse = util.rmse(target_screen, ao_recon)
                        ao_coeff_rmse = util.rmse(actual_coeffs, ao_coeffs)
                    
                        ap_data.append({
                            'Aperture Size': ap,
                            'Angle': nagle,
                            'Perturbation': p,
                            'Atmosphere ID': j,
                            'Perturbation Adjusted': perturbation_adjusted,
                            'Fitting': fitting_rmse,
                            'Open Loop Recon RMSE': ol_recon_rmse,
                            'Open Loop Coeff RMSE': ol_coeff_rmse,
                            'Type': 'GLAO',
                            'Recon RMSE': glao_recon_rmse,
                            'Coeff RMSE': glao_coeff_rmse
                        })

                        ap_data.append({
                            'Aperture Size': ap,
                            'Angle': nagle,
                            'Perturbation': p,
                            'Atmosphere ID': j,
                            'Perturbation Adjusted': perturbation_adjusted,
                            'Fitting': fitting_rmse,
                            'Open Loop Recon RMSE': ol_recon_rmse,
                            'Open Loop Coeff RMSE': ol_coeff_rmse,
                            'Type': 'AO',
                            'Recon RMSE': ao_recon_rmse,
                            'Coeff RMSE': ao_coeff_rmse,
                        })

                        for test, lsq in tests.items():
                                
                            coeffs = lsq.run(test_beacons, target, perturbed_coeffs)
                            coeffs[:3] = actual_coeffs[:3]
                            if lsq.sv is not None and lsq.method != 'all':
                                sv_data.append({
                                    'Aperture Size': ap,
                                    'Angle': nagle,
                                    'Perturbation': p,
                                    'Atmosphere ID': j,
                                    'Type': test,
                                    'Perturbation Adjusted': perturbation_adjusted,
                                    'Mean': np.mean(lsq.sv),
                                    'Min': np.amin(lsq.sv),
                                    'Max': np.amax(lsq.sv),
                                    'Median': np.median(lsq.sv),
                                })

                            recon = bss.compose(ap, coeffs)                        

                            recon_rmse = util.rmse(target_screen, recon)
                            coeff_rmse = util.rmse(actual_coeffs, coeffs)

                            ap_data.append({
                                'Aperture Size': ap,
                                'Angle': nagle,
                                'Perturbation': p,
                                'Atmosphere ID': j,
                                'Perturbation Adjusted': perturbation_adjusted,
                                'Fitting': fitting_rmse,
                                'Open Loop Recon RMSE': ol_recon_rmse,
                                'Open Loop Coeff RMSE': ol_coeff_rmse,
                                'Type': test,
                                'Recon RMSE': recon_rmse,
                                'Coeff RMSE': coeff_rmse,
                            })

        var = np.mean(variance['target'])
        calculated_r0 = D / ((var / 1.03) ** (3/5))
        print(f"For ap: {ap:3}: Actual r0: {calculated_r0:.4f}m, intended r0: {r0_total}m [var: {var:.4f}]")

        for i in range(len(alts)):
            var = np.mean(variance[f'l{i}'])
            calculated_r0 = D / ((var / 1.03) ** (3/5))
            print(f"Layer {i} r0: {calculated_r0:.4f}m, intended r0: {atms.layer_r0[i]:.4f}m [var: {var:.4f}]")

        for k, v in tests.items():
            if len(v.epsilons) > 0:
                print(f"{k} Epsilon: {np.mean(v.epsilons)}")
            else:
                print(f"{k} Epsilon: Not used")

    ap_df = pd.DataFrame(ap_data)
    ap_df.to_csv('data/tests/tomography_test_data.csv')

    sv_df = pd.DataFrame(sv_data)
    sv_df.to_csv('data/tests/tomography_singular_values.csv')


def display_leastsquares_results():

    err_df = pd.read_csv('data/tests/tomography_test_data.csv')

    # err_df['Recon Delta RMSE'] = err_df['Recon RMSE'] / err_df['Open Loop Recon RMSE'] 
    # err_df['Coeff Delta RMSE'] = err_df['Coeff RMSE'] / err_df['Open Loop Coeff RMSE']

    factors = ['Angle', 'Perturbation', 'Perturbation Adjusted', 'Atmosphere ID', 'Type']
    # Compute the reference value in each group where Aperture Size == 64
    err_df['Recon Relative RMSE'] = err_df['Recon RMSE'] / (
        err_df[err_df['Aperture Size'] == 64]
        .set_index(factors)['Open Loop Recon RMSE']
        .reindex(err_df.set_index(factors).index)
        .values
    )

    err_df['Coeff Relative RMSE'] = err_df['Coeff RMSE'] / (
        err_df[err_df['Aperture Size'] == 64]
        .set_index(factors)['Open Loop Coeff RMSE']
        .reindex(err_df.set_index(factors).index)
        .values
    )

    # # columns by aperture size
    # for tomo in err_df['Type'].unique():

    #     fig, axs = plt.subplots(1, len(err_df['Aperture Size'].unique()), sharey=True)
    #     f2, axs2 = plt.subplots(1, len(err_df['Aperture Size'].unique()), sharey=True)
    #     fig.suptitle(f'{tomo}: Reconstruction Error')
    #     f2.suptitle(f'{tomo}: Coefficient Error')
    #     for i, ap_size in enumerate(err_df['Aperture Size'].unique()):
    #         df = err_df[(err_df['Aperture Size'] == ap_size) & (err_df['Perturbation'] == '(0.5, 0.5)') & (err_df['Type'] == tomo)]
    #         sns.boxplot(ax=axs[i], data=df, x='Angle', y='Recon Relative RMSE', hue='Perturbation Adjusted')
    #         sns.boxplot(ax=axs2[i], data=df, x='Angle', y='Coeff Relative RMSE', hue='Perturbation Adjusted')
    #         axs[i].set_title(f'Aperture Size: {ap_size}')
    #         axs[i].set_yscale('log')
    #         axs2[i].set_title(f'Aperture Size: {ap_size}')
    #         axs2[i].set_yscale('log')

    # columns by perturbation adjusted
    for tomo in err_df['Type'].unique():

        fig, axs = plt.subplots(1, len(err_df['Perturbation Adjusted'].unique()), sharey=True)
        f2, axs2 = plt.subplots(1, len(err_df['Perturbation Adjusted'].unique()), sharey=True)
        fig.suptitle(f'{tomo}: Reconstruction Error')
        f2.suptitle(f'{tomo}: Coefficient Error')
        for i, p_adj in enumerate(err_df['Perturbation Adjusted'].unique()):
            df = err_df[(err_df['Perturbation Adjusted'] == p_adj) & (err_df['Perturbation'] == '(0.5, 0.5)') & (err_df['Type'] == tomo)]
            sns.boxplot(ax=axs[i], data=df, x='Angle', y='Recon Relative RMSE', hue='Aperture Size')
            sns.boxplot(ax=axs2[i], data=df, x='Angle', y='Coeff Relative RMSE', hue='Aperture Size')
            axs[i].set_title(f'Perturbation Adjusted: {p_adj}')
            axs[i].set_yscale('log')
            axs2[i].set_title(f'Perturbation Adjusted: {p_adj}')
            axs2[i].set_yscale('log')


def test_beacons():
    # check positions are valid
    pos_fig = plt.figure()
    co_fig = plt.figure()

    D = 1.8
    alt = 10000
    ap = 256
    szs, pxs = tomo.get_layer_shapes([beacon.Beacon(0, 46)], [alt], D, ap)
    mask = util.circle(ap)
    screen = np.ones((pxs[0], pxs[0]))

    rot = 12.5

    def run_test(theta, gamma):
        bcn = beacon.Beacon(theta, gamma, aperture_mask=mask)
        x, y = bcn.position(alt)
        txt = f"{theta:.2f},{gamma:.2f} ({x:.2f}, {y:.2f})"

        ax = pos_fig.gca()
        ax.plot([x], [y], marker='+')
        ax.text(x, y, txt, rotation=rot)

        ax = co_fig.gca()
        phs = bcn.cutout(D, alt, screen, szs[0], full_phase=True)

        yis, xis = np.nonzero(phs)
        x, y = np.mean(xis), np.mean(yis)
        ax.plot([x], [y], marker='+')
        ax.text(x, y, txt, rotation=rot)
        return phs

    phs = np.zeros_like(screen)

    for theta in np.linspace(0, np.pi*2, 6, endpoint=False):
        for gamma in np.linspace(15, 45, 3, endpoint=True):
            phs += run_test(theta, gamma)

    phs += run_test(0, 0)
    ax = co_fig.gca()
    ax.imshow(phs, alpha=0.3, aspect="equal")

def test_layer_shapes():

    def plotcirc(axes, pos, rad, dsp, **kwargs):
        n_points = 100
        g = [util.pol2cart(rad,t/n_points*(2*np.pi)) for t in range(n_points + 1)]
        axes.plot([pos[0] + x for x,_ in g], [pos[1] + y for _,y in g], dsp, **kwargs)

    bcns = [beacon.Beacon(t, g) for g in np.linspace(0, 30, 4, endpoint=True) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]
    alts = [1000, 500, 10000]
    d = 1.8
    ap = 64

    for bcn in bcns:
        szs, pxs = tomo.get_layer_shapes([bcn], alts, d, ap)

        fig, axs = plt.subplots(3, len(alts))

        for i, (alt, sz, pxl) in enumerate(zip(alts, szs, pxs)):
            x, y = bcn.position(alt)
            
            px, py = util.cart2imagespace(pxl, sz, (x, y))
            print(px, py, pxl, x, y, sz)
            scr = util.circle(pxl, ap/2, (px-pxl/2, py-pxl/2))

            axs[0, i].imshow(scr)
            axs[0, i].plot([0, pxl], [py, py], "r:")
            axs[0, i].plot([px, px], [0, pxl], "r:")
            axs[0, i].plot([0, pxl], [pxl/2, pxl/2], "k:")
            axs[0, i].plot([pxl/2, pxl/2], [0, pxl], "k:")
            plotcirc(axs[0, i], (pxl / 2, pxl / 2), pxl/2, "k:")
            plotcirc(axs[0, i], (px, py), ap/2, "r:")

            axs[1, i].plot([-sz/2, sz/2], [y, y], "r:")
            axs[1, i].plot([x, x], [-sz/2, sz/2], "r:")
            axs[1, i].plot([-sz/2, sz/2], [0, 0], "k:")
            axs[1, i].plot([0, 0], [-sz/2, sz/2], "k:")
            plotcirc(axs[1, i], (0, 0), sz/2, "k:")
            plotcirc(axs[1, i], (x, y), d/2, "r:")
            axs[1, i].set_aspect('equal')

            y0 = y / (sz / 2)
            x0 = x / (sz / 2)

            axs[2, i].plot([-1, 1], [y0, y0], "r:")
            axs[2, i].plot([x0, x0], [-1, 1], "r:")
            axs[2, i].plot([-1, 1], [0, 0], "k:")
            axs[2, i].plot([0, 0], [-1, 1], "k:")
            plotcirc(axs[2, i], (0, 0), 1, "k:")
            lbl = "" if i != len(alts) - 1 else f"pos: {x0:.3f}, {y0:.3f}; {d/sz:.3f} [{d:.3f}/{sz:.3f}]"
            plotcirc(axs[2, i], (x0, y0), d/sz, "r:", label=lbl)
            axs[2, i].set_aspect('equal')
        fig.legend()

def test_strehl():
    dim = 256
    sz = (dim, dim)
    nothing = np.zeros(sz)
    strehl_none, psf_none = strehl.ratio(nothing, psf_output=True)

    Z = zernike.Zernike(64)

    ps = phase_screen.kolmogorov(sz, 1.8, 0.25) * util.circle(dim)
    coeffs = Z.decompose(ps)
    recon_screen = Z.compose(sz, coeffs)

    strehl_phase, phase_psf = strehl.ratio(ps, psf_output=True)
    strehl_recon, recon_psf = strehl.ratio(recon_screen, psf_output=True)
    strehl_diff, diff_psf = strehl.ratio(ps-recon_screen, psf_output=True)
    print(strehl_phase)
    print(strehl_recon)
    print(strehl_diff)
    print(strehl.ratio_quick(ps-recon_screen))

    eps = 1e-12
    vm = -7
    # eps = 0.0
    fig, axs = plt.subplots(2, 4)
    axs[0, 0].imshow(nothing, cmap='inferno')
    axs[1, 0].imshow(np.log10(psf_none+eps), cmap='inferno', vmin=vm, vmax=0)
    axs[0, 1].imshow(ps, cmap='inferno')
    axs[1, 1].imshow(np.log10(phase_psf+eps), cmap='inferno', vmin=vm, vmax=0)
    axs[0, 2].imshow(recon_screen, cmap='inferno')
    axs[1, 2].imshow(np.log10(recon_psf+eps), cmap='inferno', vmin=vm, vmax=0)
    axs[0, 3].imshow(ps-recon_screen, cmap='inferno')
    axs[1, 3].imshow(np.log10(diff_psf+eps), cmap='inferno', vmin=vm, vmax=0)
