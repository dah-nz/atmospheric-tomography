import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from functools import lru_cache

from tomography import tomography, beacon, phase_screen, projection
from basis import zernike

import util

def run_tests():
    # test_layer_shapes()

    # test_beacons()
    test_leastsquares()
    # projection.test_projection()

    # C = zernike.covar(21)
    # D = 1.8
    # r0 = 0.07
    # C = C * (D / r0) ** (5/3)
    # plt.figure()
    # plt.imshow(C)

    plt.show()

def test_leastsquares():
    # rng = np.random.default_rng(seed=1)
    rng = np.random.default_rng()

    alts = [1000, 5000, 10000] #[1000, 5000, 10000]
    r0_total = 0.07

    D = 1.8
    N = 16
    N_TESTS = 512
    N_GS = 3

    ap = 64
    # angles = np.arange(5, 121, 5)
    angles = [15]
    nagle = 30 # asterism angle

    bss = zernike.Zernike(N)
    data = {
        "test_id": [],
        "mse": [],
        "type": [],
        "theta": [],
        "gamma": [],
        "guidestar": [],
        "mode": [],
        "mode_coeff": []
    }

    glao = tomography.GLAO(bss, D, N, alts, ap)

    print(f"Building atmospheres for {N_TESTS} tests")
    atms = phase_screen.Atmosphere(N_TESTS, alts, r0_total, D, ap, rng=rng)
    
    tests = {
        "LSQ [TA⁺]":         tomography.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="Ainv",  proj_grid=True),
        # "LSQ [(AT⁺)⁺]":      tomography.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="Tinv",  proj_grid=True),
        # "LSQ [(AᵀA+εK)⁺Aᵀ]": tomography.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="K",     proj_grid=True),
        # "LSQ [f(A,T)]":      tomography.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="dotAT", proj_grid=True),
        # "LSQ [f(A,K)]":      tomography.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="dotAK", proj_grid=True),
        # "LSQ [f(T,K)]":      tomography.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="dotTK", proj_grid=True),
        # "LSQ [all, mean]":   tomography.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=True, start_angle=30, end_angle=45, angle_mix="ave"),
        # "LSQ [all, max]":    tomography.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=True, start_angle=30, end_angle=45, angle_mix="max"),
        # "LSQ [all, min]":    tomography.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=True, start_angle=30, end_angle=45, angle_mix="min"),
        # "LSQ [all, median]": tomography.LeastSquares(bss, D, N, alts, ap, r0=r0_total, method="all",   proj_grid=True, start_angle=30, end_angle=45, angle_mix="med"),
    }

    tgt_fit = "Target : Fitting"
    tgt_glao = "Target : GLAO"
    tgt_ol = "Target : Open Loop"
    tgt_ = "Target : "

    test0 = "LSQ [TA⁺]"

    errors = {}
    for nagle in angles:
        errors[nagle] = {
            tgt_fit: [],
            tgt_glao: [],
            tgt_ol: [],
        }
        for k in tests.keys():
            errors[nagle][tgt_ + k] = []

    variance = {
        f"l{i}": [] for i in range(len(alts))
    }
    variance["target"] = []


    mask = util.circle(ap)

    for nagle in angles:
        for j, atm in enumerate(atms):
            if N_TESTS > 100:
                update_outputs = (j + 1) % 100 == 0
            else:
                update_outputs = j % 100 == 0


            target = beacon.Beacon(0, 0, aperture_mask=mask)

            # theta_o = rng.uniform(0, 2*np.pi / 3)
            # theta = np.linspace(theta_o, theta_o + 2*np.pi, N_GS, endpoint=False)
            theta = rng.uniform(0, np.pi * 2, size=N_GS) # 

            # gamma = [nagle] * N_GS
            gamma = rng.uniform(1, nagle, size=N_GS)

            beacons = [beacon.Beacon(t, g, aperture_mask=mask) for t, g in zip(theta, gamma)]

            actual_coeffs = target.cutout_coeffs(bss, D, alts, atm, atms.metapupil_diameters)
            target_screen = target.total_cutout(D, alts, atm, atms.metapupil_diameters)
            variance['target'].append(np.var(target_screen))
            asterism_coeffs = tomography.get_coeffs_for_asterism(beacons, bss, D, atm, alts, atms.metapupil_diameters)

            vars = tomography.get_layer_variance_for_asterism(beacons, D, alts, atm, atms.metapupil_diameters)
            for i in range(len(alts)):
                variance[f"l{i}"].append(vars[i])

            glao_coeffs = glao.run(beacons, target, asterism_coeffs)
            glao_coeffs[:3] = actual_coeffs[:3]
            glao_recon = bss.compose(ap, glao_coeffs)
            
            fitting = bss.compose(ap, actual_coeffs)
            target_screen = target_screen * util.circle(ap)

            mse_glao = util.mse(target_screen, glao_recon)

            errors[nagle][tgt_fit].append(util.mse(target_screen, fitting))
            errors[nagle][tgt_glao].append(mse_glao)
            errors[nagle][tgt_ol].append(util.mse(target_screen, np.zeros_like(target_screen)))

            for test, lsq in tests.items():
                # print(f"{test}: {tests[test]}")
                coeffs = lsq.run(beacons, target, asterism_coeffs)
                coeffs[:3] = actual_coeffs[:3]
                recon = bss.compose(ap, coeffs)
                mse =  util.mse(target_screen, recon)
                errors[nagle][tgt_ + test].append(mse)

            if update_outputs:
                print(f"Running test {j+1}/{N_TESTS}, angle={nagle}\", $f(V_A^T V_T^T)$={tests[test0].alignment0:.3f}°, $f(V_A V_T)$={tests[test0].alignment1:.3f}°, $f(V_A^T V_T^T)$={tests[test0].alignment2:.3f}°, $f(V_A^T V_T)$={tests[test0].alignment3:.3f}°")

    var = np.mean(variance['target'])
    calculated_r0 = D / ((var / 1.03) ** (3/5))
    print(f"Actual r0: {calculated_r0:.4f}m, intended r0: {r0_total}m [var: {var:.4f}]")

    for i in range(len(alts)):
        var = np.mean(variance[f'l{i}'])
        calculated_r0 = D / ((var / 1.03) ** (3/5))
        print(f"Layer {i} r0: {calculated_r0:.4f}m, intended r0: {atms.layer_r0[i]:.4f}m [var: {var:.4f}]")

    aves = {k: [] for k in errors[angles[0]].keys()}
    meds = {k: [] for k in errors[angles[0]].keys()}

    for angle in angles:
        plt.figure()
        mn = np.amin(errors[angle][tgt_fit])
        mx = np.amax(errors[angle][tgt_glao])
        nbins = 32

        for k in errors[angle].keys():
            aves[k].append(np.mean(errors[angle][k]))
            meds[k].append(np.median(errors[angle][k]))
            # plt.hist(errors[angle][k], bins=nbins, density=True, histtype="step", label=k, range=(mn, mx))

        plt.ylabel("Density")
        plt.xlabel("MSE (rad²)")
        plt.title(f"Results over {N_TESTS}. Angle = {angle}\"")
        plt.legend()
    
    
    for k, l in errors[angle].items():
        print(f"{k}: ave: {np.mean(l):.1f}; med: {np.median(l):.1f}")

    colours = colors.TABLEAU_COLORS
    markers = ['+', 'x', '.', '*']

    style = []
    for m in markers:
        for c in colours:
            style.append((m, c))

    plt.figure()
    i = 0

    for k in aves.keys():
        if k == tgt_fit:
            plt.plot(angles, aves[k], color="blue", linestyle='dashed', label=f"{k} - Mean")
            plt.plot(angles, meds[k], color="blue", linestyle='solid', label=f"{k} - Median")    
        elif k == tgt_glao:
            plt.plot(angles, aves[k], color="red", linestyle='dashed', label=f"{k} - Mean")
            plt.plot(angles, meds[k], color="red", linestyle='solid', label=f"{k} - Median")    
        elif k == tgt_ol:
            plt.plot(angles, aves[k], color="black", linestyle='dashed', label=f"{k} - Mean")
            plt.plot(angles, meds[k], color="black", linestyle='solid', label=f"{k} - Median")    
        else:
            m, c = style[i]
            plt.plot(angles, aves[k], color=c, marker=m, linestyle='dashed', alpha=0.6, label=f"{k} - Mean")
            plt.plot(angles, meds[k], color=c, marker=m, linestyle='solid', alpha=0.6, label=f"{k} - Median")
            i += 1
    plt.xlabel("Angle (\")")
    plt.ylabel("MSE (rad²)")
    plt.legend()

    

def test_beacons():
    # check positions are valid
    pos_fig = plt.figure()
    co_fig = plt.figure()

    D = 1.8
    alt = 10000
    ap = 256
    szs, pxs = tomography.get_layer_shapes([beacon.Beacon(0, 46)], [alt], D, ap)
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
    alts = [1000, 10000]
    d = 1.8
    ap = 64

    for bcn in bcns:
        szs, pxs = tomography.get_layer_shapes([bcn], alts, d, ap)

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
