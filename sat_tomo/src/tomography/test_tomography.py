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
    # r0 = [r0_total * (len(alts) ** (1/2)) for _ in alts] #[0.05, 0.05, 0.05]
    # r0 = [r0_total for _ in alts]

    D = 1.8
    N = 16
    N_TESTS = 256 # N ** len(alts)
    N_GS = 3

    ap = 64
    angles = np.arange(5, 121, 5)
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

    lsq = tomography.LeastSquares(bss, D, N, alts, ap)
    # lsq.set_additional(r0, layer_weightings)
    # lsq.r0 = r0[0]
    glao = tomography.GLAO(bss, D, N, alts, ap)
    print(f"Building atmospheres for {N_TESTS} tests")

    atms = phase_screen.Atmosphere(N_TESTS, alts, r0_total, D, ap, rng=rng)

    # atms = phase_screen.Atmosphere(1, alts, r0, D, ap, rng=rng)
    # @lru_cache(maxsize=256)
    # def get_single(sz, i):
    #     A = np.zeros(N)
    #     A[i] = 21.0
    #     return np.array(bss.compose(sz, A))
    # atms.screens = [[get_single(sz, (i // (N ** j)) % N) for i in range(N_TESTS)] for j, sz in enumerate(atms.layer_shapes)]
    # atms.n = N_TESTS

    errors = {}
    for nagle in angles:
        errors[nagle] = {
            "tgt:fit": [],
            "tgt:glao": [],
            "tgt:lsp": [],
            "tgt:lsg": [],
            "tgt:ol": [],
        }

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

            if update_outputs:
                print(f"Running test {j+1}/{N_TESTS}, angle={nagle}\"")

            target = beacon.Beacon(0, 0, aperture_mask=mask)

            theta_o = rng.uniform(0, 2*np.pi / 3)
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

            # if update_outputs:
            #     lsp_coeffs, A_lsp, T_lsp = lsq.run(beacons, target, asterism_coeffs, proj_grid=False, display=update_outputs)
            #     lsg_coeffs, A_lsg, T_lsg = lsq.run(beacons, target, asterism_coeffs, display=update_outputs)
            # else:
            lsp_coeffs = lsq.run(beacons, target, asterism_coeffs, proj_grid=False)
            lsg_coeffs = lsq.run(beacons, target, asterism_coeffs)

            lsp_coeffs[:3] = actual_coeffs[:3]
            lsg_coeffs[:3] = actual_coeffs[:3]

            lsp_recon = bss.compose(ap, lsp_coeffs)
            lsg_recon = bss.compose(ap, lsg_coeffs)

            fitting = bss.compose(ap, actual_coeffs)
            target_screen = target_screen * util.circle(ap)

            # if update_outputs:
            #     plt.figure()
            #     plt.title(f"Output Coefficients. Test {j+1}")
            #     # plt.plot(lsp_coeffs, label="LSQ (Pix)")
            #     plt.plot(lsg_coeffs, label="LSQ (Grid)")
            #     plt.plot(glao_coeffs, label="GLAO")
            #     plt.plot(actual_coeffs, label="Actual")
            #     [plt.plot(ast, label=f"GS{i+1}") for i, ast in enumerate(asterism_coeffs)]
            #     plt.legend()

            #     eps = 1e-6
            #     fig, axs = plt.subplots(2, 3)
            #     norm = colors.Normalize(vmin=min(np.amin(A_lsp), np.amin(A_lsg)), vmax=max(np.amax(A_lsp), np.amax(A_lsg)))
            #     axs[0, 0].set_title("$A_{pixels}$")
            #     axs[0, 0].imshow(A_lsp, norm=norm)
            #     axs[0, 1].set_title("$A_{grid}$")
            #     axs[0, 1].imshow(A_lsg, norm=norm)
            #     axs[0, 2].set_title("Difference ($A_{pixels} - A_{grid}$)")
            #     axs[0, 2].imshow(A_lsp - A_lsg)
            #     axs[1, 0].set_title("$A_{pixels}^\\dagger$")
            #     axs[1, 0].imshow(np.linalg.pinv(A_lsp, rcond=eps))
            #     axs[1, 1].set_title("$A_{grid}^\\dagger$")
            #     axs[1, 1].imshow(np.linalg.pinv(A_lsg, rcond=eps))
            #     axs[1, 2].set_title("Difference ($A_{pixels}^\\dagger - A_{grid}^\\dagger$)")
            #     axs[1, 2].imshow(np.linalg.pinv(A_lsp, rcond=eps) - np.linalg.pinv(A_lsg, rcond=eps))

            #     fig, axs = plt.subplots(2, 3)
            #     norm = colors.Normalize(vmin=min(np.amin(T_lsp), np.amin(T_lsg)), vmax=max(np.amax(T_lsp), np.amax(T_lsg)))
            #     axs[0, 0].set_title("$T_{pixels}$")
            #     axs[0, 0].imshow(T_lsp, norm=norm)
            #     axs[0, 1].set_title("$T_{grid}$")
            #     axs[0, 1].imshow(T_lsg, norm=norm)
            #     axs[0, 2].set_title("Difference ($T_{pixels} - T_{grid}$)")
            #     axs[0, 2].imshow(T_lsp - T_lsg)
            #     axs[1, 0].set_title("$T_{pixels}^\\dagger$")
            #     axs[1, 0].imshow(np.linalg.pinv(T_lsp, rcond=eps))
            #     axs[1, 1].set_title("$T_{grid}^\\dagger$")
            #     axs[1, 1].imshow(np.linalg.pinv(T_lsg, rcond=eps))
            #     axs[1, 2].set_title("Difference ($T_{pixels}^\\dagger - T_{grid}^\\dagger$)")
            #     axs[1, 2].imshow(np.linalg.pinv(T_lsp, rcond=eps) - np.linalg.pinv(T_lsg, rcond=eps))

            # recons = [lsg_recon, lsp_recon, glao_recon, fitting, target_screen]
            # lbls = ["LSQ (Grid)", "LSQ (Pix)", "GLAO", "Fitting", "Screen"]

            # recons = [lsg_recon, glao_recon, fitting, target_screen]
            # lbls = ["LSQ (Grid)", "GLAO", "Fitting", "Screen"]
            # mx = np.amax(recons)
            # mn = np.amin(recons)

            # norm = colors.Normalize(vmin=min(mn, -mx), vmax=max(mx, -mn))

            # if update_outputs:
            #     rs = len(recons)
            #     cs = len(recons) + 1
            #     fig, axs = plt.subplots(rs, cs)
            #     fig.suptitle(f"Reconstructed screen differences. Test {j+1}")
            #     for r in range(rs):
            #         for c in range(cs):
            #             axs[r,c].set_axis_off()
            #             if r == 0:
            #                 if c > 0:
            #                     axs[r, c].imshow(recons[c-1], norm=norm)
            #                     axs[r, c].set_title(lbls[c-1])
            #             elif c == 0:
            #                 if r > 0:
            #                     axs[r, c].imshow(recons[r-1], norm=norm)
            #                     axs[r, c].set_title(lbls[r-1])
            #             elif c >= r:
            #                 axs[r, c].imshow(recons[c-1]-recons[r-1], norm=norm)

            mse_glao = util.mse(target_screen, glao_recon)
            mse_lsg =  util.mse(target_screen, lsg_recon)
            mse_lsp =  util.mse(target_screen, lsp_recon)

        #     for type in ["glao", "lsp", "lsg", "diff"]:
        #        for i in range(6):
        #             for k in range(N):
        #                 data["test_id"].append(j)
        #                 data["type"].append(type)
        #                 if type == "glao":
        #                     data["mse"].append(mse_glao)
        #                 elif type == "lsg":
        #                     data["mse"].append(mse_lsg)
        #                 elif type == "lsp":
        #                     data["mse"].append(mse_lsp)
        #                 elif type == "diff":
        #                     data["mse"].append(mse_lsp - mse_glao)

        #                 # data["theta"].append(0.0)
        #                 # data["gamma"].append(np.mean(gamma))
        #                 # data["guidestar"].append(f"Target")
        #                 # data["mode_coeff"].append(actual_coeffs[k])
        #                 # data["mode"].append(k+1)

        #                 if i == 3:
        #                     data["theta"].append(0.0)
        #                     data["gamma"].append(np.mean(gamma))
        #                     data["guidestar"].append(f"Average")
        #                     data["mode_coeff"].append(actual_coeffs[k])
        #                     data["mode"].append(k+1)
        #                 elif i == 4:
        #                     data["theta"].append(0.0)
        #                     data["gamma"].append(min(gamma))
        #                     data["guidestar"].append(f"Min")
        #                     data["mode_coeff"].append(actual_coeffs[k])
        #                     data["mode"].append(k+1)
        #                 elif i == 5:
        #                     data["theta"].append(0.0)
        #                     data["gamma"].append(max(gamma))
        #                     data["guidestar"].append(f"Max")
        #                     data["mode_coeff"].append(actual_coeffs[k])
        #                     data["mode"].append(k+1)
        #                 else:
        #                     data["theta"].append(theta[i])
        #                     data["gamma"].append(gamma[i])
        #                     data["guidestar"].append(f"GS{i+1}")
        #                     data["mode_coeff"].append(asterism_coeffs[i][k])
        #                     data["mode"].append(k+1)

            errors[nagle]["tgt:fit"].append(util.mse(target_screen, fitting))
            errors[nagle]["tgt:glao"].append(mse_glao)
            errors[nagle]["tgt:lsp"].append(mse_lsp)
            errors[nagle]["tgt:lsg"].append(mse_lsg)
            errors[nagle]["tgt:ol"].append(util.mse(target_screen, np.zeros_like(target_screen)))

    # df = pd.DataFrame.from_dict(data)
    # # df2 = df.pivot_table(index="test_id", columns="guidestar", values="gamma")
    # df2 = df.pivot_table(index="test_id", columns="mode", values="mode_coeff")
    # df3 = df.pivot_table(index="test_id", columns="type", values="mse")
    # df2["diff"] = df3["glao"] > df3["lsg"]
    # # df2 = df2.sort_values(by=['diff'])

    
    # plt.figure()
    # pd.plotting.parallel_coordinates(df2, "diff", colormap="viridis")
    var = np.mean(variance['target'])
    calculated_r0 = D / ((var / 1.03) ** (3/5))
    print(f"Actual r0: {calculated_r0:.4f}m, intended r0: {r0_total}m [var: {var:.4f}]")

    for i in range(len(alts)):
        var = np.mean(variance[f'l{i}'])
        calculated_r0 = D / ((var / 1.03) ** (3/5))
        print(f"Layer {i} r0: {calculated_r0:.4f}m, intended r0: {atms.layer_r0[i]:.4f}m [var: {var:.4f}]")

    lsq_aves = []
    lsq_meds = []
    glao_aves = []
    glao_meds = []
    ol_aves = []
    ol_meds = []
    fit_aves = []
    fit_meds = []

    for angle in angles:
        # for k, l in errors[angle].items():
        #     print(f"{k}: ave: {np.mean(l):.1f}; med: {np.median(l):.1f}")

        lsq_aves.append(np.mean(errors[angle]["tgt:lsg"]))
        lsq_meds.append(np.median(errors[angle]["tgt:lsg"]))
        glao_aves.append(np.mean(errors[angle]["tgt:glao"]))
        glao_meds.append(np.median(errors[angle]["tgt:glao"]))
        ol_aves.append(np.mean(errors[angle]["tgt:ol"]))
        ol_meds.append(np.median(errors[angle]["tgt:ol"]))
        fit_aves.append(np.mean(errors[angle]["tgt:fit"]))
        fit_meds.append(np.median(errors[angle]["tgt:fit"]))
        
        plt.figure()
        mn = np.amin(errors[angle]["tgt:fit"])
        mx = np.amax(errors[angle]["tgt:glao"])
        nbins = 32

        plt.hist(errors[angle]["tgt:fit"], bins=nbins, density=True, histtype="step", label="Target:Fitting", range=(mn, mx))
        plt.hist(errors[angle]["tgt:glao"], bins=nbins, density=True, histtype="step", label="Target:GLAO", range=(mn, mx))
        # plt.hist(errors[angle]["tgt:lsp"], bins=nbins, density=True, histtype="step", label="Target:LSQ (Pix)", range=(mn, mx))
        plt.hist(errors[angle]["tgt:lsg"], bins=nbins, density=True, histtype="step", label="Target:LSQ (Grid)", range=(mn, mx))
        plt.ylabel("Density")
        plt.xlabel("MSE (rad²)")
        plt.title(f"Results over {N_TESTS}. Angle = {angle}\"")
        plt.legend()

    plt.figure()
    plt.plot(angles, lsq_aves, color='blue', linestyle='dashed', label="LSQ Mean")
    plt.plot(angles, lsq_meds, color='blue', linestyle='solid', label="LSQ Median")
    plt.plot(angles, glao_aves, color='red', linestyle='dashed', label="GLAO Mean")
    plt.plot(angles, glao_meds, color='red', linestyle='solid', label="GLAO Median")
    plt.plot(angles, ol_aves, color='black', linestyle='dashed', label="Open Loop Mean")
    plt.plot(angles, ol_meds, color='black', linestyle='solid', label="Open Loop Median")
    plt.plot(angles, fit_aves, color='green', linestyle='dashed', label="Fit Mean")
    plt.plot(angles, fit_meds, color='green', linestyle='solid', label="Fit Median")
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

