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

    plt.show()

def test_leastsquares():
    rng = np.random.default_rng(seed=1)

    alts = [1000, 10000] #[1000, 5000, 10000]
    r0_total = 0.07
    r0 = [r0_total * (len(alts) ** (1/2)) for _ in alts] #[0.05, 0.05, 0.05]
    # r0 = [r0_total for _ in alts]

    D = 1.8
    N = 16
    N_TESTS = 256 # N ** len(alts)

    ap = 64
    nagle = 10 # asterism angle

    bss = zernike.Zernike(N)
    data = {
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

    atms = phase_screen.Atmosphere(N_TESTS, alts, r0, D, ap, rng=rng)

    # atms = phase_screen.Atmosphere(1, alts, r0, D, ap, rng=rng)
    # @lru_cache(maxsize=256)
    # def get_single(sz, i):
    #     A = np.zeros(N)
    #     A[i] = 21.0
    #     return np.array(bss.compose(sz, A))
    # atms.screens = [[get_single(sz, (i // (N ** j)) % N) for i in range(N_TESTS)] for j, sz in enumerate(atms.layer_shapes)]
    # atms.n = N_TESTS

    errors = {
        "tgt:fit": [],
        "tgt:glao": [],
        "tgt:lsp": [],
        "tgt:lsg": [],
    }

    variance = []

    mask = util.circle(ap)

    for j, atm in enumerate(atms):
        if N_TESTS > 100:
            update_outputs = (j + 1) % 100 == 0
        else:
            update_outputs = j % 100 == 0

        if update_outputs:
            print(f"Running test {j+1}/{N_TESTS}")

        target = beacon.Beacon(0, 0, aperture_mask=mask)

        theta = np.linspace(0, 2*np.pi, 3, endpoint=False) # rng.uniform(0, np.pi * 2, size=3) # 
        # theta = rng.uniform(0, np.pi * 2, size=3) # 

        gamma = [nagle, nagle, nagle] # rng.uniform(0, 15, size=3)
        # gamma = rng.uniform(1, nagle, size=3)

        beacons = [beacon.Beacon(theta[i], gamma[i], aperture_mask=mask) for i in range(3)]

        actual_coeffs = target.cutout_coeffs(bss, D, alts, atm, atms.metapupil_diameters)
        target_screen = target.total_cutout(D, alts, atm, atms.metapupil_diameters)
        variance.append(np.var(target_screen))
        asterism_coeffs = tomography.get_coeffs_for_asterism(beacons, bss, D, atm, alts, atms.metapupil_diameters)

        glao_coeffs = glao.run(beacons, target, asterism_coeffs)
        glao_recon = bss.compose(ap, glao_coeffs)

        lsp_coeffs, A_lsp = lsq.run(beacons, target, asterism_coeffs, display=update_outputs)
        lsg_coeffs, A_lsg = lsq.run(beacons, target, asterism_coeffs, proj_grid=True, display=update_outputs)

        lsp_recon = bss.compose(ap, lsp_coeffs)
        lsg_recon = bss.compose(ap, lsg_coeffs)

        fitting = bss.compose(ap, actual_coeffs)
        target_screen = target_screen * util.circle(ap)

        if update_outputs:
            plt.figure()
            plt.title(f"Output Coefficients. Test {j+1}")
            # plt.plot(lsp_coeffs, label="LSQ (Pix)")
            plt.plot(lsg_coeffs, label="LSQ (Grid)")
            plt.plot(glao_coeffs, label="GLAO")
            plt.plot(actual_coeffs, label="Actual")
            [plt.plot(ast, label=f"GS{i+1}") for i, ast in enumerate(asterism_coeffs)]
            plt.legend()

            fig, axs = plt.subplots(1, 3)
            norm = colors.Normalize(vmin=min(np.amin(A_lsp), np.amin(A_lsg)), vmax=max(np.amax(A_lsp), np.amax(A_lsg)))
            axs[0].set_title("$A_{pixels}$")
            axs[0].imshow(A_lsp, norm=norm)
            axs[1].set_title("$A_{grid}$")
            axs[1].imshow(A_lsg, norm=norm)
            axs[2].set_title("Difference ($A_{pixels} - A_{grid}$)")
            axs[2].imshow(A_lsp - A_lsg)

        # recons = [lsg_recon, lsp_recon, glao_recon, fitting, target_screen]
        # lbls = ["LSQ (Grid)", "LSQ (Pix)", "GLAO", "Fitting", "Screen"]
        recons = [lsg_recon, glao_recon, fitting, target_screen]
        lbls = ["LSQ (Grid)", "GLAO", "Fitting", "Screen"]
        mx = np.amax(recons)
        mn = np.amin(recons)

        norm = colors.Normalize(vmin=min(mn, -mx), vmax=max(mx, -mn))

        if update_outputs:
            rs = len(recons)
            cs = len(recons) + 1
            fig, axs = plt.subplots(rs, cs)
            fig.suptitle(f"Reconstructed screen differences. Test {j+1}")
            for r in range(rs):
                for c in range(cs):
                    axs[r,c].set_axis_off()
                    if r == 0:
                        if c > 0:
                            axs[r, c].imshow(recons[c-1], norm=norm)
                            axs[r, c].set_title(lbls[c-1])
                    elif c == 0:
                        if r > 0:
                            axs[r, c].imshow(recons[r-1], norm=norm)
                            axs[r, c].set_title(lbls[r-1])
                    elif c >= r:
                        axs[r, c].imshow(recons[c-1]-recons[r-1], norm=norm)

        mse_glao = util.mse(target_screen, glao_recon)
        mse_lsg =  util.mse(target_screen, lsg_recon)
        mse_lsp =  util.mse(target_screen, lsp_recon)

        for type in ["glao", "lsp", "lsg", "diff"]:
            for i in range(6):
                for k in range(N):
                    data["type"].append(type)
                    if type == "glao":
                        data["mse"].append(mse_glao)
                    elif type == "lsg":
                        data["mse"].append(mse_lsg)
                    elif type == "lsp":
                        data["mse"].append(mse_lsp)
                    elif type == "diff":
                        data["mse"].append(mse_lsp - mse_glao)

                    if i == 3:
                        data["theta"].append(0.0)
                        data["gamma"].append(np.mean(gamma))
                        data["guidestar"].append(f"Average")
                        data["mode_coeff"].append(actual_coeffs[k])
                        data["mode"].append(k+1)
                    elif i == 4:
                        data["theta"].append(0.0)
                        data["gamma"].append(min(gamma))
                        data["guidestar"].append(f"Min")
                        data["mode_coeff"].append(actual_coeffs[k])
                        data["mode"].append(k+1)
                    elif i == 5:
                        data["theta"].append(0.0)
                        data["gamma"].append(max(gamma))
                        data["guidestar"].append(f"Max")
                        data["mode_coeff"].append(actual_coeffs[k])
                        data["mode"].append(k+1)
                    else:
                        data["theta"].append(theta[i])
                        data["gamma"].append(gamma[i])
                        data["guidestar"].append(f"GS{i+1}")
                        data["mode_coeff"].append(asterism_coeffs[i][k])
                        data["mode"].append(k+1)

        
        # print(f"tgt:fit  - {util.mse(target_screen, fitting)}")
        # print(f"tgt:glao - {mse_glao}")
        # print(f"tgt:lsq  - {mse_lsq}")
        # print(f"glao:fit - {util.mse(glao_recon, fitting)}")
        # print(f"lsq:fit  - {util.mse(lsq_recon, fitting)}")
        # print(f"glao:lsq - {util.mse(glao_recon, lsq_recon)}")

        errors["tgt:fit"].append(util.mse(target_screen, fitting))
        errors["tgt:glao"].append(mse_glao)
        errors["tgt:lsp"].append(mse_lsp)
        errors["tgt:lsg"].append(mse_lsg)

    # df = pd.DataFrame.from_dict(data)
    # df2 = df[df["guidestar"].isin(["Average", "Min", "Max"])]
    # sns.relplot(data=df2, x="gamma", y="mse", row="guidestar",
    #             col="type", facet_kws={'sharey': False})

    # df = pd.DataFrame.from_dict(tomography.proj_data)
    # sns.relplot(data=df, x="gamma", y="cond_p", col="altitude", hue="layer_pixels",
    #             facet_kws={'sharey': False})

    var = np.mean(variance)
    calculated_r0 = D / ((var / 1.03) ** (3/5))
    print(f"Actual r0: {calculated_r0:.4f}m (Intended: {r0_total}m) [var: {var}]")
    
    for k, l in errors.items():
        print(f"Average {k}: {np.mean(l)}")
    
    plt.figure()
    plt.hist(errors["tgt:fit"], density=True, histtype="step", label="Target:Fitting")
    plt.hist(errors["tgt:glao"], density=True, histtype="step", label="Target:GLAO")
    plt.hist(errors["tgt:lsp"], density=True, histtype="step", label="Target:LSQ (Pix)")
    plt.hist(errors["tgt:lsg"], density=True, histtype="step", label="Target:LSQ (Grid)")
    plt.ylabel("Density")
    plt.xlabel("MSE (rads²)")
    plt.title(f"Results over {N_TESTS}. Baseline: A = LW†")
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

