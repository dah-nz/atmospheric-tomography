import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

from tomography import tomography, beacon, phase_screen
from basis import zernike

import util

def run_tests():
    test_beacons()
    test_leastsquares()

def test_leastsquares():

    rng = np.random.default_rng()

    alts = [1000, 5000, 10000]
    r0 = [0.07, 0.07, 0.07]
    D = 1.8
    N = 16
    ap = 64
    bss = zernike.Zernike(N)

    lsq = tomography.LeastSquares(bss, D, N, alts, ap)
    glao = tomography.GLAO(bss, D, N, alts, ap)
    atms = phase_screen.Atmosphere(10, alts, r0, D, ap)

    mask = util.circle(ap)

    for j, atm in enumerate(atms):

        target = beacon.Beacon(0, 0, aperture_mask=mask)
        beacons = [beacon.Beacon(rng.uniform(0, np.pi * 2), rng.uniform(0, 45), aperture_mask=mask) for _ in range(3)]

        actual_coeffs = target.cutout_coeffs(bss, D, alts, atm, atms.metapupil_diameters)
        target_screen = target.total_cutout(D, alts, atm, atms.metapupil_diameters)
        asterism_coeffs = tomography.get_coeffs_for_asterism(beacons, bss, D, atm, alts, atms.metapupil_diameters)


        glao_coeffs = glao.run(beacons, target, asterism_coeffs)
        glao_recon = bss.compose(ap, glao_coeffs)

        lsq_coeffs = lsq.run(beacons, target, asterism_coeffs)
        lsq_recon = bss.compose(ap, lsq_coeffs)

        fitting = bss.compose(ap, actual_coeffs)
        target_screen = target_screen * util.circle(ap)

        plt.figure()
        plt.plot(lsq_coeffs, label="LSQ")
        plt.plot(glao_coeffs, label="GLAO")
        plt.plot(actual_coeffs, label="Actual")
        [plt.plot(ast, label=f"GS{i+1}") for i, ast in enumerate(asterism_coeffs)]
        plt.legend()

        recons = [lsq_recon, glao_recon, fitting, target_screen]
        mx = np.amax(recons)
        mn = np.amin(recons)

        norm = colors.Normalize(vmin=min(mn, -mx), vmax=max(mx, -mn))

        fig, axs = plt.subplots(4, 5)
        for r in range(4):
            for c in range(5):
                axs[r,c].set_axis_off()
                if r == 0:
                    if c > 0:
                        axs[r, c].imshow(recons[c-1], norm=norm)
                elif c == 0:
                    if r > 0:
                        axs[r, c].imshow(recons[r-1], norm=norm)
                elif c >= r:
                    axs[r, c].imshow(recons[c-1]-recons[r-1], norm=norm)

        print(f"{j+1}")
        print(f"tgt:fit  - {util.mse(target_screen, fitting)}")
        print(f"tgt:glao - {util.mse(target_screen, glao_recon)}")
        print(f"tgt:lsq  - {util.mse(target_screen, lsq_recon)}")
        print(f"glao:fit - {util.mse(glao_recon, fitting)}")
        print(f"lsq:fit  - {util.mse(lsq_recon, fitting)}")
        print(f"glao:lsq - {util.mse(glao_recon, lsq_recon)}")


    df = pd.DataFrame.from_dict(tomography.proj_data)
    sns.relplot(data=df, x="gamma", y="cond_p", col="altitude", hue="layer_pixels",
                facet_kws={'sharey': False})

    plt.show()

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

    plt.show()
