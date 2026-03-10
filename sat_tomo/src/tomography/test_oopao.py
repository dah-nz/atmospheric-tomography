import numpy as np
import matplotlib.pyplot as plt

from tomography import tomography, oopao, beacon, asterisms, phase_screen
import util
from basis import zernike
import pandas as pd
import seaborn as sns

from tqdm import tqdm

from time import perf_counter

def run_tests():
    compare_oopao_and_local_display()
    # compare_oopao_and_local_lots()
    # compare_phase_screens()
    return

def compare_phase_screens():
    pix = 100
    D = 1.8
    alts = [1000, 5000, 10000]
    r0 = 0.15
    N = 100

    target = beacon.Beacon(theta=0, gamma=0)
    target_oopao = oopao.Source("V", 10, coordinates=[0,0])

    tele = oopao.Telescope(pix, D, fov=240)
    oopao_atm = oopao.Atmosphere(tele, r0=r0, L0=1000, fractionalR0=[0.33, 0.33, 0.33], altitude=alts, windDirection=[0,0,0], windSpeed=[0,0,0], src=target_oopao)
    oopao_atm.initializeAtmosphere(tele, compute_covariance=False)

    my_atm = phase_screen.Atmosphere(N, alts, r0=r0, D=D, aperture_size=pix)
    opd_atm = phase_screen.Atmosphere(1, alts, r0=r0, D=D, aperture_size=pix)

    my_max = []
    my_min = []
    opd_max = []
    opd_min = []
    oopao_max = []
    oopao_min = []

    for n, my_screen in enumerate(my_atm):
        oopao_atm.generateNewPhaseScreen(seed=n)
        oopao_atm.relay(target_oopao)
        # oopao_atm.update()
        oopao_screen = 2*np.pi * (oopao_atm.OPD / 500e-9)
        oopao_screen -= np.mean(oopao_screen)

        atm = [getattr(oopao_atm, f'layer_{i+1}') for i in range(len(alts))]
        atm_D = [s.D_fov for s in atm]
        # screen = [s.OPD / (2 * np.pi) for s in atm]
        screen = [s.OPD for s in atm]
        opd_screen = [s - np.mean(s) for s in screen]

        opd_atm.screens = [opd_screen]

        scr = target.total_cutout(D, alts, my_screen, my_atm.metapupil_diameters)
        scr1 = target.total_cutout(D, alts, opd_atm[0], atm_D)

        if n % 30 == 0:
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(scr)
            axs[1].imshow(scr1)
            axs[2].imshow(oopao_screen)

        print(f"{n:3}: {np.amax(scr):.3f}, {np.amax(scr1):.3f}, {np.amax(oopao_screen):.3f} | {np.amin(scr):.3f}, {np.amin(scr1):.3f}, {np.amin(oopao_screen):.3f}")
        oopao_max.append(np.amax(oopao_screen))
        my_max.append(np.amax(scr))
        opd_max.append(np.amax(scr1))
        oopao_min.append(np.amin(oopao_screen))
        my_min.append(np.amin(scr))
        opd_min.append(np.amin(scr1))


    print(f"My Max:    {np.mean(my_max)}")
    print(f"OOPAO Max: {np.mean(oopao_max)}")
    print(f"OPD Max: {np.mean(opd_max)}")
    print(f"My Min:    {np.mean(my_min)}")
    print(f"OOPAO Min: {np.mean(oopao_min)}")
    print(f"OPD Min: {np.mean(opd_min)}")
    
    plt.show()


def compare_oopao_and_local_display():

    ap = 160
    n_sub_aps = 10
    D = 1.8
    N = 36
    Z = zernike.Zernike(N)
    alts = [1000, 5000, 10000]
    r0 = 0.15

    seed = np.random.random_integers(1, 65536)

    opd_atm = phase_screen.Atmosphere(1, alts, r0=r0, D=D, aperture_size=ap)

    
    LSQ = tomography.LeastSquares(Z, D, N, altitudes=alts, r0=r0, mode_skip=3)

    mask = util.circle(ap)
    asterism = list(asterisms.sample_asterisms(123, mask, seed=seed)[0])
    # asterism = [beacon.Beacon(t, 15, 0, aperture_mask=mask) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]

    target = beacon.Beacon(0, 0, 14, aperture_mask=mask)

    oopao_tomo = oopao.OOPAO_Tomo(asterism, r0, N, alts, D=D, res=ap, fov=240, seed=seed, n_subaps=n_sub_aps)

    atm = [getattr(oopao_tomo.atm, f'layer_{i+1}') for i in range(len(alts))]
    atm_D = [s.D_fov for s in atm]
    # screen = [s.OPD / (2 * np.pi) for s in atm]
    screen = [s.OPD * ((1/3)**0.5) for s in atm]
    screen = [s - np.mean(s) for s in screen]

    opd_atm.screens = [[s] for s in screen]

    # print(atm.metapupil_diameters)
    # plt.show()
    # quit()

    # screen = atm[0]
    target_screen = target.total_cutout(D, alts, screen, atm_D)
    target_scr1 = target.total_cutout(D, alts, opd_atm[0], atm_D)
    ptt = target.cutout_coeffs(Z, D, alts, screen, atm_D)[:3]
    ptt1 = target.cutout_coeffs(Z, D, alts, opd_atm[0], atm_D)[:3]
    coeffs = tomography.get_coeffs_for_asterism(asterism, Z, D, screen, alts, atm_D)
    coeffs1 = tomography.get_coeffs_for_asterism(asterism, Z, D, opd_atm[0], alts, atm_D)

    # print([np.mean(s) for s in screen])
    # print(np.mean(target_screen))

    lsq_coeffs = LSQ.run(asterism, target, coeffs, ptt)
    lsq_coeffs1 = LSQ.run(asterism, target, coeffs1, ptt1)

    oopao_base, oopao_resid = oopao_tomo.run()
    oopao_base = oopao_base - np.mean(oopao_base, where=np.array(mask, dtype=bool))
    oopao_resid = oopao_resid - np.mean(oopao_resid, where=np.array(mask, dtype=bool))

    lsq_rec = Z.compose(ap, lsq_coeffs)
    lsq_resid = lsq_rec - target_screen
    oopao_rec = oopao_base - oopao_resid
    oopao_rec = oopao_rec - np.mean(oopao_rec, where=np.array(mask, dtype=bool))
    oopao_coeffs = Z.decompose(oopao_rec)
    oopao_coeffs[:3] = ptt
    oopao_Z_rec = Z.compose(ap, oopao_coeffs)
    oopao_Z_resid = oopao_Z_rec - target_screen

    lsq1_rec = Z.compose(ap, lsq_coeffs1)
    lsq1_resid = lsq1_rec - target_screen

    ideal_coeffs = Z.decompose(target_screen)
    ideal_screen = Z.compose(ap, ideal_coeffs)

    wfe_oopao = np.round(np.std(oopao_resid, where=np.array(mask, dtype=bool)), 4)
    wfe_lsq = np.round(np.std(lsq_resid, where=np.array(mask, dtype=bool)), 4)
    wfe_lsq1 = np.round(np.std(lsq1_resid, where=np.array(mask, dtype=bool)), 4)
    wfe_Z_oopao = np.round(np.std(oopao_Z_resid, where=np.array(mask, dtype=bool)), 4)


    fig, axs = plt.subplots(4, 4)
    fig.suptitle(f"Figure {seed}")
    axs[0, 0].imshow(target_screen * mask)
    axs[0, 0].set_title(f"Target")
    axs[0, 1].imshow(lsq_rec * mask)
    axs[0, 1].set_title(f"LSQ Recon")
    axs[0, 2].imshow(lsq_resid * mask)
    axs[0, 2].set_title(f"LSQ Resid (WFE: {wfe_lsq:.3f})")

    axs[1, 0].imshow(target_scr1 * mask)
    axs[1, 0].set_title(f"Target 1")
    axs[1, 1].imshow(lsq1_rec * mask)
    axs[1, 1].set_title(f"LSQ1 Recon")
    axs[1, 2].imshow(lsq1_resid * mask)
    axs[1, 2].set_title(f"LSQ1 Resid (WFE: {wfe_lsq1:.3f})")

    axs[2, 0].imshow(ideal_screen * mask)
    axs[2, 0].set_title(f"Ideal Recon")
    axs[2, 1].imshow(oopao_rec * mask)
    axs[2, 1].set_title(f"OOPAO Recon")
    axs[2, 2].imshow(oopao_resid * mask)
    axs[2, 2].set_title(f"OOPAO Resid (WFE: {wfe_oopao:.3f})")

    axs[3, 0].imshow(oopao_base * mask)
    axs[3, 0].set_title(f"OOPAO Base")
    axs[3, 1].imshow(oopao_Z_rec * mask)
    axs[3, 1].set_title(f"OOPAO Zernike Recon")
    axs[3, 2].imshow(oopao_Z_resid * mask)
    axs[3, 2].set_title(f"OOPAO Zernike resid (WFE: {wfe_Z_oopao:.3f})")

    axs[0, 3].imshow((target_screen - ideal_screen) * mask)
    axs[0, 3].set_title(f"Target - Ideal (diff)")
    axs[1, 3].imshow((target_screen - target_scr1) * mask)
    axs[1, 3].set_title(f"Target - Target1 (diff)")
    axs[2, 3].imshow((target_screen - oopao_base) * mask)
    axs[2, 3].set_title(f"Target - OOPAO Base (diff)")
    axs[3, 3].imshow((oopao_Z_resid - lsq_resid) * mask)
    axs[3, 3].set_title(f"OOPAO Zernike resid - LSQ resid (diff)")

    plt.figure()
    plt.plot(lsq_coeffs, 'r', label='LSQ')
    plt.plot(lsq_coeffs1, 'g', label='LSQ1')
    plt.plot(oopao_coeffs, 'b', label='OOPAO')
    plt.plot(ideal_coeffs, 'k', label='Ideal')
    plt.legend()

    plt.show()

def compare_oopao_and_local_lots():

    lots = 10
    ap = 256
    D = 1.8
    N = 32
    Z = zernike.Zernike(N)

    alts = [1000, 5000, 10000]
    r0 = 0.15
    LSQ = tomography.LeastSquares(Z, D, N, altitudes=alts, r0=r0, mode_skip=3)
    mask = util.circle(ap)

    data = {
        "Seed": [],
        "Type": [],
        "Recon MSE": [],
        "Coeff MSE": [],
        "WFE": [],
        "Time": [],
    }

    screen_diff = []
    asterism_list = asterisms.generate_asterisms(lots, mask=mask)

    for seed, asterism in tqdm(enumerate(asterism_list)):

        asterism = list(asterism)
        target = beacon.Beacon(0, 0, 14, aperture_mask=mask)

        oopao_tomo = oopao.OOPAO_Tomo(asterism, r0, N, alts, D=D, res=ap, fov=max([b._gamma_arcsec*2 for b in asterism]), seed=seed)

        atm = [getattr(oopao_tomo.atm, f'layer_{i+1}') for i in range(len(alts))]
        atm_D = [s.D_fov for s in atm]
        # screen = [s.OPD / (2*np.pi) for s in atm]
        screen = [s.OPD for s in atm]
        screen = [s - np.mean(s) for s in screen]

        target_screen = target.total_cutout(D, alts, screen, atm_D)
        ptt = target.cutout_coeffs(Z, D, alts, screen, atm_D)[:3]
        coeffs = tomography.get_coeffs_for_asterism(asterism, Z, D, screen, alts, atm_D)
        
        start = perf_counter()
        lsq_coeffs = LSQ.run(asterism, target, coeffs, ptt)
        lsq_done = perf_counter()
        oopao_base, oopao_resid = oopao_tomo.run()
        oopao_done = perf_counter()
        print(f"Mean Base: {np.mean(oopao_base, where=np.array(mask, dtype=bool))}, Mean Resid: {np.mean(oopao_resid, where=np.array(mask, dtype=bool))}")
        oopao_base = oopao_base - np.mean(oopao_base, where=np.array(mask, dtype=bool))
        oopao_resid = oopao_resid - np.mean(oopao_resid, where=np.array(mask, dtype=bool))

        screen_diff.append(np.max(oopao_base) / np.max(target_screen))

        lsq_rec = Z.compose(ap, lsq_coeffs)
        lsq_resid = lsq_rec - target_screen
        oopao_rec = oopao_base - oopao_resid
        oopao_rec = oopao_rec - np.mean(oopao_rec, where=np.array(mask, dtype=bool))
        oopao_coeffs = Z.decompose(oopao_rec)
        oopao_Z_rec = Z.compose(ap, oopao_coeffs)
        oopao_Z_resid = oopao_Z_rec - target_screen

        ideal_coeffs = Z.decompose(target_screen)
        oopao_ideal_coeffs = Z.decompose(oopao_base)
        ideal_screen = Z.compose(ap, ideal_coeffs)

        wfe_oopao = np.round(np.std(oopao_resid, where=np.array(mask, dtype=bool)), 4)
        wfe_lsq = np.round(np.std(lsq_resid, where=np.array(mask, dtype=bool)), 4)
        wfe_Z_oopao = np.round(np.std(oopao_Z_resid, where=np.array(mask, dtype=bool)), 4)

        data["Type"].append("LSQ")
        data["Seed"].append(seed)
        data["Time"].append(lsq_done - start)
        data["Recon MSE"].append(util.mse(target_screen, lsq_rec))
        data["Coeff MSE"].append(util.mse(ideal_coeffs, lsq_coeffs))
        data["WFE"].append(wfe_lsq)

        data["Type"].append("OOPAO Base Compare")
        data["Seed"].append(seed)
        data["Recon MSE"].append(util.mse(oopao_base, oopao_rec))
        data["Coeff MSE"].append(util.mse(oopao_ideal_coeffs, oopao_coeffs))
        data["Time"].append(oopao_done - lsq_done)
        data["WFE"].append(wfe_oopao)

        data["Type"].append("OOPAO Target Compare")
        data["Seed"].append(seed)
        data["Recon MSE"].append(util.mse(target_screen, oopao_Z_rec))
        data["Coeff MSE"].append(util.mse(ideal_coeffs, oopao_coeffs))
        data["WFE"].append(wfe_Z_oopao)
        data["Time"].append(None)

    df = pd.DataFrame(data)
    df.to_csv('temp_data.csv')

    sns.displot(data=df,
                x='Recon MSE',
                col='Type',
                facet_kws={'sharex': False},
                common_bins=False)
    
    sns.displot(data=df,
                x='Coeff MSE',
                col='Type',
                facet_kws={'sharex': False},
                common_bins=False)
    
    sns.displot(data=df,
                x='WFE',
                col='Type',
                facet_kws={'sharex': False},
                common_bins=False)
    
    sns.displot(data=df,
                x='Time',
                col='Type',
                facet_kws={'sharex': False},
                common_bins=False)
    
    plt.figure()
    plt.plot(screen_diff)
    print(np.mean(screen_diff))

    plt.show()