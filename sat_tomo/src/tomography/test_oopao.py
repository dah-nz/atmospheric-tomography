import numpy as np
import matplotlib.pyplot as plt

from tomography import tomography, oopao, beacon, asterisms
import util
from basis import zernike
import pandas as pd
import seaborn as sns

from tqdm import tqdm

from time import perf_counter

def run_tests():
    # compare_oopao_and_local_display()
    compare_oopao_and_local_lots()
    return

def compare_oopao_and_local_display():

    ap = 256
    D = 1.8
    N = 32
    Z = zernike.Zernike(N)

    seed = np.random.random_integers()


    alts = [1000, 5000, 10000]
    r0 = 0.15
    
    LSQ = tomography.LeastSquares(Z, D, N, altitudes=alts, r0=r0, mode_skip=3)

    mask = util.circle(ap)
    asterism = list(asterisms.sample_asterisms(123, mask, seed=seed)[0])
    # asterism = [beacon.Beacon(t, 15, 0, aperture_mask=mask) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]

    target = beacon.Beacon(0, 0, 14, aperture_mask=mask)

    oopao_tomo = oopao.OOPAO_Tomo(asterism, r0, N, alts, D=D, res=ap, fov=max([b._gamma_arcsec*2 for b in asterism]), seed=seed)

    atm = [getattr(oopao_tomo.atm, f'layer_{i+1}') for i in range(len(alts))]
    atm_D = [s.D_fov for s in atm]
    screen = [s.OPD / (2 * np.pi) for s in atm]

    # print(atm.metapupil_diameters)
    # plt.show()
    # quit()

    # screen = atm[0]
    target_screen = target.total_cutout(D, alts, screen, atm_D)
    ptt = target.cutout_coeffs(Z, D, alts, screen, atm_D)[:3]
    coeffs = tomography.get_coeffs_for_asterism(asterism, Z, D, screen, alts, atm_D)
    print([np.mean(s) for s in screen])
    print(np.mean(target_screen))

    lsq_coeffs = LSQ.run(asterism, target, coeffs, ptt)

    oopao_base, oopao_resid = oopao_tomo.run()

    lsq_rec = Z.compose(ap, lsq_coeffs)
    lsq_resid = lsq_rec - target_screen
    oopao_rec = oopao_base - oopao_resid
    oopao_coeffs = Z.decompose(oopao_rec)
    oopao_Z_rec = Z.compose(ap, oopao_coeffs)
    oopao_Z_resid = oopao_Z_rec - target_screen

    ideal_coeffs = Z.decompose(target_screen)
    ideal_screen = Z.compose(ap, ideal_coeffs)

    wfe_oopao = np.round(np.std(oopao_resid, where=np.array(mask, dtype=bool)), 4)
    wfe_lsq = np.round(np.std(lsq_resid, where=np.array(mask, dtype=bool)), 4)
    wfe_Z_oopao = np.round(np.std(oopao_Z_resid, where=np.array(mask, dtype=bool)), 4)


    fig, axs = plt.subplots(3, 3)
    fig.suptitle(f"Figure {seed}")
    axs[0, 0].imshow(target_screen * mask)
    axs[0, 1].imshow(lsq_rec * mask)
    axs[0, 2].imshow(lsq_resid * mask)
    axs[0, 2].set_title(f"LSQ WFE: {wfe_lsq}")

    axs[1, 0].imshow(ideal_screen * mask)
    axs[1, 1].imshow(oopao_rec * mask)
    axs[1, 2].imshow(oopao_resid * mask)
    axs[1, 2].set_title(f"OOPAO WFE: {wfe_oopao}")

    axs[2, 0].imshow(oopao_base * mask)
    axs[2, 1].imshow(oopao_Z_rec * mask)
    axs[2, 2].imshow(oopao_Z_resid * mask)
    axs[2, 2].set_title(f"OOPAO (Zernike) WFE: {wfe_Z_oopao}")

    plt.figure()
    plt.plot(lsq_coeffs, 'r', label='LSQ')
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
        screen = [s.OPD for s in atm]

        target_screen = target.total_cutout(D, alts, screen, atm_D)
        ptt = target.cutout_coeffs(Z, D, alts, screen, atm_D)[:3]
        coeffs = tomography.get_coeffs_for_asterism(asterism, Z, D, screen, alts, atm_D)
        
        start = perf_counter()
        lsq_coeffs = LSQ.run(asterism, target, coeffs, ptt)
        lsq_done = perf_counter()
        oopao_base, oopao_resid = oopao_tomo.run()
        oopao_done = perf_counter()

        screen_diff.append(np.max(oopao_base) / np.max(target_screen))

        lsq_rec = Z.compose(ap, lsq_coeffs)
        lsq_resid = lsq_rec - target_screen
        oopao_rec = oopao_base + oopao_resid
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

        data["Type"].append("OOPAO")
        data["Seed"].append(seed)
        data["Recon MSE"].append(util.mse(oopao_base, oopao_rec))
        data["Coeff MSE"].append(util.mse(oopao_ideal_coeffs, oopao_coeffs))
        data["Time"].append(oopao_done - lsq_done)
        data["WFE"].append(wfe_oopao)

        data["Type"].append("Baseline Compare")
        data["Seed"].append(seed)
        data["Coeff MSE"].append(util.mse(ideal_coeffs, oopao_ideal_coeffs))
        data["Recon MSE"].append(util.mse(ideal_screen, wfe_Z_oopao))
        data["WFE"].append(util.mse(target_screen, oopao_base))
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