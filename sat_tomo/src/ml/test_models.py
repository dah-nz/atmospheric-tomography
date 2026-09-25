import numpy as np
import pandas as pd
import os
import torch
# import ml.models as models
from ml import models, runner, ml
import torch
from torchinfo import summary, ModelStatistics
import cfg as config
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplot2tikz as tikzplotlib
from tomography import asterisms, phase_screen, tomography, beacon, psf, strehl

def size(info):
    return ModelStatistics.to_megabytes(
                info.total_input + info.total_output_bytes + info.total_param_bytes
            )

def run_tests():
    # model_sizes()

    # run_models_over_different_heuristics()
    # display_models_over_different_heuristics()
    # run_models_over_different_Dr0()
    # display_models_over_different_Dr0()

    run_and_display_single_example()

def model_sizes():
    data = []

    cfg = config.Config()

    for m in cfg.train_models:
        model = eval(f"models.{m}(cfg)")


        info = summary(model, input_size=model.input_size, verbose=0)
        print(info)

    #     print(f"{m}: {info.trainable_params}/{info.total_params} [{size(info):.2f} MB]")
    #     data.append((m,info.trainable_params,size(info)))
    #     if "ResNet" in m:
    #         p0 = info.trainable_params

    # for m, p, s in data:
    #     print(f"{m},{s:.2f},{p},{(p/p0-1.0)*100.0:.4f}")
        
    print("TEST")


def run_models_over_different_Dr0():
    from training import run_epoch
    

    N = 200

    cfg = config.Config()

    ml_run = ml.Trainer(cfg, skip_lsq=True)

    results = {
        "Profile": [],
        "D_r0": [],
        "Model": [],
        "Type": [],
        "Loss": [],
    }


    print(ml_run.model_file_names)
    ml_run.load_models(suffix="_Variable_Final", dir=cfg.directory + "/current")

    mask = cfg.basis.get_mask(cfg.aperture_size)

    layer_conditions = {
        "UCMJO": ([0.88, 0.03, 0.09], [1000, 5000, 10000]),
        "Ellerbroek": ([0.230, 0.296, 0.246, 0.110, 0.064, 0.050, 0.004], [4.5, 463, 1483, 4840, 11122, 14906, 18635])
    }

    for profile, (cn2_pc, alts) in layer_conditions.items():
        for D_r0 in [5., 10., 15., 20., 25., 30., 35., 40., 45., 50.]:
            total_r0 = cfg.diameter / D_r0
            var = total_r0 ** (-5/3)

            r0 = [(var * layer_cn2_pc) ** (-3/5) for layer_cn2_pc in cn2_pc]

            atm = phase_screen.Atmosphere(N, alts, r0, cfg.diameter, cfg.aperture_size)

            asts = asterisms.generate_asterisms(N, 
                                                mask=mask,
                                                quality_threshold=cfg.asterism_quality_threshold,
                                                quality_limit=cfg.minimum_asterism_quality,
                                                use_regime_model=True,
                                                model_location=cfg.asterism_model_location,
                                                quality_model=cfg.quality_model,
                                                regime_model=cfg.regime_model)

            result_data = run_epoch(ml_run, N, asts, atm, cfg.basis, train=False, wfs=None)

            for model, data_dict in result_data.items():
                for key, value in data_dict.items():
                    results["Profile"].append(profile)
                    results["D_r0"].append(D_r0)
                    results["Model"].append(model)
                    results["Type"].append(key)
                    results["Loss"].append(np.mean(value))

    df = pd.DataFrame.from_dict(results, orient="columns")

    df.to_parquet(os.path.join("D:/Data/2026/cross_comparison/constant", f"CrossComparisonForDR0_{int(cfg.diameter/cfg.total_r0)}.parquet"))
    print(df.head())

    # Build a set of different atmospheres with different D/r0 profiles

    # Pull a number of variable asterisms

    # Run the different models with the different asterism and atmospheres

def display_models_over_different_Dr0():

    # Display results from above
    import glob
    import matplotlib.lines as mlines

    search_pattern = os.path.join("D:/Data/2026/cross_comparison/constant", "CrossComparisonForDR0*.parquet")
    files = glob.glob(search_pattern)

    df = pd.DataFrame()

    for file in files:
        Dr0 = int(file.split(".")[0].split("_")[-1])
        df0 = pd.read_parquet(file)
        df0["Model_D_r0"] = Dr0

        df = pd.concat([df, df0])

    # Set embedding state
    df["Embed"] = "No"
    df.loc[df['Model'].str.contains("Embed"), "Embed"] = "Yes"
    df.loc[df['Model'].isin(['Open Loop', 'LeastSquares', 'MMSE', "Fitting Error", 'GLAO', 'AO']), "Embed"] = "NA"
    df["Model"] = df["Model"].replace(r'(?:^GuideStarEmbed|Embed)_?', '', regex=True)
    print(df['Embed'].unique())

    # Set size variable
    df["Size"] = "Standardised"
    df.loc[df['Model'].str.contains("smol"), "Size"] = "Small"
    df.loc[df['Model'].isin(['Open Loop', 'LeastSquares', 'MMSE', "Fitting Error", 'GLAO', 'AO']), "Size"] = "NA"
    print(df['Size'].unique())
    df["Model"] = df["Model"].replace(r'smol_', '', regex=True)
    df["Model"] = df["Model"].replace(r'_smol', '', regex=True)

    # Remove unwanted junk from Model names
    df["Model"] = df["Model"].replace(r'MLNN_', '', regex=True)
    df["Model"] = df["Model"].replace(r'ResNet(?!_|$)', 'ResNet_', regex=True)
    df["Model"] = df["Model"].replace(r'_Variable.*', '', regex=True)
    
    df = df[df['Model'] != 'singlemode']

    df["Model"] = df["Model"].map({
        'baseline': "Direct MLNN",
        'pseudo_lsq': "Pseudo-LS",
        'pseudo_mmse': "Pseudo-MMSE",
        'NoInverseKzz': "PQ + Priors",
        'LeastSquares': "LS"
    }).fillna(df["Model"])

    print(df.head())
    print(df["Model"].unique())

    size_excl = "Standardised"
    embed_excl = "No"

    df0 = df[(df["Size"] != size_excl) 
              & (df["Type"] == "Reconstruction")
              & (df["Embed"] != embed_excl)
              & (df["Model"] == "Direct MLNN")]
    # df0 = df0.sort_values(by=["Model"])
    df0 = df0.rename(columns= {"Model_D_r0": "Model $D/r_0$"})
    

    g = sns.relplot(data=df0, kind='line',
                    x="D_r0", y="Loss", hue="Model $D/r_0$", style="Profile")
    g.set(ylabel="Reconstruction Loss (rads$^2$)", xlabel="$D/r_0$")
    g._legend.remove()

    group_cols = [c for c in df.columns if c not in ["Model_D_r0", "Profile", "Loss"]]
    dfm = df.groupby(group_cols, as_index=False)['Loss'].mean()
    dfm = dfm[(dfm["Size"] != size_excl)
              & (dfm["Type"] == "Reconstruction")
              & (dfm["Embed"] != embed_excl)
              & (dfm["Model"].isin(["GLAO", "Fitting Error"]))]

    r0_list = dfm[dfm["Model"] == "GLAO"]["D_r0"].tolist()
    glao_list = dfm[dfm["Model"] == "GLAO"]["Loss"].tolist()

    r0_fe_list = dfm[dfm["Model"] == "Fitting Error"]["D_r0"].tolist()
    fe_list = dfm[dfm["Model"] == "Fitting Error"]["Loss"].tolist()

    for ax in g.axes.flat:
        ax.plot(r0_fe_list, fe_list, label="Fitting Error")
        ax.plot(r0_list, glao_list, label="GLAO")

        handles, labels = ax.get_legend_handles_labels()
        blank_handle = mlines.Line2D([], [], color='none')

        handles.insert(-2, blank_handle)
        labels.insert(-2, '')   # use ' ' (a space) if '' gets collapsed/stripped

        ax.legend(handles, labels)
    pass

def run_and_display_single_example():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 7 #7

    cfg = config.Config()

    mask = cfg.basis.get_mask(cfg.aperture_size)

    target = beacon.Beacon(0,0)

    # r0 = [0.194032125, 1.49358721, 0.772677337]
    r0 = [0.064677, 0.497862, 0.257559]

    heuristic = cfg.directory.split('/')[-1]

    model = models.GuideStarEmbed_pseudo_lsq(cfg).to(device)
    filepath = os.path.join(cfg.directory + "/current/parameters", "GuideStarEmbed_pseudo_lsq_Zernike_36_0.06_1.8__Variable_Final.mdl")
    model.load_state_dict(torch.load(filepath, weights_only=True))

    MMSE = tomography.MMSE(cfg.basis, cfg.diameter, cfg.num_modes, cfg.altitudes, cfg.aperture_size, r0=cfg.total_r0)
    LSQ = tomography.LeastSquares(cfg.basis, cfg.diameter, cfg.num_modes, cfg.altitudes, cfg.aperture_size, r0=cfg.total_r0)
    
    atmosphere = phase_screen.Atmosphere(1, cfg.altitudes, r0, cfg.diameter, cfg.aperture_size, rng=seed)
    angle_norm = lambda x: [(x.theta % (2*np.pi)) / (2*np.pi), x._gamma_arcsec / atmosphere.limit_angle]

    # # Method for finding a decent asterism
    # plt.figure()
    # ax = plt.gca()
    # x = [120 * np.cos(theta) for theta in np.linspace(0, 2*np.pi, endpoint=True)]
    # y = [120 * np.sin(theta) for theta in np.linspace(0, 2*np.pi, endpoint=True)]
    # plt.plot(x, y, linestyle='dotted', color='k')
    # plt.plot([0], [0], marker="o")

    # for seed in range(0, 10):
    #     asterism = asterisms.generate_asterisms(1, 
    #                                             mask=mask,
    #                                             quality_threshold=cfg.asterism_quality_threshold,
    #                                             quality_limit=cfg.minimum_asterism_quality,
    #                                             model_location=cfg.asterism_model_location,
    #                                             quality_model=cfg.quality_model,
    #                                             regime_model=cfg.regime_model,
    #                                             rng=seed)[0]
    #     gs_x = [a.position_arcsec()[0] for a in asterism]
    #     gs_y = [a.position_arcsec()[1] for a in asterism]
    #     plt.plot(gs_x, gs_y, "--", marker="x", label=seed)

    # plt.xlabel('Position (")')
    # plt.ylabel('Position (")')
    # ax.set_aspect('equal', 'box')
    # plt.legend()
    # plt.tight_layout()
    # return

    asterism = asterisms.generate_asterisms(1, 
                                        mask=mask,
                                        quality_threshold=cfg.asterism_quality_threshold,
                                        quality_limit=cfg.minimum_asterism_quality,
                                        model_location=cfg.asterism_model_location,
                                        quality_model=cfg.quality_model,
                                        regime_model=cfg.regime_model,
                                        rng=seed)[0]

    gs_x = [a.position_arcsec()[0] for a in asterism]
    gs_y = [a.position_arcsec()[1] for a in asterism]

    target_coeffs = target.cutout_coeffs(cfg.basis, cfg.diameter, atmosphere.altitudes, atmosphere[0], atmosphere.metapupil_diameters)
    target_screen = target.total_cutout(cfg.diameter, atmosphere.altitudes, atmosphere[0], atmosphere.metapupil_diameters)
    fitted_screen = cfg.basis.compose(target_screen.shape, target_coeffs)

    asterism_coeffs = tomography.get_coeffs_for_asterism(asterism, cfg.basis, cfg.diameter, atmosphere[0], atmosphere.altitudes, atmosphere.metapupil_diameters)

    with torch.no_grad():
        gs = np.array([angle_norm(a) for a in asterism], dtype=np.float32)
        X = asterism_coeffs

        ptt_y = target_coeffs[:cfg.mode_skip]
        X = X[:, cfg.mode_skip:]

        gs = torch.from_numpy(gs).unsqueeze(0).to(device)
        X = torch.from_numpy(X).unsqueeze(0).to(device)
    
        pred_A = model(gs)
        X = torch.flatten(X, start_dim=1)
        pred = torch.einsum("bnm,bm->bn", pred_A, X)

        pred = pred.to("cpu").detach().numpy().ravel()
        pred_A = pred_A.to("cpu").detach().numpy()

        res = np.zeros_like(target_coeffs)
        np.concatenate((ptt_y, pred), out=res)

        recon_screen = cfg.basis.compose(target_screen.shape, res)

    lsq_coeffs = LSQ.run(asterism, target, asterism_coeffs, ptt_y)
    lsq_screen = cfg.basis.compose(target_screen.shape, lsq_coeffs)

    mmse_coeffs = MMSE.run(asterism, target, asterism_coeffs, ptt_y)
    mmse_screen = cfg.basis.compose(target_screen.shape, mmse_coeffs)
    
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(mmse_screen-lsq_screen)
    axs[1].imshow(mmse_screen)
    axs[2].imshow(lsq_screen)

    height = 2.2
    cmap = 'twilight_shifted'
    psf_cmap = 'bone'

    save_dir = "//file/UsersD$/dah158/Home/My Documents/03 - Doctorate/2026/Journal Paper/fig"

    LSQ_A = LSQ.ATii.squeeze()
    MINN_A = pred_A.squeeze()

    vmax = max(np.amax(LSQ_A), np.amax(MINN_A))
    vmin = min(np.amin(LSQ_A), np.amin(MINN_A))
    vmax, vmin = max(abs(vmin), abs(vmax)), -max(abs(vmin), abs(vmax))

    sz = 0.8
    plt.figure(figsize=(sz*2*height, sz*height))
    plt.imshow(LSQ_A, vmax=vmax, vmin=vmin, cmap=cmap)
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'example_LSQ_matrix_{heuristic}.pgf'), backend='pgf')
    plt.figure(figsize=(sz*2*height, sz*height))
    plt.imshow(MINN_A, vmax=vmax, vmin=vmin, cmap=cmap)
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'example_pseudo_ls_matrix_{heuristic}.pgf'), backend='pgf')

    fig_mat_cb, ax_mat_cb = plt.subplots(figsize=(1.2, height))
    # fig2 = plt.figure(figsize=(1, 4.5))
    # ax2 = fig2.add_axes() #[0.1, 0.1, 0.3, 0.8])
    cmap2 = eval(f"mpl.cm.{cmap}")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cbar = mpl.colorbar.ColorbarBase(ax_mat_cb, cmap=cmap2, norm=norm, orientation='vertical')

    cbar.set_label("Value")
    fig_mat_cb.tight_layout()
    plt.savefig(os.path.join(save_dir, f"example_matrix_cb_{heuristic}.pgf"), backend='pgf')

    # Plot different parts
    # cmap = 'viridis'


    vmax = max(np.max(target_screen), np.max(fitted_screen), np.max(recon_screen), np.max(mmse_screen), np.max(lsq_screen))
    vmin = min(np.min(target_screen), np.min(fitted_screen), np.min(recon_screen), np.min(mmse_screen), np.min(lsq_screen))
    vmax, vmin = max(vmax, -vmin), min(-vmax, vmin)
    extent_a = (120, -120, 120, -120)

    plt.figure(figsize=(height, height))
    plt.imshow(fitted_screen, vmax=vmax, vmin=vmin, cmap=cmap, extent=extent_a)
    plt.xlabel('Position (")')
    plt.ylabel('Position (")')
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "fitted_screen.tex"))
    plt.savefig(os.path.join(save_dir, f"example_fitted_screen_{heuristic}.pgf"), backend='pgf')

    plt.figure(figsize=(height, height))
    plt.imshow(recon_screen, vmax=vmax, vmin=vmin, cmap=cmap, extent=extent_a)
    plt.xlabel('Position (")')
    plt.ylabel('Position (")')
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "recon_screen.tex"))
    plt.savefig(os.path.join(save_dir, f"example_recon_screen_{heuristic}.pgf"), backend='pgf')

    plt.figure(figsize=(height, height))
    plt.imshow(lsq_screen, vmax=vmax, vmin=vmin, cmap=cmap, extent=extent_a)
    plt.xlabel('Position (")')
    plt.ylabel('Position (")')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"example_lsq_screen_{heuristic}.pgf"), backend='pgf')

    plt.figure(figsize=(height, height))
    plt.imshow(mmse_screen, vmax=vmax, vmin=vmin, cmap=cmap, extent=extent_a)
    plt.xlabel('Position (")')
    plt.ylabel('Position (")')
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "mmse_screen.tex"))
    plt.savefig(os.path.join(save_dir, f"example_mmse_screen_{heuristic}.pgf"), backend='pgf')

    target_screen *= mask

    plt.figure(figsize=(height, height))
    scr_im = plt.imshow(target_screen, vmax=vmax, vmin=vmin, cmap=cmap, extent=extent_a)
    plt.xlabel('Position (")')
    plt.ylabel('Position (")')
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "target_screen.tex"))
    plt.savefig(os.path.join(save_dir, f"example_target_screen_{heuristic}.pgf"), backend='pgf')

    # dmax = max(np.max(target_screen - fitted_screen),
    #            np.max(target_screen - recon_screen),
    #            np.max(target_screen - mmse_screen),
    #            np.max(target_screen - lsq_screen))
    # dmin = min(np.min(target_screen - fitted_screen),
    #            np.min(target_screen - recon_screen),
    #            np.min(target_screen - mmse_screen),
    #            np.min(target_screen - lsq_screen))
    # dmax, dmin = max(dmax, -dmin), min(-dmax, dmin)
    dmax, dmin = vmax, vmin
    
    plt.figure(figsize=(height, height))
    dif_im = plt.imshow(target_screen - fitted_screen, vmax=dmax, vmin=dmin, cmap=cmap, extent=extent_a)
    sr = strehl.ratio(target_screen - fitted_screen, mask=mask)
    print(f"Ideal SR: {sr}")
    plt.xlabel('Position (")')
    plt.ylabel('Position (")')
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "fitted_resid.tex"))
    plt.savefig(os.path.join(save_dir, f"example_fitted_resid_{heuristic}.pgf"), backend='pgf')

    plt.figure(figsize=(height, height))
    plt.imshow(target_screen - lsq_screen, vmax=dmax, vmin=dmin, cmap=cmap, extent=extent_a)
    sr = strehl.ratio(target_screen - lsq_screen, mask=mask)
    print(f"LSQ SR: {sr}")
    plt.xlabel('Position (")')
    plt.ylabel('Position (")')
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "fitted_resid.tex"))
    plt.savefig(os.path.join(save_dir, f"example_lsq_resid_{heuristic}.pgf"), backend='pgf')

    plt.figure(figsize=(height, height))
    plt.imshow(target_screen - recon_screen, vmax=dmax, vmin=dmin, cmap=cmap, extent=extent_a)
    sr = strehl.ratio(target_screen - recon_screen, mask=mask)
    print(f"MINN-LS SR: {sr}")
    plt.xlabel('Position (")')
    plt.ylabel('Position (")')
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "recon_resid.tex"))
    plt.savefig(os.path.join(save_dir, f"example_recon_resid_{heuristic}.pgf"), backend='pgf')

    plt.figure(figsize=(height, height))
    plt.imshow(target_screen - mmse_screen, vmax=dmax, vmin=dmin, cmap=cmap, extent=extent_a)
    sr = strehl.ratio(target_screen - mmse_screen, mask=mask)
    print(f"MMSE SR: {sr}")
    plt.xlabel('Position (")')
    plt.ylabel('Position (")')
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "mmse_resid.tex"))
    plt.savefig(os.path.join(save_dir, f"example_mmse_resid_{heuristic}.pgf"), backend='pgf')

    # wavelength = 550e-9
    wavelength = 1
    res = 1000 * wavelength / cfg.diameter
    # extent_l = (-res/2 * 1e6, res/2 * 1e6, -res/2 * 1e6, res/2 * 1e6)
    extent_l = (-res/2, res/2, -res/2, res/2)
    

    print(f"Ideal MSE: {cfg.loss_function(torch.from_numpy(target_screen), torch.from_numpy(fitted_screen)).item()}")
    print(f"LSQ MSE: {cfg.loss_function(torch.from_numpy(target_screen), torch.from_numpy(lsq_screen)).item()}")
    print(f"MINN-LS MSE: {cfg.loss_function(torch.from_numpy(target_screen), torch.from_numpy(recon_screen)).item()}")
    print(f"MMSE MSE: {cfg.loss_function(torch.from_numpy(target_screen), torch.from_numpy(mmse_screen)).item()}")


    airy_disc = psf.psf(mask)
    fitted_psf = psf.psf(target_screen - fitted_screen)
    recon_psf = psf.psf(target_screen - recon_screen)
    mmse_psf = psf.psf(target_screen - mmse_screen)
    lsq_psf = psf.psf(target_screen - lsq_screen)

    plt.figure(figsize=(height, height))
    airy_psf = np.log10(airy_disc / airy_disc.max())
    psf_im = plt.imshow(airy_psf, vmax=0, vmin=-6, cmap=psf_cmap, extent=extent_l)
    plt.xlabel("Position ($\mu$m)")
    plt.ylabel("Position ($\mu$m)")
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "airy_psf.tex"))
    plt.savefig(os.path.join(save_dir, f"example_airy_psf_{heuristic}.pgf"), backend='pgf')

    plt.figure(figsize=(height, height))
    lsq_psf = np.log10(lsq_psf / airy_disc.max())
    plt.imshow(lsq_psf, vmax=0, vmin=-6, cmap=psf_cmap, extent=extent_l)
    plt.xlabel("Position ($\mu$m)")
    plt.ylabel("Position ($\mu$m)")
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "airy_psf.tex"))
    plt.savefig(os.path.join(save_dir, f"example_lsq_psf_{heuristic}.pgf"), backend='pgf')

    plt.figure(figsize=(height, height))
    fitted_psf = np.log10(fitted_psf / airy_disc.max())
    plt.imshow(fitted_psf, vmax=0, vmin=-6, cmap=psf_cmap, extent=extent_l)
    plt.xlabel("Position ($\mu$m)")
    plt.ylabel("Position ($\mu$m)")
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "fitted_psf.tex"))
    plt.savefig(os.path.join(save_dir, f"example_fitted_psf_{heuristic}.pgf"), backend='pgf')

    plt.figure(figsize=(height, height))
    recon_psf = np.log10(recon_psf / airy_disc.max())
    plt.imshow(recon_psf, vmax=0, vmin=-6, cmap=psf_cmap, extent=extent_l)
    plt.xlabel("Position ($\mu$m)")
    plt.ylabel("Position ($\mu$m)")
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "recon_psf.tex"))
    plt.savefig(os.path.join(save_dir, f"example_recon_psf_{heuristic}.pgf"), backend='pgf')

    plt.figure(figsize=(height, height))
    mmse_psf = np.log10(mmse_psf / airy_disc.max())
    plt.imshow(mmse_psf, vmax=0, vmin=-6, cmap=psf_cmap, extent=extent_l)
    plt.xlabel("Position ($\mu$m)")
    plt.ylabel("Position ($\mu$m)")
    plt.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "mmse_psf.tex"))
    plt.savefig(os.path.join(save_dir, f"example_mmse_psf_{heuristic}.pgf"), backend='pgf')

    plt.figure(figsize=(height, height))
    ax = plt.gca()
    plt.scatter([0], [0], marker="x", color='k', label="Target")
    plt.scatter(gs_x, gs_y, marker="+", color='r', label='Guide-stars')
    x = [120 * np.cos(theta) for theta in np.linspace(0, 2*np.pi, endpoint=True)]
    y = [120 * np.sin(theta) for theta in np.linspace(0, 2*np.pi, endpoint=True)]
    plt.plot(x, y, linestyle='dotted', color='k')
    plt.xlabel('Position (")')
    plt.ylabel('Position (")')
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"example_asterism_{heuristic}.pgf"), backend='pgf')
    # plt.legend(loc='lower left')
    # 3. Collect handles and labels from the data axis
    handles, labels = ax.get_legend_handles_labels()

    plt.figure(figsize=(height, 1))
    ax = plt.gca()
    ax.legend(handles, labels, loc='center')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"example_asterism_{heuristic}_legend.pgf"), backend='pgf')
    
    psf_cmap = eval(f"mpl.cm.{psf_cmap}")
    cmap = eval(f"mpl.cm.{cmap}")

    fig2, ax2 = plt.subplots(figsize=(height, 1))
    # fig2 = plt.figure(figsize=(1, 4.5))
    # ax2 = fig2.add_axes() #[0.1, 0.1, 0.3, 0.8])
    norm = mpl.colors.Normalize(vmin=-6, vmax=0)

    cbar = mpl.colorbar.ColorbarBase(ax2, cmap=psf_cmap, norm=norm, orientation='horizontal')

    cbar.set_label("Normalised Intensity [$\\log(I/I_{max})$]")
    fig2.tight_layout()
    plt.savefig(os.path.join(save_dir, f"example_psf_cb_{heuristic}.pgf"), backend='pgf')
    # tikzplotlib.save(os.path.join(save_dir, "psf_cb.tex"))

    # cbar = fig2.colorbar(psf_im, ax=ax2, cax=ax2, orientation='vertical', cmap=cmap)

    # cbar = fig2.colorbar(psf_im, ax=ax2, cax=ax2, orientation='vertical', cmap=cmap)
    # cbar.set_label("Normalised Intensity [$\\log(I/I_{max})$]")
    # fig2.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "psf_cb.tex"))

    fig3, ax3 = plt.subplots(figsize=(2.5*height, 1))
    # fig3 = plt.figure(figsize=(1, 4.5))
    # ax3 = fig3.add_axes() #[0.1, 0.1, 0.3, 0.8])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cbar = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, orientation='horizontal')

    cbar.set_label("Phase (radians)")
    fig3.tight_layout()
    plt.savefig(os.path.join(save_dir, f"example_screen_cb_{heuristic}.pgf"), backend='pgf')
    # tikzplotlib.save(os.path.join(save_dir, "screen_cb.tex"))

    # cbar = fig3.colorbar(scr_im, ax=ax3, cax=ax3, orientation='vertical', cmap=cmap)
    # cbar.set_label("Phase (radians)")
    # fig3.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "screen_cb.tex"))

    fig4, ax4 = plt.subplots(figsize=(height, 1))
    # fig4 = plt.figure(figsize=(1, 4.5))
    # ax4 = fig4.add_axes() #[0.1, 0.1, 0.3, 0.8])
    norm = mpl.colors.Normalize(vmin=dmin, vmax=dmax)

    cbar = mpl.colorbar.ColorbarBase(ax4, cmap=cmap, norm=norm, orientation='horizontal')

    cbar.set_label("Phase (radians)")
    fig4.tight_layout()
    plt.savefig(os.path.join(save_dir, f"example_resid_cb_{heuristic}.pgf"), backend='pgf')
    # tikzplotlib.save(os.path.join(save_dir, "resid_cb.tex"))

    # cbar = fig4.colorbar(dif_im, ax=ax4, cax=ax4, orientation='vertical', cmap=cmap)
    # cbar.set_label("Phase (radians)")
    # fig4.tight_layout()
    # tikzplotlib.save(os.path.join(save_dir, "resid_cb.tex"))

    pass

def run_models_over_different_heuristics():
    from training import run_epoch
    
    N = 256

    cfg = config.Config()

    ml_run = ml.Trainer(cfg, skip_lsq=False)

    results = {
        "Model_Heuristic": [],
        "Asterism_Heuristic": [],
        "Model": [],
        "Type": [],
        "Loss": [],
        "stdev": []
    }

    heuristics = ["all", "quality"]

    directory_base = cfg.directory

    directories = [os.path.join(directory_base, h + "_asterisms") for h in heuristics]
    print(ml_run.model_file_names)

    mask = cfg.basis.get_mask(cfg.aperture_size)

    # layer_conditions = {
    #     "UCMJO": ([0.88, 0.03, 0.09], [1000, 5000, 10000]),
    #     "Ellerbroek": ([0.230, 0.296, 0.246, 0.110, 0.064, 0.050, 0.004], [4.5, 463, 1483, 4840, 11122, 14906, 18635])
    # }

    # for profile, (cn2_pc, alts) in layer_conditions.items():
        # for D_r0 in [5., 10., 15., 20., 25., 30., 35., 40., 45., 50.]:
            # total_r0 = cfg.diameter / D_r0
            # var = total_r0 ** (-5/3)

            # r0 = [(var * layer_cn2_pc) ** (-3/5) for layer_cn2_pc in cn2_pc]

    for heuristic in heuristics:
        atm = phase_screen.Atmosphere(N, cfg.altitudes, cfg.r0, cfg.diameter, cfg.aperture_size)

        if heuristic == "both" or heuristic == "quality":
            quality_model = cfg.quality_model
        else:
            quality_model = None

        if heuristic == "both" or heuristic == "regime":
            regime_model = cfg.regime_model
        else:
            regime_model = None

        asts = asterisms.generate_asterisms(N, 
                                            mask=mask,
                                            quality_threshold=cfg.asterism_quality_threshold,
                                            quality_limit=cfg.minimum_asterism_quality,
                                            use_regime_model=True,
                                            model_location=cfg.asterism_model_location,
                                            quality_model=quality_model,
                                            regime_model=regime_model)

        for model_h, directory in zip(heuristics, directories):
            ml_run.load_models(suffix="_Variable_Final", dir=directory + "/current")

            result_data = run_epoch(ml_run, N, asts, atm, cfg.basis, train=False, wfs=None)

            for model, data_dict in result_data.items():
                for key, value in data_dict.items():
                    results["Model_Heuristic"].append(model_h)
                    results["Asterism_Heuristic"].append(heuristic)
                    results["Model"].append(model)
                    results["Type"].append(key)
                    results["Loss"].append(np.mean(value))
                    results['stdev'].append(np.std(value))

    df = pd.DataFrame.from_dict(results, orient="columns")

    df.to_parquet(os.path.join("D:/Data/2026/heuristic_comparison", f"HeuristicComparison.parquet"))
    print(df.head())

def display_models_over_different_heuristics():

    # Display results from above
    import glob
    import matplotlib.lines as mlines

    file_path = os.path.join("D:/Data/2026/heuristic_comparison", "HeuristicComparison.parquet")
    df = pd.read_parquet(file_path)
    
    # Set embedding state
    df["Embed"] = "No"
    df.loc[df['Model'].str.contains("Embed"), "Embed"] = "Yes"
    df.loc[df['Model'].isin(['Open Loop', 'LeastSquares', 'MMSE', "Fitting Error", 'GLAO', 'AO']), "Embed"] = "NA"
    df["Model"] = df["Model"].replace(r'(?:^GuideStarEmbed|Embed)_?', '', regex=True)
    print(df['Embed'].unique())

    # Set size variable
    df["Size"] = "Standardised"
    df.loc[df['Model'].str.contains("smol"), "Size"] = "Small"
    df.loc[df['Model'].isin(['Open Loop', 'LeastSquares', 'MMSE', "Fitting Error", 'GLAO', 'AO']), "Size"] = "NA"
    print(df['Size'].unique())
    df["Model"] = df["Model"].replace(r'smol_', '', regex=True)
    df["Model"] = df["Model"].replace(r'_smol', '', regex=True)

    # Remove unwanted junk from Model names
    df["Model"] = df["Model"].replace(r'MLNN_', '', regex=True)
    df["Model"] = df["Model"].replace(r'ResNet(?!_|$)', 'ResNet_', regex=True)
    df["Model"] = df["Model"].replace(r'_Variable.*', '', regex=True)
    
    df = df[df['Model'] != 'singlemode']

    df["Model"] = df["Model"].map({
        'baseline': "Direct MLNN",
        'pseudo_lsq': "Pseudo-LS",
        'pseudo_mmse': "Pseudo-MMSE",
        'NoInverseKzz': "PQ + Priors",
        'LeastSquares': "LS",
        "AO": "SCAO"
    }).fillna(df["Model"])

    print(df.head())
    print(df["Model"].unique())

    size_excl = "Small"
    embed_excl = "No"


    # df0 = df0.sort_values(by=["Model"])

    df["Model_Heuristic"] = df["Model_Heuristic"].map({
        'both': "Both",
        'none': "None",
        'regime': "Tomographic Regime",
        'quality': "Quality",
    }).fillna(df["Model_Heuristic"])

    df["Asterism_Heuristic"] = df["Asterism_Heuristic"].map({
        'both': "Both",
        'none': "None",
        'regime': "Tomographic Regime",
        'quality': "Quality",
    }).fillna(df["Asterism_Heuristic"])

    df = df.rename(columns= {"Model_Heuristic": "Model Heuristic", "Asterism_Heuristic": "Asterism Heuristic"})

    df = df[(df["Type"] == "Reconstruction")]
    #            & ((df["Model Heuristic"] == "Both") | (df["Model Heuristic"] == "None"))
    #        & ((df["Asterism Heuristic"] == "Both") | (df["Asterism Heuristic"] == "None"))]

    for model in ["Pseudo-LS", "Pseudo-MMSE", "Direct MLNN", "PQ + Priors", "ResNet", "SCAO", "GLAO", "LS", "MMSE"]:
        df0 = df[df["Model"] == model]
        if model != "SCAO" and model != "GLAO" and model != "LS" and model != "MMSE":
            # df1 = df0[df0["Size"] == "Small"]
            # df1 = df1[df1["Embed"] == "Yes"]
            # print(df1)
            # df1 = df0[df0["Size"] == "Small"]
            # df1 = df1[df1["Embed"] == "No"]
            # print(df1)
            df1 = df0[df0["Size"] == "Standardised"]
            df1 = df1[df1["Embed"] == "Yes"]
            print(df1)
            df1 = df0[df0["Size"] == "Standardised"]
            df1 = df1[df1["Embed"] == "No"]
            print(df1)
        # print(df0.groupby(["Model Heuristic", "Asterism Heuristic"])["Loss"].mean())
        else:
            print(df0)

    # g = sns.relplot(data=df0, kind='line',
    #                 x="D_r0", y="Loss", hue="Model $D/r_0$", style="Profile")
    # g.set(ylabel="Reconstruction Loss (rads$^2$)", xlabel="$D/r_0$")
    # g._legend.remove()

    # group_cols = [c for c in df.columns if c not in ["Model_D_r0", "Profile", "Loss"]]
    # dfm = df.groupby(group_cols, as_index=False)['Loss'].mean()
    # dfm = dfm[(dfm["Size"] != size_excl)
    #           & (dfm["Type"] == "Reconstruction")
    #           & (dfm["Embed"] != embed_excl)
    #           & (dfm["Model"].isin(["GLAO", "Fitting Error"]))]

    # r0_list = dfm[dfm["Model"] == "GLAO"]["D_r0"].tolist()
    # glao_list = dfm[dfm["Model"] == "GLAO"]["Loss"].tolist()

    # r0_fe_list = dfm[dfm["Model"] == "Fitting Error"]["D_r0"].tolist()
    # fe_list = dfm[dfm["Model"] == "Fitting Error"]["Loss"].tolist()

    # for ax in g.axes.flat:
    #     ax.plot(r0_fe_list, fe_list, label="Fitting Error")
    #     ax.plot(r0_list, glao_list, label="GLAO")

    #     handles, labels = ax.get_legend_handles_labels()
    #     blank_handle = mlines.Line2D([], [], color='none')

    #     handles.insert(-2, blank_handle)
    #     labels.insert(-2, '')   # use ' ' (a space) if '' gets collapsed/stripped

    #     ax.legend(handles, labels)
    pass