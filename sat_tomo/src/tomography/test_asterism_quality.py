import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from tomography import asterisms, beacon
import ml.regression

# sklearn models and utility functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import util

DATA_FOLDER = 'C:/Code/PhD/Data/2026'
MODEL_FOLDER = 'C:/Code/PhD/Data/2026/Models'

METHODS = ['lsq'] # ['glao', 'lsq_T', 'lsq_K']

ALTITUDES = [1000, 5000, 10000]

def run_tests():
    # test_overlap_percentage()
    # return

    # This is set up for the results from test_regime_data (2026-06-26)
    build_quality_models()

    # plt.show()

def build_quality_models():
    files = glob.glob(os.path.join(DATA_FOLDER, '*transformed.parquet'))

    training_file = 'training_data_merge'
    validation_file = 'validation_data_merge'

    test_df = pd.DataFrame()
    test_files = [f for f in files if 'training' in f]
    for n, file in enumerate(test_files):
        tdf = pd.read_parquet(os.path.join(DATA_FOLDER, file))
        tdf['file_id'] = f'test_{n}'

        test_df = pd.concat((test_df, tdf))
    test_df.to_parquet(os.path.join(DATA_FOLDER, training_file + '.parquet'), index=False)

    valid_df = pd.DataFrame()
    validation_files = [f for f in files if 'validation' in f]
    for n, file in enumerate(validation_files):
        vdf = pd.read_parquet(os.path.join(DATA_FOLDER, file))
        vdf['file_id'] = f'test_{n}'

        valid_df = pd.concat((valid_df, vdf))
    valid_df.to_parquet(os.path.join(DATA_FOLDER, validation_file + '.parquet'), index=False)

    print("Analysing results")
    analyse_results(training_file, validation_file)

    # plot_results(training_file)
    # analyse_asterisms(training_file, validation_file)
    # data_correlations(validation_file)


# Perform regression on a particular model
def run_reg(data, model, name, scale=False):
    ###
    print(f"Running regression on model: {name}")
    ###
    X_train, X_test, y_train, y_test = data

    star_cols = ['file_id', 'group_id', 'asterism_id']
    star_data_train = X_train[star_cols].copy()
    star_data_train = star_data_train.reset_index(drop=True)
    star_data_train['id'] = star_data_train.index.to_numpy()
    star_data_test = X_test[star_cols].copy()
    star_data_test = star_data_test.reset_index(drop=True)
    star_data_test['id'] = star_data_test.index.to_numpy()
    star_data = pd.merge(star_data_test, star_data_train, how='inner', on='id', suffixes=('_test', '_train'))
    star_data = star_data.drop(columns='id')

    X_train = X_train.drop(columns=star_cols)
    X_test = X_test.drop(columns=star_cols)

    features = list(X_train.columns)

    if scale == True:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).flatten()

    predictions = list(y_pred)
    tests = list(y_test)

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    reg_data = [{
        'Model': name,
        # 'Model Data': model,
        'R²': r2,
        'RMSE': rmse,
        'Features': features,
        'idx': i,
        'Predictions': predictions[i],
        'Test': tests[i],
        'Test File ID': star_data['file_id_test'].iat[i],
        'Test Asterism ID': star_data['asterism_id_test'].iat[i],
        'Test Group ID': star_data['group_id_test'].iat[i],
        'Train File ID': star_data['file_id_train'].iat[i],
        'Train Asterism ID': star_data['asterism_id_train'].iat[i],
        'Train Group ID': star_data['group_id_train'].iat[i]
        } for i in range(len(predictions))]
    print(f"Regression results: R²={r2}; RMSE={rmse}")

    return reg_data, model, scaler

def analyse_asterisms(file, validation_file):

    os.makedirs('./img/asterisms', exist_ok=True)

    df_r = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_prediction_results.parquet'))
    # df = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_noise_CLEAN.parquet'))
    v_df = pd.read_parquet(os.path.join(DATA_FOLDER, validation_file + f'.parquet'))

    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",  # or "lualatex", "xelatex"
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "figure.figsize": (3, 3),
    })

    # df['validation'] = False
    # v_df['validation'] = True
    # TOP_N = 25

    test_variables = {
        # "recon_rmse_mean": True,
        # "recon_rmse_log_mean": True,
        # "recon_rmse_rel_mean": True,
        # "recon_rmse_rel_log_mean": True,
        "strehl_ratio_mean": False,
        "strehl_ratio_log_mean": False,
        "strehl_ratio_rel_mean": False,
        "strehl_ratio_rel_log_mean": False,
    }

    n_samples = 200000

    for min_distance, TOP_N in zip([0, 20], [16, 8]):
        df_r1 = df_r[(df_r['Method'] == 'lsq')]
        v_df1 = v_df[(v_df['tomo_method'] == 'lsq')]

        v_df1 = v_df1[(v_df1['f_distance_0'] >= min_distance) & (v_df1['f_distance_1'] >= min_distance) & (v_df1['f_distance_2'] >= min_distance)]
        print(v_df1.columns)
        columns = ['file_id', 'group_id', 'asterism_id']
        for i in range(3):
            columns.append(f'star_{i}_gamma')
            columns.append(f'star_{i}_theta')
            columns.append(f'star_{i}_mag')
        v_df1 = v_df1[columns]

        try:
            df_r1 = df_r1.sample(n_samples)
        except ValueError:
            pass

        # df0 = pd.concat([df1, v_df1])

        for test_var, ascending in test_variables.items():

            top_test_asterisms = df_r1[df_r1['Metric'] == test_var].sort_values(by='Test', ascending=ascending).drop_duplicates(subset='Test Group ID', keep='first')
            top_test_asterisms = top_test_asterisms.merge(v_df1, left_on=['Test File ID', 'Test Group ID', 'Test Asterism ID'], right_on=['file_id', 'group_id', 'asterism_id'], how='inner')
            top_test_asterisms = top_test_asterisms.head(TOP_N)
            top_test_asterisms = top_test_asterisms.drop(columns=['file_id', 'group_id', 'asterism_id'])

            bottom_test_asterisms = df_r1[df_r1['Metric'] == test_var].sort_values(by='Test', ascending=(not ascending)).drop_duplicates(subset='Test Group ID', keep='first')
            bottom_test_asterisms = bottom_test_asterisms.merge(v_df1, left_on=['Test File ID', 'Test Group ID', 'Test Asterism ID'], right_on=['file_id', 'group_id', 'asterism_id'], how='inner')
            bottom_test_asterisms = bottom_test_asterisms.head(TOP_N)
            bottom_test_asterisms = bottom_test_asterisms.drop(columns=['file_id', 'group_id', 'asterism_id'])

            top_pred_asterisms = df_r1[df_r1['Metric'] == test_var].sort_values(by='Predictions', ascending=ascending).drop_duplicates(subset='Test Group ID', keep='first')
            top_pred_asterisms = top_pred_asterisms.merge(v_df1, left_on=['Test File ID', 'Test Group ID', 'Test Asterism ID'], right_on=['file_id', 'group_id', 'asterism_id'], how='inner')
            top_pred_asterisms = top_pred_asterisms.head(TOP_N)
            top_pred_asterisms = top_pred_asterisms.drop(columns=['file_id', 'group_id', 'asterism_id'])

            bottom_pred_asterisms = df_r1[df_r1['Metric'] == test_var].sort_values(by='Predictions', ascending=(not ascending)).drop_duplicates(subset='Test Group ID', keep='first')
            bottom_pred_asterisms = bottom_pred_asterisms.merge(v_df1, left_on=['Test File ID', 'Test Group ID', 'Test Asterism ID'], right_on=['file_id', 'group_id', 'asterism_id'], how='inner')
            bottom_pred_asterisms = bottom_pred_asterisms.head(TOP_N)
            bottom_pred_asterisms = bottom_pred_asterisms.drop(columns=['file_id', 'group_id', 'asterism_id'])


            # mean_val = df0[test_var].mean()
            # mid_asterisms = df0.assign(diff=np.abs(df0[test_var] - mean_val)).sort_values(by='diff').head(TOP_N)

            def recreate_asterism(row):
                return [beacon.Beacon(row[f'star_{i}_theta'], row[f'star_{i}_gamma'], row[f'star_{i}_mag']) for i in range(3)]

            def plot_circ(ax):
                theta = np.linspace(0, 2*np.pi, 100, endpoint=True)
                rho = np.ones_like(theta) * 120
                x, y = util.pol2cart(rho, theta)

                ax.plot(x, y, "k--", alpha=0.2)

            fig_data = [
                ("best_test", top_test_asterisms),
                ("worst_test", bottom_test_asterisms),
                ("best_pred", top_pred_asterisms),
                ("worst_pred", bottom_pred_asterisms),
            ]

            for label, data in fig_data:
                fig, ax = plt.subplots()
                asts = data.apply(recreate_asterism, axis=1)

                ax.plot([0], [0], 'ko')
                for a in asts:
                    x = [b.position_arcsec()[0] for b in a]
                    y = [b.position_arcsec()[1] for b in a]
                    ax.plot(x, y, '+:', alpha=0.5)

                if label == "best_pred" and min_distance == 0:
                    test_psfs(asts[0], label)

                plot_circ(ax)
                # ax.set_xticks([-120, 0, 120])
                # ax.set_yticks([-120, 0, 120])
                # if 'worst' in label:
                #     ax.set_xlabel('Position (")')
                # if 'test' in label:
                #     ax.set_ylabel('Position (")')
                ax.set_aspect('equal')
                ax.axis('off')
                fig.tight_layout()
                fig.savefig(f"./img/asterisms/{test_var}_{label}_min{min_distance}.pgf", bbox_inches='tight')
                fig.savefig(f"./img/asterisms/{test_var}_{label}_min{min_distance}.pdf", bbox_inches='tight')
                plt.close(fig)

    return

def test_psfs(asterism, label):

    os.makedirs('./img/psf', exist_ok=True)

    from tomography import phase_screen, psf, tomography
    from basis import zernike
    D = 1.8
    A = [1000, 5000, 10000]
    ap = 256
    N = 21
    total_r0 = 0.15
    layer_r0 = util.total_r0_to_layer_r0(0.15, 3)
    atm = phase_screen.Atmosphere(1, A, r0=layer_r0, D=D, aperture_size=ap)

    target = beacon.Beacon(0, 0, 10)
    screen = target.total_cutout(D, A, atm[0], atm.metapupil_diameters)
    screen = screen * util.circle(ap)

    extent_m = (-D/2, D/2, -D/2, D/2)
    wavelength = 550e-9
    res = ap*wavelength/D
    extent_l = (-res/2, res/2, -res/2, res/2)

    z = zernike.Zernike(N)
    zcoeffs = z.decompose(screen)
    zscreen = z.compose(ap, zcoeffs)
    zresid = screen - zscreen

    z_psf = psf.psf(zresid)
    z_psf = np.log10(z_psf / z_psf.max())

    lsq = tomography.LeastSquares(z, D, N, A, ap, r0=total_r0, method='Tinv')
    tcoeffs = tomography.get_coeffs_for_asterism(asterism, z, D, atm[0], A, atm.metapupil_diameters)
    
    lsq_coeffs = lsq.run(asterism, target, tcoeffs, ptt=zcoeffs[:3])
    lsq_recon = z.compose(ap, lsq_coeffs)
    lsq_resid = screen - lsq_recon
    lsq_psf = psf.psf(lsq_resid)
    lsq_psf = np.log10(lsq_psf / lsq_psf.max())

    screen_vmin = min(np.amin(screen), np.amin(lsq_recon))
    screen_vmax = max(np.amax(screen), np.amax(lsq_recon))

    sz = 2.2
    cmap = 'inferno'

    fig, ax = plt.subplots(figsize=(sz, sz))
    screen_im = ax.imshow(screen, extent=extent_m, vmin=screen_vmin, vmax=screen_vmax, cmap=cmap)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_target_phase_screen.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_target_phase_screen.pdf", bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(sz, sz))
    ax.imshow(lsq_recon, extent=extent_m, vmin=screen_vmin, vmax=screen_vmax, cmap=cmap)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_tomo_recon.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_tomo_recon.pdf", bbox_inches='tight')
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(sz/6, sz))
    cbar = fig2.colorbar(screen_im, cax=ax2, orientation='vertical', cmap=cmap)
    cbar.set_label("Phase (radians)")
    fig2.savefig(f"./img/psf/{label}_recon_colorbar.pgf", bbox_inches='tight')
    fig2.savefig(f"./img/psf/{label}_recon_colorbar.pdf", bbox_inches='tight')
    plt.close(fig2)

    resid_vmin = min(np.amin(zresid), np.amin(lsq_resid))
    resid_vmax = max(np.amax(zresid), np.amax(lsq_resid))

    fig, ax = plt.subplots(figsize=(2,2))
    resid_im = ax.imshow(zresid, extent=extent_m, vmin=resid_vmin, vmax=resid_vmax, cmap=cmap)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_target_fitting_residuals.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_target_fitting_residuals.pdf", bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(sz, sz))
    ax.imshow(lsq_resid, extent=extent_m, vmin=resid_vmin, vmax=resid_vmax, cmap=cmap)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_tomo_residuals.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_tomo_residuals.pdf", bbox_inches='tight')
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(sz/6, sz))
    cbar = fig2.colorbar(resid_im, cax=ax2, orientation='vertical', cmap=cmap)
    cbar.set_label("Phase (radians)")
    fig2.savefig(f"./img/psf/{label}_resid_colorbar.pgf", bbox_inches='tight')
    fig2.savefig(f"./img/psf/{label}_resid_colorbar.pdf", bbox_inches='tight')
    plt.close(fig2)

    
    fig, ax = plt.subplots(figsize=(sz, sz))
    psf_im = ax.imshow(z_psf, vmin=-6, vmax=0, extent=extent_l, cmap=cmap)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_fitting_psf.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_fitting_psf.pdf", bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(sz, sz))
    ax.imshow(lsq_psf, vmin=-6, vmax=0, extent=extent_l, cmap=cmap)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_tomo_psf.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_tomo_psf.pdf", bbox_inches='tight')
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(sz/6, sz))
    cbar = fig2.colorbar(psf_im, cax=ax2, orientation='vertical', cmap=cmap)
    cbar.set_label("Normalised Intensity [$\\log(I/I_{max})$]")
    fig2.savefig(f"./img/psf/{label}_psf_colorbar.pgf", bbox_inches='tight')
    fig2.savefig(f"./img/psf/{label}_psf_colorbar.pdf", bbox_inches='tight')
    plt.close(fig2)

    return

def analyse_results(training_file, validation_file):
    df = pd.read_parquet(os.path.join(DATA_FOLDER, training_file + '.parquet'))
    v_df = pd.read_parquet(os.path.join(DATA_FOLDER, validation_file + '.parquet'))

    # Find regression models for different sets of features
    test_variables = [
        # "recon_rmse_mean",
        # "recon_rmse_rel_mean",
        # "recon_rmse_log_mean",
        # "recon_rmse_rel_log_mean",
        "strehl_ratio_mean",
        "strehl_ratio_rel_mean",
        "strehl_ratio_log_mean",
        "strehl_ratio_rel_log_mean",
    ]

    feature_groups = {
        # 'Geometry Only': ['f_distance_0',
        #              'f_distance_1',
        #              'f_distance_2',
        #              'f_angle_0',
        #              'f_angle_1',
        #              'f_angle_2'],
        # 'Geometry + Overlap': 
        #             ['f_distance_0',
        #              'f_distance_1',
        #              'f_distance_2',
        #              'f_angle_0',
        #              'f_angle_1',
        #              'f_angle_2',
        #              'f_overlap'],
        'All without Overlap':
        [
            'f_distance_0',
            'f_distance_1',
            'f_distance_2',
            'f_angle_0',
            'f_angle_1',
            'f_angle_2',
            'f_perimeter',
            'f_area',
            'f_mean_distance',
            'f_signed_distance_function',
            'f_distance_to_closest_line',
            'f_centroid_distance',
            'f_incentre_distance',
            'f_incircle_radius',
            ],
        # 'All Features':
        # [
        #     'f_distance_0',
        #     'f_distance_1',
        #     'f_distance_2',
        #     'f_angle_0',
        #     'f_angle_1',
        #     'f_angle_2',
        #     'f_perimeter',
        #     'f_area',
        #     'f_mean_distance',
        #     'f_signed_distance_function',
        #     'f_distance_to_closest_line',
        #     'f_centroid_distance',
        #     'f_incentre_distance',
        #     'f_incircle_radius',
        #     'f_overlap'
        #     ]
    }

    from torch import nn
    C = 10.0
    eps = 0.01

    models = {
        "Partial Least Squares": (PLSRegression(n_components=2), False),
        "Principal Component": (make_pipeline(PCA(n_components=2), LinearRegression()), False),
        "Ridge": (Ridge(alpha=1.0), False),
        "Lasso": (Lasso(alpha=1.0), False),
        # "SVR (Linear)": (SVR(kernel='linear', C=1.0, epsilon=0.1), False),
        # "SVR (Poly)": (SVR(kernel='poly', C=C, epsilon=eps), True),
        "SVR (RBF)": (SVR(kernel='rbf', C=C, epsilon=eps), True),
        # "SVR (Poly) [Scaled Inputs]": (SVR(kernel='poly', C=C, epsilon=eps), True),
        # "SVR (RBF) [Scaled Inputs]": (SVR(kernel='rbf', C=C, epsilon=eps), True),
        # "sklearn Neural Network": (MLPRegressor(hidden_layer_sizes=[32,64,32], learning_rate_init=0.01, activation="relu"), True),
        "Neural Network": (ml.regression.ML_Regressor(layer_dims=[32,64,32], learning_rate=0.02, weight_decay=1e-4, activation_func=nn.LeakyReLU), True),
    }

    star_columns = ["file_id", "group_id", "asterism_id"]

    results = []

    for method in ['lsq']:
        df0 = df[df['tomo_method'] == method]
        v_df0 = v_df[v_df['tomo_method'] == method]
        
        for test_column in test_variables:
            for feature_group, feature_columns in feature_groups.items(): 
                df_train = df0
                df_test = v_df0

                X_test = df_test[feature_columns + star_columns]
                X_train = df_train[feature_columns + star_columns]

                y_test = df_test[test_column]
                y_train = df_train[test_column]

                data = (X_train, X_test, y_train, y_test)
                
                for name, (model, scale) in models.items():
                    label = f"{name}, {method}, {test_column}, {feature_group}"
                    print(label)
                    test_results, trained_model, scaler = run_reg(data, model, name, scale)
                    for tr in test_results:
                        tr['Method'] = method
                        tr['Metric'] = test_column
                        tr["Feature Group"] = feature_group
                    results += test_results

                    name = name.replace('(', '').replace(')', '').replace(' ', '_')
                    filename = os.path.join(MODEL_FOLDER, f'{name}_{method}_{test_column}_{feature_group}.pkl')
                    with open(filename, 'wb') as f:
                        pickle.dump((trained_model, scaler), f)

    print("Finishing analysis, displaying plots")
    df_r = pd.DataFrame(results)
    # df_rs = df_r.drop(columns=['Model Data'])

    df_r.to_parquet(os.path.join(DATA_FOLDER, training_file + '_prediction_results.parquet'))

    df_summary = df_r.drop(columns=['idx', 'Predictions', 'Test', 'Features', 'Test Asterism ID', 'Test Group ID', 'Train Asterism ID', 'Train Group ID'])
    df_summary = df_summary.drop_duplicates(subset=['Model', 'Metric', 'Feature Group', 'Method'])
    
    df_summary.to_csv(os.path.join(DATA_FOLDER, training_file + f'_R2_summary_results.csv'))

def plot_results(file):
    os.makedirs(f'./img/{file}', exist_ok=True)

    df_r = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_prediction_results.parquet'))
    fig_idx = 1

    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",  # or "lualatex", "xelatex"
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "figure.figsize": (3, 3),
    })

    strehl = ["Strehl Ratio, $\\bar{S}$", "$\\ln(\\textrm{Strehl Ratio})$, $\\ln(\\bar{S})$", "$\\frac{\\textrm{Strehl Ratio}}{\\textrm{Open Loop Strehl Ratio}}$, $\\bar{\\xi}$", "$\\ln(\\frac{\\textrm{Strehl Ratio}}{\\textrm{Open Loop Strehl Ratio}})$, $\\overline{\\ln{\\xi}}$"]
    rmse = ["Reconstruction RMSE", "Log (Reconstruction RMSE)", "Reconstruction RMSE / Open Loop RMSE", "Log(Reconstruction RMSE / Open Loop RMSE)"]

    metric_labels = {
        # "recon_rmse_mean": rmse[0],
        # "recon_rmse_log_mean": rmse[1],
        # "recon_rmse_rel_mean": rmse[2],
        # "recon_rmse_rel_log_mean": rmse[3],
        "strehl_ratio_mean": strehl[0],
        "strehl_ratio_log_mean": strehl[1],
        "strehl_ratio_rel_mean": strehl[2],
        "strehl_ratio_rel_log_mean": strehl[3],
        # "Recon Rmse Mean": rmse[0],
        # "Recon Rmse Log Mean": rmse[1],
        # "Recon Rmse Rel Mean": rmse[2],
        # "Recon Rmse Rel Log Mean": rmse[3],
        "Strehl Ratio Mean": strehl[0],
        "Strehl Ratio Log Mean": strehl[1],
        "Strehl Ratio Rel Mean": strehl[2],
        "Strehl Ratio Rel Log Mean": strehl[3]
    }

    df_r['Model'] = df_r['Model'].replace({
        "Principal Component": "Principal Component Analysis",
        "Ridge": "Linear Regression (Ridge)",
        "Lasso": "Linear Regression (Lasso)",
        "SVR (Poly)": "SVR (Polynomial)",
        # "sklearn Neural Network": "Neural Network A (Adam, ReLU)",
        # "My Neural Network": "Neural Network",
    })

    metrics = [m for m in df_r['Metric'].unique() if 'strehl' in m]

    for metric in metrics:
        hist_of_test_variable = True
        
        for method in df_r['Method'].unique():
            for feature_group in ['Geometry Only', 'All without Overlap', 'All Features']:
                for model in df_r['Model'].unique():

                    df_plot = df_r[(df_r['Method'] == method) 
                                    & (df_r['Metric'] == metric) 
                                    & (df_r['Feature Group'] == feature_group)
                                    & (df_r['Model'] == model)]
                    
                    fig_name = f"_{method}_{metric}_{model}_{feature_group}"
                    replacements = {
                        '/': '',
                        ' ': '_',
                        '(': '_',
                        ')': '',
                    }

                    for i, n in replacements.items():
                        fig_name = fig_name.replace(i, n)
                    
                    if hist_of_test_variable:
                        fig, axs = plt.subplots(1, 1)
                        axs.hist(x=df_plot['Test'], bins=50)
                        axs.set_xlabel(metric)
                            
                        hist_of_test_variable = False
                        fig.tight_layout()
                        fig.savefig(f"./img/{file}/fig{fig_idx}_histogram_{fig_name}.pgf", bbox_inches='tight')
                        fig.savefig(f"./img/{file}/fig{fig_idx}_histogram_{fig_name}.pdf", bbox_inches='tight')
                        print(f"Figure {fig_idx}: Histogram of test variables for metric {metric}")
                        fig_idx += 1

                    fig, axs = plt.subplots(1, 1, figsize=(3, 3))
                    n_samples = 500
                    df_sampled = df_plot.sample(n_samples, replace=True)
                    y_test = df_sampled['Test']
                    y_pred = df_sampled['Predictions']
                    axs.scatter(y_test, y_pred, marker='x', alpha=0.8)
                    max_extent = min(df_plot['Test'].max(), df_plot['Predictions'].max())
                    k = 1.03
                    max_extent = max_extent * k if max_extent > 0.0 else max_extent / k
                    min_extent = max(df_plot['Test'].min(), df_plot['Predictions'].min())
                    min_extent = min_extent / k if min_extent > 0.0 else min_extent * k
                    axs.plot([min_extent, max_extent], [min_extent, max_extent], 'k--', linewidth=0.7)
                    axs.set_xlabel(metric_labels[metric])
                    axs.set_ylabel(f"Predictions of {model}")
                    # axs.grid(color='gray', linestyle='--', linewidth=0.5)
                    # axs.spines['top'].set_visible(False)
                    # axs.spines['right'].set_visible(False)
                    # axs.spines['bottom'].set_visible(False)
                    # axs.spines['left'].set_visible(False)

                    # axs[1].hist(x=df_plot['Test'], bins=50)
                    # axs[1].set_xlabel(metric)

                    # axs[2].hist(x=df_plot['Predictions'], bins=50)
                    # axs[2].set_xlabel(model)
                    # # g = sns.lmplot(df_plot, x=metric, y='Model Prediction', palette='dark', markers='+', hue='Model')
                    print(f"Figure {fig_idx}: {method}, {metric}, {model} and {feature_group}")

                    fig.tight_layout()
                    fig.savefig(f"./img/{file}/fig{fig_idx}_{fig_name}.pgf", bbox_inches='tight')
                    fig.savefig(f"./img/{file}/fig{fig_idx}_{fig_name}.pdf", bbox_inches='tight')

                    fig_idx += 1
                    plt.close()
                    # plt.show()
    return

 
def data_correlations(file):

    test_variables = [
        # "recon_rmse_mean",
        # "recon_rmse_log_mean",
        # "recon_rmse_rel_mean",
        # "recon_rmse_rel_log_mean",
        "strehl_ratio_mean",
        "strehl_ratio_log_mean",
        "strehl_ratio_rel_mean",
        "strehl_ratio_rel_log_mean",
    ]

    feature_columns = [
            'f_distance_0',
            'f_distance_1',
            'f_distance_2',
            'f_angle_0',
            'f_angle_1',
            'f_angle_2',
            'f_perimeter',
            'f_area',
            'f_mean_distance',
            'f_signed_distance_function',
            'f_distance_to_closest_line',
            'f_centroid_distance',
            'f_circumcentre_distance',
            'f_circumcircle_radius',
            'f_incentre_distance',
            'f_incircle_radius',
            'f_overlap',
            # 'f_min_distance' : asterisms.min_distance(a),
            # 'f_max_distance' : asterisms.max_distance(a),
            # 'f_length_ratio' : asterisms.length_ratio(a),
            # 'f_average_side_length' : asterisms.average_side_length(a),
            # 'f_minimum_side_length' : asterisms.minimum_side_length(a),
            # 'f_maximum_side_length' : asterisms.maximum_side_length(a),
            # 'f_perimeter_area_ratio' : asterisms.perimeter_area_ratio(a),
            # 'f_closest_furthest_side_length' : asterisms.closest_furthest_side_length(a),
            # 'f_distance_to_closest_edge_obsolete' : asterisms.distance_to_closest_line(a),
            # 'f_min_interior_angle' : asterisms.min_angle(a),
            # 'f_maximum_interior_angle' : asterisms.max_angle(a),
            # 'f_gaussian_area_within_triangle' : asterisms.gaussian_area_within_triangle(a),
    ]

    df = pd.read_parquet(os.path.join(DATA_FOLDER, file + '.parquet'))
    # df0 = df.filter(regex='^f_.*')
    # feature_columns = list(df0.columns)

    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",  # or "lualatex", "xelatex"
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })

    fig, ax = plt.subplots()

    correlations = df[test_variables + feature_columns].corr().loc[test_variables, feature_columns]
    sns.heatmap(correlations, ax=ax, annot=True)

    correlations.to_csv(os.path.join(DATA_FOLDER, file + f'_correlations.csv'))
    print(correlations)

    fig.savefig(f"./img/correlations_heatmap.pgf", bbox_inches='tight')
    fig.savefig(f"./img/correlations_heatmap.pdf", bbox_inches='tight')

    # Find prediction results for best model
    # df_v = pd.read_parquet(os.path.join(DATA_FOLDER, validation_file + '_CLEAN.parquet'))

    # TODO: Predict results for some set asterisms.
    # Line, circle, off-centre cluster, half-circle

def test_overlap_percentage():

    os.makedirs('./img/overlap', exist_ok=True)

    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely import plotting
    from tomography import tomography

    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",  # or "lualatex", "xelatex"
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })

    D = 1.8
    asterism = [beacon.Beacon(3.0, 30),
                beacon.Beacon(2.1, 35),
                beacon.Beacon(5.0, 50)]#asterisms.get_base_asterisms(1, asterism_size=5)
    
    # asterism = asterism[0][:3]
    for s in asterism:
        print(f'{s.theta *180/np.pi}, {s._gamma_arcsec}')
    
    resolution = 256
    circle = lambda centre, radius, resolution: Point(centre).buffer(radius, resolution)
    overlaps = []

    fig, axs = plt.subplots(1, len(ALTITUDES))
    metapupil_sizes = tomography.get_layer_shapes(asterism, ALTITUDES, D, resolution)[0]

    for i, (alt, m_dia) in enumerate(zip(ALTITUDES, metapupil_sizes)):
        ax = axs[i]
        radius = D / 2

        circles = [circle(a.position(alt), radius, resolution) for a in asterism]
        metapupil = circle((0, 0), m_dia/2, resolution)
        
        # build the target circle
        C4 = circle((0, 0), radius, resolution)
        
        # union of the three source circles
        U123 = unary_union(circles)
        
        # intersect that union with the target circle
        overlap = U123.intersection(C4)

        # Plot metapupil
        patch = plotting.plot_polygon(metapupil, ax=ax, color='k', linewidth=0.5, add_points=False)
        patch.set_facecolor((0.0, 0.0, 0.0, 0.0))

        # Plot circle faces for targets
        for c in circles:
            patch = plotting.plot_polygon(c, ax=ax, color='cyan', linewidth=0.8, add_points=False)
            patch.set_facecolor((0.32, 0.32, 0.46, 0.3))
            patch.set_edgecolor((0.32, 0.32, 0.46, 0.0))

        # Plot white background for the overlap union
        patch = plotting.plot_polygon(overlap, ax=ax, color='white', linewidth=3.0, add_points=False)
        patch.set_facecolor((1.0, 1.0, 1.0, 1.0))
        patch.set_edgecolor((0.0, 0.0, 0.0, 0.0))

        # Plot the union itself
        patch = plotting.plot_polygon(overlap, ax=ax, color='orange', linewidth=3.0, add_points=False)
        patch.set_facecolor((1.0, 0.58, 0.16, 0.3))
        patch.set_edgecolor((1.0, 0.58, 0.16, 1.0))

        # Plot the edges of the guide-stars
        for c in circles:
            patch = plotting.plot_polygon(c, ax=ax, color='cyan', linewidth=0.8, add_points=False)
            patch.set_facecolor((0.2, 0.74, 0.85, 0.0))
            patch.set_edgecolor((0.32, 0.32, 0.46, 1.0))

        # Plot the target edges
        patch = plotting.plot_polygon(C4, ax=ax, color='k', linewidth=1.0, add_points=False)
        patch.set_facecolor((0.0, 0.0, 0.0, 0.0))


        ax.set_aspect('equal')
        ax.set_xlabel('Position (m)')
        if i == 0:
            ax.set_ylabel('Position (m)')
        ax.set_title(f'Layer at {int(alt/1000)}km')
        
        # compute ratio
        overlaps.append(overlap.area / C4.area)

    print(overlaps)
    print(np.mean(overlaps))

    fig.tight_layout()
    fig.savefig("./img/overlap/overlap.pgf", bbox_inches='tight')
    fig.savefig("./img/overlap/overlap.pdf", bbox_inches='tight')
    plt.close(fig)
    plt.show()