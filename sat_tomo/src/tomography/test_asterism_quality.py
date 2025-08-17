import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

DATA_FOLDER = 'C:/Code/PhD/Data'

METHODS = ['lsq_T'] # ['glao', 'lsq_T', 'lsq_K']

ALTITUDES = [1000, 5000, 10000]
# ATM_SAMPLE_SIZES = [32, 64, 128, 256, 512]
ATM_SAMPLE_SIZES = [100]

def run_tests():
    # test_overlap_percentage()
    # return

    files = ['testing_data_21r015', 'validation_data_21r015']
    noise_types = ['none'] #, 'measurement', 'position']
    for file in files:
        for noise_type in noise_types:
            print(f"Cleaning data in file: {file}_{noise_type}_noise")
            clean_data(file, noise_type)
    
    print("Analysing results")
    analyse_results(files[0], files[1])
    plot_results(files[0])

    analyse_asterisms(files[0], files[1])

    # data_correlations(files[1])

    plt.show()

def clean_data(file, noise_type):

    df = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_noise.parquet'))

    def compute_asterism_features(row):
        a = [beacon.Beacon.from_tuple(row[f'star_{i}']) for i in range(3)]
        # b = [beacon.Beacon.from_tuple(row[f'star_{i}_baseline']) for i in range(3)]

        data = {
            'f_distance_0': a[0]._gamma_arcsec,
            'f_distance_1': a[1]._gamma_arcsec,
            'f_distance_2': a[2]._gamma_arcsec,
            'f_angle_0':    a[0].theta,
            'f_angle_1':    a[1].theta,
            'f_angle_2':    a[2].theta,
            'f_mean_distance': asterisms.ave_distance(a),
            'f_min_distance' : asterisms.min_distance(a),
            'f_max_distance' : asterisms.max_distance(a),
            'f_area' : asterisms.area(a),
            'f_centroid_distance' : asterisms.centroid_distance(a),
            'f_circumcentre_distance' : asterisms.circumcentre_distance(a),
            'f_circumcircle_radius' : asterisms.circumcircle_radius(a),
            'f_length_ratio' : asterisms.length_ratio(a),
            'f_average_side_length' : asterisms.average_side_length(a),
            'f_minimum_side_length' : asterisms.minimum_side_length(a),
            'f_maximum_side_length' : asterisms.maximum_side_length(a),
            'f_incentre_distance' : asterisms.incentre_distance(a),
            'f_incircle_radius' : asterisms.incircle_radius(a),
            'f_perimeter_area_ratio' : asterisms.perimeter_area_ratio(a),
            'f_closest_furthest_side_length' : asterisms.closest_furthest_side_length(a),
            'f_distance_to_closest_edge_obsolete' : asterisms.distance_to_closest_line(a),
            'f_perimeter' : asterisms.perimeter(a),
            'f_min_interior_angle' : asterisms.min_angle(a),
            'f_maximum_interior_angle' : asterisms.max_angle(a),
            'f_gaussian_area_within_triangle' : asterisms.gaussian_area_within_triangle(a),
            'f_overlap' : asterisms.overlap(a, ALTITUDES),
            'f_signed_distance_function' : asterisms.is_target_in_triangle(a),
            'f_distance_to_closest_line' : np.abs(asterisms.is_target_in_triangle(a)),
        }

        for i in range(3):
            data[f'star_{i}'] = row[f'star_{i}']

        return data
    
    ###
    print("Select correct noise profile results")
    ###
    df = df[(df['noise'] == noise_type)]
    
    ###
    print("Drop perturbation data")
    ###
    df = df[(df['perturbation_type'] == 'none')]
    perturbation_columns = df.columns[df.columns.str.contains('perturbation_')]
    df = df.drop(columns=perturbation_columns)
    df = df.drop(columns=['aperture_pixels'])


    # ###
    # print("Strehl ratio for different asterism and atmospheres for sanity checking")
    # ###
    # df = df[df['tomo_method'].isin(['fitting', 'open_loop', 'lsq_T', 'glao'])]
    # df = df[df['asterism_type'].str.contains('raw')]
    # id_factors = ['group_id', 'asterism_id', 'atmosphere_id', 'tomo_method']
    # agg_dict = {
    #     'strehl_ratio': 'mean'
    # }

    # df = df.groupby(id_factors)[['strehl_ratio']].agg(agg_dict).reset_index()
    # df.columns = ['_'.join(col).rstrip('_') if isinstance(col, tuple) else col for col in df.columns]
    # df = df.pivot(index=['atmosphere_id', 'group_id', 'asterism_id'], columns='tomo_method', values='strehl_ratio').reset_index()
    # df = df.sample(100000)
    # df.to_csv(os.path.join(DATA_FOLDER, file + f'_fitting_data.csv'), index=False)
    # quit()

    ###
    print("Drop unused results")
    ###
    result_columns = ['recon_rmse', 'strehl_ratio']
    # df = df.drop(columns='strehl_ratio')

    ###
    print("Calculate performance relative to baseline (open loop) performance")
    ###
    baseline_factors = ['group_id', 'asterism_id', 'atmosphere_id', 'asterism_type']
    baseline = df[df['tomo_method'] == 'open_loop'].copy()
    baseline = baseline.rename(columns={col: f"{col}_baseline" for col in result_columns})
    baseline = baseline.drop(columns=['tomo_method', 'star_0', 'star_1', 'star_2'])  # to allow joining on all ID columns
    df = df.merge(baseline, on=baseline_factors)

    for col in result_columns:
        df[f'{col}_rel'] = df[col] / df[f'{col}_baseline']
        df[f'{col}_rel_log'] = (df[col] / df[f'{col}_baseline']).apply(np.log)
        df[f'{col}_log'] = (df[col]).apply(np.log)

    result_col_rel = [f'{col}{t}' for col in result_columns for t in ['_rel', '_rel_log', '_log']]

    # TODO: Display plots of result columns in histograms

    ###
    print("Drop extraneous testing methods (open loop, AO, fitting error, etc)")
    ###


    df = df[df['tomo_method'].isin(METHODS)]

    ###
    print("Aggregate results along the atmosphere column, finding results for different numbers of atmospheres")
    ###
    id_factors = ['group_id', 'asterism_id', 'tomo_method', 'asterism_type'] #,'star_0', 'star_1', 'star_2']
    sample_ids = df['atmosphere_id'].dropna().unique()
    agged_dfs = []
    print(f"Total atmosphere IDs: {len(sample_ids)}")

    star_df = df[id_factors + ['star_0', 'star_1', 'star_2']]
    star_df = star_df.drop_duplicates(subset=id_factors)
        
    for n in ATM_SAMPLE_SIZES:
        print(f"Sampling for an atmosphere sample size of {n}")
        # Step 1: Sample from unique values in 'atmosphere_id' column
        try:
            samples = np.random.choice(sample_ids, size=n, replace=False)
        except ValueError:
            samples = df['atmosphere_id'].unique()

        # Step 2: Filter DataFrame based on sampled groups
        df0 = df[df['atmosphere_id'].isin(samples)]
        # df0 = df0.drop(columns=['star_0', 'star_1', 'star_2'])

        agg_dict = {
            col: ['mean', 'std', 'median', 'sem'] for col in result_col_rel + result_columns
        }

        df0 = df0.groupby(id_factors)[result_col_rel + result_columns].agg(agg_dict).reset_index()
        df0.columns = ['_'.join(col).rstrip('_') if isinstance(col, tuple) else col for col in df0.columns]
        df0['atmosphere_sample_size'] = n
        agged_dfs.append(df0)

    df = pd.concat(agged_dfs, ignore_index=True)
    df = df.merge(star_df, on=id_factors)
    
    ###
    print("Find unique asterisms")
    ###
    for i in range(3):
        df[f'star_{i}'] = df[f'star_{i}'].apply(tuple)

    ast_df = df[['group_id', 'asterism_id', 'asterism_type','star_0', 'star_1', 'star_2']].copy()
    ast_df = ast_df.drop_duplicates(subset=['star_0', 'star_1', 'star_2'])
    star_cols = [f'star_{i}' for i in range(3)]

    ###
    print("Compute features of unique asterisms")
    ###
    df0 = ast_df.apply(compute_asterism_features, axis=1)
    df0 = pd.DataFrame(df0.tolist(), index=df0.index)
    df = df.merge(df0, on=star_cols, how='inner')

    ###
    print("Group asterism types into different groups, and augmented status")
    ###
    asterism_group = {f'star_{i}_circular': 'Single Star Circular' for i in range(5)}
    augmented = {f'star_{i}_circular': 'Yes' for i in range(5)}
    for i in range(10):
        asterism_group[f'sg{i}_raw'] = 'Raw'
        asterism_group[f'sg{i}_equal_angle'] = 'Equal Angle'
        asterism_group[f'sg{i}_mean_dist'] = 'Mean Distance'
        asterism_group[f'sg{i}_mean_circular'] = 'Mean Circular'

        augmented[f'sg{i}_raw'] = 'No'
        augmented[f'sg{i}_equal_angle'] = 'Yes'
        augmented[f'sg{i}_mean_dist'] = 'Yes'
        augmented[f'sg{i}_mean_circular'] = 'Yes'

    df['asterism_group'] = df['asterism_type'].map(asterism_group)
    df['asterism_augmented'] = df['asterism_type'].map(augmented)

    df.to_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_noise_CLEAN.parquet'), index=False)
    df.to_csv(os.path.join(DATA_FOLDER, file + f'_{noise_type}_noise_CLEAN.csv'), index=False)

# Perform regression on a particular model
def run_reg(data, model, name, scale=False):
    ###
    print(f"Running regression on model: {name}")
    ###
    X_train, X_test, y_train, y_test = data

    star_cols = ['group_id', 'asterism_id']
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
        X_test = scaler.fit_transform(X_test)

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
        'Test Asterism ID': star_data['asterism_id_test'].iat[i],
        'Test Group ID': star_data['group_id_test'].iat[i],
        'Train Asterism ID': star_data['asterism_id_train'].iat[i],
        'Train Group ID': star_data['group_id_train'].iat[i]
        } for i in range(len(predictions))]
    print(f"Regression results: R²={r2}; RMSE={rmse}")

    return reg_data

def analyse_asterisms(file, validation_file):


    noise_type = 'none'
    df_r = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_prediction_results.parquet'))
    # df = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_noise_CLEAN.parquet'))
    v_df = pd.read_parquet(os.path.join(DATA_FOLDER, validation_file + f'_{noise_type}_noise_CLEAN.parquet'))

    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",  # or "lualatex", "xelatex"
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "figure.figsize": (3, 3),
    })

    # df['validation'] = False
    # v_df['validation'] = True
    TOP_N = 25

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

    for augmented in ['No']: # 'All', 
        if augmented != 'All':
            df_r1 = df_r[(df_r['Augmented'] == augmented) & (df_r['Method'] == 'lsq_T') & (df_r['Atmosphere Sample Size'] == ATM_SAMPLE_SIZES[-1])]
            # df1 = df[(df['asterism_augmented'] == augmented) & (df['tomo_method'] == 'lsq_T') & (df['atmosphere_sample_size'] == 512)]
            v_df1 = v_df[(v_df['asterism_augmented'] == augmented) & (v_df['tomo_method'] == 'lsq_T') & (v_df['atmosphere_sample_size'] == ATM_SAMPLE_SIZES[-1])]
        else:
            df_r1 = df_r[(df_r['Method'] == 'lsq_T') & (df_r['Atmosphere Sample Size'] == ATM_SAMPLE_SIZES[-1])]
            # df1 = df[(df['tomo_method'] == 'lsq_T') & (df['atmosphere_sample_size'] == 512)]
            v_df1 = v_df[(v_df['tomo_method'] == 'lsq_T') & (v_df['atmosphere_sample_size'] == ATM_SAMPLE_SIZES[-1])]

        min_distance = 0
        v_df1 = v_df1[(v_df1['f_distance_0'] >= min_distance) & (v_df1['f_distance_1'] >= min_distance) & (v_df1['f_distance_2'] >= min_distance)]
        v_df1 = v_df1[['group_id', 'asterism_id', 'star_0', 'star_1', 'star_2']]

        try:
            df_r1 = df_r1.sample(n_samples)
        except ValueError:
            pass

        # df0 = pd.concat([df1, v_df1])

        for test_var, ascending in test_variables.items():

            top_test_asterisms = df_r1[df_r1['Metric'] == test_var].sort_values(by='Test', ascending=ascending).drop_duplicates(subset='Test Group ID', keep='first')
            top_test_asterisms = top_test_asterisms.merge(v_df1, left_on=['Test Group ID', 'Test Asterism ID'], right_on=['group_id', 'asterism_id'], how='inner')
            top_test_asterisms = top_test_asterisms.head(TOP_N)
            top_test_asterisms = top_test_asterisms.drop(columns=['group_id', 'asterism_id'])

            bottom_test_asterisms = df_r1[df_r1['Metric'] == test_var].sort_values(by='Test', ascending=(not ascending)).drop_duplicates(subset='Test Group ID', keep='first')
            bottom_test_asterisms = bottom_test_asterisms.merge(v_df1, left_on=['Test Group ID', 'Test Asterism ID'], right_on=['group_id', 'asterism_id'], how='inner')
            bottom_test_asterisms = bottom_test_asterisms.head(TOP_N)
            bottom_test_asterisms = bottom_test_asterisms.drop(columns=['group_id', 'asterism_id'])

            top_pred_asterisms = df_r1[df_r1['Metric'] == test_var].sort_values(by='Predictions', ascending=ascending).drop_duplicates(subset='Test Group ID', keep='first')
            top_pred_asterisms = top_pred_asterisms.merge(v_df1, left_on=['Test Group ID', 'Test Asterism ID'], right_on=['group_id', 'asterism_id'], how='inner')
            top_pred_asterisms = top_pred_asterisms.head(TOP_N)
            top_pred_asterisms = top_pred_asterisms.drop(columns=['group_id', 'asterism_id'])

            bottom_pred_asterisms = df_r1[df_r1['Metric'] == test_var].sort_values(by='Predictions', ascending=(not ascending)).drop_duplicates(subset='Test Group ID', keep='first')
            bottom_pred_asterisms = bottom_pred_asterisms.merge(v_df1, left_on=['Test Group ID', 'Test Asterism ID'], right_on=['group_id', 'asterism_id'], how='inner')
            bottom_pred_asterisms = bottom_pred_asterisms.head(TOP_N)
            bottom_pred_asterisms = bottom_pred_asterisms.drop(columns=['group_id', 'asterism_id'])


            # mean_val = df0[test_var].mean()
            # mid_asterisms = df0.assign(diff=np.abs(df0[test_var] - mean_val)).sort_values(by='diff').head(TOP_N)

            def recreate_asterism(row):
                return [beacon.Beacon.from_tuple(row[f'star_{i}']) for i in range(3)]

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
                fig.savefig(f"./img/asterism_performance_{test_var}_{label}_min{min_distance}.pgf", bbox_inches='tight')
                fig.savefig(f"./img/asterism_performance_{test_var}_{label}_min{min_distance}.pdf", bbox_inches='tight')
                plt.close(fig)

    return

def test_psfs(asterism, label):

    os.makedirs('psf', exist_ok=True)

    from tomography import phase_screen, psf, tomography
    from basis import zernike
    D = 1.8
    A = [1000, 5000, 10000]
    ap = 256
    N = 21
    r0 = 0.25
    atm = phase_screen.Atmosphere(1, A, r0=r0, D=D, aperture_size=ap)

    target = beacon.Beacon(0, 0, 10)
    screen = target.total_cutout(D, A, atm[0], atm.metapupil_diameters)
    screen = screen * util.circle(ap)

    extent_m = (-D/2, D/2, -D/2, D/2)
    wavelength = 550e-9
    res = ap*wavelength/D
    extent_l = (-res/2, res/2, -res/2, res/2)

    fig, ax = plt.subplots()
    ax.imshow(screen, extent=extent_m)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_target_phase_screen.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_target_phase_screen.pdf", bbox_inches='tight')
    plt.close(fig)

    # a_psf = psf.psf(screen)
    # a_psf = np.log10(a_psf / a_psf.max())

    # fig, ax = plt.subplots()
    # ax.imshow(a_psf, vmin=-6, extent=extent_l)
    # ax.set_xlabel('Position (m)')
    # ax.set_ylabel('Position (m)')
    # fig.savefig(f"./img/psf/{label}_target_psf.pgf", bbox_inches='tight')
    # fig.savefig(f"./img/psf/{label}_target_psf.pdf", bbox_inches='tight')
    # plt.close(fig)

    z = zernike.Zernike(N)
    zcoeffs = z.decompose(screen)
    zscreen = z.compose(ap, zcoeffs)
    zresid = screen - zscreen

    fig, ax = plt.subplots()
    ax.imshow(zresid, extent=extent_m)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_target_fitting_residuals.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_target_fitting_residuals.pdf", bbox_inches='tight')
    plt.close(fig)

    z_psf = psf.psf(zresid)
    z_psf = np.log10(z_psf / z_psf.max())

    fig, ax = plt.subplots()
    ax.imshow(z_psf, vmin=-6, extent=extent_l)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_fitting_psf.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_fitting_psf.pdf", bbox_inches='tight')
    plt.close(fig)

    # plt.figure()
    flat = util.circle(ap)
    # plt.imshow(flat, extent=(-D/2, D/2, -D/2, D/2))

    f_psf = psf.psf(flat)
    f_psf = np.log10(f_psf / f_psf.max())

    fig, ax = plt.subplots()
    ax.imshow(f_psf, vmin=-6, extent=extent_l)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_airy_psf.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_airy_psf.pdf", bbox_inches='tight')
    plt.close(fig)

    lsq = tomography.LeastSquares(z, D, N, A, ap, r0=r0, method='Tinv')
    tcoeffs = tomography.get_coeffs_for_asterism(asterism, z, D, atm[0], A, atm.metapupil_diameters)
    
    lsq_coeffs = lsq.run(asterism, target, tcoeffs, ptt=zcoeffs[:3])
    lsq_recon = z.compose(ap, lsq_coeffs)

    fig, ax = plt.subplots()
    ax.imshow(lsq_recon, extent=extent_m)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_tomo_recon.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_tomo_recon.pdf", bbox_inches='tight')
    plt.close(fig)

    lsq_resid = screen - lsq_recon
    fig, ax = plt.subplots()
    ax.imshow(lsq_resid, extent=extent_m)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_tomo_residuals.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_tomo_residuals.pdf", bbox_inches='tight')
    plt.close(fig)

    lsq_psf = psf.psf(lsq_resid)
    lsq_psf = np.log10(lsq_psf / lsq_psf.max())
    
    fig, ax = plt.subplots()
    ax.imshow(lsq_psf, vmin=-6, extent=extent_l)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    fig.savefig(f"./img/psf/{label}_tomo_psf.pgf", bbox_inches='tight')
    fig.savefig(f"./img/psf/{label}_tomo_psf.pdf", bbox_inches='tight')
    plt.close(fig)

    return

def analyse_results(file, validation_file):
    noise_type = 'none'
    df = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_noise_CLEAN.parquet'))
    v_df = pd.read_parquet(os.path.join(DATA_FOLDER, validation_file + f'_{noise_type}_noise_CLEAN.parquet'))

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
        'Geometry Only': ['f_distance_0',
                     'f_distance_1',
                     'f_distance_2',
                     'f_angle_0',
                     'f_angle_1',
                     'f_angle_2'],
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
        'All Features':
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
            'f_overlap'
            ]
    }

    from torch import nn
    C = 10.0
    eps = 0.01

    models = {
        "Partial Least Squares": (PLSRegression(n_components=2), False),
        "Principal Component": (make_pipeline(PCA(n_components=2), LinearRegression()), False),
        "Ridge": (Ridge(alpha=0.1), False),
        "Lasso": (Lasso(alpha=0.1), False),
        # "SVR (Linear)": (SVR(kernel='linear', C=1.0, epsilon=0.1), False),
        "SVR (Poly)": (SVR(kernel='poly', C=C, epsilon=eps), False),
        "SVR (RBF)": (SVR(kernel='rbf', C=C, epsilon=eps), False),
        "SVR (Poly) [Scaled Inputs]": (SVR(kernel='poly', C=C, epsilon=eps), True),
        "SVR (RBF) [Scaled Inputs]": (SVR(kernel='rbf', C=C, epsilon=eps), True),
        # "sklearn Neural Network": (MLPRegressor(hidden_layer_sizes=[32,64,32], learning_rate_init=0.01, activation="relu"), True),
        "My Neural Network": (ml.regression.ML_Regressor(layer_dims=[32,64,32], learning_rate=0.01, weight_decay=1e-4, activation_func=nn.LeakyReLU), True),
    }

    star_columns = ["group_id", "asterism_id"]

    results = []

    for augmented in ['No', 'All']:
        if augmented != "All":
            df2 = df[df['asterism_augmented'] == augmented]
            # v_df2 = v_df[v_df['asterism_augmented'] == augmented]
        else:
            df2 = df
            # v_df2 = v_df
        for method in ['lsq_T']:
            df0 = df2[df2['tomo_method'] == method]
            v_df0 = v_df[v_df['tomo_method'] == method]
            for atmosphere_sample_size in [ATM_SAMPLE_SIZES[-1]]:
                df1 = df0[df0['atmosphere_sample_size'] == atmosphere_sample_size]
                v_df1 = v_df0[v_df0['atmosphere_sample_size'] == atmosphere_sample_size]
                for test_column in test_variables:
                    for feature_group, feature_columns in feature_groups.items(): 
                        # test_asterism_types = ['sg0_raw', 'sg1_raw', 'sg2_raw']
                        # df_test = df1[df1['asterism_type'].isin(test_asterism_types)]
                        # df_train = df1[~df1['asterism_type'].isin(test_asterism_types)]
                        df_train = df1
                        df_test = v_df1


                        # X = df1[feature_columns]
                        # y = df1[test_column]
                        # data = train_test_split(X, y, test_size=0.3)

                        X_test = df_test[feature_columns + star_columns]
                        X_train = df_train[feature_columns + star_columns]

                        y_test = df_test[test_column]
                        y_train = df_train[test_column]

                        # print(X_test.shape)

                        data = (X_train, X_test, y_train, y_test)

                        for name, (model, scale) in models.items():
                            label = f"{name}, {method}, {test_column}, {feature_group}, {atmosphere_sample_size}"
                            print(label)
                            test_results = run_reg(data, model, name, scale)
                            for tr in test_results:
                                tr['Method'] = method
                                tr['Atmosphere Sample Size'] = atmosphere_sample_size
                                tr['Metric'] = test_column
                                tr["Feature Group"] = feature_group
                                tr["Augmented"] = augmented
                            results += test_results

    print("Finishing analysis, displaying plots")
    df_r = pd.DataFrame(results)
    # df_rs = df_r.drop(columns=['Model Data'])
    df_r.to_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_prediction_results.parquet'))

    df_summary = df_r.drop(columns=['idx', 'Predictions', 'Test', 'Features', 'Test Asterism ID', 'Test Group ID', 'Train Asterism ID', 'Train Group ID'])
    df_summary = df_summary.drop_duplicates(subset=['Model', 'Atmosphere Sample Size', 'Metric', 'Feature Group', 'Augmented', 'Method'])
    
    df_summary.to_csv(os.path.join(DATA_FOLDER, file + f'_{noise_type}_R2_summary_results.csv'))

def plot_results(file):
    noise_type = 'none'

    df_r = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_prediction_results.parquet'))
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
        "recon_rmse_mean": rmse[0],
        "recon_rmse_log_mean": rmse[1],
        "recon_rmse_rel_mean": rmse[2],
        "recon_rmse_rel_log_mean": rmse[3],
        "strehl_ratio_mean": strehl[0],
        "strehl_ratio_log_mean": strehl[1],
        "strehl_ratio_rel_mean": strehl[2],
        "strehl_ratio_rel_log_mean": strehl[3],
        "Recon Rmse Mean": rmse[0],
        "Recon Rmse Log Mean": rmse[1],
        "Recon Rmse Rel Mean": rmse[2],
        "Recon Rmse Rel Log Mean": rmse[3],
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
        "My Neural Network": "Neural Network",
    })

    metrics = [m for m in df_r['Metric'].unique() if 'strehl' in m]

    for metric in metrics:
    # for metric in df_r['Metric'].unique():

        hist_of_test_variable = True
        for augmented in ['No']:
        # for augmented in df_r['Augmented'].unique():

            for method in df_r['Method'].unique():
                for sample_size in df_r['Atmosphere Sample Size'].unique():

                    for feature_group in ['Geometry Only', 'All without Overlap', 'All Features']:
                    # for feature_group in df_r['Feature Group'].unique():

                        for model in df_r['Model'].unique():

                            df_plot = df_r[(df_r['Method'] == method) 
                                            & (df_r['Metric'] == metric) 
                                            & (df_r['Atmosphere Sample Size'] == sample_size) 
                                            & (df_r['Feature Group'] == feature_group) 
                                            & (df_r['Augmented'] == augmented)
                                            & (df_r['Model'] == model)]
                            
                            fig_name = f"_{method}_{metric}_{model}_{feature_group}_{sample_size}_aug_{augmented}"
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
                                fig.savefig(f"./img/fig{fig_idx}_histogram_{fig_name}.pgf", bbox_inches='tight')
                                fig.savefig(f"./img/fig{fig_idx}_histogram_{fig_name}.pdf", bbox_inches='tight')
                                print(f"Figure {fig_idx}: Histogram of test variables for metric {metric}")
                                fig_idx += 1

                            fig, axs = plt.subplots(1, 1) #, figsize=(5, 5))
                            n_samples = 500
                            df_sampled = df_plot.sample(n_samples)
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
                            axs.set_ylabel(model)
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
                            print(f"Figure {fig_idx}: {method}, {metric}, {model}, {feature_group} and {sample_size}. Augmented: {augmented}")

                            fig.tight_layout()
                            fig.savefig(f"./img/fig{fig_idx}_{fig_name}.pgf", bbox_inches='tight')
                            fig.savefig(f"./img/fig{fig_idx}_{fig_name}.pdf", bbox_inches='tight')

                            fig_idx += 1
                            plt.close()
                            # plt.show()

    # for augmented in df_r['Augmented'].unique():
    #     for method in df_r['Method'].unique():
    #         for metric in df_r['Metric'].unique():
    #             for sample_size in df_r['Atmosphere Sample Size'].unique():
    #                 for model in df_r['Model'].unique():
    #                     df_slice = df_r[(df_r['Method'] == method) & (df_r['Metric'] == metric) & (df_r['Atmosphere Sample Size'] == sample_size) & (df_r['Model'] == model) & (df_r['Augmented'] == augmented)]
    #                     df_plot = df_slice.rename(columns={'Test': metric, 'Predictions': 'Model Prediction'})
                        
    #                     g = sns.lmplot(df_plot, x=metric, y='Model Prediction', col='Feature Group', col_wrap=2, palette='dark', markers='+', hue='Model')
    #                     print(f"Figure {fig_idx}: {method}, {metric}, {model} and {sample_size}. Augmented: {augmented}")
    #                     fig_idx += 1

    # for augmented in df_r['Augmented'].unique():
    #     for method in df_r['Method'].unique():
    #         for model in df_r['Model'].unique():
    #             for sample_size in df_r['Atmosphere Sample Size'].unique():
    #                 for feature_group in df_r['Feature Group'].unique():
    #                     df_slice = df_r[(df_r['Method'] == method) & (df_r['Model'] == model) & (df_r['Atmosphere Sample Size'] == sample_size) & (df_r['Feature Group'] == feature_group) & (df_r['Augmented'] == augmented)]
    #                     df_plot = df_slice.rename(columns={'Test': 'Test Value', 'Predictions': 'Model Prediction'})
                        
    #                     g = sns.lmplot(df_plot, x='Test Value', y='Model Prediction', col='Metric', palette='dark', markers='+', hue='Model', sharey=False)
    #                     print(f"Figure {fig_idx}: {method}, {model}, {feature_group} and {sample_size}. Augmented: {augmented}")
    #                     fig_idx += 1

                        # for model in df_plot['Model'].unique():
                        #     print(df_summary[['Model', 'Feature Group', 'R²', 'RMSE']][:1])

    return

 
def data_correlations(file, noise_type='none'):

    test_variables = [
        "recon_rmse_mean",
        "recon_rmse_log_mean",
        "recon_rmse_rel_mean",
        "recon_rmse_rel_log_mean",
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

    df = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_noise_CLEAN.parquet'))
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

    correlations.to_csv(os.path.join(DATA_FOLDER, file + f'_{noise_type}_correlations.csv'))
    print(correlations)

    fig.savefig(f"./img/correlations_heatmap.pgf", bbox_inches='tight')
    fig.savefig(f"./img/correlations_heatmap.pdf", bbox_inches='tight')

    # Find prediction results for best model
    # df_v = pd.read_parquet(os.path.join(DATA_FOLDER, validation_file + '_CLEAN.parquet'))

    # TODO: Predict results for some set asterisms.
    # Line, circle, off-centre cluster, half-circle

def test_overlap_percentage():
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
            patch.set_facecolor((0.2, 0.74, 0.85, 0.3))
            patch.set_edgecolor((0.2, 0.74, 0.85, 0.0))

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
            patch.set_edgecolor((0.2, 0.74, 0.85, 1.0))

        # Plot the target edges
        patch = plotting.plot_polygon(C4, ax=ax, color='k', linewidth=1.0, add_points=False)
        patch.set_facecolor((0.0, 0.0, 0.0, 0.0))


        ax.set_aspect('equal')
        ax.set_xlabel('Position (m)')
        if i == 0:
            ax.set_ylabel('Position (m)')
        ax.set_title(f'Layer at {alt}m')
        
        # compute ratio
        overlaps.append(overlap.area / C4.area)

    print(overlaps)
    print(np.mean(overlaps))

    fig.tight_layout()
    fig.savefig("./img/overlap.pgf", bbox_inches='tight')
    fig.savefig("./img/overlap.pdf", bbox_inches='tight')
    # plt.show()