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
from sklearn.svm import SVR

import util

DATA_FOLDER = 'C:/Code/PhD/Data'

METHODS = ['glao', 'lsq_T', 'lsq_K']

ALTITUDES = [1000, 5000, 10000]
ATM_SAMPLE_SIZES = [32, 64, 128, 256, 512]

def run_tests():

    files = ['testing_data', 'validation_data']
    noise_types = ['none'] #, 'measurement', 'position']
    # for file in files:
    #     for noise_type in noise_types:
    #         print(f"Cleaning data in file: {file}_{noise_type}_noise")
    #         clean_data(file, noise_type)
    
    print("Analysing results")
    analyse_results(files[0], files[1])

    analyse_asterisms(files[0], files[1])

    data_correlations(files[1])

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
            'f_distance_to_closest_edge' : asterisms.distance_to_closest_line(a),
            'f_perimeter' : asterisms.perimeter(a),
            'f_min_interior_angle' : asterisms.min_angle(a),
            'f_maximum_interior_angle' : asterisms.max_angle(a),
            'f_gaussian_area_within_triangle' : asterisms.gaussian_area_within_triangle(a),
            'f_overlap' : asterisms.overlap(a, ALTITUDES),
            'f_target_in_triangle' : asterisms.is_target_in_triangle(a),
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

    ###
    print("Drop unused results")
    ###
    result_columns = ['recon_rmse', 'coeff_rmse', 'strehl_ratio']
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

    result_col_rel = [f'{col}_rel{t}' for col in result_columns for t in ['', '_log']]

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
        samples = np.random.choice(sample_ids, size=n, replace=False)

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
def run_reg(data, model, name):
    ###
    print(f"Running regression on model: {name}")
    ###
    X_train, X_test, y_train, y_test = data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).flatten()

    predictions = list(y_pred)
    tests = list(y_test)

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    reg_data = [{
        'Model': name,
        'Model Data': model,
        'R²': r2,
        'RMSE': rmse,
        'Features': list(X_train.columns),
        'idx': i,
        'Predictions': predictions[i],
        'Test': tests[i] }
        for i in range(len(predictions))]
    print(f"Regression results: R²={r2}; RMSE={rmse}")

    return reg_data

def analyse_asterisms(file, validation_file):
    noise_type = 'none'
    df = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_noise_CLEAN.parquet'))
    v_df = pd.read_parquet(os.path.join(DATA_FOLDER, validation_file + f'_{noise_type}_noise_CLEAN.parquet'))
    df['validation'] = False
    v_df['validation'] = True
    TOP_N = 25

    test_variables = {
        "recon_rmse_mean": True,
        "strehl_ratio_rel_log_mean": False
    }

    for augmented in ['Yes', 'No']:
        df1 = df[(df['asterism_augmented'] == augmented) & (df['tomo_method'] == 'lsq_T') & (df['atmosphere_sample_size'] == 512)]
        v_df1 = v_df[(v_df['asterism_augmented'] == augmented) & (v_df['tomo_method'] == 'lsq_T') & (v_df['atmosphere_sample_size'] == 512)]

        df0 = pd.concat([df1, v_df1])

        for test_var, ascending in test_variables.items():

            top_asterisms = df0.sort_values(by=test_var, ascending=ascending).drop_duplicates(subset='group_id', keep='first').head(TOP_N)
            bottom_asterisms = df0.sort_values(by=test_var, ascending=(not ascending)).drop_duplicates(subset='group_id', keep='first').head(TOP_N)

            mean_val = df0[test_var].mean()
            mid_asterisms = df0.assign(diff=np.abs(df0[test_var] - mean_val)).sort_values(by='diff').head(TOP_N)

            def recreate_asterism(row):
                return [beacon.Beacon.from_tuple(row[f'star_{i}']) for i in range(3)]

            def plot_circ(ax):
                theta = np.linspace(0, 2*np.pi, 100, endpoint=True)
                rho = np.ones_like(theta) * 120
                x, y = util.pol2cart(rho, theta)
                ax.plot(x, y, "k--", alpha=0.2)

            fig, axs = plt.subplots(1, 3)
            fig.suptitle(f"Asterism Performance: {test_var}")
            good_asterisms = top_asterisms.apply(recreate_asterism, axis=1)

            axs[0].plot([0], [0], 'ko')
            for a in good_asterisms:
                x = [b.position_arcsec()[0] for b in a]
                y = [b.position_arcsec()[1] for b in a]
                axs[0].plot(x, y, '+:', alpha=0.5)

            plot_circ(axs[0])
            axs[0].set_aspect('equal')
            axs[0].set_title("Best Asterisms")

            worst_asterisms = bottom_asterisms.apply(recreate_asterism, axis=1)

            axs[1].plot([0], [0], 'ko')
            for a in worst_asterisms:
                x = [b.position_arcsec()[0] for b in a]
                y = [b.position_arcsec()[1] for b in a]
                axs[1].plot(x, y, '+:', alpha=0.5)

            plot_circ(axs[1])
            axs[1].set_aspect('equal')
            axs[1].set_title("Worst Asterisms")

            mean_asterisms = mid_asterisms.apply(recreate_asterism, axis=1)

            axs[2].plot([0], [0], 'ko')
            for a in mean_asterisms:
                x = [b.position_arcsec()[0] for b in a]
                y = [b.position_arcsec()[1] for b in a]
                axs[2].plot(x, y, '+:', alpha=0.5)

            plot_circ(axs[2])
            axs[2].set_aspect('equal')
            axs[2].set_title("Average Asterisms")
    
    return


def analyse_results(file, validation_file):
    noise_type = 'none'
    df = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_noise_CLEAN.parquet'))
    v_df = pd.read_parquet(os.path.join(DATA_FOLDER, validation_file + f'_{noise_type}_noise_CLEAN.parquet'))

    # Find regression models for different sets of features
    test_variables = [
        "recon_rmse_mean",
        "recon_rmse_rel_log_mean",
        "strehl_ratio_rel_log_mean",
        "strehl_ratio_mean",
        "recon_rmse_rel_mean",
        "strehl_ratio_rel_mean",
        # "recon_rmse_median",
        # "recon_rmse_rel_median",
        # "recon_rmse_rel_log_mean",
        # "recon_rmse_rel_log_median",
    ]

    feature_groups = {
        'Geometry Only': ['f_distance_0',
                     'f_distance_1',
                     'f_distance_2',
                     'f_angle_0',
                     'f_angle_1',
                     'f_angle_2'],
        'Geometry + Overlap': 
                    ['f_distance_0',
                     'f_distance_1',
                     'f_distance_2',
                     'f_angle_0',
                     'f_angle_1',
                     'f_angle_2',
                     'f_overlap']
    }

    models = {
        "Partial Least Squares": PLSRegression(n_components=2),
        # "Principal Component": make_pipeline(PCA(n_components=2), LinearRegression()),
        "Ridge": Ridge(alpha=0.1),
        "Lasso": Lasso(alpha=0.1),
        # "SVR (Linear)": SVR(kernel='linear', C=1.0, epsilon=0.1),
        "SVR (Poly)": SVR(kernel='poly', C=1.0, epsilon=0.1),
        "SVR (RBF)": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        # "SVR (Sigmoid)": SVR(kernel='sigmoid', C=1.0, epsilon=0.1),
        "Neural Network": ml.regression.ML_Regressor(layer_dims=[32,64,32], learning_rate=0.01, weight_decay=1e-4),
    }

    results = []

    for augmented in ['All', 'No']:
        if augmented != "All":
            df2 = df[df['asterism_augmented'] == augmented]
            # v_df2 = v_df[v_df['asterism_augmented'] == augmented]
        else:
            df2 = df
            # v_df2 = v_df
        for method in ['lsq_T']:
            df0 = df2[df2['tomo_method'] == method]
            v_df0 = v_df[v_df['tomo_method'] == method]
            for atmosphere_sample_size in [512]:
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

                        X_test = df_test[feature_columns]
                        X_train = df_train[feature_columns]

                        y_test = df_test[test_column]
                        y_train = df_train[test_column]

                        # print(X_test.shape)

                        data = (X_train, X_test, y_train, y_test)

                        for name, model in models.items():
                            label = f"{name}, {method}, {test_column}, {feature_group}, {atmosphere_sample_size}"
                            print(label)
                            test_results = run_reg(data, model, name)
                            for tr in test_results:
                                tr['Method'] = method
                                tr['Atmosphere Sample Size'] = atmosphere_sample_size
                                tr['Metric'] = test_column.replace('_', " ").title()
                                tr["Feature Group"] = feature_group
                                tr["Augmented"] = augmented
                            results += test_results

    print("Finishing analysis, displaying plots")
    df_r = pd.DataFrame(results)
    fig_idx = 1
    df_summary = pd.DataFrame()

    for augmented in df_r['Augmented'].unique():
        for method in df_r['Method'].unique():
            for metric in df_r['Metric'].unique():
                for sample_size in df_r['Atmosphere Sample Size'].unique():
                    for feature_group in df_r['Feature Group'].unique():
                        df_slice = df_r[(df_r['Method'] == method) & (df_r['Metric'] == metric) & (df_r['Atmosphere Sample Size'] == sample_size) & (df_r['Feature Group'] == feature_group) & (df_r['Augmented'] == augmented)]
                        df_plot = df_slice.rename(columns={'Test': metric, 'Predictions': 'Model Prediction'})
                        
                        g = sns.lmplot(df_plot, x=metric, y='Model Prediction', col='Model', col_wrap=2, palette='dark', markers='+')
                        print(f"Figure {fig_idx}: {method}, {metric}, {feature_group} and {sample_size}. Augmented: {augmented}")
                        fig_idx += 1

                        
                    
                        df_summary = pd.concat([df_summary, df_slice], ignore_index=True)
                        # for model in df_plot['Model'].unique():
                        #     print(df_summary[['Model', 'Feature Group', 'R²', 'RMSE']][:1])
    
    df_summary = df_summary.drop(columns=['idx', 'Predictions', 'Test', 'Features', 'Model Data'])
    df_summary = df_summary.drop_duplicates(subset=['Model', 'Atmosphere Sample Size', 'Metric', 'Feature Group', 'Augmented'])
    
    df_summary.to_csv(os.path.join(DATA_FOLDER, file + f'_{noise_type}_R2_results.csv'))
    print(df_summary)

    return

 
def data_correlations(file, noise_type='none'):

    test_variables = [
        "recon_rmse_mean",
        "recon_rmse_rel_mean",
        "strehl_ratio_rel_log_mean"
    ]

    df = pd.read_parquet(os.path.join(DATA_FOLDER, file + f'_{noise_type}_noise_CLEAN.parquet'))
    df0 = df.filter(regex='^f_.*')
    feature_columns = list(df0.columns)

    plt.figure()

    correlations = df[test_variables + feature_columns].corr().loc[feature_columns, test_variables]
    sns.heatmap(correlations, annot=True)

    correlations.to_csv(os.path.join(DATA_FOLDER, file + f'_{noise_type}_correlations.csv'))
    print(correlations)

    # Find prediction results for best model
    # df_v = pd.read_parquet(os.path.join(DATA_FOLDER, validation_file + '_CLEAN.parquet'))

    # TODO: Predict results for some set asterisms.
    # Line, circle, off-centre cluster, half-circle
