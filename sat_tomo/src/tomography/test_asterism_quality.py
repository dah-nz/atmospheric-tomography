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

DATA_FOLDER = 'C:/Code/PhD/Data'

METHODS = ['glao', 'lsq_T', 'lsq_K']

ALTITUDES = [1000, 5000, 10000]
ATM_SAMPLE_SIZES = [32, 64, 128, 256, 512]

def run_tests():

    files = ['testing_data', 'validation_data']
    noise_types = ['none', 'measurement', 'position']
    for file in files:
        for noise_type in noise_types:
            print(f"Cleaning data in file: {file}_{noise_type}_noise")
            clean_data(file, noise_type)
    
    # print("Analysing results")
    # analyse_results(files[0]) #, files[1])

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
            # 'f_circular_similarity': asterisms.circular_similarity(a, b),
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

def analyse_results(file): #, validation_file):
    df = pd.read_parquet(os.path.join(DATA_FOLDER, file + '_CLEAN.parquet'))

    # Find regression models for different sets of features
    test_variables = [
        "recon_rmse_mean",
        # "recon_rmse_median",
        # "recon_rmse_rel_mean",
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
        # "SVM (Linear)": SVR(kernel='linear', C=1.0, epsilon=0.1),
        "SVM (Poly)": SVR(kernel='poly', C=1.0, epsilon=0.1),
        "SVM (RBF)": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        # "SVM (Sigmoid)": SVR(kernel='sigmoid', C=1.0, epsilon=0.1),
        "Neural Network": ml.regression.ML_Regressor(layer_dims=[32,64,32], learning_rate=0.01, weight_decay=1e-4),
    }

    results = []

    for method in ['lsq_T']:
        df0 = df[df['tomo_method'] == method]
        for atmosphere_sample_size in [512]:
            df1 = df0[df0['atmosphere_sample_size'] == atmosphere_sample_size]
            for test_column in test_variables:
                for feature_group, feature_columns in feature_groups.items():
                    test_asterism_types = ['sg0_raw', 'sg1_raw', 'sg2_raw']
                    df_test = df1[df1['asterism_type'].isin(test_asterism_types)]
                    df_train = df1[~df1['asterism_type'].isin(test_asterism_types)]

                    # X = df1[feature_columns]
                    # y = df1[test_column]
                    # data = train_test_split(X, y, test_size=0.3)

                    X_test = df_test[feature_columns]
                    X_train = df_train[feature_columns]

                    y_test = df_test[test_column]
                    y_train = df_train[test_column]

                    print(X_test.shape)

                    data = (X_train, X_test, y_train, y_test)

                    for name, model in models.items():
                        print(f"Regression test for model: {name}, Method: {method}, Metric: {test_column}, Feature Group: {feature_group}, Sample Size: {atmosphere_sample_size}")
                        test_results = run_reg(data, model, name)
                        for tr in test_results:
                            tr['Method'] = method
                            tr['Atmosphere Sample Size'] = atmosphere_sample_size
                            tr['Metric'] = test_column.replace('_', " ").title()
                            tr["Feature Group"] = feature_group
                        results += test_results

    print("Finishing analysis, displaying plots")
    df_r = pd.DataFrame(results)
    fig_idx = 1

    for method in df_r['Method'].unique():
        for metric in df_r['Metric'].unique():
            for sample_size in df_r['Atmosphere Sample Size'].unique():
                for feature_group in df_r['Feature Group'].unique():
                    df_plot = df_r[(df_r['Method'] == method) & (df_r['Metric'] == metric) & (df_r['Atmosphere Sample Size'] == sample_size) & (df_r['Feature Group'] == feature_group)]
                    
                    g = sns.lmplot(df_plot, x='Test', y='Predictions', col='Model', col_wrap=2)
                    print(f"Figure {fig_idx}: {method}, {metric}, {feature_group} and {sample_size}")
                    fig_idx += 1
                
                    for model in df_plot['Model'].unique():
                        df_summary = df_plot[(df_plot['Model'] == model)] 
                        print(df_summary[['Model', 'Feature Group', 'R²', 'RMSE']][:1])

    
    # Find prediction results for best model
    # df_v = pd.read_parquet(os.path.join(DATA_FOLDER, validation_file + '_CLEAN.parquet'))

    # TODO: Predict results for some set asterisms.
    # Line, circle, off-centre cluster, half-circle

