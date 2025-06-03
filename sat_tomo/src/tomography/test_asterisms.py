import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
import time
import traceback

import pandas as pd
import seaborn as sns

from tomography import asterisms, beacon, phase_screen
from tomography.tomography import GLAO, LeastSquares, get_coeffs_for_asterism, AO
from basis import zernike
import util
import joblib

def run_tests():

    # test_asterism_generation()
    # test_phase_variance()
    
    f = Filter()
    f.run_asterism_comparison()
    f.filter_asterisms()
    # load_angle_data_test()

    plt.show()


def load_angle_data_test():
    with open(f'data/satellites/angle_data_mag6.pak', 'rb') as f:
        data = joblib.load(f)

    print(data[:5])

    with open(f'data/satellites/angle_data_filtered.pak', 'rb') as f:
        data = joblib.load(f)

    print(data[:5])

class Filter():
    def __init__(self):
        self.n_asterisms = 400
        self.n_atm = 10

        self.N = 21
        self.Z = zernike.Zernike(self.N)
        self.D = 1.8
        self.r0 = 0.25
        self.ap_pix = 100
        self.alts = [1000, 5000, 10000]
        self.glao = GLAO(self.Z, self.D, self.N, self.alts, self.ap_pix)
        self.ao = AO(self.Z, self.D, self.N, self.alts, self.ap_pix)
        self.lsq = LeastSquares(self.Z, self.D, self.N, self.alts, self.ap_pix, r0=self.r0)
        self.mask = util.circle(self.ap_pix)
        self.target = beacon.Beacon(0, 0, 10, aperture_mask=self.mask)
        self.total = -1

        self.tomo_methods = ['glao', 'tiptilt', 'ao', 'lsq', 'ol', 'ideal']
        self.metrics = ['coeff_rmse', 'recon_rmse', 'strehl']
        self.statistics = ['mean', 'std', 'min', 'lq', 'med', 'uq', 'max']

    def worker(self, atm, aq, rq, l):
        try:
            tstart = time.time()
            id = os.getpid()
            l.acquire()
            while not aq.empty():
                i, asterism = aq.get()
                l.release()
                t0 = time.time()

                results = {
                    t: {m: [] for m in self.metrics} for t in self.tomo_methods
                }

                stats = {
                    t: {m: {} for m in self.metrics} for t in self.tomo_methods
                }
                
                for screens in atm:
                    target_coeffs = self.target.cutout_coeffs(self.Z, self.D, self.alts, screens, atm.metapupil_diameters)
                    target_screen = self.target.total_cutout(self.D, self.alts, screens, atm.metapupil_diameters)
                    ast_coeffs = get_coeffs_for_asterism(asterism, self.Z, self.D, screens, self.alts, atm.metapupil_diameters)

                    for tomo in self.tomo_methods:
                        match tomo:
                            case 'glao':
                                coeffs = self.glao.run(asterism, self.target, ast_coeffs)
                            case 'tiptilt':
                                coeffs = np.zeros_like(target_coeffs)
                                coeffs[:3] = target_coeffs[:3]
                            case 'ao':
                                coeffs = self.ao.run(asterism, self.target, ast_coeffs)
                            case 'lsq':
                                coeffs = self.lsq.run(asterism, self.target, ast_coeffs)
                            case 'ideal':
                                coeffs = target_coeffs
                            case _:
                                coeffs = np.zeros_like(target_coeffs)

                        for metric in self.metrics:
                            match metric:
                                case 'recon_rmse':
                                    recon = self.Z.compose(target_screen.shape, coeffs)
                                    result = util.rmse(recon, target_screen, mask=self.mask)
                                case 'coeff_rmse':
                                    result = util.rmse(coeffs, target_coeffs)
                                case 'strehl':
                                    recon = self.Z.compose(target_screen.shape, coeffs)
                                    # fig, axs = plt.subplots(1, 3)
                                    # fig.suptitle(tomo)
                                    r = recon - target_screen
                                    # axs[0].imshow(recon)
                                    # axs[1].imshow(target_screen)
                                    # axs[2].imshow(r)
                                    var = np.var(r, where=self.mask > 0)
                                    print(f"{tomo}\n\
                                          {target_coeffs}\n{coeffs}\n\
                                          target r0: {phase_screen.calculate_r0(target_screen, self.D)}, recon r0: {phase_screen.calculate_r0(recon, self.D)}, diff r0: {phase_screen.calculate_r0(r, self.D)}\n\
                                          Variance: {var}, Strehl: {np.exp(-var)}")
                                    result = np.exp(-var)
                                    # plt.show()
                                case _:
                                    result = np.nan
                            results[tomo][metric].append(result)

                for tomo in self.tomo_methods:
                    for metric in self.metrics:
                        for statistic in self.statistics:
                            match statistic:
                                case 'std':
                                    r = np.std(results[tomo][metric])
                                case 'mean':
                                    r = np.mean(results[tomo][metric])
                                case 'min':
                                    r = np.amin(results[tomo][metric])
                                case 'lq':
                                    r = np.percentile(results[tomo][metric], 25)
                                case 'med':
                                    r = np.median(results[tomo][metric])
                                case 'uq':
                                    r = np.percentile(results[tomo][metric], 75)
                                case 'max':
                                    r = np.amax(results[tomo][metric])

                            stats[tomo][metric][statistic] = r
                
                rq.put((i, stats))

                t1 = time.time()
                dt = t1 - t0
                dt_total = t1 - tstart 
                print(f"[{id:5}] Simulated asterism {i+1}/{self.total}. Took {dt:5.2f}s. Total time since start: {dt_total:.2f}s")
                l.acquire()
            l.release()
        except Exception as e:
            print(type(e))    # the exception type
            print(e.args)     # arguments stored in .args
            print(e)
            print(traceback.format_exc(), flush=True)
        return

    def run_asterism_comparison(self):

        # all_asterisms = asterisms.generate_variable_asterisms(n=self.n_asterisms, zero_rate=0.0, fixed_rate=0.0)
        # fixed_asterisms = asterisms.generate_fixed_asterisms(self.n_asterisms, 5, theta_0_random=True)
        # # for angle in range(10, 120, 5):
        # #     fixed_asterisms += asterisms.generate_fixed_asterisms(10, angle, theta_0_random=True)
        # all_asterisms += fixed_asterisms
        all_asterisms = asterisms.get_test_asterisms(self.n_asterisms, fixed_rate=0.1, uniform_rate=0.2, offset_rate=0.5)
        self.total = len(all_asterisms)


        print(f"Generating {self.n_atm} simulated atmospheres")
        atm = phase_screen.Atmosphere(self.n_atm, self.alts, r0=self.r0, D=self.D, aperture_size=self.ap_pix)
        print(f"Completed simulating {self.n_atm} atmospheres")

        manager = Manager()
        asterism_queue = manager.Queue()
        result_queue = manager.Queue()
        lock = manager.Lock()

        for i, asterism in tqdm(enumerate(all_asterisms)):
            asterism_queue.put((i, asterism))

        n_processes = 8
        print("Run multi-processing for asterism quality filtering")

        with Pool(processes=n_processes) as p:
            workers = [p.apply_async(self.worker, args=(atm, asterism_queue, result_queue, lock)) for _ in range(n_processes)]

            res = [p.get() for p in workers]

        print("Multi-processing completed")

        results = []

        while not result_queue.empty():
            j, stats = result_queue.get()

            if j % 100 == 0:
                print(f"Loading results for asterism {j+1}/{self.total}")
            asterism = all_asterisms[j]

            r_dict = {
                'asterism_id':  j,
                'gs_1':         asterism[0].to_simple_str(),
                'gs_2':         asterism[1].to_simple_str(),
                'gs_3':         asterism[2].to_simple_str()
            }
            for t in self.tomo_methods:
                for m in self.metrics:
                    for s in self.statistics:
                        r_dict[f'{t}_{m}_{s}'] = stats[t][m][s]
                    r_dict[f'{t}_{m}_iqr'] = stats[t][m]['uq'] - stats[t][m]['lq']

            results.append(r_dict)

        df = pd.DataFrame(results)
        df.to_csv('data/satellites/asterism_quality_data.csv')

    def filter_asterisms(self):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
        
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.pipeline import make_pipeline
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.svm import SVC, SVR
        from sklearn.manifold import TSNE
        import umap

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader

        df = pd.read_csv('data/satellites/asterism_quality_data.csv', index_col="asterism_id")

        asterism_list = [[beacon.Beacon.from_simple_str(df.loc[idx, f'gs_{i+1}']) for i in range(3)] for idx in range(df.shape[0])]
        distance_0 = [a[0]._gamma_arcsec for a in asterism_list]
        distance_1 = [a[1]._gamma_arcsec for a in asterism_list]
        distance_2 = [a[2]._gamma_arcsec for a in asterism_list]
        angle_01 = [a[1].theta - a[0].theta for a in asterism_list]
        angle_12 = [a[2].theta - a[1].theta for a in asterism_list]
        
        tomo_method = 'lsq'
        metric = 'recon_rmse'
        statistic = 'mean'


        def norm(a):
            # ZScore normalisation
            a = np.array(a)
            return (a - a.mean()) / a.std()
        
        performance = df[f'{tomo_method}_{metric}_{statistic}']

        # performance = (0.8 * df['ol_mse']) - df['lsq_mse']
        # performance = (performance - performance.mean()) / performance.std()

        df0 = pd.DataFrame()

        df0['performance'] = performance

        df0['distance_0'] = norm(distance_0)
        df0['distance_1'] = norm(distance_1)
        df0['distance_2'] = norm(distance_2)
        df0['angle_01'] =   norm(angle_01)
        df0['angle_12'] =   norm(angle_12)

        df0['ave_distance'] =          norm([asterisms.ave_distance(a) for a in asterism_list])
        df0['centroid_distance'] =     norm([asterisms.centroid_distance(a) for a in asterism_list])
        df0['circumcentre_distance'] = norm([asterisms.circumcentre_distance(a) for a in asterism_list])
        df0['circumcircle_radius'] =   norm([asterisms.circumcircle_radius(a) for a in asterism_list])
        df0['length_ratio'] =          norm([asterisms.length_ratio(a) for a in asterism_list])

        df0['overlap'] =                      norm([asterisms.overlap(a, [1000, 5000, 10000]) for a in asterism_list])
        df0['closest_furthest_side_length'] = norm([asterisms.closest_furthest_side_length(a) for a in asterism_list])
        df0['ave_side_length'] =              norm([asterisms.average_side_length(a) for a in asterism_list])
        df0['min_side_length'] =              norm([asterisms.minimum_side_length(a) for a in asterism_list])
        df0['max_side_length'] =              norm([asterisms.maximum_side_length(a) for a in asterism_list])

        df0['distance_to_closest_line'] = norm([asterisms.distance_to_closest_line(a) for a in asterism_list])
        df0['incentre_distance'] =        norm([asterisms.incentre_distance(a) for a in asterism_list])
        df0['incircle_radius'] =          norm([asterisms.incircle_radius(a) for a in asterism_list])
        df0['perimeter_area_ratio'] =     norm([asterisms.perimeter_area_ratio(a) for a in asterism_list])
        df0['area'] =                     norm([asterisms.area(a) for a in asterism_list])

        df0['perimeter'] =     norm([asterisms.perimeter(a) for a in asterism_list])
        df0['in_triangle'] =   norm([asterisms.is_target_in_triangle(a) for a in asterism_list])
        df0['min_angle'] =     norm([asterisms.min_angle(a) for a in asterism_list])
        df0['max_angle'] =     norm([asterisms.max_angle(a) for a in asterism_list])
        df0['gaussian_area'] = norm([asterisms.gaussian_area_within_triangle(a) for a in asterism_list])

        named_columns = [col for col in df0.columns if not col.startswith("Unnamed")]
        num_named_columns = len(named_columns)
        print(f"Number of named columns: {num_named_columns}")
        n_features = num_named_columns - 1

        df0 = df0.sample(frac=0.2)

        # X = df0.drop(columns='performance')
        # y = df0['performance']

        X = df0.drop(columns='performance')
        y = df0['performance']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_c = df0.drop(columns='performance')
        y_c = (df0['performance'] < 0.5 * (df[f'ol_{metric}_{statistic}'].mean())).astype(int)
        # y_c = (df0['performance'] > 0.0).astype(int)
        X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_c, y_c, test_size=0.2)

        
        print("Calculating 2D UMAP")
        plt.figure()
        umap_reducer = umap.UMAP(n_components=2)
        X_umap = umap_reducer.fit_transform(X_c)
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_c, cmap='viridis')
        plt.title("UMAP 2D")

        print("Calculating 3D UMAP")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        umap_reducer = umap.UMAP(n_components=3)
        X_umap = umap_reducer.fit_transform(X)
        ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], c=y, cmap='viridis')
        fig.suptitle("UMAP 3D")



        print("Calculating 2D t-SNE")
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X_c)

        plt.figure()
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_c, cmap='viridis')
        plt.title("t-SNE 2D")

        print("Calculating 3D t-SNE")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        tsne2 = TSNE(n_components=3)
        X_tsne2 = tsne2.fit_transform(X)
        ax.scatter(X_tsne2[:, 0], X_tsne2[:, 1], X_tsne2[:, 2], c=y, cmap='viridis')
        fig.suptitle("t-SNE 3D")

        # Display higher dimensional data as 2D using Principal Component Analysis
        print("Calculating 2D PCA")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_c)

        plt.figure()
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_c, cmap='viridis')
        plt.title("PCA 2D")

        print("Calculating 3D PCA")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        PCA_model = PCA(n_components=3).fit(X)
        X_reduced = PCA_model.transform(X)
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='viridis')
        fig.suptitle("PCA 3D")

        # Show correlations between variables
        plt.figure()
        correlations = df0.corr()
        sns.heatmap(correlations, annot=True)

        def regression(X_train, X_test, y_train, y_test, model, name):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test).flatten()
        
            df = pd.DataFrame()
            df['prediction'] = y_pred
            df['test'] = y_test.to_numpy()

            print(name)
            print("R² score:", r2_score(y_test, y_pred))
            print("MSE:", mean_squared_error(y_test, y_pred))
            print("\n")

            sns.lmplot(data=df, x='test', y='prediction')
            plt.plot([df['test'].min(), df['test'].max()], [df['test'].min(), df['test'].max()], 'r:')
            plt.title(name)

        def classification(X_c_train, X_c_test, y_c_train, y_c_test, model, name):
            model.fit(X_c_train, y_c_train)

            y_pred = model.predict(X_c_test)
            
            df = pd.DataFrame()
            df['prediction'] = y_pred
            df['test'] = y_c_test.to_numpy()

            sns.lmplot(data=df, x='test', y='prediction')
            plt.plot([df['test'].min(), df['test'].max()], [df['test'].min(), df['test'].max()], 'r:')
            plt.title(name)

            accuracy = accuracy_score(df['test'], df['prediction'])
            precision = precision_score(df['test'], df['prediction'])
            recall = recall_score(df['test'], df['prediction'])

            print(name)
            print(f"Accuracy:  {accuracy:.2f}      Proportion of correct predictions (both positives and negatives.)")
            print(f"Precision: {precision:.2f}      TP / (TP + FP) — How many predicted positives are truly positive?")
            print(f"Recall:    {recall:.2f}      TP / (TP + FN) — How many actual positives were correctly predicted?")

            cm = confusion_matrix(df['test'], df['prediction'])
            print("Confusion Matrix:\n", cm)

            # F1 Score
            f1 = f1_score(df['test'], df['prediction'])
            print(f"F1 Score: {f1:.2f}\n\n")

        # Partial Least Squares Regression (PLS)
        model = PLSRegression(n_components=2)
        regression(X_train, X_test, y_train, y_test, model, "Partial Least Squares Regression")

        # Principal Component Regression (PCR)
        pcr = make_pipeline(PCA(n_components=2), LinearRegression())
        regression(X_train, X_test, y_train, y_test, pcr, "Principal Component Regression")

        # Ridge Regression
        rdg = Ridge(alpha=0.1)
        regression(X_train, X_test, y_train, y_test, rdg, "Ridge Regression")
        
        # Lasso Regression
        lso = Lasso(alpha=0.1)
        regression(X_train, X_test, y_train, y_test, lso, "Lasso Regression")

        # Linear Discriminant Analysis
        lda = LinearDiscriminantAnalysis()
        classification(X_c_train, X_c_test, y_c_train, y_c_test, lda, "Linear Discriminant Analysis")

        # Support Vector Machine
        # for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for kernel in ['linear', 'rbf']:
            clf = SVC(kernel=kernel)
            classification(X_c_train, X_c_test, y_c_train, y_c_test, clf, f"Support Vector Classifier [{kernel}]")

            svr = SVR(kernel=kernel, C=1.0, epsilon=0.1)
            regression(X_train, X_test, y_train, y_test, svr, f"Support Vector Regression [{kernel}]")
        
        # Machine Learning Regression
        class ML_Model():
            def __init__(self, nn_model, loss, cls=False):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.criterion = loss()
                self.model = nn_model()
                self.optimizer = optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=1e-4)
                self.classifier = cls
            def fit(self, X, y):
                X_nn = torch.tensor(X.to_numpy(), dtype=torch.float32)
                y_nn = torch.tensor(y.to_numpy(), dtype=torch.float32)

                if self.classifier:
                    self.unique_vals, indices = y_nn.unique(sorted=True, return_inverse=True)
                    y_nn = nn.functional.one_hot(indices, num_classes=len(self.unique_vals)).float()

                dataset = TensorDataset(X_nn, y_nn)
                batch_size = 100
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                self.model.to(self.device)

                epochs = 16
                self.model.train()
                for epoch in range(epochs):
                    for batch_X, batch_y in loader:
                        outputs = self.model(batch_X.to(self.device))
                        loss = self.criterion(outputs, batch_y.to(self.device))
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    if epoch == 0 or (epoch % 10 == 9):
                        print(f"Epoch {epoch+1}/{epochs} complete")
            def predict(self, X_test):
                self.model.eval()
                with torch.no_grad():
                    X_nn = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
                    y_pred = self.model(X_nn.to(self.device)).detach()
                if self.classifier:
                    y_pred = self.unique_vals[torch.argmax(y_pred.to('cpu'), dim=1)]
                return y_pred.squeeze().to('cpu').numpy()

        class RegressionModel(nn.Module):
            def __init__(self):
                super(RegressionModel, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(n_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            def forward(self, x):
                return self.network(x)
            
        reg_model = ML_Model(RegressionModel, nn.MSELoss)
        regression(X_train, X_test, y_train, y_test, reg_model, "ML Regression")
            
        class ClassificationModel(nn.Module):
            def __init__(self):
                super(ClassificationModel, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(n_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2)
                )
            def forward(self, x):
                return self.network(x)

        cls_model = ML_Model(ClassificationModel, nn.CrossEntropyLoss, cls=True)
        classification(X_c_train, X_c_test, y_c_train, y_c_test, cls_model, "ML Classification")
        
        return


def test_asterism_generation():

    fixed_asts =    asterisms.get_test_asterisms(100, fixed_rate=1.0, uniform_rate=0.0, offset_rate=0.0)
    uniform_asts =  asterisms.get_test_asterisms(100, fixed_rate=0.0, uniform_rate=1.0, offset_rate=0.0)
    offset_asts =   asterisms.get_test_asterisms(100, fixed_rate=0.0, uniform_rate=0.0, offset_rate=1.0)
    real_asts =     asterisms.get_test_asterisms(100, fixed_rate=0.0, uniform_rate=0.0, offset_rate=0.0)

    mixed_asts =     asterisms.get_test_asterisms(100, fixed_rate=0.1, uniform_rate=0.2, offset_rate=0.5)
    mix_2_asts =     asterisms.get_test_asterisms(100, fixed_rate=0.2, uniform_rate=0.2, offset_rate=0.4)
    

    def getCircle(radius, n_points, centre=(0,0), angle=2*np.pi):
        circle = [util.pol2cart(radius, t/n_points*angle) for t in range(n_points + 1)]
        X = [x+centre[0] for x, _ in circle]
        Y = [y+centre[1] for _, y in circle]
        return X, Y
    
    xcirc, ycirc = getCircle(120, 64)
    
    fig, axs = plt.subplots(2, 3)
    fig.suptitle("Asterism Distribution")

    def plot_asterisms(ax, title, asts):
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlabel('Arcseconds')
        ax.set_ylabel('Arcseconds')
        for ast in asts:
            x, y = asterisms.get_asterism_coords(ast, theta_offset=True)
            ax.scatter(x, y, c='r', alpha=0.2)
        for ast in asts:
            x, y = asterisms.get_asterism_coords(ast)
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ax.plot(x, y, 'b', alpha=0.1)
        ax.plot(xcirc, ycirc, 'k:', alpha=0.2)

    plot_asterisms(axs[0, 0], "Fixed Asterisms", fixed_asts)
    plot_asterisms(axs[0, 1], "Uniform Asterisms", uniform_asts)
    plot_asterisms(axs[1, 0], "Offset Asterisms", offset_asts)
    plot_asterisms(axs[1, 1], "Real Asterisms", real_asts)
    plot_asterisms(axs[0, 2], "Mix 1 Asterisms", mixed_asts)
    plot_asterisms(axs[1, 2], "Mix 2 Asterisms", mix_2_asts)

    fig.tight_layout()

def test_phase_variance():
    cutout = 100
    phase = 120
    n_modes = 21
    Z = zernike.Zernike(n_modes)

    for i in range(1, n_modes):
        c = np.array([0]*n_modes)
        c[i] = 1
        s = Z.compose(cutout, c)
        # print(np.var(s, where=util.circle(cutout) > 0))

        s = Z.compose(phase, c)
        p_var = np.var(s, where=util.circle(phase) > 0)
        # print(p_var)

        s2 = Z.compose(phase, c)
        c0 = (phase - cutout) // 2
        c1 = (phase + cutout) // 2
        c_var = np.var(s2[c0:c1, c0:c1], where=util.circle(cutout) > 0)
        # print(c_var)

        k = np.sqrt(p_var / c_var)
        s2 = s2 * k

        c_var2 = np.var(s2[c0:c1, c0:c1], where=util.circle(cutout) > 0)
        # print(c_var2)

        print(f"Mode: {i+1}\n\tPhase Size: {phase}\n\tCutout Size: {cutout}\n\tPhase Var: {p_var}\n\tCutout Var: {c_var}\n\tScaled Var: {c_var2}\n\tScaling Factor: {k}")

    c = np.ones_like(c) * np.sqrt(1/20)
    s = Z.compose(phase, c)
    p_var = np.var(s, where=util.circle(phase) > 0)
    c0 = (phase - cutout) // 2
    c1 = (phase + cutout) // 2
    c_var = np.var(s[c0:c1, c0:c1], where=util.circle(cutout) > 0)
    k = np.sqrt(p_var / c_var)
    s2 = s * k
    c_var2 = np.var(s2[c0:c1, c0:c1], where=util.circle(cutout) > 0)

    print(f"Modes: {n_modes}\n\tPhase Size: {phase}\n\tCutout Size: {cutout}\n\tPhase Var: {p_var}\n\tCutout Var: {c_var}\n\tScaled Var: {c_var2}\n\tScaling Factor: {k}")

    k_data = []
    calc_data = []
    for i in range(100):
        ks = []
        p_vars = []
        c_vars = []
        scaled_vars = []
        p_gen = phase_screen.PhaseScreenGenerator(phase, 1.8, 0.07)
        screens = p_gen.generate(n_modes)

        for screen in screens:
            # screen = phase_screen.kolmogorov((phase, phase), 1.8, 0.07)
            p_var = np.var(screen, where=util.circle(phase) > 0)
            p_vars.append(p_var)
            c0 = (phase - cutout) // 2
            c1 = (phase + cutout) // 2
            c_var = np.var(screen[c0:c1, c0:c1], where=util.circle(cutout) > 0)
            c_vars.append(c_var)
            k = np.sqrt(p_var / c_var)
            s2 = screen * k
            c_var2 = np.var(s2[c0:c1, c0:c1], where=util.circle(cutout) > 0)
            scaled_vars.append(c_var2)
            ks.append(k)
        calc_k = p_gen.find_scaling((cutout, cutout), screens)

        k_data.append(np.mean(ks))
        calc_data.append(np.mean(calc_k))

    from scipy.stats import ttest_rel

    # Perform dependent two-sample t-test
    t_stat, p_value = ttest_rel(k_data, calc_data)

    print("t-statistic:", t_stat)
    print("p-value:", p_value)

    # print(f"Total Screens: {n_modes}\n\tPhase Size: {phase}\n\tCutout Size: {cutout}\n\tPhase Var: {np.mean(p_vars)}\n\tCutout Var: {np.mean(c_vars)}\n\tScaled Var: {np.mean(scaled_vars)}\n\tScaling Factor: {np.mean(ks)} [{calc_k}]")
