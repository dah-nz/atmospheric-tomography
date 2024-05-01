import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from parameters import test
from astro import data
from util import util
from tomography import beacon, phase_screen
from ml import ml
from tomography import tomo_LSQ
import torch
from torchinfo import summary

def model_summary(m):
    s = ""
    for k, v in m.t.items():
        s += f"{k}|{v}\n"
    s += f"loss_fn,{m.loss_fn.__class__.__name__}\n"
    s += f"optimiser,{m.optimiser.__class__.__name__}\n"
    s += str(summary(m.model, input_size=(1, 6+3*m.t["n_coeffs"]), verbose=0))
    return s

def main():
    seed1 = np.random.randint((2**31)-1)
    seed2 = np.random.randint((2**31)-1)
    model = ml.Model(seed=seed1)
    print(model_summary(model))
    model_sd = ml.Model(seed=seed2, static_train=True)
    model_ss = ml.Model(seed=seed2, static_train=True, static_test=True)

    model.train()
    model.evaluate()

    model_sd.train()
    model_sd.evaluate()

    model_ss.train()
    model_ss.evaluate()

    tomo = tomo_LSQ.LSQ(model.t["basis"], model.t["D"], model.t["n_coeffs"], model.t["aperture_size"])
    tomo_variable = []
    tomo_fixed = []
    open_loop = []
    ol_coeffs = np.zeros(model.t["n_coeffs"])

    target = beacon.Beacon(0, 0)
    for n in range(model.t["epochs"]):
        total_loss = 0
        openloop_loss = 0
        print(f"Calculating variable asterism LSQ epoch {n+1}/{model.t['epochs']}")
        for X, y in model.test_dataloader:
            X = X.numpy()
            y = y.numpy()
            for i in range(model.t["testing_batch"]):
                data = X[i, :]
                target_coeffs = y[i, :]
                beacon_data = np.reshape(data[:6], (-1, 2))
                beacon_coeffs = np.reshape(data[6:], (3, -1))
                beacons = [beacon.Beacon(t, g) for t, g in beacon_data]
                tomo_coeffs = tomo.run(beacons, target, np.ravel(beacon_coeffs), model.t['layer_altitudes'])
                total_loss += model.loss_fn(torch.from_numpy(tomo_coeffs), torch.from_numpy(target_coeffs)).item()
                openloop_loss += model.loss_fn(torch.from_numpy(ol_coeffs), torch.from_numpy(target_coeffs))
            break
        now = datetime.now()
        now = now.strftime("%H:%M:%S")
        loss = total_loss / model.t["testing_batch"]
        ol_loss = openloop_loss / model.t["testing_batch"]
        print(f"{now} -> loss: {loss:>7f}")
        tomo_variable.append(loss)
        open_loop.append(ol_loss)

    for n in range(model_ss.t["epochs"]):
        total_loss = 0
        print(f"Calculating fixed asterism LSQ epoch {n+1}/{model_ss.t['epochs']}")
        for X, y in model_ss.test_dataloader:
            X = X.numpy()
            y = y.numpy()
            for i in range(model_ss.t["testing_batch"]):
                data = X[i, :]
                target_coeffs = y[i, :]
                beacon_data = np.reshape(data[:6], (-1, 2))
                beacon_coeffs = np.reshape(data[6:], (3, -1))
                beacons = [beacon.Beacon(t, g) for t, g in beacon_data]
                tomo_coeffs = tomo.run(beacons, target, np.ravel(beacon_coeffs), model.t['layer_altitudes'])
                total_loss += model.loss_fn(torch.from_numpy(tomo_coeffs), torch.from_numpy(target_coeffs)).item()
            break
        now = datetime.now()
        now = now.strftime("%H:%M:%S")
        loss = total_loss / model.t["testing_batch"]
        print(f"{now} -> loss: {loss:>7f}")
        tomo_fixed.append(loss)

    plt.figure()
    plt.plot(model.test_results, 'r-', label="Variable Asterism Model")
    plt.plot(model_sd.test_results, 'b-', label="Fixed Asterism Model (Variable Test)")
    plt.plot(model_ss.test_results, 'g-', label="Fixed Asterism Model")
    plt.plot(model.train_results, 'r:', label="Variable Asterism Model [Training}")
    plt.plot(model_sd.train_results, 'b:', label="Fixed Asterism Model (Variable Test) [Training]")
    plt.plot(model_ss.train_results, 'g:', label="Fixed Asterism Model [Training]")
    plt.plot(tomo_fixed, 'k--', label="LSQ Fixed Asterism Tomography")
    plt.plot(tomo_variable, 'k:', label="LSQ Variable Asterism Tomography")
    plt.plot(open_loop, 'y--', label="Open Loop")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    data_file = "./results/ml_models/" + now + "_" + model.t["model_name"] + model.t["data_file_suffix"]
    print(f"Saving data to {data_file}")
    with open(data_file, 'w', encoding="utf-8", errors='replace') as df:
        summ = f"{now}\n"
        summ += model_summary(model)
        df.write(summ)
        df.write('\nEpoch,Variable Training Loss,Variable Test Loss,Fixed Training Loss,Fixed (Var) Test Loss,Fixed (Var) Training Loss,Fixed Test Loss,Open Loop,LSQ Fixed Asterism,LSQ Variable Asterism\n')
        for e, vtr, vts, ftr, fts, fvtr, fvts, ol, fa, va in zip(range(model.t["epochs"]), 
                                                                    model.train_results, 
                                                                    model.test_results, 
                                                                    model_ss.train_results, 
                                                                    model_ss.test_results, 
                                                                    model_sd.train_results, 
                                                                    model_sd.test_results, 
                                                                    open_loop,
                                                                    tomo_fixed,
                                                                    tomo_variable):
            df.write(f"{e},{vtr},{vts},{ftr},{fts},{fvtr},{fvts},{ol},{fa},{va}\n")
                

    plt.show()
    return

    # Settings
    N = 100
    star_magnitude = 14
    angular_limit = 120 # arcseconds
    altitudes = [1000, 8000]
    D = 0.6
    aperture_size = 64 
    r0 = 0.05 # m

    # Generate a batch of asterisms
    #   - Load satellite/star distribution data
    #   - Draw sets of three stars from asterism data.
    #   - Limit to furthest star within X arcseconds
    #   - Have N^2 sets of asterisms per batch


    star_distances = data.load_asterism_data(star_magnitude, N**2, angular_limit)
    star_distances = util.arcsec2rad(star_distances)
    star_angles = np.random.uniform(low=0.0, high=2*np.pi, size=star_distances.shape)

    fig, ax = plt.subplots(1, 1)

    star_x, star_y = util.azimuth2cart(7500, star_distances, star_angles)

    # star_x = star_distances * np.cos(star_angles)
    # star_y = star_distances * np.sin(star_angles)

    for x, y in zip(star_x[:], star_y[:]):
        ax.plot(x, y, alpha=0.1)


    # Generate a batch of phase screens
    #   - Use two layers based on Mohr data
    #   - Build N screens for each layer, with the correct sigma for each.
    #   - This results in N^2 combinations of phase screens, matching the number of asterisms
    outer = [beacon.Beacon(0, angular_limit, gamma_arcseconds=True)]
    layer_diameters, layer_pixels = util.get_layer_shapes(altitudes, outer, D, aperture_size)
    screen_gens = [phase_screen.PhaseScreenGenerator(px, D, r0) for px in layer_pixels]
    screens = [gen.generate(N, cut_out=(aperture_size, aperture_size)) for gen in screen_gens]
    print(len(screens))
    print(len(screens[0]))
    print(len(screens[1]))

    fig, axs = plt.subplots(2, 4)
    for i, n in enumerate([3, 15, 42, 56]):
        axs[0, i].imshow(screens[0][n])
        axs[1, i].imshow(screens[1][n])

    # Train an ML system to estimate target mode aberrations
    #   - Train on infinite batches with N = 100
    #   - Save and reload the model after finishing a batch

    phase_screen_combos = np.array(np.meshgrid(np.array(range(N)), np.array(range(N)))).T.reshape(-1,2)
    # shuffle
    l = phase_screen_combos.shape[0]
    indices = np.random.choice(l, size=l)
    phase_screen_combos = phase_screen_combos[indices, :]

    # After each batch completes, message a comparison process that runs in parallel
    #   - Compare system to Ragazzoni LSQ approach after each batch completes

    # ? Train a companion model on a static asterism to compare loss of precision when adding variance of the star field
    #   - Static asterism will be an equilateral triangle of 3 stars at the mode distance of all stars

if __name__ == "__main__":
    main()