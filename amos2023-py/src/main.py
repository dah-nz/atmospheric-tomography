import numpy as np
import matplotlib.pyplot as plt

from parameters import test
from astro import data
from util import util
from tomography import beacon, phase_screen
from ml import ml
from tomography import tomo_LSQ
import torch

def main():
    seed1 = np.random.randint((2**31)-1)
    seed2 = np.random.randint((2**31)-1)
    model = ml.Model(seed=seed1)
    model_sd = ml.Model(seed=seed2, static_train=True)
    model_ss = ml.Model(seed=seed2, static_train=True, static_test=True)

    model.train()
    model.evaluate()

    model_sd.train()
    model_sd.evaluate()

    model_ss.train()
    model_ss.evaluate()

    tomo = tomo_LSQ.LSQ(model.t["basis"], model.t["D"], model.t["n_coeffs"], model.t["aperture_size"])
    tomo_results = []


    target = beacon.Beacon(0, 0)
    for n in range(model.t["epochs"]):
        total_loss = 0
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
                # print(tomo_coeffs)
                # print(target_coeffs)
                total_loss += model.loss_fn(torch.from_numpy(tomo_coeffs), torch.from_numpy(target_coeffs)).item()
            break
        tomo_results.append(total_loss / model.t["testing_batch"])

    plt.figure()
    plt.plot(model.results, label="Dynamic Model")
    plt.plot(model_sd.results, label="Static Model (Dynamic Test)")
    plt.plot(model_ss.results, label="Static Model")
    plt.plot(tomo_results, label="LSQ Tomography")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

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