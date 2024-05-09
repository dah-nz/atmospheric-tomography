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
        if type(v) not in [int, float, list, str]:
            n = v.__class__.__name__
        else:
            n = v
        s += f"{k},{n}\n"
    s += f"loss_fn,{m.loss_fn.__class__.__name__}\n"
    s += f"optimiser,{m.optimiser.__class__.__name__}\n"
    s += str(summary(m.model, input_size=(1, 3, m.t["n_coeffs"], 3), verbose=0))
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
    tomo_mixed_loss = []
    open_loop = []
    ol_coeffs = np.zeros(model.t["n_coeffs"])

    fixed_asterism = [beacon.Beacon(t, model.t["fixed_asterism_angle"]) for t in np.linspace(0, 2*np.pi, 3, endpoint=False)]

    # Reset output type for LSQ
    model.test_data.output_type = '1d'
    model_ss.test_data.output_type = '1d'
    
    target = beacon.Beacon(0, 0)
    for n in range(model.t["epochs"]):
        total_loss = 0
        mixed_loss = 0
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
                beacons = [beacon.Beacon(t, g, gamma_arcseconds=False) for t, g in beacon_data]
                tomo_coeffs = tomo.run(beacons, target, np.ravel(beacon_coeffs), model.t['layer_altitudes'])
                tomo_mixed = tomo.run(fixed_asterism, target, np.ravel(beacon_coeffs), model.t['layer_altitudes'])
                total_loss += model.loss_fn(torch.from_numpy(tomo_coeffs), torch.from_numpy(target_coeffs)).item()
                mixed_loss += model.loss_fn(torch.from_numpy(tomo_mixed), torch.from_numpy(target_coeffs)).item()
                openloop_loss += model.loss_fn(torch.from_numpy(ol_coeffs), torch.from_numpy(target_coeffs)).item()
            break
        now = datetime.now()
        now = now.strftime("%H:%M:%S")
        loss = total_loss / model.t["testing_batch"]
        md_loss = mixed_loss / model.t["testing_batch"]
        ol_loss = openloop_loss / model.t["testing_batch"]
        if md_loss > ol_loss:
            md_loss = 100
        if loss > ol_loss:
            loss = 100
        tomo_variable.append(loss)
        open_loop.append(ol_loss)
        tomo_mixed_loss.append(md_loss)

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

    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    data_file = "./results/ml_models/" + now + "_" + model.t["ml_network"]+ "_" + model.t["basis_type"] + str(model.t["n_coeffs"]) + model.t["data_file_suffix"]
    print(f"Saving data to {data_file}")
    with open(data_file, 'w', encoding="utf-8", errors='replace') as df:
        summ = f"{now}\n"
        summ += model_summary(model)
        df.write(summ)
        df.write("\nDATA FOLLOWS HERE\n")
        df.write('Epoch,Variable Training Loss,Variable Test Loss,Fixed Training Loss,Fixed Test Loss,Mixed Training Loss,Mixed Test Loss,Open Loop,LSQ Fixed Asterism,LSQ Variable Asterism,LSQ Mixed Asterism\n')
        for t in zip(range(model.t["epochs"]), 
                model.train_results, 
                model.test_results, 
                model_ss.train_results, 
                model_ss.test_results, 
                model_sd.train_results, 
                model_sd.test_results, 
                open_loop,
                tomo_fixed,
                tomo_variable,
                tomo_mixed_loss):
            st = ",".join(map(str,t))
            df.write(st + "\n")

    plt.figure()
    plt.plot(model.test_results, 'r-', label="Variable Asterism Model")
    plt.plot(model_sd.test_results, 'b-', label="Mixed Asterism Model")
    plt.plot(model_ss.test_results, 'g-', label="Fixed Asterism Model")
    plt.plot(model.train_results, 'r:', label="Variable Asterism Model [Training]")
    plt.plot(model_sd.train_results, 'b:', label="Mixed Asterism Model [Training]")
    plt.plot(model_ss.train_results, 'g:', label="Fixed Asterism Model [Training]")
    plt.plot(tomo_variable, 'k--', label="LSQ Variable Asterism Tomography")
    plt.plot(tomo_mixed_loss, 'c:', label="LSQ Mixed Asterism Tomography")
    plt.plot(tomo_fixed, 'k:', label="LSQ Fixed Asterism Tomography")
    plt.plot(open_loop, 'y--', label="Open Loop")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

    plt.show()
    return

if __name__ == "__main__":
    main()