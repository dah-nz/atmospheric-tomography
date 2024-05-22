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
    seed = np.random.randint((2**31)-1)
    
    print("Building model configurations")
    model_var = ml.Model(seed=seed)
    model_mix = ml.Model(seed=seed, static_train=True)
    model_fix = ml.Model(seed=seed, static_train=True, static_test=True)
    print(model_summary(model_var))

    print(">>> Variable model")
    model_var.train()
    model_var.evaluate()

    print(">>> Mixed model")
    model_mix.train()
    model_mix.evaluate()

    print(">> Fixed model")
    model_fix.train()
    model_fix.evaluate()

    print(">>> ML model training complete")

    tomo = tomo_LSQ.LSQ(model_var.t["basis"], model_var.t["D"], model_var.t["n_coeffs"], model_var.t["aperture_size"])
    tomo_variable_loss = []
    tomo_fixed_loss = []
    tomo_mixed_loss = []
    tomo_open_loop_loss = []
    ol_coeffs = np.zeros(model_var.t["n_coeffs"])
    
    target = beacon.Beacon(0, 0)

    def process_data(X, y, i):
        data = X[i]
        target_coeffs = y[i]
        theta = np.array([data[0, 0, 1], data[1, 0, 1], data[2, 0, 1]])
        gamma = np.array([data[0, 0, 2], data[1, 0, 2], data[2, 0, 2]])

        beacon_coeffs = np.ravel(data[:, :, 0])
        beacons = [beacon.Beacon(t, g, gamma_arcseconds=False) for t, g in zip(theta, gamma)]

        return target_coeffs, beacon_coeffs, beacons

    for n in range(model_var.t["epochs"]):
        fixed_loss = 0
        variable_loss = 0
        mixed_loss = 0
        openloop_loss = 0
        print(f"Calculating LSQ data for this epoch {n+1}/{model_var.t['epochs']}")
        for (X, y), (Xf, yf) in zip(model_var.validation_dataloader, model_fix.validation_dataloader):
            X, y = X.numpy(), y.numpy()
            Xf, yf = Xf.numpy(), yf.numpy()
            for i in range(model_var.t["validation_batch"]):
                target_coeffs, beacon_coeffs, beacons = process_data(X, y, i)
                target_coeffs_f, beacon_coeffs_f, beacons_f = process_data(Xf, yf, i)
                
                tomo_variable = tomo.run(beacons,   target, beacon_coeffs,   model_var.t['layer_altitudes'])
                tomo_mixed =    tomo.run(beacons_f, target, beacon_coeffs,   model_var.t['layer_altitudes'])
                tomo_fixed =    tomo.run(beacons_f, target, beacon_coeffs_f, model_var.t['layer_altitudes'])

                variable_loss += model_var.loss_fn(torch.from_numpy(tomo_variable), 
                                                   torch.from_numpy(target_coeffs)).item()
                mixed_loss    += model_var.loss_fn(torch.from_numpy(tomo_mixed),
                                                   torch.from_numpy(target_coeffs)).item()
                openloop_loss += model_var.loss_fn(torch.from_numpy(ol_coeffs),
                                                   torch.from_numpy(target_coeffs)).item()
                fixed_loss    += model_var.loss_fn(torch.from_numpy(tomo_fixed),
                                                   torch.from_numpy(target_coeffs_f)).item()
                
            break
        now = datetime.now().strftime("%H:%M:%S")
        va_loss = variable_loss / model_var.t["validation_batch"]
        ma_loss = mixed_loss    / model_var.t["validation_batch"]
        fa_loss = fixed_loss    / model_var.t["validation_batch"]
        ol_loss = openloop_loss / model_var.t["validation_batch"]
        if ma_loss > 100:
            ma_loss = 100
        if va_loss > 100:
            va_loss = 100
        if fa_loss > 100:
            fa_loss = 100
        tomo_variable_loss.append(va_loss)
        tomo_open_loop_loss.append(ol_loss)
        tomo_mixed_loss.append(ma_loss)
        tomo_fixed_loss.append(fa_loss)
        print(f"{now} ->  variable loss: {va_loss:>7f}")
        print(f"         ->     mixed loss: {ma_loss:>7f}")
        print(f"         ->     fixed loss: {fa_loss:>7f}")
        print(f"         -> open loop loss: {ol_loss:>7f}")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = "./results/ml_models/{}_{}_{}{}.{}".format(now,
                                                          model_var.t["ml_network"],
                                                          model_var.t["basis_type"],
                                                          str(model_var.t["n_coeffs"]),
                                                          model_var.t["data_file_suffix"])
    print(f"Saving data to {data_file}")

    with open(data_file, 'w', encoding="utf-8", errors='replace') as df:
        summ = f"{now}\n"
        summ += model_summary(model_var)
        df.write(summ)
        df.write("\nDATA FOLLOWS HERE\n")
        df.write('Epoch,Variable Training Loss,Variable Test Loss,Variable Validation Loss,Fixed Training Loss,Fixed Test Loss,Fixed Validation Loss,Mixed Training Loss,Mixed Test Loss,Mixed Validation Loss,Open Loop,LSQ Fixed Asterism,LSQ Variable Asterism,LSQ Mixed Asterism\n')
        for t in zip(range(model_var.t["epochs"]), 
                model_var.train_results, 
                model_var.test_results,
                model_var.validation_results,
                model_fix.train_results, 
                model_fix.test_results, 
                model_fix.validation_results,
                model_mix.train_results, 
                model_mix.test_results, 
                model_mix.validation_results,
                tomo_open_loop_loss,
                tomo_fixed_loss,
                tomo_variable_loss,
                tomo_mixed_loss):
            st = ",".join(map(str,t))
            df.write(st + "\n")

    plt.figure()
    plt.plot(model_var.validation_results, 'r-',  label="Variable Asterism Model")
    plt.plot(model_mix.validation_results, 'b-',  label="Mixed Asterism Model")
    plt.plot(model_fix.validation_results, 'g-',  label="Fixed Asterism Model")
    plt.plot(model_var.train_results,      'r:',  label="Variable Asterism Model [Training]")
    plt.plot(model_mix.train_results,      'b:',  label="Mixed Asterism Model [Training]")
    plt.plot(model_fix.train_results,      'g:',  label="Fixed Asterism Model [Training]")
    plt.plot(model_var.test_results,       'r--', label="Variable Asterism Model [Test]")
    plt.plot(model_mix.test_results,       'b--', label="Mixed Asterism Model [Test]")
    plt.plot(model_fix.test_results,       'g--', label="Fixed Asterism Model [Test]")
    plt.plot(tomo_variable_loss,           'm:',  label="LSQ Variable Asterism Tomography")
    plt.plot(tomo_mixed_loss,              'c:',  label="LSQ Mixed Asterism Tomography")
    plt.plot(tomo_fixed_loss,              'k:',  label="LSQ Fixed Asterism Tomography")
    plt.plot(tomo_open_loop_loss,          'y:',  label="Open Loop")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

    plt.show()
    return

if __name__ == "__main__":
    main()