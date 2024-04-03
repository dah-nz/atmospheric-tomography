import numpy as np
import matplotlib.pyplot as plt
import time

from tomography.tomo_MLNN import TomographyData, MLNN
from tomography.tomo_LSQ import LSQ

from parameters import test
from util import util
from tomography import beacon
from astro import astro

t = test.load()

starmap = astro.StarMap(mag=12)

test_data = TomographyData(train=False, starmap=starmap, output_ps=True)

mask = util.circle(t["aperture_size"]/2, t["aperture_size"])

nn = MLNN(t["model_file"])
lsq = LSQ()

print("RMSE            Time")
print("NN      LSQ     NN      LSQ")

for i in range(10):
    x, y = test_data[i]

    beacons = [beacon.Beacon(x[0], x[1]), beacon.Beacon(x[2], x[3]), beacon.Beacon(x[4], x[5])]
    coeffs = x[6:]

    t0 = time.perf_counter()

    pred = nn.run(beacons, coeffs)

    t1 = time.perf_counter()

    pred_lsq = lsq.run(beacons, coeffs)

    t2 = time.perf_counter()

    basis = t["basis"]

    predicted = basis.compose((t["aperture_size"], t["aperture_size"]), pred)
    predicted_lsq = basis.compose((t["aperture_size"], t["aperture_size"]), pred_lsq)
    # actual = basis.compose((t["aperture_size"], t["aperture_size"]), y)

    masked_actual = mask * (y - np.mean(y))
    masked_nn = mask * (predicted - np.mean(predicted))
    masked_lsq = mask * (predicted_lsq - np.mean(predicted_lsq))

    # plt.figure()
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(predicted)
    axs[1].imshow(y)
    axs[2].imshow(predicted_lsq)

    rmse_nn = util.rmse(masked_actual, masked_nn, mask=mask)
    rmse_lsq = util.rmse(masked_actual, masked_lsq, mask=mask)

    t_nn = t1 - t0
    t_lsq = t2 - t1

    print("{:.4f}, {:.4f}, {:.6f}, {:.6f}".format(rmse_nn, rmse_lsq, t_nn, t_lsq))

plt.show()