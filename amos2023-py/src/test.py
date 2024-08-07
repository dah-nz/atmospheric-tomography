import matplotlib.pyplot as plt
import numpy as np

from basis import zernike

n_modes = 32
sz = (256, 256)

z = zernike.Zernike(n_modes)

fig, axs = plt.subplots(4,8)

i = 0
for row in range(4):
    for col in range(8):
        zn = z.out(sz, i)
        np.savetxt(f"data/zernike/python_z{i+1}.csv", zn, delimiter=",")
        axs[row, col].imshow(zn)
        i += 1
        axs[row, col].set_title(f"Mode {i}")

in_coeffs = [(-1) ** i for i in range(n_modes)]
in_coeffs[0] = 0
screen = z.compose(sz, in_coeffs)

out_coeffs = z.decompose(screen)
print(out_coeffs)

out_screen = z.compose(sz, out_coeffs)

fig, axs = plt.subplots(1, 4)

vmx = max(np.amax(screen), np.amax(out_screen))
vmn = min(np.amin(screen), np.amin(out_screen))

axs[1].imshow(screen, vmax=vmx, vmin=vmn)
axs[1].set_title("Input Screen")
axs[1].axis("off")

axs[2].imshow(out_screen, vmax=vmx, vmin=vmn)
axs[2].set_title("Output Screen")
axs[2].axis("off")

axs[3].imshow(screen - out_screen, vmax=vmx, vmin=vmn)
axs[3].set_title("Difference")
axs[3].axis("off")

axs[0].plot(in_coeffs, label="input")
axs[0].plot(out_coeffs, label="output")
axs[0].legend()

plt.show()