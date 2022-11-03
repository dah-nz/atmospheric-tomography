import numpy as np
import phase_screen as ps
import util

def strehl_from_pupil(phase_screen, size=256, wavelength=600e-9):
    j = complex(0, 1)
    wf_ideal = util.circle(size / 2, size)
    wf = wf_ideal * np.exp(j * phase_screen * 2 * np.pi / wavelength)

    return np.abs(np.sum(wf) / np.sum(wf_ideal)) ** 2

def strehl_from_modes(zernike_modes, size=256, wavelength=600e-9, ignoreModes=1):
    phase_screen = ps.createFromModes(zernike_modes, size, ignoreModes)
    return strehl_from_pupil(phase_screen, size, wavelength)

def main():
    for m in range(20):
        modes = [0] * 20
        modes[m] = 1
        print("Strehl ratio for mode {}: {}".format(m + 1, strehl_from_modes(modes, 1, 600e-9, ignoreModes=0)))

    for i in range(5):
        from zernike import randomModes
        modes = randomModes(20, 0)
        print("Strehl ratio for random test {} A: {}".format(i + 1, strehl_from_modes(modes, 1, 600e-9, ignoreModes=0)))
        modes = list(0.1 * np.array(modes))
        print("Strehl ratio for random test {} B: {}".format(i + 1, strehl_from_modes(modes, 1, 600e-9, ignoreModes=0)))
    

if __name__ == "__main__":
    main()