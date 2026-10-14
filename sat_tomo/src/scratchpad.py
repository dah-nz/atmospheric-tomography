from tomography import phase_screen, wavefront, asterisms
from basis import zernike
import util
import matplotlib.pyplot as plt

altitudes = [1000, 5000, 10000]
r0 = [0.064677, 0.497862, 0.257559]
D = 1.8
aperture_size = 100
n = 36

atm = phase_screen.Atmosphere(1, altitudes, r0, D, aperture_size)

asts = asterisms.generate_asterisms(1)

wfs = wavefront.SHWFS(D, n)
wfs_n = wavefront.SHWFS(D, n, noise=True)

z = zernike.Zernike(n)

mask = util.circle(aperture_size)

for a in asts:
    for b in a:
        co = b.total_cutout(D, altitudes, atm[0], atm.metapupil_diameters)
        coeffs, spots = wfs.run(co, b.magnitude)
        coeffs_n, spots_n = wfs_n.run(co, b.magnitude)
        coeffs_ideal = z.decompose(co)
        ptt = coeffs_ideal[:3]
        coeffs_n[:3] = ptt
        coeffs[:3] = ptt
        fig, axs = plt.subplots(3,4)
        axs[0,0].imshow(co * mask)

        axs[0,1].imshow(spots)
        recon = z.compose(aperture_size, coeffs)
        axs[0,2].imshow(recon)
        axs[0,3].imshow(co * mask - recon)

        axs[1,0].plot(coeffs, label="WFS w/o Photon Noise")
        axs[1,0].plot(coeffs_n, label=f"WFS w/ Photon Noise  [mag = {b.magnitude}]")
        axs[1,0].plot(coeffs_ideal, label="Ideal Coeffs")
        axs[1,0].legend()
        
        axs[1,1].imshow(spots_n)
        recon_n = z.compose(aperture_size, coeffs_n)
        axs[1,2].imshow(recon_n)
        axs[1,3].imshow(co * mask - recon_n)

        recon_ideal = z.compose(aperture_size, coeffs_ideal)
        axs[2, 0].imshow(recon_ideal)
        axs[2, 1].imshow(co * mask - recon_ideal)
        axs[2, 2].imshow(recon_ideal - recon)
        axs[2, 3].imshow(recon_ideal - recon_n)

plt.show()
