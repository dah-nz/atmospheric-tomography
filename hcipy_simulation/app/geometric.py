from hcipy.wavefront_sensing.wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from hcipy.propagation import FresnelPropagator

class GeometricWavefrontSensorOptics(WavefrontSensorOptics):
    def __init__(self, input_grid, separation):
        self.prop_extra_focal = FresnelPropagator(input_grid, separation)
        self.prop_intra_focal = FresnelPropagator(input_grid, -separation)
    def forward(self, wavefront):
        return (self.prop_intra_focal.forward(wavefront), self.prop_extra_focal.forward(wavefront))
    def backward(self, wavefront_intra, wavefront_extra):
        return (self.prop_intra_focal.backward(wavefront_intra), self.prop_extra_focal.backward(wavefront_extra))

class GeometricWavefrontSensorEstimator(WavefrontSensorEstimator):
    def __init__(self):
        pass
    def estimate(self, images):
        return super().estimate(images)

def main():
    import hcipy
    import numpy as np
    import matplotlib.pyplot as plt

    def double_field(a, b, title='', xlabel='', ylabel='', **kwargs):
        double_plot(a, b, field=True, title='', xlabel='', ylabel='', **kwargs)

    def double_plot(a, b, field=True, title='', xlabel='', ylabel='', **kwargs):
        ''' A function to nicely show two fields side-to-side. '''
        fig, axes = plt.subplots(1, 2, gridspec_kw={'left': 0.14, 'right': 0.98, 'top': 0.95, 'bottom': 0.07, 'wspace': 0.02})
        fig.suptitle(title)

        if field:
            hcipy.imshow_field(a, **kwargs, ax=axes[0])
            hcipy.imshow_field(b, **kwargs, ax=axes[1])
        else:
            hcipy.imshow_psf(a, **kwargs, ax=axes[0])
            hcipy.imshow_psf(b, **kwargs, ax=axes[1])

        axes[1].yaxis.set_ticks([])
        axes[0].set_xlabel(xlabel)
        axes[1].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)

        return fig
    
    telescope_diameter = 0.62 #m
    wavelength = 650e-9 #m
    f_ratio = 6.3 #m
    focal_length = f_ratio * telescope_diameter

    zero_magnitude_flux = 3.9e10
    stellar_magnitude = 0

    num_pupil_pixels = 240
    pupil_grid_diameter = 256/240 * telescope_diameter

    pupil_grid = hcipy.make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

    aperture_luvoir = hcipy.evaluate_supersampled(hcipy.make_luvoir_a_aperture(True), pupil_grid.scaled(1 / pupil_grid_diameter), 8)
    aperture_luvoir.grid = pupil_grid
    aperture_circ = hcipy.evaluate_supersampled(hcipy.make_circular_aperture(pupil_grid_diameter), pupil_grid, 8)

    double_field(aperture_circ, aperture_luvoir,
            xlabel='x [m]', ylabel='y [m]',
            grid_units=1, cmap='gray')
    
    num_actuators_across_pupil = 10
    actuator_spacing = telescope_diameter / num_actuators_across_pupil
    influence_functions = hcipy.make_gaussian_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing)
    deformable_mirror = hcipy.DeformableMirror(influence_functions)
    num_modes = deformable_mirror.num_actuators

    wf_circ = hcipy.Wavefront(aperture_circ, wavelength)
    wf_luvoir = hcipy.Wavefront(aperture_luvoir, wavelength)

    spatial_resolution = wavelength / telescope_diameter
    focal_grid = hcipy.make_focal_grid(q=12, num_airy=100, spatial_resolution=spatial_resolution)

    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

    img_c = prop(wf_circ)
    norm_c = img_c.power.max()
    img_l = prop(wf_luvoir)
    norm_l = img_l.power.max()

    double_field(np.log10(img_c.power / img_c.power.max()), np.log10(img_l.power / img_l.power.max()),
            xlabel='x [um]', ylabel='y [um]',
            vmin=-6, cmap='inferno', grid_units=1e-6)
    
    propagation_distance = 1e-3 #m
    gwfs = GeometricWavefrontSensorOptics(focal_grid, propagation_distance)

    intra_c, extra_c = gwfs.forward(img_c)
    intra_l, extra_l = gwfs.forward(img_l)
    
    double_field(np.log10(extra_c.power / extra_c.power.max()), np.log10(extra_l.power / extra_l.power.max()),
            xlabel='x [um]', ylabel='y [um]',
            vmin=-6, cmap='inferno', grid_units=1e-6)

    double_field(np.log10(intra_c.power / intra_c.power.max()), np.log10(intra_l.power / intra_l.power.max()),
            xlabel='x [um]', ylabel='y [um]',
            vmin=-6, cmap='inferno', grid_units=1e-6)
    
    deformable_mirror.random(0.1 * wavelength)
    PSF_c = prop(deformable_mirror.forward(wf_circ))
    PSF_l = prop(deformable_mirror.forward(wf_luvoir))
    
    double_field(np.log10(PSF_c.power / norm_c), np.log10(PSF_l.power / norm_l),
            xlabel='x [um]', ylabel='y [um]',
            vmin=-6, cmap='inferno', grid_units=1e-6)

    intra_c, extra_c = gwfs.forward(PSF_c)
    intra_l, extra_l = gwfs.forward(PSF_l)

    double_field(np.log10(extra_c.power / extra_c.power.max()), np.log10(extra_l.power / extra_l.power.max()),
            xlabel='x [um]', ylabel='y [um]',
            vmin=-6, cmap='inferno', grid_units=1e-6)

    double_field(np.log10(intra_c.power / intra_c.power.max()), np.log10(intra_l.power / intra_l.power.max()),
            xlabel='x [um]', ylabel='y [um]',
            vmin=-6, cmap='inferno', grid_units=1e-6)
    
    diff_c = intra_c.power - extra_c.power
    diff_l = intra_l.power - extra_l.power
    
    double_field(diff_c / diff_c.max(), diff_l / diff_l.max(),
            xlabel='x [um]', ylabel='y [um]',
            vmin=-6, cmap='inferno', grid_units=1e-6)
    

    plt.show()


if __name__ == "__main__":
    main()