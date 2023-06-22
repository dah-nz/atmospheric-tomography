import numpy as np
import matplotlib.pyplot as plt

from tomography import tomography, phase_screen
from util import util
from basis import zernike

Z = zernike.Zernike(50)
T = tomography.Tomography([0], Z, 50, 1.0)

n_points = 250
def getCircle(radius, n_points, centre=(0,0), angle=2*np.pi):
    circle = [util.pol2cart(radius, t/n_points*angle) for t in range(n_points + 1)]
    return [(x + centre[0], y + centre[1]) for x, y in circle]
circlePoints = getCircle(1, n_points)
grid = T._getGrid(n_points)
def plotgrid(grid, l, d, dsp):
    g = T._scaleGrid(grid, d)
    g = T._translateGrid(g, l)
    X = [x for x,_ in g]
    Y = [y for _,y in g]
    plt.scatter(X, Y, c=dsp)
    plt.hist2d(X, Y, bins=21, alpha=0.5)

size = 256
nModes = 50
np.random.seed(3)
test_modes = [list(m * np.exp(-(i+1)/nModes) for i, m in enumerate(np.random.rand(nModes) - 0.5))]
scale = 0.54
l = 0.25
# location = np.array((-0.25, 0.10))
locs = [np.array((0, 0)),
        np.array((l, 0)),
        np.array((l, l)),
        np.array((0, l)),
        np.array((-l, l)),
        np.array((-l, 0)),
        np.array((-l, -l)),
        np.array((0, -l)),
        np.array((l, -l))]

cmap = "RdBu"
# matplotlib.use('pgf')

modes = test_modes[0]
errors = []
for loc in locs:
    P = T._projectionMatrix(tuple(loc), scale)
    # P = np.transpose(P)
    mode_screen = np.zeros((size, size))
    mode_screen = phase_screen.fromCoefficients(Z, mode_screen, modes)
    sub_modes = np.matmul(P, modes)
    sub_mode_screen = np.zeros((int(scale*size), int(scale*size)))
    sub_mode_screen = phase_screen.fromCoefficients(Z, sub_mode_screen, sub_modes)
    reconstructed_modes = np.matmul(np.linalg.pinv(P), sub_modes)
    reconstructed_screen = phase_screen.fromCoefficients(Z, mode_screen, reconstructed_modes)
    error = mode_screen - reconstructed_screen
    errors.append(np.sqrt(np.average(error * error)))

fig = plt.figure()
fig.set_size_inches(5.5,3)
plt.ylabel('RMS Error [radians]')
plt.xlabel("Location")
plt.plot(errors, 'k-')
plt.tight_layout()
# plt.savefig("img/ProjectionMatrixErrors.pgf")

location = locs[2]
modes = test_modes[0]
P = T._projectionMatrix(tuple(location), scale)
# P = np.transpose(P)
# Calculate subpupils
mode_screen = np.zeros((size, size))
mode_screen = phase_screen.fromCoefficients(Z, mode_screen, modes)
sub_modes = np.matmul(P, modes)
sub_mode_screen = np.zeros((int(scale*size), int(scale*size)))
sub_mode_screen = phase_screen.fromCoefficients(Z, sub_mode_screen, sub_modes)
reconstructed_modes = np.matmul(np.linalg.pinv(P), sub_modes)
reconstructed_screen = phase_screen.fromCoefficients(Z, mode_screen, reconstructed_modes)

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Plot projection matrix
fig, ax = plt.subplots(1)
divider = make_axes_locatable(ax)
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig = ax.get_figure()
fig.add_axes(ax_cb)
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
ax.set_title("Projection Matrix")
im = ax.imshow(P, cmap='RdBu', vmax=1.0, vmin=-1.0)
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
# plt.savefig("img/ProjectionMatrix.pgf")

# Show modes for subpupil and metapupil
fig = plt.figure()
fig.set_size_inches(5.5,3)
plt.ylabel('Mode Coefficients [radians]')
plt.xlabel("Zernike Mode Index")
plt.plot(test_modes[0], 'k-', label="Metapupil Modes")
plt.plot(sub_modes, 'r-', label="Subpupil Modes")
plt.legend()
plt.tight_layout()
# plt.savefig("img/ProjectionMatrixModes.pgf")

# Show initial, reconstructed and errors of metapupils
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(5.8,3)
for ax in axes:
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

axes[0].set_title("Initial Metapupil")
divider = make_axes_locatable(axes[0])
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig = axes[0].get_figure()
fig.add_axes(ax_cb)
im = axes[0].imshow(mode_screen, cmap=cmap)
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()

axes[1].set_title("Reconstructed Metapupil")
divider = make_axes_locatable(axes[1])
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig = axes[1].get_figure()
fig.add_axes(ax_cb)
im = axes[1].imshow(reconstructed_screen, cmap=cmap)
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()

axes[2].set_title("Reconstruction Error\n")
divider = make_axes_locatable(axes[2])
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig = axes[2].get_figure()
fig.add_axes(ax_cb)
im = axes[2].imshow(mode_screen - reconstructed_screen, cmap=cmap)
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
plt.tight_layout()
# plt.savefig("img/ProjectionMatrixReconstruction.pgf")

# Show subpupil location and projection
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(5.5,4)
for ax in axes:
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

axes[0].set_title("Subpupil Location")
divider = make_axes_locatable(axes[0])
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig = axes[0].get_figure()
fig.add_axes(ax_cb)
mx = np.amax(mode_screen)
mn = np.amin(mode_screen)
im = axes[0].imshow(mode_screen, cmap=cmap, vmax=mx, vmin=mn)
loc = ((size*location[0] + size) / 2, (size*-location[1] + size) / 2)
util.plotcirc(axes[0], loc, scale * size / 2, 'k--', alpha=0.5)
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()

axes[1].set_title("Projected Subpupil")
divider = make_axes_locatable(axes[1])
ax_cb = divider.append_axes("right", size="5%", pad=0.05)
fig = axes[1].get_figure()
fig.add_axes(ax_cb)
im = axes[1].imshow(sub_mode_screen, cmap=cmap, vmax=mx, vmin=mn)
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
plt.tight_layout()
# plt.savefig("img/ProjectionMatrixSubpupil.pgf")

# sub_mode_mask = np.zeros((size, size))

# img_loc = util.cart2imagespace(size, location)
# radius = int(size * scale / 2)

# c_loc = (size * -location[1] / 2, size * location[0] / 2)

# c = circle(int(size * scale / 2), size, c_loc)
# mode_mask = c * mode_screen

# sub_mode_mask[(img_loc[0]-radius):(img_loc[0]+radius), (img_loc[1]-radius):(img_loc[1]+radius)] = sub_mode_screen

# plt.figure()
# plt.title("Metapupil Phase Screen Masked")
# plt.imshow(mode_mask)
# plt.figure()
# plt.title("Subpupil Phase Screen in Metapupil")
# plt.imshow(sub_mode_mask)
# plt.figure()
# plt.title("Metapupil Phase Screen difference from Masked Metapupil Phase Screen")
# plt.imshow(mode_screen - mode_mask)
# plt.figure()
# plt.title("Metapupil Phase Screen difference from calculated phase screen")
# plt.imshow(mode_screen - sub_mode_mask)
# plt.figure()


plt.show()