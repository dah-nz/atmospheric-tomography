import numpy as np
import matplotlib.pyplot as plt

from astro import data


# Generate a batch of asterisms
#   - Load satellite/star distribution data
#   - Draw sets of three stars from asterism data.
#   - Limit to furthest star within X arcseconds
#   - Have N^2 sets of asterisms per batch

N = 100
star_magnitude = 14
angular_limit = 120 # arcseconds
star_distances = data.load_asterism_data(star_magnitude, N**2, angular_limit)
print(star_distances)
star_angles = np.random.uniform(low=0.0, high=2*np.pi, size=star_distances.shape)

fig, ax = plt.subplots(1, 1)

star_x = star_distances * np.cos(star_angles)
star_y = star_distances * np.sin(star_angles)

for x, y in zip(star_x[:], star_y[:]):
    ax.plot(x, y, alpha=0.1)

plt.show()

# Generate a batch of phase screens
#   - Use two layers based on Mohr data
#   - Build N screens for each layer, with the correct sigma for each.
#   - This results in N^2 combinations of phase screens, matching the number of asterisms



# Train an ML system to estimate target mode aberrations
#   - Train on infinite batches with N = 100
#   - Save and reload the model after finishing a batch

# After each batch completes, message a comparison process that runs in parallel
#   - Compare system to Ragazzoni LSQ approach after each batch completes

# ? Train a companion model on a static asterism to compare loss of precision when adding variance of the star field
#   - Static asterism will be an equilateral triangle of 3 stars at the mode distance of all stars