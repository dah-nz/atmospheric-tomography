import numpy as np
import matplotlib.pyplot as plt

from util import util

import constellations


wavelength = 600e-9
r_0 = 0.05

# SET UP GUIDE STARS
guidestar_sets = {}
for i, (n, g) in enumerate([(3, 30), (6, 60)]):
    guidestars = [(t, g) for t in np.linspace(i, (2*np.pi)+i, n, endpoint=False)]
    guidestar_sets[(n, g)] = guidestars

cons = constellations.get_natural()

for i, con in enumerate(cons):
    guidestar_sets[(len(con), -1-i)] = [(t, g * 3600 * 180 / np.pi) for t, g in con]

# TARGET POSITIONS
targets = []
targets.append((0, 0))
for t in np.linspace(0, 2*np.pi, 8, endpoint=False):   # theta angle (acw from horizon)
    for g in np.linspace(5, 25, 5, endpoint=True):      # gamma angle (from telescope axis)
        target = (t + g, g)
        targets.append(target)

for t in np.linspace(0, 2*np.pi, 8, endpoint=False):    # theta angle (acw from horizon)
    for g in np.linspace(35, 95, 7, endpoint=True):     # gamma angle (from telescope axis)
        target = (t + g, g)
        targets.append(target)

# calculate isoplanatic angle
isoplanatic_angle = 2.44 * wavelength / r_0
isoplanatic_angle = util.rad2arcsec(isoplanatic_angle)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(7,7)
ax.grid(which="both", color='#D8D8D8', linestyle='-', zorder=0)

for k, v in guidestar_sets.items():
    if k[1] != -1:
        lbl = 'Symmetric, {} guide stars, {}" radius'.format(k[0], k[1])
    else:
        lbl = 'Asymmetric, 5 guide stars'
    x = [g * np.sin(t) for t, g in v]
    y = [g * np.cos(t) for t, g in v]
    ax.scatter(x, y, label=lbl, marker='*', zorder=3.0)

print(targets)

x = [g * np.sin(t) for t, g in targets]
y = [g * np.cos(t) for t, g in targets]
ax.scatter(x, y, label='Target Positions', zorder=3.5)

util.plotcirc(ax, (0, 0), 30, '--', alpha=0.2)
util.plotcirc(ax, (0, 0), 60, '--', alpha=0.2)
util.plotcirc(ax, (0, 0), isoplanatic_angle, '--', alpha=0.2)

gs = guidestar_sets[(5, -1)]
gs.sort(key=lambda x: x[0])
x = [g * np.sin(t) for t, g in gs]
y = [g * np.cos(t) for t, g in gs]
x.append(x[0])
y.append(y[0])
ax.plot(x, y, 'g--', alpha=0.2)

# target_line = [(line_theta, line_spread), (line_theta+np.pi, line_spread)]
# x = [g * np.sin(t) for t, g in target_line]
# y = [g * np.cos(t) for t, g in target_line]
# ax.plot(x, y, 'r--', alpha=0.2)

# target_line = [(line_theta+(np.pi/2.0)-phi, h), (line_theta+(np.pi/2.0)+phi, h)]
# x = [g * np.sin(t) for t, g in target_line]
# y = [g * np.cos(t) for t, g in target_line]
# ax.plot(x, y, 'r--', alpha=0.2)

ax.set_xlabel("Angle (arcseconds)")
ax.set_ylabel("Angle (arcseconds)")
ax.set_xticks(np.arange(-100, 101, 20))
ax.set_yticks(np.arange(-100, 101, 20))
ax.axis("equal")
ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0.95))

plt.show()