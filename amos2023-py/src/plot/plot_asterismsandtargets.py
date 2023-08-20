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
    for g in np.linspace(5, 30, 6, endpoint=True):      # gamma angle (from telescope axis)
        target = (t + g, g)
        targets.append(target)

for t in np.linspace(0, 2*np.pi, 6, endpoint=False):    # theta angle (acw from horizon)
    for g in np.linspace(35, 95, 7, endpoint=True):     # gamma angle (from telescope axis)
        target = (t + g, g)
        targets.append(target)

# calculate isoplanatic angle
isoplanatic_angle = 2.44 * wavelength / r_0
isoplanatic_angle = util.rad2arcsec(isoplanatic_angle)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(7,7)
ax.grid(which="both", color='#000000', alpha=0.1, linestyle='-', zorder=0)
clrs = ["#ff0000", "#00ff00", "#0000ff"]
for (k, v), c in zip(guidestar_sets.items(), clrs):
    if k[1] != -1:
        lbl = 'Symmetric, {} guide stars, {}" radius'.format(k[0], k[1])
    else:
        lbl = 'Asymmetric, 5 guide stars'
    x = [g * np.sin(t) for t, g in v]
    y = [g * np.cos(t) for t, g in v]
    ax.scatter(x, y, label=lbl, marker='*', c=c, s=100, zorder=3.0)

print(targets)

x = [g * np.sin(t) for t, g in targets]
y = [g * np.cos(t) for t, g in targets]
ax.scatter(x, y, label='Target Positions', c="#000000", s=20, marker='+', alpha=0.75, zorder=3.5)

util.plotcirc(ax, (0, 0), 30, '--', color=clrs[0], alpha=0.2)
util.plotcirc(ax, (0, 0), 60, '--', color=clrs[1], alpha=0.2)
util.plotcirc(ax, (0, 0), isoplanatic_angle, '', linestyle='dashdot', color="#223344", alpha=0.5, label="Isoplanatic Patch")

gs = guidestar_sets[(5, -1)]
gs.sort(key=lambda x: x[0])
x = [g * np.sin(t) for t, g in gs]
y = [g * np.cos(t) for t, g in gs]
x.append(x[0])
y.append(y[0])
ax.plot(x, y, '--', color=clrs[2], alpha=0.2)

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
ax.set_xticks(np.arange(-90, 91, 30))
ax.set_yticks(np.arange(-90, 91, 30))
ax.axis("equal")
ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0.93))

plt.show()