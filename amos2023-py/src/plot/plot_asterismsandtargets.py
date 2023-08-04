import numpy as np
import matplotlib.pyplot as plt

from util import util

import constellations

# TARGET POSITIONS
line_theta = 30 * (np.pi/180)   # radians
line_spread = 3 # arcseconds
line_offset = 3 # arcseconds

# Set up tests
guidestar_sets = {}
for i, (n, g) in enumerate([(3, 7.5), (6, 15)]):
    guidestars = [(t, g) for t in np.linspace(i, (2*np.pi)+i, n, endpoint=False)]
    guidestar_sets[(n, g)] = guidestars

cons = constellations.get_natural()

for i, con in enumerate(cons):
    guidestar_sets[(len(cons), -1-i)] = [(t, g * 3600 * 180 / np.pi) for t, g in con]

targets = []
# line through origin
targets.append((0, 0))
targets.append((line_theta, line_spread))
targets.append((line_theta+np.pi, line_spread))
for t in np.linspace(0, 2*np.pi, 4, endpoint=False):    # theta angle (acw from horizon)
    for g in [line_spread, 2 * line_spread]: # gamma angle (from telescope axis)
        targets.append((t, g))

#offset line
h = np.sqrt(line_spread * line_spread + line_offset * line_offset) #distance from origin of spread out points
phi = np.arctan(line_spread/line_offset) # angle of line points from closest point
targets.append((line_theta+(np.pi/2.0), line_offset))
targets.append((line_theta+(np.pi/2.0)-phi, h))
targets.append((line_theta+(np.pi/2.0)+phi, h))

ax = plt.subplot()
ax.grid(which="both", color='#D8D8D8', linestyle='-', zorder=0)

for k, v in guidestar_sets.items():
    if k[0] > 1:
        lbl = 'Test Asterism: {} guidestars, {} arcsecond radius'.format(k[0], k[1])
    else:
        lbl = 'Natural Guidestar Constellation'
    x = [g * np.sin(t) for t, g in v]
    y = [g * np.cos(t) for t, g in v]
    ax.scatter(x, y, label=lbl, marker='*', zorder=3.0)

print(targets)

x = [g * np.sin(t) for t, g in targets]
y = [g * np.cos(t) for t, g in targets]
ax.scatter(x, y, label='Science Target Positions', zorder=3.5)

util.plotcirc(ax, (0, 0), 7.5, '--', alpha=0.2)
util.plotcirc(ax, (0, 0), 15, '--', alpha=0.2)

target_line = [(line_theta, line_spread), (line_theta+np.pi, line_spread)]
x = [g * np.sin(t) for t, g in target_line]
y = [g * np.cos(t) for t, g in target_line]
ax.plot(x, y, 'r--', alpha=0.2)

target_line = [(line_theta+(np.pi/2.0)-phi, h), (line_theta+(np.pi/2.0)+phi, h)]
x = [g * np.sin(t) for t, g in target_line]
y = [g * np.cos(t) for t, g in target_line]
ax.plot(x, y, 'r--', alpha=0.2)

ax.set_xlabel("Angle (arcseconds)")
ax.set_ylabel("Angle (arcseconds)")
ax.set_xticks(np.arange(-40, 41, 10))
ax.set_yticks(np.arange(-40, 41, 10))
ax.axis("equal")
ax.legend()

plt.show()