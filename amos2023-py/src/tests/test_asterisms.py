import numpy as np

import matplotlib.pyplot as plt

import constellations
from tomography import beacon

guidestar_amount = [3, 6]
beacon_radii = [7.5, 15]

guidestar_sets = {}
for n in guidestar_amount:
    for g in beacon_radii:   # angle of radius
        guidestars = [beacon.Beacon(t, g, gamma_arcseconds=True) for t in np.linspace(0, 2*np.pi, n, endpoint=False)]
        guidestar_sets[(n, g)] = guidestars

cons = constellations.get_natural()

for i, con in enumerate(cons):
    guidestar_sets[(len(con), -1-i)] = [beacon.Beacon(t, g) for t, g in con]

targets = {}
line_theta = 30 * (np.pi/180)   # radians
line_spread = 3 # arcseconds
line_offset = 3 # arcseconds
targets = {}
# line through origin
targets[(0, 0)] = beacon.Beacon(0, 0, gamma_arcseconds=True)
targets[(line_theta, line_spread)] = beacon.Beacon(line_theta, line_spread, gamma_arcseconds=True)
targets[(line_theta+np.pi, line_spread)] = beacon.Beacon(line_theta+np.pi, line_spread, gamma_arcseconds=True)

#offset line
h = np.sqrt(line_spread * line_spread + line_offset * line_offset) #distance from origin of spread out points
phi = np.arctan(line_spread/line_offset) # angle of line points from closest point
targets[(line_theta+(np.pi/2.0), line_offset)] = beacon.Beacon(line_theta+(np.pi/2.0), line_offset, gamma_arcseconds=True)
targets[(line_theta+(np.pi/2.0)-phi, h)] = beacon.Beacon(line_theta+(np.pi/2.0)-phi, h, gamma_arcseconds=True)
targets[(line_theta+(np.pi/2.0)+phi, h)] = beacon.Beacon(line_theta+(np.pi/2.0)+phi, h, gamma_arcseconds=True)
for t in np.linspace(0, 2*np.pi, 4, endpoint=False):    # theta angle (acw from horizon)
    for g in [line_spread, 2*line_spread]: # gamma angle (from telescope axis)
        target = beacon.Beacon(t, g, gamma_arcseconds=True)
        targets[(t, g)] = target


natural = constellations.get_natural()

altitude = 1000
colours = ['#2058b3', '#117348', '#a63a0f', '#830fa6']
i = 0

fig, axs = plt.subplots(1, 2)
fig.suptitle("Guide Star Asterisms\nat 500m altitude")
for k, v in guidestar_sets.items():
    pos = [s.position(altitude) for s in v]
    if k[0] == 3:
        a = 0
    if k[0] == 6:
        a = 1
    if k[1] == 7.5:
        clr = colours[0]
    elif k[1] == 15:
        clr = colours[1]
    else:
        clr = colours[2]
    x = [p[0] for p in pos]
    y = [p[1] for p in pos]
    if -1 in k:
        lbl = 'Natural Constellation'
    else:
        lbl = "{} arcseconds".format(k[1])
    axs[a].scatter(x, y, c=clr, label=lbl, marker='*')

axs[0].set_title("3 Guidestars")
axs[0].axis('equal')
axs[0].legend()
axs[0].set_xlabel("Guidestar Position (m)")
axs[0].set_ylabel("Guidestar Position (m)")

axs[1].set_title("6 Guidestars")
axs[1].axis('equal')
axs[1].legend()
axs[1].set_xlabel("Guidestar Position (m)")
axs[1].set_ylabel("Guidestar Position (m)")


for k, v in targets.items():
    pos = v.position(altitude)
    x = pos[0]
    y = pos[1]
    axs[0].scatter(x, y, c=colours[3], marker='8')
    axs[1].scatter(x, y, c=colours[3], marker='8')

plt.show()