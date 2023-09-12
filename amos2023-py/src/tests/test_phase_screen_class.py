import matplotlib.pyplot as plt

from tomography.phase_screen import PhaseScreen

ps_size = (15000, 15000)
co_size = (256, 256)

ps = PhaseScreen(ps_size, 0.6, 0.1)

ps.generate([0, 1, 2, 3, 4])
print(ps.f.keys())

for key in ps.f.keys():
    print(ps.f[key].keys())

ps.find_scaling(co_size)
print(ps._exp_sigma)
print(ps._est_sigma)
print(ps.k)

screen = ps.get(0)

plt.imshow(screen)
plt.show()
