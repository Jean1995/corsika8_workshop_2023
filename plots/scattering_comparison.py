import proposal as pp


### check correct proposal version. uncomment this line if you want to run the scripts with newer/other versions
assert(pp.__version__=='7.5.1')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlibconfig as conf

import matplotlib.colors as mcolors
import scipy.stats as stats
from tqdm import trange

from uncertainties import ufloat
import time


mpl.use("pgf")
plt.rcParams.update(conf.params)


# plots-specific size (width should always be taken from conf)
plt.rcParams["figure.figsize"] = (conf.width, 3)

pp.RandomGenerator.get().set_seed(42)

particle = pp.particle.EMinusDef()
medium = pp.medium.Air()
cuts = pp.EnergyCutSettings(np.inf, 1, False)

cross = pp.crosssection.make_std_crosssection(particle, medium, cuts, True)

moliere = pp.make_multiple_scattering("moliere", particle, medium, cross, True)
moliere_interpol = pp.make_multiple_scattering("moliereinterpol", particle, medium, cross, True)
highland = pp.make_multiple_scattering("highland", particle, medium, cross, True)

displacement = pp.make_displacement(cross, False)

E_i = 1e3
E_f = 1e3 * 0.9
X = displacement.solve_track_integral(E_i, E_f) # calculate grammage
print(f"grammage: {X} g/^cm^2")
print(f"distance: {X/ medium.mass_density} cm")

STAT = int(1e7)
BATCHES = 100 # split into this number of batches, take mean and std of these batches as times

theta_list_moliere = []
theta_list_moliere_interpol = []
theta_list_highland = []

times = []

for param, theta_list in zip([moliere, moliere_interpol, highland], 
                             [theta_list_moliere, theta_list_moliere_interpol, theta_list_highland]):
    individual_times = []
    for batch in trange(BATCHES):
        start_time = time.time()
        for i in range(int(STAT/BATCHES)):
            scattering_offset = param.scatter(X, E_i, E_f, np.random.rand(4))
            theta_list.append(scattering_offset.tx)
            theta_list.append(scattering_offset.ty)
        end_time = time.time()
        individual_times.append((end_time - start_time)/(STAT/BATCHES)*1e6) # save time per calculation in µs
    mean = np.mean(individual_times)
    std = np.std(individual_times)
    times.append(ufloat(mean, std))
    
theta_minmax = 6
NUM_BINS = 30

bins = np.linspace(-theta_minmax, theta_minmax, NUM_BINS)
labels = (r'$\texttt{Molière}$', r'$\texttt{Highland}$')
colors = ('tab:blue', 'tab:orange')

for theta_list, label, color in zip([theta_list_moliere, theta_list_highland], labels, colors):
    plt.hist(np.rad2deg(theta_list), density=False, bins=bins, histtype='step', label=label, color=color)

# this can be used to plot the theory curve for highland
#theta0 = highland.CalculateTheta0(X, E_i, E_f)
#minmax_highland = max(theta_list_highland)
#theta_list = np.linspace(-minmax_highland, minmax_highland, 500)
#plt.plot(np.rad2deg(x_list), stats.norm.pdf(x_list, 0, theta0))

plt.legend(loc='best')
plt.yscale('log')
plt.grid(conf.grid_conf)
plt.ylabel(r'Frequency')
plt.xlabel(r'$\theta \,/\, \mathrm{deg} $')

plt.savefig('scattering_comparison.pdf', bbox_inches='tight')

### plot times

labels_time = (r'$\texttt{Molière}$', r'$\texttt{Highland}$', r'$\texttt{MolièreInterpol}$')

fig, ax = plt.subplots()
y_pos = np.arange(len(labels_time))

time = np.array([times[0].n, times[2].n, times[1].n])
error = np.array([times[0].s,  times[2].s, times[1].s])

ax.barh(y_pos, time, xerr=error, align='center') # one could add color=colors here, but this might be a tick too much
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_time)
ax.invert_yaxis()
ax.set_aspect(1.5)
ax.set_xlabel('Computing time per scattering angle calculation in µs')
#plt.xscale('log')
#plt.xlim(1e4, 1e6)
plt.savefig('scattering_runtimes.pdf', bbox_inches='tight')
