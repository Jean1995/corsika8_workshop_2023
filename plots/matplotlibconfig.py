font_size = 10
label_size = 9
tick_size = 9
width = 5.45

params = {
    'backend': 'pdf',
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': font_size,
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage{amsmath}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[detect-all,locale=US,separate-uncertainty=true,per-mode=symbol-or-fraction]{siunitx}",
        ]),
    'axes.labelsize': font_size,
    'legend.numpoints': 1,
    'legend.shadow': False,
    'legend.fontsize': font_size,
    'xtick.labelsize': tick_size,
    'ytick.labelsize': tick_size,
    'axes.unicode_minus': True,
    'axes.labelsize' : label_size,
}


grid_conf = 'ls=":", lw=0.2, zorder=0'

grid_conf = {
    'ls': ':',
    'lw': 0.2,
    'zorder': 0,
}