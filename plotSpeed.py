import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import fitting as fit
import matplotlib


filename = "speedDat"
outfile = open(filename, 'rb')
ns, timesPy, timesC = pkl.load(outfile)

outfile.close()

obj_Model = fit.models
obj_Model.setMod(obj_Model, 0, [3])

data = np.array([ns, timesPy, [1]*len(ns)])
data = np.transpose(data)

# Optimise model
dict_Result = opt.minimize(fit.Chi2,
                           obj_Model.arr_Vals,
                           method='nelder-mead',
                           args=(obj_Model, data))

# Storing model data
obj_Model.arr_Vals = dict_Result.x
obj_Model.float_Chi2 = dict_Result.fun

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 22
})

plt.locator_params(nbins=4)
plt.xlabel("Number of particles")
plt.ylabel("t/$m$s")
plt.plot(ns, timesPy*1e3, color='#68246D', linewidth=3.0)
plt.plot(ns, obj_Model.f(obj_Model, ns, obj_Model.arr_Vals)*1e3, color='#BE1E2D', linewidth=3.0)
plt.tight_layout()

plt.savefig("BF.pgf")
obj_Model.setMod(obj_Model, 2, [])

data = np.array([ns, timesC, [1]*len(ns)])
data = np.transpose(data)

# Optimise model
dict_Result = opt.minimize(fit.Chi2,
                           obj_Model.arr_Vals,
                           method='nelder-mead',
                           args=(obj_Model, data))

# Storing model data
obj_Model.arr_Vals = dict_Result.x
obj_Model.float_Chi2 = dict_Result.fun

# plt.xlabel("Number of particles")
# plt.ylabel("t/$m$s")
# plt.plot(ns, timesC*1e3, color='#68246D', linewidth=3.0)
# plt.plot(ns, obj_Model.f(obj_Model, ns, obj_Model.arr_Vals)*1e3)
plt.tight_layout()

# plt.savefig("tree.pgf")

