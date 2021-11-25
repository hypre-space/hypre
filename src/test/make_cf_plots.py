import matplotlib.pyplot as plt
import numpy as np

maxIter = 100

# Intermediate CF markers
for it in range(maxIter):
    try:
        cf_marker = np.loadtxt(f"CF_marker_it{it}_rank0.txt")

        n = int(np.sqrt(len(cf_marker)))

        x = np.arange(n)
        X, Y = np.meshgrid(x, x)
        X = X.flatten()
        Y = Y.flatten()

        fpts_x = X[cf_marker < 0]
        fpts_y = Y[cf_marker < 0]

        cpts_x = X[cf_marker > 0]
        cpts_y = Y[cf_marker > 0]

        upts_x = X[cf_marker == 0]
        upts_y = Y[cf_marker == 0]

        plt.figure()
        plt.scatter(cpts_x, cpts_y, marker='s', s=6, c='r')
        plt.scatter(fpts_x, fpts_y, marker='x', s=4, c='k')
        plt.scatter(upts_x, upts_y, marker='.', s=1, c='b')



        filename = f"plot_cf_it{it}_rank0.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    except:
        break


# Tentative C-pts
for it in range(maxIter):
    try:
        cf_marker = np.loadtxt(f"tentative_CF_marker_it{it}_rank0.txt")

        n = int(np.sqrt(len(cf_marker)))

        x = np.arange(n)
        X, Y = np.meshgrid(x, x)
        X = X.flatten()
        Y = Y.flatten()

        fpts_x = X[cf_marker < 0]
        fpts_y = Y[cf_marker < 0]

        cpts_x = X[cf_marker > 0]
        cpts_y = Y[cf_marker > 0]

        upts_x = X[cf_marker == 0]
        upts_y = Y[cf_marker == 0]

        plt.figure()
        plt.scatter(cpts_x, cpts_y, marker='s', s=6, c='r')
        plt.scatter(fpts_x, fpts_y, marker='x', s=4, c='k')
        plt.scatter(upts_x, upts_y, marker='.', s=1, c='b')



        filename = f"plot_tentative_c_it{it}_rank0.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    except:
        break


# Intermediate measure arrays
for it in range(maxIter):
    try:
        measure = np.loadtxt(f"measure_it{it}_rank0.txt")

        n = int(np.sqrt(len(measure)))

        x = np.arange(n)
        X, Y = np.meshgrid(x, x)
        X = X.flatten()
        Y = Y.flatten()

        measure_x = X[measure > 0]
        measure_y = Y[measure > 0]

        plt.figure()
        plt.scatter(measure_x, measure_y, c=measure[measure > 0])
        plt.colorbar()

        filename = f"plot_measure_it{it}_rank0.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    except:
        break


# Rejection measures
for it in range(maxIter):
    try:
        measure = np.loadtxt(f"rejection_measure_it{it}_rank0.txt")

        n = int(np.sqrt(len(measure)))

        x = np.arange(n)
        X, Y = np.meshgrid(x, x)
        X = X.flatten()
        Y = Y.flatten()

        measure_x = X[measure > 0]
        measure_y = Y[measure > 0]

        plt.figure()
        plt.scatter(measure_x, measure_y, c=measure[measure > 0])
        plt.colorbar()

        filename = f"plot_rejection_measure_it{it}_rank0.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    except:
        break
