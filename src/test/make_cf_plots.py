import matplotlib.pyplot as plt
import numpy as np

maxIter = 20
numRanks = 4

# Intermediate CF markers
for it in range(maxIter):
    try:
        plt.figure()
        for rank in range(numRanks):
            cf_marker = np.loadtxt(f"CF_marker_it{it}_rank{rank}.txt")

            n = int(np.sqrt(len(cf_marker)))
            rank_n = int(np.sqrt(numRanks))
            rank_x = rank % rank_n
            rank_y = rank // rank_n

            x = np.arange(n)
            X, Y = np.meshgrid(x, x)
            X = X.flatten() + (rank_x * n)
            Y = Y.flatten() + (rank_y * n)

            fpts_x = X[cf_marker < 0]
            fpts_y = Y[cf_marker < 0]

            cpts_x = X[cf_marker > 0]
            cpts_y = Y[cf_marker > 0]

            upts_x = X[cf_marker == 0]
            upts_y = Y[cf_marker == 0]

            plt.scatter(cpts_x, cpts_y, marker='s', s=6, c='r')
            plt.scatter(fpts_x, fpts_y, marker='x', s=4, c='k')
            plt.scatter(upts_x, upts_y, marker='.', s=1, c='b')

        filename = f"plot_cf_it{it}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    except:
        plt.close()
        break


# Tentative C-pts
for it in range(maxIter):
    try:
        plt.figure()
        for rank in range(numRanks):
            cf_marker = np.loadtxt(f"tentative_CF_marker_it{it}_rank{rank}.txt")

            n = int(np.sqrt(len(cf_marker)))
            rank_n = int(np.sqrt(numRanks))
            rank_x = rank % rank_n
            rank_y = rank // rank_n

            x = np.arange(n)
            X, Y = np.meshgrid(x, x)
            X = X.flatten() + (rank_x * n)
            Y = Y.flatten() + (rank_y * n)

            fpts_x = X[cf_marker < 0]
            fpts_y = Y[cf_marker < 0]

            cpts_x = X[cf_marker > 0]
            cpts_y = Y[cf_marker > 0]

            upts_x = X[cf_marker == 0]
            upts_y = Y[cf_marker == 0]

            plt.scatter(cpts_x, cpts_y, marker='s', s=6, c='r')
            plt.scatter(fpts_x, fpts_y, marker='x', s=4, c='k')
            plt.scatter(upts_x, upts_y, marker='.', s=1, c='b')

        filename = f"plot_tentative_c_it{it}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    except:
        break


# Intermediate measure arrays
for it in range(maxIter):
    try:
        plt.figure()
        for rank in range(numRanks):
            measure = np.loadtxt(f"measure_it{it}_rank{rank}.txt")

            n = int(np.sqrt(len(measure)))
            rank_n = int(np.sqrt(numRanks))
            rank_x = rank % rank_n
            rank_y = rank // rank_n

            x = np.arange(n)
            X, Y = np.meshgrid(x, x)
            X = X.flatten() + (rank_x * n)
            Y = Y.flatten() + (rank_y * n)

            measure_x = X[measure > 0]
            measure_y = Y[measure > 0]

            plt.scatter(measure_x, measure_y, c=measure[measure > 0])

        plt.colorbar()

        filename = f"plot_measure_it{it}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    except:
        break


# Rejection measures
for it in range(maxIter):
    try:
        plt.figure()
        for rank in range(numRanks):
            measure = np.loadtxt(f"rejection_measure_it{it}_rank{rank}.txt")

            n = int(np.sqrt(len(measure)))
            rank_n = int(np.sqrt(numRanks))
            rank_x = rank % rank_n
            rank_y = rank // rank_n

            x = np.arange(n)
            X, Y = np.meshgrid(x, x)
            X = X.flatten() + (rank_x * n)
            Y = Y.flatten() + (rank_y * n)

            measure_x = X[measure > 0]
            measure_y = Y[measure > 0]

            plt.scatter(measure_x, measure_y, c=measure[measure > 0])

        plt.colorbar()

        filename = f"plot_rejection_measure_it{it}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    except:
        break
