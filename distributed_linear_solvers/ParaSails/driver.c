#include <stdlib.h>
#include <assert.h>
#include "Common.h"
#include "Matrix.h"
#include "ParaSails.h"
#include "ConjGrad.h"

int main(int argc, char *argv[])
{
    int mype, npes;
    Matrix *A;
    ParaSails *ps;
    FILE *file;
    int n, beg_row, end_row;
    double time0, time1;

    double *x, *y, *b;
    int i, j;
    double thresh;
    double selparam;
    int nlevels;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    /* Read number of rows in matrix */
    file = fopen(argv[1], "r");
    assert(file != NULL);
    fscanf(file, "%d\n", &n);
    fclose(file);
    assert(n >= npes);

    beg_row = (int) ((double)(mype*n) / npes) + 1; /* assumes 1-based */
    end_row = (int) ((double)((mype+1)* n) / npes);
    if (mype == 0)
        assert(beg_row == 1);
    if (mype == npes-1)
        assert(end_row == n);

    x = (double *) malloc((end_row-beg_row+1) * sizeof(double));
    b = (double *) malloc((end_row-beg_row+1) * sizeof(double));

    A = MatrixCreate(MPI_COMM_WORLD, beg_row, end_row);

    MatrixRead(A, argv[1]);
    /* MatrixPrint(A, "A"); */

    /* Right-hand side */
    if (argc > 2)
        RhsRead(b, A, argv[2]);
    else
        for (i=0; i<end_row-beg_row+1; i++)
            b[i] = (double) (2*rand()) / (double) RAND_MAX - 1.0;

    while (1)
    {
        /* Initial guess */
        for (i=0; i<end_row-beg_row+1; i++)
            x[i] = 0.0;

#if ONE_TIME
        selparam = 1.0;
	nlevels = 0;
#else
        if (mype == 0)
        {
	    fflush(stdout);
            printf("Enter parameters selparam (0.75), nlevels (1): ");
            scanf("%lf %d", &selparam, &nlevels);
            printf("selparam %f, nlevels %d\n", selparam, nlevels);
            fflush(stdout);
	}

	MPI_Bcast(&selparam, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nlevels,  1, MPI_INT,    0, MPI_COMM_WORLD);

        if (nlevels < 0)
            break;
#endif

        time0 = MPI_Wtime();
        ps = ParaSailsCreate(A);
        thresh = ParaSailsSelectThresh(ps, selparam);
        ParaSailsSetupPattern(ps, thresh, nlevels);
        ParaSailsSetupValues(ps, A);
        time1 = MPI_Wtime();
        printf("%d: Total time for ParaSails: %f\n", mype, time1-time0);
        i = MatrixNnz(ps->M);
        j = (MatrixNnz(A) - n) / 2 + n;
        if (mype == 0) printf("number of nonzeros: %d (%.2f)\n", i, i/(double)j);
        /*MatrixPrint(ps->M, "M");*/

        time0 = MPI_Wtime();
        PCG_ParaSails(A, ps, b, x, 1.e-8, 1500);
        time1 = MPI_Wtime();
        printf("%d: Total time for it sol: %f\n", mype, time1-time0);

        MatrixMatvecComplete(A); /* convert matrix back to global numbering */
        ParaSailsDestroy(ps);

#if ONE_TIME
        break;
#endif
    }

    free(x);
    free(b);

    MatrixDestroy(A);
    MPI_Finalize();

    return 0;
}
