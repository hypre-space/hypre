#include <assert.h>
#include "mpi.h"
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
    int i;
    double thresh;

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

    A = MatrixCreate(MPI_COMM_WORLD, beg_row, end_row);

    MatrixRead(A, argv[1]);
    /* MatrixPrint(A, "A"); */

    x = (double *) malloc((end_row-beg_row+1) * sizeof(double));
    y = (double *) malloc((end_row-beg_row+1) * sizeof(double));
    b = (double *) malloc((end_row-beg_row+1) * sizeof(double));

    for (i=0; i<end_row-beg_row+1; i++)
    {
        x[i] = (double) (i+beg_row);
        y[i] = 0.0;
        b[i] = (double) (i+beg_row);
    }

/*
    MatrixMatvecSetup(A);

    MatrixMatvec(A, x, y);
    for (i=0; i<end_row-beg_row+1; i++)
        printf("%d:  %d %f\n", mype, i, y[i]);

    MatrixMatvecTrans(A, x, y);
    for (i=0; i<end_row-beg_row+1; i++)
        printf("%d:  %d %f\n", mype, i, y[i]);

    MatrixMatvecComplete(A);
*/

    time0 = MPI_Wtime();
    ps = ParaSailsCreate(A);
    thresh = ParaSailsSelectThresh(ps, 0.75);
    ParaSailsSetupPattern(ps, thresh, 1);
    ParaSailsSetupValues(ps, A);
    time1 = MPI_Wtime();
    printf("%d: Total time for ParaSails: %f\n", mype, time1-time0);
    /*MatrixPrint(ps->M, "M");*/

    time0 = MPI_Wtime();
    PCG_ParaSails(A, ps, b, y, 1.e-8, 500);
    time1 = MPI_Wtime();
    printf("%d: Total time for it sol: %f\n", mype, time1-time0);

    ParaSailsDestroy(ps);

    MatrixDestroy(A);
    MPI_Finalize();

    return 0;
}
