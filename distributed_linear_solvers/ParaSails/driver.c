#include <stdlib.h>
#include <assert.h>
#include "Common.h"
#include "Matrix.h"
#include "ParaSails.h"
#include "ConjGrad.h"

/*
 * Usage: driver symmetric num_runs matrixfile [rhsfile]
 *
 * If num_runs == 1, then hard-coded parameters will be used; else the
 * user will be prompted for parameters (except on the final run).
 *
 * To simulate diagonal preconditioning, use a large value of thresh, 
 * e.g., thresh > 10.
 */

int main(int argc, char *argv[])
{
    int mype, npes;
    int symmetric;
    int num_runs;
    Matrix *A;
    ParaSails *ps;
    FILE *file;
    int n, beg_row, end_row;
    int nnza, nnz0;
    double time0, time1;
    double setup_time, solve_time;
    double max_setup_time, max_solve_time;

    double *x, *y, *b;
    int i, j;
    double thresh;
    int nlevels;
    double filter;
    double loadbal;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    /* Read number of rows in matrix */
    symmetric = atoi(argv[1]);
    num_runs  = atoi(argv[2]);

    file = fopen(argv[3], "r");
    assert(file != NULL);
#ifdef EMSOLVE
    fscanf(file, "%*d %d\n", &n);
#else
    fscanf(file, "%d\n", &n);
#endif
    fclose(file);
    assert(n >= npes);

    beg_row = (int) ((double)(mype*n) / npes) + 1; /* assumes 1-based */
    end_row = (int) ((double)((mype+1)* n) / npes);

    if (mype == 0)
        assert(beg_row == 1);
    if (mype == npes-1)
        assert(end_row == n);

#ifdef EMSOLVE
    beg_row--;
    end_row--;
#endif

    x = (double *) malloc((end_row-beg_row+1) * sizeof(double));
    b = (double *) malloc((end_row-beg_row+1) * sizeof(double));

    A = MatrixCreate(MPI_COMM_WORLD, beg_row, end_row);

    MatrixRead(A, argv[3]);
    if (mype == 0) 
        printf("%s\n", argv[3]);

    /* MatrixPrint(A, "A"); */

    /* Right-hand side */
    if (argc > 4)
    {
        RhsRead(b, A, argv[4]);
        if (mype == 0) 
            printf("Using rhs from %s\n", argv[4]);
    }
    else
    {
        for (i=0; i<end_row-beg_row+1; i++)
            b[i] = (double) (2*rand()) / (double) RAND_MAX - 1.0;
    }

    while (num_runs--)
    {
        /* Initial guess */
        for (i=0; i<end_row-beg_row+1; i++)
            x[i] = 0.0;

	if (num_runs == 0)
	{
            thresh = 0.0;
	    nlevels = 1;
	    filter = 0.0;
            loadbal = 0.0;
	}
	else
	{
            if (mype == 0)
            {
                printf("Enter parameters thresh (0.75), nlevels (1), "
	            "filter (0.1), beta (0.0):\n");
	        fflush(NULL);
                scanf("%lf %d %lf %lf", &thresh, &nlevels, 
		    &filter, &loadbal);
	    }

	    MPI_Bcast(&thresh,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(&nlevels, 1, MPI_INT,    0, MPI_COMM_WORLD);
	    MPI_Bcast(&filter,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(&loadbal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (nlevels < 0)
                break;
	}

        /**************
	 * Setup phase   
	 **************/

        MPI_Barrier(MPI_COMM_WORLD);
        time0 = MPI_Wtime();

        ps = ParaSailsCreate(MPI_COMM_WORLD, beg_row, end_row, symmetric);

        ps->loadbal_beta = loadbal;

        ParaSailsSetupPattern(ps, A, thresh, nlevels);
        ParaSailsSetupValues(ps, A, filter);

        time1 = MPI_Wtime();
	setup_time = time1-time0;
        printf("SETUP %3d %8.1f\n", mype, setup_time);
	fflush(NULL);
        MPI_Barrier(MPI_COMM_WORLD);

	if (!strncmp(argv[3], "testpsmat", 8))
            MatrixPrint(ps->M, "M");

#if 0
        if (mype == 0) 
            printf("SETTING UP VALUES AGAIN WITH FILTERED PATTERN\n");
        ParaSailsSetupValues(ps, A, 0.0);
#endif

        /*****************
	 * Solution phase
	 *****************/

        MPI_Barrier(MPI_COMM_WORLD);
        time0 = MPI_Wtime();

        if (!symmetric)
            FGMRES_ParaSails(A, ps, b, x, 50, 1.e-8, 1500);
	else
            PCG_ParaSails(A, ps, b, x, 1.e-8, 1700);

        time1 = MPI_Wtime();
	solve_time = time1-time0;

        MPI_Reduce(&setup_time, &max_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0, 
	    MPI_COMM_WORLD);
        MPI_Reduce(&solve_time, &max_solve_time, 1, MPI_DOUBLE, MPI_MAX, 0, 
	    MPI_COMM_WORLD);

	if (mype == 0)
	{
            printf("**********************************************\n");
            printf("***    Setup    Solve    Total\n");
            printf("III %8.1f %8.1f %8.1f\n", max_setup_time, max_solve_time, 
		max_setup_time+max_solve_time);
            printf("**********************************************\n");
	}

        ParaSailsDestroy(ps);
    }

    free(x);
    free(b);

    MatrixDestroy(A);
    MPI_Finalize();

    return 0;
}
