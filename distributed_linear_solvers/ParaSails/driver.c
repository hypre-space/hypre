#include <stdlib.h>
#include <assert.h>
#include "Common.h"
#include "Matrix.h"
#include "ParaSails.h"
#include "ConjGrad.h"

extern double parasails_loadbal_beta;

void report_times(MPI_Comm comm, double setup_time, double solve_time)
{
    int mype, npes;
    double setup_times[1024];
    double max_solve_time;
    double m = 0.0, tot = 0.0;
    int i;

    MPI_Comm_rank(comm, &mype);
    MPI_Comm_size(comm, &npes);

    MPI_Gather(&setup_time, 1, MPI_DOUBLE, setup_times, 1, MPI_DOUBLE, 0, comm);

    MPI_Reduce(&solve_time, &max_solve_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    for (i=0; i<npes; i++)
        m = MAX(m, setup_times[i]);

    if (mype != 0)
	return;

    printf("**********************************************\n");
    printf("***    Setup    Solve    Total\n");
    printf("III %8.1f %8.1f %8.1f\n", m, max_solve_time, m+max_solve_time);
    printf("**********************************************\n");
}

int main(int argc, char *argv[])
{
    int mype, npes;
    Matrix *A;
    ParaSails *ps;
    FILE *file;
    int n, beg_row, end_row;
    int nnza, nnz0, nnz1;
    double time0, time1;
    double setup_time, solve_time;

    double *x, *y, *b;
    int i, j;
    double thresh;
    double selparam;
    double filter;
    int nlevels;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    /* Read number of rows in matrix */
    file = fopen(argv[1], "r");
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
        selparam = 0.00;
	nlevels = 1;
	filter = 0.0;
        parasails_loadbal_beta = 0.0;
#else
        if (mype == 0)
        {
            printf("Enter parameters selparam (0.75), nlevels (1), "
	        "filter (0.1), beta (0.0): ");
	    fflush(NULL);
            scanf("%lf %d %lf %lf", &selparam, &nlevels, 
		&filter, &parasails_loadbal_beta);
	}

	MPI_Bcast(&selparam, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nlevels,  1, MPI_INT,    0, MPI_COMM_WORLD);
	MPI_Bcast(&filter,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&parasails_loadbal_beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (nlevels < 0)
            break;
#endif

        if (mype == 0) 
	{
            printf("selparam %f, nlevels %d, filter %f, beta %f\n", 
		selparam, nlevels, filter, parasails_loadbal_beta);
            fflush(stdout);
	}

        MPI_Barrier(MPI_COMM_WORLD);
        time0 = MPI_Wtime();
        ps = ParaSailsCreate(A);

#ifdef NONSYM
        ParaSailsSetSym(ps, 0);
#else
        ParaSailsSetSym(ps, 1);
#endif

        /* thresh = ParaSailsSelectThresh(ps, selparam); */
        thresh=selparam;
#ifdef DIAG_PRECON
        thresh=10.0;
#endif
        if (mype == 0) 
            printf("thresh: %e\n", thresh);
        ParaSailsSetupPattern(ps, thresh, nlevels);
        ParaSailsSetupValues(ps, A);
        nnz0 = MatrixNnz(ps->M);

#if 0
        /* filtration step */
        filter = ParaSailsSelectFilter(ps, filter);
        if (mype == 0) 
            printf("filter: %f\n", filter);
#endif

	ParaSailsFilterValues(ps, filter);
        nnz1 = MatrixNnz(ps->M);

#if 0
        if (mype == 0) 
            printf("SETTING UP VALUES AGAIN WITH FILTERED PATTERN\n");
        ParaSailsSetupValues(ps, A);
#endif

	ParaSailsComplete(ps);
        time1 = MPI_Wtime();
	setup_time = time1-time0;
        printf("SETUP %3d %8.1f\n", mype, setup_time);

        /* MatrixPrint(ps->M, "M"); */

#ifdef NONSYM
        nnza = MatrixNnz(A);
#else
        nnza = (MatrixNnz(A) - n) / 2 + n;
#endif
        if (mype == 0) 
        {
            printf("%s\n", argv[1]);
            printf("Inumber of nonzeros: %d (%.2f)\n", nnz0, nnz0/(double)nnza);
            printf("Innz after filter  : %d (%.2f)\n", nnz1, nnz1/(double)nnza);
        }

        time0 = MPI_Wtime();
#ifdef NONSYM
        FGMRES_ParaSails(A, ps, b, x, 50, 1.e-8, 1500);
#else
        PCG_ParaSails(A, ps, b, x, 1.e-8, 1700);
#endif
        time1 = MPI_Wtime();
	solve_time = time1-time0;

	report_times(MPI_COMM_WORLD, setup_time, solve_time);

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
