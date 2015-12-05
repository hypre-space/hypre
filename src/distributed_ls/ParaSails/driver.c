/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




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

HYPRE_Int main(HYPRE_Int argc, char *argv[])
{
    HYPRE_Int mype, npes;
    HYPRE_Int symmetric;
    HYPRE_Int num_runs;
    Matrix *A;
    ParaSails *ps;
    FILE *file;
    HYPRE_Int n, beg_row, end_row;
    double time0, time1;
    double setup_time, solve_time;
    double max_setup_time, max_solve_time;
    double cost;

    double *x, *b;
    HYPRE_Int i, niter;
    double thresh;
    double threshg;
    HYPRE_Int nlevels;
    double filter;
    double loadbal;

    hypre_MPI_Init(&argc, &argv);
    hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mype);
    hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &npes);

    /* Read number of rows in matrix */
    symmetric = atoi(argv[1]);
    num_runs  = atoi(argv[2]);

    file = fopen(argv[3], "r");
    assert(file != NULL);
#ifdef EMSOLVE
    hypre_fscanf(file, "%*d %d\n", &n);
#else
    hypre_fscanf(file, "%d\n", &n);
#endif
    fclose(file);
    assert(n >= npes);

    beg_row = (HYPRE_Int) ((double)(mype*n) / npes) + 1; /* assumes 1-based */
    end_row = (HYPRE_Int) ((double)((mype+1)* n) / npes);

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

    A = MatrixCreate(hypre_MPI_COMM_WORLD, beg_row, end_row);

    MatrixRead(A, argv[3]);
    if (mype == 0) 
        hypre_printf("%s\n", argv[3]);

    /* MatrixPrint(A, "A"); */

    /* Right-hand side */
    if (argc > 4)
    {
        RhsRead(b, A, argv[4]);
        if (mype == 0) 
            hypre_printf("Using rhs from %s\n", argv[4]);
    }
    else
    {
        for (i=0; i<end_row-beg_row+1; i++)
            b[i] = (double) (2*rand()) / (double) RAND_MAX - 1.0;
    }

    while (num_runs && num_runs >= -1)
    {
        /* Initial guess */
        for (i=0; i<end_row-beg_row+1; i++)
            x[i] = 0.0;

	if (num_runs == -1)
	{
            thresh = 0.0;
	    nlevels = 0;
	    filter = 0.0;
            loadbal = 0.0;
	}
	else
	{
            if (mype == 0)
            {
#if PARASAILS_EXT_PATTERN
                hypre_printf("Enter parameters threshg, thresh, nlevels, "
	            "filter, beta:\n");
	        fflush(stdout);
                hypre_scanf("%lf %lf %d %lf %lf", &threshg, &thresh, &nlevels, 
		    &filter, &loadbal);
#else
                hypre_printf("Enter parameters thresh, nlevels, "
	            "filter, beta:\n");
	        fflush(stdout);
                hypre_scanf("%lf %d %lf %lf", &thresh, &nlevels, 
		    &filter, &loadbal);
#endif
	    }

	    hypre_MPI_Bcast(&threshg, 1, hypre_MPI_DOUBLE, 0, hypre_MPI_COMM_WORLD);
	    hypre_MPI_Bcast(&thresh,  1, hypre_MPI_DOUBLE, 0, hypre_MPI_COMM_WORLD);
	    hypre_MPI_Bcast(&nlevels, 1, HYPRE_MPI_INT,    0, hypre_MPI_COMM_WORLD);
	    hypre_MPI_Bcast(&filter,  1, hypre_MPI_DOUBLE, 0, hypre_MPI_COMM_WORLD);
	    hypre_MPI_Bcast(&loadbal, 1, hypre_MPI_DOUBLE, 0, hypre_MPI_COMM_WORLD);

            if (nlevels < 0)
                break;
	}

        /**************
	 * Setup phase   
	 **************/

        hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
        time0 = hypre_MPI_Wtime();

        ps = ParaSailsCreate(hypre_MPI_COMM_WORLD, beg_row, end_row, symmetric);

        ps->loadbal_beta = loadbal;

#if PARASAILS_EXT_PATTERN
        ParaSailsSetupPatternExt(ps, A, threshg, thresh, nlevels);
#else
        ParaSailsSetupPattern(ps, A, thresh, nlevels);
#endif

        time1 = hypre_MPI_Wtime();
	setup_time = time1-time0;

        cost = ParaSailsStatsPattern(ps, A);
	if (cost > 5.e11)
	{
            hypre_printf("Aborting setup and solve due to high cost.\n");
	    goto cleanup;
	}

        hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
        time0 = hypre_MPI_Wtime();

        err = ParaSailsSetupValues(ps, A, filter);
        if (err != 0)
	{
            hypre_printf("ParaSailsSetupValues returned error.\n");
	    goto cleanup;
	}

        time1 = hypre_MPI_Wtime();
	setup_time += (time1-time0);

        ParaSailsStatsValues(ps, A);

	if (!strncmp(argv[3], "testpsmat", 8))
            MatrixPrint(ps->M, "M");

#if 0
        if (mype == 0) 
            hypre_printf("SETTING UP VALUES AGAIN WITH FILTERED PATTERN\n");
        ps->loadbal_beta = 0;
        ParaSailsSetupValues(ps, A, 0.0);
#endif

        /*****************
	 * Solution phase
	 *****************/

	niter = 3000;
        if (MatrixNnz(ps->M) == n) /* if diagonal preconditioner */
	    niter = 5000;

        hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
        time0 = hypre_MPI_Wtime();

        if (symmetric == 1)
            PCG_ParaSails(A, ps, b, x, 1.e-8, niter);
	else
            FGMRES_ParaSails(A, ps, b, x, 50, 1.e-8, niter);

        time1 = hypre_MPI_Wtime();
	solve_time = time1-time0;

        hypre_MPI_Reduce(&setup_time, &max_setup_time, 1, hypre_MPI_DOUBLE, hypre_MPI_MAX, 0, 
	    hypre_MPI_COMM_WORLD);
        hypre_MPI_Reduce(&solve_time, &max_solve_time, 1, hypre_MPI_DOUBLE, hypre_MPI_MAX, 0, 
	    hypre_MPI_COMM_WORLD);

	if (mype == 0)
	{
            hypre_printf("**********************************************\n");
            hypre_printf("***    Setup    Solve    Total\n");
            hypre_printf("III %8.1f %8.1f %8.1f\n", max_setup_time, max_solve_time, 
		max_setup_time+max_solve_time);
            hypre_printf("**********************************************\n");
	}

cleanup:
        ParaSailsDestroy(ps);

        num_runs--;
    }

    free(x);
    free(b);

    MatrixDestroy(A);
    hypre_MPI_Finalize();

    return 0;
}
