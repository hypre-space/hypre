/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




#include <stdio.h>
/* Convert - conversion routines from triangular formats */
/* assumes the matrix has a diagonal */

#define MM_MAX_LINE_LENGTH 1000

HYPRE_Int convert(FILE *infile, FILE *outfile)
{
    char line[MM_MAX_LINE_LENGTH];
    HYPRE_Int num_items_read, ret;
    HYPRE_Int M, N, nz, nnz;
    hypre_longint offset;
    HYPRE_Int *counts, *pointers;
    HYPRE_Int row, col;
    double value;
    HYPRE_Int *ind;
    double *val;
    HYPRE_Int i, j;

    /* skip the comment section */
    do 
    {
        if (fgets(line, MM_MAX_LINE_LENGTH, infile) == NULL) 
            return -1;
    }
    while (line[0] == '%');

    hypre_sscanf(line, "%d %d %d", &M, &N, &nz); 

    hypre_printf("%d %d %d\n", M, N, nz);
    nnz = 2*nz - M;

    /* save this position in the file */
    offset = ftell(infile);

    /* allocate space for row counts */
    counts   = (HYPRE_Int *) calloc(M+1, sizeof(HYPRE_Int));
    pointers = (HYPRE_Int *) malloc((M+1) * sizeof(HYPRE_Int));

    /* read the entire matrix */
    ret = hypre_fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    while (ret != EOF)
    {
        counts[row]++;
        if (row != col) /* do not count the diagonal twice */
           counts[col]++;

        ret = hypre_fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    }

    /* allocate space for whole matrix */
    ind = (HYPRE_Int *)    malloc(nnz * sizeof(HYPRE_Int));
    val = (double *) malloc(nnz * sizeof(double));
    
    /* set pointer to beginning of each row */
    pointers[1] = 0;
    for (i=2; i<=M; i++)
        pointers[i] = pointers[i-1] + counts[i-1];

    /* traverse matrix again, putting in the values */
    fseek(infile, offset, SEEK_SET);
    ret = hypre_fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    while (ret != EOF)
    {
        val[pointers[row]] = value;
        ind[pointers[row]++] = col;

        if (row != col)
        {
           val[pointers[col]] = value;
           ind[pointers[col]++] = row;
        }

        ret = hypre_fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    }

    /* print out the matrix to the output file */
    hypre_fprintf(outfile, "%d %d %d\n", M, M, nnz);
    for (i=1; i<=M; i++)
        for (j=0; j<counts[i]; j++)
            hypre_fprintf(outfile, "%d %d %.15e\n", i, *ind++, *val++);

    free(counts);
    free(pointers);
    free(ind);
    free(val);

    return 0;
}

main(HYPRE_Int argc, char *argv[])
{
    HYPRE_Int ret;
    FILE *infile  = fopen(argv[1], "r");
    FILE *outfile = fopen(argv[2], "w");

    ret = convert(infile, outfile);
    if (ret)
	hypre_printf("Conversion failed\n");

    fclose(infile);
    fclose(outfile);
}
