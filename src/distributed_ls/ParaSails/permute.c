/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




#include <stdio.h>
/* Permute - permute matrix */
/* Usage: permute permfile infile outfile */
/* where perm is new2old, the inverse permutation, or the matlab permutation */
/* 0-based ordering vectors */

#define MM_MAX_LINE_LENGTH 80

HYPRE_Int permute(FILE *permfile, FILE *infile, FILE *outfile)
{
    char line[MM_MAX_LINE_LENGTH];
    HYPRE_Int ret;
    HYPRE_Int M, N, nnz;

    HYPRE_Int *old2new, *new2old;
    HYPRE_Int row;

    HYPRE_Int *ptr, *ind;
    double *val;
    HYPRE_Int i, j;

    HYPRE_Int oldrow, k;

    /* skip the comment section */
    do 
    {
        if (fgets(line, MM_MAX_LINE_LENGTH, infile) == NULL) 
            return -1;
    }
    while (line[0] == '%');

    hypre_sscanf(line, "%d %d %d", &M, &N, &nnz);

    hypre_printf("%d %d %d\n", M, N, nnz);

    /* allocate space for whole matrix */
    ptr = (HYPRE_Int *)    malloc((M+1) * sizeof(HYPRE_Int));
    ind = (HYPRE_Int *)    malloc(nnz * sizeof(HYPRE_Int));
    val = (double *) malloc(nnz * sizeof(double));
    
    /* read the entire matrix */
    k = 0;
    ptr[0] = 0;
    oldrow = 1; /* input row numbers are 1-based */
    ret = hypre_fscanf(infile, "%d %d %lf", &row, &ind[k], &val[k]);
    while (ret != EOF)
    {
        if (row != oldrow)
	{
	    /* set beginning of new row */
	    ptr[oldrow] = k;
	    oldrow = row;
	}

	k++;
        ret = hypre_fscanf(infile, "%d %d %lf", &row, &ind[k], &val[k]);
    }
    /* set end of last row */
    ptr[M] = k;

    /* allocate space for permutation vectors */
    new2old = (HYPRE_Int *) malloc(M * sizeof(HYPRE_Int));
    old2new = (HYPRE_Int *) malloc(M * sizeof(HYPRE_Int));

    /* read the new2old permutation vector, 0-based */
    for (i=0; i<M; i++)
        ret = hypre_fscanf(permfile, "%d", &new2old[i]);

    /* construct the original ordering, 0-based */
    for (i=0; i<M; i++)
        old2new[new2old[i]] = i;

    /* print out the matrix to the output file with the correct permutation */
    hypre_fprintf(outfile, "%d %d %d\n", M, M, nnz);
    for (i=0; i<M; i++)
    {
        for (j=ptr[new2old[i]]; j<ptr[new2old[i]+1]; j++)
            hypre_fprintf(outfile, "%d %d %.15e\n", i+1, old2new[ind[j]-1]+1, val[j]);
    }

    free(ptr);
    free(ind);
    free(val);
    free(new2old);
    free(old2new);

    return 0;
}

main(HYPRE_Int argc, char *argv[])
{
    HYPRE_Int ret;
    FILE *permfile = fopen(argv[1], "r");
    FILE *infile   = fopen(argv[2], "r");
    FILE *outfile  = fopen(argv[3], "w");

    ret = permute(permfile, infile, outfile);
    if (ret)
	hypre_printf("Permutation failed\n");

    fclose(permfile);
    fclose(infile);
    fclose(outfile);
}
