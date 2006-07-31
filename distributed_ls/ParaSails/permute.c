/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include <stdio.h>
/* Permute - permute matrix */
/* Usage: permute permfile infile outfile */
/* where perm is new2old, the inverse permutation, or the matlab permutation */
/* 0-based ordering vectors */

#define MM_MAX_LINE_LENGTH 80

int permute(FILE *permfile, FILE *infile, FILE *outfile)
{
    char line[MM_MAX_LINE_LENGTH];
    int ret;
    int M, N, nnz;

    int *old2new, *new2old;
    int row;

    int *ptr, *ind;
    double *val;
    int i, j;

    int oldrow, k;

    /* skip the comment section */
    do 
    {
        if (fgets(line, MM_MAX_LINE_LENGTH, infile) == NULL) 
            return -1;
    }
    while (line[0] == '%');

    sscanf(line, "%d %d %d", &M, &N, &nnz);

    printf("%d %d %d\n", M, N, nnz);

    /* allocate space for whole matrix */
    ptr = (int *)    malloc((M+1) * sizeof(int));
    ind = (int *)    malloc(nnz * sizeof(int));
    val = (double *) malloc(nnz * sizeof(double));
    
    /* read the entire matrix */
    k = 0;
    ptr[0] = 0;
    oldrow = 1; /* input row numbers are 1-based */
    ret = fscanf(infile, "%d %d %lf", &row, &ind[k], &val[k]);
    while (ret != EOF)
    {
        if (row != oldrow)
	{
	    /* set beginning of new row */
	    ptr[oldrow] = k;
	    oldrow = row;
	}

	k++;
        ret = fscanf(infile, "%d %d %lf", &row, &ind[k], &val[k]);
    }
    /* set end of last row */
    ptr[M] = k;

    /* allocate space for permutation vectors */
    new2old = (int *) malloc(M * sizeof(int));
    old2new = (int *) malloc(M * sizeof(int));

    /* read the new2old permutation vector, 0-based */
    for (i=0; i<M; i++)
        ret = fscanf(permfile, "%d", &new2old[i]);

    /* construct the original ordering, 0-based */
    for (i=0; i<M; i++)
        old2new[new2old[i]] = i;

    /* print out the matrix to the output file with the correct permutation */
    fprintf(outfile, "%d %d %d\n", M, M, nnz);
    for (i=0; i<M; i++)
    {
        for (j=ptr[new2old[i]]; j<ptr[new2old[i]+1]; j++)
            fprintf(outfile, "%d %d %.15e\n", i+1, old2new[ind[j]-1]+1, val[j]);
    }

    free(ptr);
    free(ind);
    free(val);
    free(new2old);
    free(old2new);

    return 0;
}

main(int argc, char *argv[])
{
    int ret;
    FILE *permfile = fopen(argv[1], "r");
    FILE *infile   = fopen(argv[2], "r");
    FILE *outfile  = fopen(argv[3], "w");

    ret = permute(permfile, infile, outfile);
    if (ret)
	printf("Permutation failed\n");

    fclose(permfile);
    fclose(infile);
    fclose(outfile);
}
