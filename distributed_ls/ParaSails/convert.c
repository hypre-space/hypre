/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#include <stdio.h>
/* Convert - conversion routines from triangular formats */
/* assumes the matrix has a diagonal */

#define MM_MAX_LINE_LENGTH 1000

int convert(FILE *infile, FILE *outfile)
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read, ret;
    int M, N, nz, nnz;
    long offset;
    int *counts, *pointers;
    int row, col;
    double value;
    int *ind;
    double *val;
    int i, j;

    /* skip the comment section */
    do 
    {
        if (fgets(line, MM_MAX_LINE_LENGTH, infile) == NULL) 
            return -1;
    }
    while (line[0] == '%');

    sscanf(line, "%d %d %d", &M, &N, &nz); 

    printf("%d %d %d\n", M, N, nz);
    nnz = 2*nz - M;

    /* save this position in the file */
    offset = ftell(infile);

    /* allocate space for row counts */
    counts   = (int *) calloc(M+1, sizeof(int));
    pointers = (int *) malloc((M+1) * sizeof(int));

    /* read the entire matrix */
    ret = fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    while (ret != EOF)
    {
        counts[row]++;
        if (row != col) /* do not count the diagonal twice */
           counts[col]++;

        ret = fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    }

    /* allocate space for whole matrix */
    ind = (int *)    malloc(nnz * sizeof(int));
    val = (double *) malloc(nnz * sizeof(double));
    
    /* set pointer to beginning of each row */
    pointers[1] = 0;
    for (i=2; i<=M; i++)
        pointers[i] = pointers[i-1] + counts[i-1];

    /* traverse matrix again, putting in the values */
    fseek(infile, offset, SEEK_SET);
    ret = fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    while (ret != EOF)
    {
        val[pointers[row]] = value;
        ind[pointers[row]++] = col;

        if (row != col)
        {
           val[pointers[col]] = value;
           ind[pointers[col]++] = row;
        }

        ret = fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    }

    /* print out the matrix to the output file */
    fprintf(outfile, "%d %d %d\n", M, M, nnz);
    for (i=1; i<=M; i++)
        for (j=0; j<counts[i]; j++)
            fprintf(outfile, "%d %d %.15e\n", i, *ind++, *val++);

    free(counts);
    free(pointers);
    free(ind);
    free(val);

    return 0;
}

main(int argc, char *argv[])
{
    int ret;
    FILE *infile  = fopen(argv[1], "r");
    FILE *outfile = fopen(argv[2], "w");

    ret = convert(infile, outfile);
    if (ret)
	printf("Conversion failed\n");

    fclose(infile);
    fclose(outfile);
}
