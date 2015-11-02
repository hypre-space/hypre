/*BHEADER**********************************************************************
 * Copyright (c) 2014,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "_hypre_struct_mv.h"
#include "HYPRE_struct_mv.h"

#define MAXDIM 3
#define DEBUG 0

/*--------------------------------------------------------------------------
 * Data structures
 *--------------------------------------------------------------------------*/

/* Globals */
char      infile_default[50]  = "structmat.in";
char      outfile_default[50] = "structmat.out";
HYPRE_Int ndim                = 0;

typedef HYPRE_Int Index[MAXDIM];

typedef struct
{
   /* Grid data */
   HYPRE_Int        nboxes;
   Index           *ilowers;
   Index           *iuppers;
   HYPRE_Int       *boxsizes;
   HYPRE_Int        max_boxsize;
   Index            periodic;

   /* Matrix data */
   HYPRE_Int        nmatrices;
   HYPRE_Int       *matrix_sizes;
   Index           *matrix_rstrides;
   Index           *matrix_dstrides;
   Index          **matrix_offsets;
   HYPRE_Real     **matrix_values;
   HYPRE_Int       *matrix_ncentries; /* num constant entries */
   HYPRE_Int      **matrix_centries;  /* constant entries */
   HYPRE_Int       *matrix_symmetric;

   /* Vector data */
   HYPRE_Int        nvectors;
   Index           *vector_strides;
   HYPRE_Real      *vector_values;

} Data;
 
/*--------------------------------------------------------------------------
 * Read routines
 *--------------------------------------------------------------------------*/

HYPRE_Int
SScanIntArray( char       *sdata_ptr,
               char      **sdata_ptr_ptr,
               HYPRE_Int   size,
               HYPRE_Int  *array )
{
   HYPRE_Int i;

   sdata_ptr += strspn(sdata_ptr, " \t\n(");
   for (i = 0; i < size; i++)
   {
      array[i] = strtol(sdata_ptr, &sdata_ptr, 10);
   }
   sdata_ptr += strcspn(sdata_ptr, ")") + 1;

   *sdata_ptr_ptr = sdata_ptr;
   return 0;
}

HYPRE_Int
ReadData( char  *filename,
          Data  *data_ptr )
{
   Data        data;

   HYPRE_Int   myid;
   FILE       *file;

   char       *sdata = NULL;
   char       *sdata_line;
   char       *sdata_ptr;
   HYPRE_Int   sdata_size;
   HYPRE_Int   memchunk = 10000;
   HYPRE_Int   maxline  = 250;

   char        key[250];

   HYPRE_Int   bi, mi, vi, ei, n, d, s;

   /*-----------------------------------------------------------
    * Read data file from process 0, then broadcast
    *-----------------------------------------------------------*/
 
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   if (myid == 0)
   {
      if ((file = fopen(filename, "r")) == NULL)
      {
         hypre_printf("Error: can't open input file %s\n", filename);
         exit(1);
      }

      /* allocate initial space, and read first input line */
      sdata_size = 0;
      sdata = hypre_TAlloc(char, memchunk);
      sdata_line = fgets(sdata, maxline, file);

      s = 0;
      while (sdata_line != NULL)
      {
         sdata_size += strlen(sdata_line) + 1;

         /* allocate more space, if necessary */
         if ((sdata_size + maxline) > s)
         {
            sdata = hypre_TReAlloc(sdata, char, (sdata_size + memchunk));
            s= sdata_size + memchunk;
         }
         
         /* read the next input line */
         sdata_line = fgets((sdata + sdata_size), maxline, file);
      }
      fclose(file);
   }

   /* broadcast the data size */
   hypre_MPI_Bcast(&sdata_size, 1, HYPRE_MPI_INT, 0, hypre_MPI_COMM_WORLD);

   /* broadcast the data */
   sdata = hypre_TReAlloc(sdata, char, sdata_size);
   hypre_MPI_Bcast(sdata, sdata_size, hypre_MPI_CHAR, 0, hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Parse the data and fill Data structure
    *-----------------------------------------------------------*/

   data.nboxes = 0;
   data.max_boxsize = 0;
   data.nmatrices = 0;
   data.nvectors = 0;

   sdata_line = sdata;
   while (sdata_line < (sdata + sdata_size))
   {
      sdata_ptr = sdata_line;
      
      if ( ( hypre_sscanf(sdata_ptr, "%s", key) > 0 ) && ( sdata_ptr[0] != '#' ) )
      {
         sdata_ptr += strcspn(sdata_ptr, " \t\n");

         if ( strcmp(key, "GridCreate:") == 0 )
         {
            ndim = strtol(sdata_ptr, &sdata_ptr, 10);
            data.nboxes = strtol(sdata_ptr, &sdata_ptr, 10);
            n = data.nboxes;
            data.ilowers  = hypre_CTAlloc(Index, n);
            data.iuppers  = hypre_CTAlloc(Index, n);
            data.boxsizes = hypre_CTAlloc(HYPRE_Int, n);
         }
         else if ( strcmp(key, "GridSetExtents:") == 0 )
         {
            bi = strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr, ndim, data.ilowers[bi]);
            SScanIntArray(sdata_ptr, &sdata_ptr, ndim, data.iuppers[bi]);
            data.boxsizes[bi] = 1;
            for (d = 0; d < ndim; d++)
            {
               data.boxsizes[bi] *= (data.iuppers[bi][d] - data.ilowers[bi][d] + 1);
            }
            data.max_boxsize = hypre_max(data.max_boxsize, data.boxsizes[bi]);
         }
         else if ( strcmp(key, "GridSetPeriodic:") == 0 )
         {
            SScanIntArray(sdata_ptr, &sdata_ptr, ndim, data.periodic);
         }
         else if ( strcmp(key, "Matrix:") == 0 )
         {
            data.nmatrices = strtol(sdata_ptr, &sdata_ptr, 10);
            n = data.nmatrices;
            data.matrix_sizes     = hypre_CTAlloc(HYPRE_Int, n);
            data.matrix_rstrides  = hypre_CTAlloc(Index, n);
            data.matrix_dstrides  = hypre_CTAlloc(Index, n);
            data.matrix_offsets   = hypre_CTAlloc(Index *, n);
            data.matrix_values    = hypre_CTAlloc(HYPRE_Real *, n);
            data.matrix_ncentries = hypre_CTAlloc(HYPRE_Int, n);
            data.matrix_centries  = hypre_CTAlloc(HYPRE_Int *, n);
            data.matrix_symmetric = hypre_CTAlloc(HYPRE_Int, n);
         }
         else if ( strcmp(key, "MatrixCreate:") == 0 )
         {
            mi = strtol(sdata_ptr, &sdata_ptr, 10);
            data.matrix_sizes[mi] = strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr, ndim, data.matrix_rstrides[mi]);
            SScanIntArray(sdata_ptr, &sdata_ptr, ndim, data.matrix_dstrides[mi]);
            n = data.matrix_sizes[mi];
            data.matrix_offsets[mi] = hypre_CTAlloc(Index, n);
            data.matrix_values[mi]  = hypre_CTAlloc(HYPRE_Real, n);
         }
         else if ( strcmp(key, "MatrixSetCoeff:") == 0 )
         {
            mi = strtol(sdata_ptr, &sdata_ptr, 10);
            ei = strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          ndim, data.matrix_offsets[mi][ei]);
            data.matrix_values[mi][ei] = strtod(sdata_ptr, &sdata_ptr);
         }
         else if ( strcmp(key, "MatrixSetConstant:") == 0 )
         {
            mi = strtol(sdata_ptr, &sdata_ptr, 10);
            data.matrix_ncentries[mi] = strtol(sdata_ptr, &sdata_ptr, 10);
            n = data.matrix_ncentries[mi];
            data.matrix_centries[mi] = hypre_CTAlloc(HYPRE_Int, n);
            SScanIntArray(sdata_ptr, &sdata_ptr, n, data.matrix_centries[mi]);
         }
         else if ( strcmp(key, "MatrixSetSymmetric:") == 0 )
         {
            mi = strtol(sdata_ptr, &sdata_ptr, 10);
            data.matrix_symmetric[mi] = strtol(sdata_ptr, &sdata_ptr, 10);
         }
         else if ( strcmp(key, "Vector:") == 0 )
         {
            data.nvectors = strtol(sdata_ptr, &sdata_ptr, 10);
            n = data.nvectors;
            data.vector_strides = hypre_CTAlloc(Index, n);
            data.vector_values  = hypre_CTAlloc(HYPRE_Real, n);
         }
         else if ( strcmp(key, "VectorCreate:") == 0 )
         {
            vi = strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr, ndim, data.vector_strides[vi]);
            data.vector_values[vi] = strtod(sdata_ptr, &sdata_ptr);
         }
      }

      sdata_line += strlen(sdata_line) + 1;
   }

   hypre_TFree(sdata);

   *data_ptr = data; 
   return 0;
}
 
/*--------------------------------------------------------------------------
 * Distribute routines
 *--------------------------------------------------------------------------*/

HYPRE_Int
MapIndex( Index     index,
          Index     stride,
          HYPRE_Int upper )
{
   HYPRE_Int d, i = 0;

   if (upper)
   {
      i = 1;
   }
   for (d = 0; d < ndim; d++)
   {
      index[d] = stride[d]*index[d] + (stride[d]-1)*i;
   }

   return 0;
}

HYPRE_Int
DistributeData( Data       global_data,
                Index      refine,
                Index      distribute,
                Index      block,
                HYPRE_Int  num_procs,
                HYPRE_Int  myid,
                Data      *data_ptr )
{
   Data             data = global_data;
   HYPRE_Int        np, pid;
   HYPRE_Int        box, i, j, d, s, size, rem, div;
   Index            m, p, n;

   /* check number of processes */
   np = 1;
   for (d = 0; d < ndim; d++)
   {
      np *= distribute[d];
   }
   if (np != num_procs)
   {
      hypre_printf("Error: Invalid number of processes or process topology \n");
      exit(1);
   }

   pid = myid;

   /* refine boxes */
   s = 1;
   for (d = 0; d < ndim; d++)
   {
      m[d] = refine[d];
      s *= m[d];
   }
   if (s > 1)
   {
      for (box = 0; box < data.nboxes; box++)
      {
         MapIndex(data.ilowers[box], m, 0);
         MapIndex(data.iuppers[box], m, 1);
      }
   }

   /* refine and distribute boxes */
   s = 1;
   for (d = 0; d < ndim; d++)
   {
      m[d] = distribute[d];
      s *= m[d];
   }
   if (s > 1)
   {
      div = s;
      rem = pid;
      for (d = ndim-1; d >= 0; d--)
      {
         div /= m[d];
         p[d] = rem / div;
         rem %= div;
      }

      for (box = 0; box < data.nboxes; box++)
      {
         for (d = 0; d < ndim; d++)
         {
            n[d] = data.iuppers[box][d] - data.ilowers[box][d] + 1;
         }

         /* Compute base box */
         MapIndex(data.ilowers[box], m, 0);
         for (d = 0; d < ndim; d++)
         {
            data.iuppers[box][d] = data.ilowers[box][d] + n[d] - 1;
         }

         /* Shift */
         for (d = 0; d < ndim; d++)
         {
            data.ilowers[box][d] = data.ilowers[box][d] + p[d]*n[d];
            data.iuppers[box][d] = data.iuppers[box][d] + p[d]*n[d];
         }
      }
   }

   /* refine and block boxes */
   s = 1;
   for (d = 0; d < ndim; d++)
   {
      m[d] = block[d];
      s *= m[d];
   }
   if (s > 1)
   {
      size = s*data.nboxes;
      data.ilowers = hypre_TReAlloc(data.ilowers, Index, size);
      data.iuppers = hypre_TReAlloc(data.iuppers, Index, size);
      data.boxsizes = hypre_TReAlloc(data.boxsizes, HYPRE_Int, size);
      for (box = 0; box < data.nboxes; box++)
      {
         for (d = 0; d < ndim; d++)
         {
            n[d] = data.iuppers[box][d] - data.ilowers[box][d] + 1;
         }

         /* Compute base box */
         MapIndex(data.ilowers[box], m, 0);
         for (d = 0; d < ndim; d++)
         {
            data.iuppers[box][d] = data.ilowers[box][d] + n[d] - 1;
         }

         /* Shift */
         i = box;
         for (d = 0; d < ndim; d++)
         {
            p[d] = 0;
         }
         for (j = 0; j < s; j++)
         {
            for (d = 0; d < ndim; d++)
            {
               data.ilowers[i][d] = data.ilowers[box][d] + p[d]*n[d];
               data.iuppers[i][d] = data.iuppers[box][d] + p[d]*n[d];
            }
            i += data.nboxes;

            /* update p */
            for (d = 0; (d < ndim-1) && (p[d] == m[d]-1); d++)
            {
               p[d] = 0;
            }
            p[d]++;
         }
      }
      data.nboxes *= s;
   }

   /* compute box sizes, etc. */
   data.max_boxsize = 0;
   for (box = 0; box < data.nboxes; box++)
   {
      data.boxsizes[box] = 1;
      for (d = 0; d < ndim; d++)
      {
         data.boxsizes[box] *= (data.iuppers[box][d] - data.ilowers[box][d] + 1);
      }
      data.max_boxsize = hypre_max(data.max_boxsize, data.boxsizes[box]);
   }

   /* refine periodicity */
   for (d = 0; d < ndim; d++)
   {
      data.periodic[d] *= refine[d]*block[d]*distribute[d];
   }

   if (data.nboxes == 0)
   {
      hypre_TFree(data.ilowers);
      hypre_TFree(data.iuppers);
      hypre_TFree(data.boxsizes);
      data.max_boxsize = 0;
   }

   *data_ptr = data; 
   return 0;
}

/*--------------------------------------------------------------------------
 * Destroy data
 *--------------------------------------------------------------------------*/

HYPRE_Int
DestroyData( Data data )
{
   HYPRE_Int mi;

   if (data.nboxes > 0)
   {
      hypre_TFree(data.ilowers);
      hypre_TFree(data.iuppers);
      hypre_TFree(data.boxsizes);
   }

   if (data.nmatrices > 0)
   {
      for (mi = 0; mi < data.nmatrices; mi++)
      {
         hypre_TFree(data.matrix_offsets[mi]);
         hypre_TFree(data.matrix_values[mi]);
         hypre_TFree(data.matrix_centries[mi]);
      }
      hypre_TFree(data.matrix_sizes);
      hypre_TFree(data.matrix_rstrides);
      hypre_TFree(data.matrix_dstrides);
      hypre_TFree(data.matrix_offsets);
      hypre_TFree(data.matrix_values);
      hypre_TFree(data.matrix_ncentries);
      hypre_TFree(data.matrix_centries);
      hypre_TFree(data.matrix_symmetric);
   }

   if (data.nvectors > 0)
   {
      hypre_TFree(data.vector_strides);
      hypre_TFree(data.vector_values);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Routine to load cosine function
 *--------------------------------------------------------------------------*/

HYPRE_Int
SetCosineVector( HYPRE_Real  scale,
                 Index       ilower,
                 Index       iupper,
                 HYPRE_Real *values )
{
   HYPRE_Int    i, j, k;
   HYPRE_Int    count = 0;

   for (k = ilower[2]; k <= iupper[2]; k++)
   {
      for (j = ilower[1]; j <= iupper[1]; j++)
      {
         for (i = ilower[0]; i <= iupper[0]; i++)
         {
            values[count] = scale * cos((i+j+k)/10.0);
            count++;
         }
      }
   }

   return(0);
}

/*--------------------------------------------------------------------------
 * Print usage info
 *--------------------------------------------------------------------------*/

HYPRE_Int
PrintUsage( char      *progname,
            HYPRE_Int  myid )
{
   if ( myid == 0 )
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [-in <filename>] [<options>]\n", progname);
      hypre_printf("\n");
      hypre_printf("  -in  <filename> : input file  (default is `%s')\n",
                   infile_default);
      hypre_printf("  -out <filename> : output file (default is `%s')\n",
                   outfile_default);
      hypre_printf("  -outlev <level> : level = 0 (none), 1 (default), 2 (all)\n");
      hypre_printf("\n");
      hypre_printf("  -P <Px> <Py> ...     : refine and distribute part(s)\n");
      hypre_printf("  -r <rx> <ry> ...     : refine part(s)\n");
      hypre_printf("  -b <bx> <by> ...     : refine and block part(s)\n");
      hypre_printf("\n");
      hypre_printf("  -mat-vec <A> <x> <y> : compute A*x + y\n");
      hypre_printf("  -matTvec <A> <x> <y> : compute A^T*x + y\n");
      hypre_printf("\n");
      hypre_printf("  -mat-mat <n> <A>[T] <B>[T] ... : compute A*B*... or A^T*B*..., etc. \n");
      hypre_printf("                                 : for n possibly transposed matrices \n");
      hypre_printf("                                 : example P^T*A*P: -mat-mat 3 1T 0 1 \n");
      hypre_printf("\n");
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface
 *--------------------------------------------------------------------------*/
 
hypre_int
main( hypre_int  argc,
      char      *argv[] )
{
   char                 *infile, *outfile, filename[255];
   Data                  global_data;
   Data                  data;
   Index                 refine;
   Index                 distribute;
   Index                 block;
                        
   HYPRE_StructGrid      grid;
   HYPRE_StructStencil  *stencils;
   HYPRE_StructMatrix   *matrices;
   HYPRE_StructGrid     *vgrids;
   HYPRE_StructVector   *vectors;
   HYPRE_StructMatrix    M;

   HYPRE_Real           *values;

   HYPRE_Int             num_procs, myid, outlev, ierr;
   HYPRE_Int             time_index;
   HYPRE_Int             arg_index, box, mi, vi, ei, d, i, k;
   HYPRE_Int             do_matvec, do_matvecT, do_matmat;
   HYPRE_Int             mv_A, mv_x, mv_y;
   HYPRE_Int             nterms, *terms, *trans;
   char                  transposechar;
                        
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   hypre_InitMemoryDebug(myid);

   infile  = infile_default;
   outfile = outfile_default;
   outlev  = 1;

   /*-----------------------------------------------------------
    * Read input file
    *-----------------------------------------------------------*/

   arg_index = 1;

   /* parse command line for input file name */
   if (argc > 1)
   {
      if ( strcmp(argv[arg_index], "-in") == 0 )
      {
         arg_index++;
         infile = argv[arg_index++];
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         PrintUsage(argv[0], myid);
         exit(1);
      }
   }

   ReadData(infile, &global_data);

   /*-----------------------------------------------------------
    * Check some things
    *-----------------------------------------------------------*/

   if (ndim == 0)
   {
      hypre_printf("Error: number of dimensions not specified!\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   for (d = 0; d < ndim; d++)
   {
      refine[d]     = 1;
      distribute[d] = 1;
      block[d]      = 1;
   }

   do_matvec = 0;
   do_matvecT = 0;
   do_matmat = 0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-out") == 0 )
      {
         arg_index++;
         outfile = argv[arg_index++];
      }
      else if ( strcmp(argv[arg_index], "-outlev") == 0 )
      {
         arg_index++;
         outlev = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         k = arg_index;
         for (d = 0; d < ndim; d++)
         {
            distribute[d] = atoi(argv[k++]);
         }
         arg_index += ndim;
      }
      else if ( strcmp(argv[arg_index], "-r") == 0 )
      {
         arg_index++;
         k = arg_index;
         for (d = 0; d < ndim; d++)
         {
            refine[d] = atoi(argv[k++]);
         }
         arg_index += ndim;
      }
      else if ( strcmp(argv[arg_index], "-b") == 0 )
      {
         arg_index++;
         k = arg_index;
         for (d = 0; d < ndim; d++)
         {
            block[d] = atoi(argv[k++]);
         }
         arg_index += ndim;
      }
      else if ( strcmp(argv[arg_index], "-mat-vec") == 0 )
      {
         arg_index++;
         do_matvec = 1;
         mv_A = atoi(argv[arg_index++]);
         mv_x = atoi(argv[arg_index++]);
         mv_y = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-matTvec") == 0 )
      {
         arg_index++;
         do_matvecT = 1;
         mv_A = atoi(argv[arg_index++]);
         mv_x = atoi(argv[arg_index++]);
         mv_y = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mat-mat") == 0 )
      {
         arg_index++;
         do_matmat = 1;
         nterms = atoi(argv[arg_index++]);
         terms = hypre_CTAlloc(HYPRE_Int, nterms);
         trans = hypre_CTAlloc(HYPRE_Int, nterms);
         for (i = 0; i < nterms; i++)
         {
            transposechar = ' ';
            sscanf(argv[arg_index++], "%d%c", &terms[i], &transposechar);
            if (transposechar == 'T')
            {
               trans[i] = 1;
            }
         }
      }
      else
      {
         arg_index++;
         /*break;*/
      }
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("\n");
      hypre_printf("P =");
      for (d = 0; d < ndim; d++)
      {
         hypre_printf(" %d", distribute[d]);
      }
      hypre_printf("\n");
      hypre_printf("r =");
      for (d = 0; d < ndim; d++)
      {
         hypre_printf(" %d", refine[d]);
      }
      hypre_printf("\n");
      hypre_printf("b =");
      for (d = 0; d < ndim; d++)
      {
         hypre_printf(" %d", block[d]);
      }
      hypre_printf("\n");
      hypre_printf("\n");
   }

   /*-----------------------------------------------------------
    * Distribute data
    *-----------------------------------------------------------*/

   DistributeData(global_data, refine, distribute, block, num_procs, myid, &data);

   /*-----------------------------------------------------------
    * Set up the grid
    *-----------------------------------------------------------*/

   HYPRE_StructGridCreate(hypre_MPI_COMM_WORLD, ndim, &grid);
   for (box = 0; box < data.nboxes; box++)
   {
      HYPRE_StructGridSetExtents(grid, data.ilowers[box], data.iuppers[box]);
   }
   HYPRE_StructGridSetPeriodic(grid, data.periodic);
#if 1 /* Remove eventually */
   {
      HYPRE_Int num_ghost[2*MAXDIM];
      for (i = 0; i < 2*MAXDIM; i++)
      {
         num_ghost[i] = 0;
      }
      HYPRE_StructGridSetNumGhost(grid, num_ghost);
   }
#endif
   HYPRE_StructGridAssemble(grid);

   /*-----------------------------------------------------------
    * Set up the matrices and vectors
    *-----------------------------------------------------------*/

   values = hypre_TAlloc(HYPRE_Real, data.max_boxsize);

   stencils = hypre_CTAlloc(HYPRE_StructStencil, data.nmatrices);
   matrices = hypre_CTAlloc(HYPRE_StructMatrix, data.nmatrices);
   for (mi = 0; mi < data.nmatrices; mi++)
   {
      HYPRE_StructStencilCreate(ndim, data.matrix_sizes[mi], &stencils[mi]);
      for (ei = 0; ei < data.matrix_sizes[mi]; ei++)
      {
         HYPRE_StructStencilSetEntry(stencils[mi], ei,
                                     data.matrix_offsets[mi][ei]);
      }

      HYPRE_StructMatrixCreate(
         hypre_MPI_COMM_WORLD, grid, stencils[mi], &matrices[mi]);
      HYPRE_StructMatrixSetRangeStride(matrices[mi], data.matrix_rstrides[mi]);
      HYPRE_StructMatrixSetDomainStride(matrices[mi], data.matrix_dstrides[mi]);
      HYPRE_StructMatrixSetSymmetric(matrices[mi], data.matrix_symmetric[mi]);
      HYPRE_StructMatrixSetConstantEntries(
         matrices[mi], data.matrix_ncentries[mi], data.matrix_centries[mi]);
      HYPRE_ClearAllErrors();
      ierr = HYPRE_StructMatrixInitialize(matrices[mi]);
      if (ierr)
      {
         if (myid == 0)
         {
            hypre_printf("Error constructing matrix %d: skipping...\n", mi);
         }
         matrices[mi] = NULL;
         continue;
      }
      
      for (ei = 0; ei < data.matrix_sizes[mi]; ei++)
      {
         Index ilower, iupper, origin, stride;

         /* Compute origin and stride.  This assumes that at least one of
          * rstride and dstride is all ones. */
         for (d = 0; d < ndim; d++)
         {
            if (data.matrix_dstrides[mi][d] > 1)
            {
               origin[d] = -data.matrix_offsets[mi][ei][d];
               stride[d] =  data.matrix_dstrides[mi][d];
            }
            else
            {
               origin[d] = 0;
               stride[d] = data.matrix_rstrides[mi][d];
            }
         }
         for (i = 0; i < data.max_boxsize; i++)
         {
            values[i] = data.matrix_values[mi][ei];
         }
         for (box = 0; box < data.nboxes; box++)
         {
            for (d = 0; d < ndim; d++)
            {
               ilower[d] = data.ilowers[box][d];
               iupper[d] = data.iuppers[box][d];
            }
            HYPRE_StructGridProjectBox(grid, ilower, iupper, origin, stride);
            HYPRE_StructMatrixSetBoxValues(
               matrices[mi], ilower, iupper, 1, &ei, values);
         }
      }
      HYPRE_StructMatrixAssemble(matrices[mi]);
      /* Zero out coefficients that reach outside of the grid */
      hypre_StructMatrixClearBoundary(matrices[mi]);
   }

   vgrids = hypre_CTAlloc(HYPRE_StructGrid, data.nvectors);
   vectors = hypre_CTAlloc(HYPRE_StructVector, data.nvectors);
   for (vi = 0; vi < data.nvectors; vi++)
   {
      HYPRE_StructGridCoarsen(grid, data.vector_strides[vi], &vgrids[vi]);

      HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, vgrids[vi], &vectors[vi]);
      HYPRE_StructVectorInitialize(vectors[vi]);
      for (i = 0; i < data.max_boxsize; i++)
      {
         values[i] = data.vector_values[vi];
      }
      HYPRE_StructVectorSetConstantValues(vectors[vi], values[0]);
      /*for (box = 0; box < data.nboxes; box++)
      {
         HYPRE_StructVectorSetBoxValues(
            vectors[vi], data.ilowers[box], data.iuppers[box], values);
      }*/
      HYPRE_StructVectorAssemble(vectors[vi]);
   }

   hypre_TFree(values);

   /*-----------------------------------------------------------
    * Print matrices and vectors
    *-----------------------------------------------------------*/

   if (outlev >= 2)
   {
      for (mi = 0; mi < data.nmatrices; mi++)
      {
         if (matrices[mi] != NULL)
         {
            hypre_sprintf(filename, "%s.matrix%d", outfile, mi);
            HYPRE_StructMatrixPrint(filename,  matrices[mi], 0);
         }
      }
      for (vi = 0; vi < data.nvectors; vi++)
      {
         hypre_sprintf(filename, "%s.vector%d", outfile, vi);
         HYPRE_StructVectorPrint(filename,  vectors[vi], 0);
      }
   }

   /*-----------------------------------------------------------
    * Matrix-vector multiply
    *-----------------------------------------------------------*/

   if (do_matvec)
   {
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      time_index = hypre_InitializeTiming("Matrix-vector multiply");
      hypre_BeginTiming(time_index);

#if DEBUG
      /* First, set num_ghost to zero for both x and y */
      {
         HYPRE_Int        num_ghost[2*MAXDIM];
         hypre_BoxArray  *data_space;

         for (i = 0; i < 2*MAXDIM; i++)
         {
            num_ghost[i] = 0;
         }
         hypre_StructVectorComputeDataSpace(vectors[mv_x], num_ghost, &data_space);
         hypre_StructVectorResize(vectors[mv_x], data_space);
         hypre_StructVectorComputeDataSpace(vectors[mv_y], num_ghost, &data_space);
         hypre_StructVectorResize(vectors[mv_y], data_space);
         hypre_StructVectorForget(vectors[mv_x]);
         hypre_StructVectorForget(vectors[mv_y]);
         HYPRE_StructVectorPrint("zvec-x-resize0", vectors[mv_x], 1);
         HYPRE_StructVectorPrint("zvec-y-resize0", vectors[mv_y], 1);
      }
      /* Now, test reindex, etc. and add appropriate num_ghost */
      {
         HYPRE_Int       *num_ghost;
         hypre_BoxArray  *data_space;

         hypre_StructVectorReindex(vectors[mv_x], grid, data.vector_strides[mv_x]);
         hypre_StructVectorRestore(vectors[mv_x]);

         hypre_StructVectorReindex(vectors[mv_x], grid, data.vector_strides[mv_x]);
         hypre_StructNumGhostFromStencil(stencils[mv_A], &num_ghost);
         hypre_StructVectorComputeDataSpace(vectors[mv_x], num_ghost, &data_space);
         hypre_StructVectorResize(vectors[mv_x], data_space);
         HYPRE_StructVectorPrint("zvec-x-resize1", vectors[mv_x], 1);
         hypre_StructVectorRestore(vectors[mv_x]);
         HYPRE_StructVectorPrint("zvec-x-restore0", vectors[mv_x], 1);

         hypre_StructVectorComputeDataSpace(vectors[mv_x], num_ghost, &data_space);
         hypre_StructVectorResize(vectors[mv_x], data_space);
         HYPRE_StructVectorPrint("zvec-x-resize2", vectors[mv_x], 1);

         /* Currently need to add ghost to y (but shouldn't in the future) */
         hypre_StructVectorComputeDataSpace(vectors[mv_y], num_ghost, &data_space);
         hypre_StructVectorResize(vectors[mv_y], data_space);
         hypre_StructVectorForget(vectors[mv_y]);
         HYPRE_StructVectorPrint("zvec-y-resize1", vectors[mv_y], 1);

         hypre_TFree(num_ghost);
      }
#endif

      HYPRE_StructMatrixMatvec(1.0, matrices[mv_A], vectors[mv_x], 1.0, vectors[mv_y]);
      
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Matrix-vector multiply", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      if (outlev >= 1)
      {
         hypre_sprintf(filename, "%s.matvec", outfile);
         HYPRE_StructVectorPrint(filename, vectors[mv_y], 0);
      }
   }

   /*-----------------------------------------------------------
    * Transpose matrix-vector multiply
    *-----------------------------------------------------------*/

   if (do_matvecT)
   {
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      time_index = hypre_InitializeTiming("Transpose matrix-vector multiply");
      hypre_BeginTiming(time_index);

      HYPRE_StructMatrixMatvecT(1.0, matrices[mv_A], vectors[mv_x], 1.0, vectors[mv_y]);
      
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Transpose matrix-vector multiply", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      if (outlev >= 1)
      {
         hypre_sprintf(filename, "%s.matvecT", outfile);
         HYPRE_StructVectorPrint(filename, vectors[mv_y], 0);
      }
   }

   /*-----------------------------------------------------------
    * Matrix-matrix multiply
    *-----------------------------------------------------------*/

   if (do_matmat)
   {
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      time_index = hypre_InitializeTiming("Matrix-matrix multiply");
      hypre_BeginTiming(time_index);

      hypre_StructMatmult(data.nmatrices, matrices, nterms, terms, trans, &M);
      
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Matrix-matrix multiply", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      if (outlev >= 1)
      {
         hypre_sprintf(filename, "%s.matmat", outfile);
         HYPRE_StructMatrixPrint(filename, M, 0);
      }

      HYPRE_StructMatrixDestroy(M);
      hypre_TFree(terms);
      hypre_TFree(trans);
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_StructGridDestroy(grid);
   for (mi = 0; mi < data.nmatrices; mi++)
   {
      HYPRE_StructStencilDestroy(stencils[mi]);
      HYPRE_StructMatrixDestroy(matrices[mi]);
   }
   hypre_TFree(stencils);
   hypre_TFree(matrices);
   for (vi = 0; vi < data.nvectors; vi++)
   {
      HYPRE_StructGridDestroy(vgrids[vi]);
      HYPRE_StructVectorDestroy(vectors[vi]);
   }
   hypre_TFree(vgrids);
   hypre_TFree(vectors);

   DestroyData(data);

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return (0);
}
