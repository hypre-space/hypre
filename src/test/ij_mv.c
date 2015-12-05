/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.6 $
 ***********************************************************************EHEADER*/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix-vector interface.
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

HYPRE_Int BuildParFromFile (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParDifConv (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParFromOneFile (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildRhsParFromOneFile (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_Int *partitioning , HYPRE_ParVector *b_ptr );
HYPRE_Int BuildParLaplacian9pt (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian27pt (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );

#define SECOND_TIME 0
 
hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int                 arg_index;
   HYPRE_Int                 print_usage;
   HYPRE_Int                 sparsity_known = 0;
   HYPRE_Int                 build_matrix_type;
   HYPRE_Int                 build_matrix_arg_index;
   HYPRE_Int                 build_rhs_type;
   HYPRE_Int                 build_rhs_arg_index;
   double              norm;
   void		      *object;

   HYPRE_IJMatrix      ij_A; 
   HYPRE_IJVector      ij_b;
   HYPRE_IJVector      ij_x;
   HYPRE_IJVector      ij_v;

   HYPRE_ParCSRMatrix  parcsr_A;
   HYPRE_ParVector     b;
   HYPRE_ParVector     x;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 local_row;
   HYPRE_Int                *indices;
   HYPRE_Int                *row_sizes;
   HYPRE_Int                *diag_sizes;
   HYPRE_Int                *offdiag_sizes;
   HYPRE_Int                *rows;
   HYPRE_Int                 size;
   HYPRE_Int                *ncols;
   HYPRE_Int                *col_inds;

   MPI_Comm            comm = hypre_MPI_COMM_WORLD;
   HYPRE_Int		       time_index;
   HYPRE_Int                 ierr = 0;
   HYPRE_Int                 M, N, i, j;
   HYPRE_Int                 first_local_row, last_local_row, local_num_rows;
   HYPRE_Int                 first_local_col, last_local_col, local_num_cols;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
/*
   hypre_InitMemoryDebug(myid);
*/
   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   build_matrix_type = 2;
   build_matrix_arg_index = argc;
   build_rhs_type = 2;
   build_rhs_arg_index = argc;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-fromijfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = -1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromparcsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 0;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromonecsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-laplacian") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-9pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 3;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 4;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-difconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 5;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-exact_size") == 0 )
      {
         arg_index++;
         sparsity_known = 1;
      }
      else if ( strcmp(argv[arg_index], "-storage_low") == 0 )
      {
         arg_index++;
         sparsity_known = 2;
      }
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 0;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromonefile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 1;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 2;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsrand") == 0 )
      {
         arg_index++;
         build_rhs_type      = 3;
         build_rhs_arg_index = arg_index;
      }   
      else if ( strcmp(argv[arg_index], "-xisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 4;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhszero") == 0 )
      {
         arg_index++;
         build_rhs_type      = 5;
         build_rhs_arg_index = arg_index;
      }   
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/
 
   if ( (print_usage) && (myid == 0) )
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [<options>]\n", argv[0]);
      hypre_printf("\n");
      hypre_printf("  -fromijfile <filename>     : ");
      hypre_printf("matrix read in IJ format from distributed files\n");
      hypre_printf("  -fromparcsrfile <filename> : ");
      hypre_printf("matrix read in ParCSR format from distributed files\n");
      hypre_printf("  -fromonecsrfile <filename> : ");
      hypre_printf("matrix read in CSR format from a file on one processor\n");
      hypre_printf("\n");
      hypre_printf("  -laplacian [<options>] : build laplacian problem\n");
      hypre_printf("  -9pt [<opts>] : build 9pt 2D laplacian problem\n");
      hypre_printf("  -27pt [<opts>] : build 27pt 3D laplacian problem\n");
      hypre_printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
      hypre_printf("    -n <nx> <ny> <nz>    : total problem size \n");
      hypre_printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      hypre_printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      hypre_printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      hypre_printf("\n");
      hypre_printf("  -exact_size           : inserts immediately into ParCSR structure\n");
      hypre_printf("  -storage_low          : allocates not enough storage for aux struct\n");
      hypre_printf("  -concrete_parcsr      : use parcsr matrix type as concrete type\n");
      hypre_printf("\n");
      hypre_printf("  -rhsfromfile           : rhs read in IJ form from distributed files\n");
      hypre_printf("  -rhsfromonefile        : rhs read from a file one one processor\n");
      hypre_printf("  -rhsrand               : rhs is random vector\n");
      hypre_printf("  -rhsisone              : rhs is vector with unit components (default)\n");
      hypre_printf("  -xisone                : solution of all ones\n");
      hypre_printf("  -rhszero               : rhs is zero vector\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("Running with these driver parameters:\n");
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   if ( build_matrix_type == -1 )
   {
      HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                          HYPRE_PARCSR, &ij_A );
   }
   else if ( build_matrix_type == 0 )
   {
      BuildParFromFile(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildParFromOneFile(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 2 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 3 )
   {
      BuildParLaplacian9pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 4 )
   {
      BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 5 )
   {
      BuildParDifConv(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else
   {
      hypre_printf("You have asked for an unsupported test with\n");
      hypre_printf("build_matrix_type = %d.\n", build_matrix_type);
      return(-1);
   }

   time_index = hypre_InitializeTiming("Spatial Operator");
   hypre_BeginTiming(time_index);
    
   if (build_matrix_type < 2)
   {
     ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
     parcsr_A = (HYPRE_ParCSRMatrix) object;

     ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
               &first_local_row, &last_local_row ,
               &first_local_col, &last_local_col );

     local_num_rows = last_local_row - first_local_row + 1;
     local_num_cols = last_local_col - first_local_col + 1;

     ierr = HYPRE_IJMatrixInitialize( ij_A );

   }
   else
   {

     /*--------------------------------------------------------------------
      * Copy the parcsr matrix into the IJMatrix through interface calls
      *--------------------------------------------------------------------*/
 
     ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
               &first_local_row, &last_local_row ,
               &first_local_col, &last_local_col );

     local_num_rows = last_local_row - first_local_row + 1;
     local_num_cols = last_local_col - first_local_col + 1;

     ierr += HYPRE_ParCSRMatrixGetDims( parcsr_A, &M, &N );

     ierr += HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                   first_local_col, last_local_col,
                                   &ij_A );

     ierr += HYPRE_IJMatrixSetObjectType( ij_A, HYPRE_PARCSR );
 

/* the following shows how to build an IJMatrix if one has only an
   estimate for the row sizes */
     if (sparsity_known == 1)
     {
/*  build IJMatrix using exact row_sizes for diag and offdiag */

       diag_sizes = hypre_CTAlloc(HYPRE_Int, local_num_rows);
       offdiag_sizes = hypre_CTAlloc(HYPRE_Int, local_num_rows);
       local_row = 0;
       for (i=first_local_row; i<= last_local_row; i++)
       {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                           &col_inds, &values );

         for (j=0; j < size; j++)
         {
           if (col_inds[j] < first_local_row || col_inds[j] > last_local_row)
             offdiag_sizes[local_row]++;
           else
             diag_sizes[local_row]++;
         }
         local_row++;
         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                               &col_inds, &values );
       }
       ierr += HYPRE_IJMatrixSetDiagOffdSizes( ij_A,
                                        (const HYPRE_Int *) diag_sizes,
                                        (const HYPRE_Int *) offdiag_sizes );
       hypre_TFree(diag_sizes);
       hypre_TFree(offdiag_sizes);

       ierr = HYPRE_IJMatrixInitialize( ij_A );

       for (i=first_local_row; i<= last_local_row; i++)
       {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                           &col_inds, &values );

         ierr += HYPRE_IJMatrixSetValues( ij_A, 1, &size, &i,
                                          (const HYPRE_Int *) col_inds,
                                          (const double *) values );

         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                               &col_inds, &values );
       }
     }
     else
     {
       row_sizes = hypre_CTAlloc(HYPRE_Int, local_num_rows);

       size = 5; /* this is in general too low, and supposed to test
                    the capability of the reallocation of the interface */

       if (sparsity_known == 0) /* tries a more accurate estimate of the
                                    storage */
       {
         if (build_matrix_type == 2) size = 7;
         if (build_matrix_type == 3) size = 9;
         if (build_matrix_type == 4) size = 27;
       }

       for (i=0; i < local_num_rows; i++)
         row_sizes[i] = size;

       ierr = HYPRE_IJMatrixSetRowSizes ( ij_A, (const HYPRE_Int *) row_sizes );

       hypre_TFree(row_sizes);

       ierr = HYPRE_IJMatrixInitialize( ij_A );

       /* Loop through all locally stored rows and insert them into ij_matrix */
       for (i=first_local_row; i<= last_local_row; i++)
       {
         ierr += HYPRE_ParCSRMatrixGetRow( parcsr_A, i, &size,
                                           &col_inds, &values );

         ierr += HYPRE_IJMatrixSetValues( ij_A, 1, &size, &i,
                                          (const HYPRE_Int *) col_inds,
                                          (const double *) values );

         ierr += HYPRE_ParCSRMatrixRestoreRow( parcsr_A, i, &size,
                                               &col_inds, &values );
       }
     }

     ierr += HYPRE_IJMatrixAssemble( ij_A );

   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Initial IJ Matrix Setup", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
  
   if (ierr)
   {
       hypre_printf("Error in driver building IJMatrix from parcsr matrix. \n");
       return(-1);
   }

   time_index = hypre_InitializeTiming("Backward Euler Time Step");
   hypre_BeginTiming(time_index);
    
   /* This is to emphasize that one can IJMatrixAddToValues after an
      IJMatrixRead or an IJMatrixAssemble.  After an IJMatrixRead,
      assembly is unnecessary if the sparsity pattern of the matrix is
      not changed somehow.  If one has not used IJMatrixRead, one has
      the opportunity to IJMatrixAddTo before a IJMatrixAssemble. */

   ncols    = hypre_CTAlloc(HYPRE_Int, last_local_row - first_local_row + 1);
   rows     = hypre_CTAlloc(HYPRE_Int, last_local_row - first_local_row + 1);
   col_inds = hypre_CTAlloc(HYPRE_Int, last_local_row - first_local_row + 1);
   values   = hypre_CTAlloc(double, last_local_row - first_local_row + 1);

   for (i = first_local_row; i <= last_local_row; i++)
   {
     j = i - first_local_row;
     rows[j] = i;
     ncols[j] = 1;
     col_inds[j] = i;
     values[j] = -27.8;
   }
     
   ierr += HYPRE_IJMatrixAddToValues( ij_A,
                                      local_num_rows,
                                      ncols, rows,
                                      (const HYPRE_Int *) col_inds,
                                      (const double *) values );

   hypre_TFree(values);
   hypre_TFree(col_inds);
   hypre_TFree(rows);
   hypre_TFree(ncols);

   /* If sparsity pattern is not changed since last IJMatrixAssemble call,
      this should be a no-op */

   ierr += HYPRE_IJMatrixAssemble( ij_A );

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Matrix Diagonal Augmentation", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
  
   /*-----------------------------------------------------------
    * Fetch the resulting underlying matrix out
    *-----------------------------------------------------------*/

   if (build_matrix_type > 1)
     ierr += HYPRE_ParCSRMatrixDestroy(parcsr_A);

   ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
   parcsr_A = (HYPRE_ParCSRMatrix) object;

   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_v);
   HYPRE_IJVectorSetObjectType(ij_v, HYPRE_PARCSR );
   HYPRE_IJVectorInitialize(ij_v);

   values  = hypre_CTAlloc(double, local_num_cols);

  /*-------------------------------------------------------------------
   * Check HYPRE_IJVectorSet(Get)Values calls
   *
   * All local components changed -- NULL indices
   *-------------------------------------------------------------------*/

   for (i = 0; i < local_num_cols; i++)
     values[i] = 1.;

   HYPRE_IJVectorSetValues(ij_v, local_num_cols, NULL, values);

   for (i = 0; i < local_num_cols; i++)
     values[i] = (double)i;

   HYPRE_IJVectorAddToValues(ij_v, local_num_cols/2, NULL, values);

   HYPRE_IJVectorGetValues(ij_v, local_num_cols, NULL, values);

   ierr = 0;
   for (i = 0; i < local_num_cols/2; i++)
     if (values[i] != (double)i + 1.) ++ierr;
   for (i = local_num_cols/2; i < local_num_cols; i++)
     if (values[i] != 1.) ++ierr;
   if (ierr)
   {
     hypre_printf("One of HYPRE_IJVectorSet(AddTo,Get)Values\n");
     hypre_printf("calls with NULL indices bad\n");
     hypre_printf("IJVector Error 1 with ierr = %d\n", ierr);
     exit(1);
   }

  /*-------------------------------------------------------------------
   * All local components changed, assigned reverse-ordered values
   *   as specified by indices
   *-------------------------------------------------------------------*/

   indices = hypre_CTAlloc(HYPRE_Int, local_num_cols);

   for (i = first_local_col; i <= last_local_col; i++)
   {
     j = i - first_local_col;
     values[j] = (double)i;
     indices[j] = last_local_col - i;
   }

   HYPRE_IJVectorSetValues(ij_v, local_num_cols, indices, values);

   for (i = first_local_col; i <= last_local_col; i++)
   {
     j = i - first_local_col;
     values[j] = (double)i*i;
   }

   HYPRE_IJVectorAddToValues(ij_v, local_num_cols, indices, values);

   HYPRE_IJVectorGetValues(ij_v, local_num_cols, indices, values);

   hypre_TFree(indices);

   ierr = 0;
   for (i = first_local_col; i <= last_local_col; i++)
   {
     j = i - first_local_col;
     if (values[j] != (double)(i*i + i)) ++ierr;
   }

   if (ierr)
   {
     hypre_printf("One of HYPRE_IJVectorSet(Get)Values\n");
     hypre_printf("calls bad\n");
     hypre_printf("IJVector Error 2 with ierr = %d\n", ierr);
     exit(1);
   }

   HYPRE_IJVectorDestroy(ij_v);

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("RHS and Initial Guess");
   hypre_BeginTiming(time_index);

   if ( build_rhs_type == 0 )
   {
      if (myid == 0)
      {
        hypre_printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
        hypre_printf("  Initial guess is 0\n");
      }

/* RHS */
      ierr = HYPRE_IJVectorRead( argv[build_rhs_arg_index], hypre_MPI_COMM_WORLD,
                                 HYPRE_PARCSR, &ij_b );
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 1 )
   {
      hypre_printf("build_rhs_type == 1 not currently implemented\n");
      return(-1);

#if 0
/* RHS */
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, part_b, &b);
#endif
   }
   else if ( build_rhs_type == 2 )
   {
      if (myid == 0)
      {
        hypre_printf("  RHS vector has unit components\n");
        hypre_printf("  Initial guess is 0\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      values = hypre_CTAlloc(double, local_num_rows);
      for (i = 0; i < local_num_rows; i++)
         values[i] = 1.;
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 3 )
   {
      if (myid == 0)
      {
        hypre_printf("  RHS vector has random components and unit 2-norm\n");
        hypre_printf("  Initial guess is 0\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* For purposes of this test, HYPRE_ParVector functions are used, but these are
   not necessary.  For a clean use of the interface, the user "should"
   modify components of ij_x by using functions HYPRE_IJVectorSetValues or
   HYPRE_IJVectorAddToValues */

      HYPRE_ParVectorSetRandomValues(b, 22775);
      HYPRE_ParVectorInnerProd(b,b,&norm);
      norm = 1./sqrt(norm);
      ierr = HYPRE_ParVectorScale(norm, b);

/* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 4 )
   {
      if (myid == 0)
      {
        hypre_printf("  RHS vector set for solution with unit components\n");
        hypre_printf("  Initial guess is 0\n");
      }

/* Temporary use of solution vector */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 1.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

/* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      HYPRE_ParCSRMatrixMatvec(1.,parcsr_A,x,0.,b);

/* Initial guess */
      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);
   }
   else if ( build_rhs_type == 5 )
   {
      if (myid == 0)
      {
        hypre_printf("  RHS vector is 0\n");
        hypre_printf("  Initial guess has unit components\n");
      }

/* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      values = hypre_CTAlloc(double, local_num_rows);
      for (i = 0; i < local_num_rows; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

/* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(double, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 1.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Vector Setup", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   HYPRE_IJVectorGetObjectType(ij_b, &j);
   HYPRE_IJVectorPrint(ij_b, "driver.out.b");
   HYPRE_IJVectorPrint(ij_x, "driver.out.x");

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_IJMatrixDestroy(ij_A);
   HYPRE_IJVectorDestroy(ij_b);
   HYPRE_IJVectorDestroy(ij_x);

/*
   hypre_FinalizeMemoryDebug();
*/

   hypre_MPI_Finalize();

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from file. Expects three files on each processor.
 * filename.D.n contains the diagonal part, filename.O.n contains
 * the offdiagonal part and filename.INFO.n contains global row
 * and column numbers, number of columns of offdiagonal matrix
 * and the mapping of offdiagonal column numbers to global column numbers.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParFromFile( HYPRE_Int                  argc,
                  char                *argv[],
                  HYPRE_Int                  arg_index,
                  HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   HYPRE_ParCSRMatrix A;

   HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  FromFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   HYPRE_ParCSRMatrixRead(hypre_MPI_COMM_WORLD, filename,&A);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian( HYPRE_Int                  argc,
                   char                *argv[],
                   HYPRE_Int                  arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   double              cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  Laplacian:\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 4);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.0;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD, 
		nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point convection-diffusion operator 
 * Parameters given in command line.
 * Operator:
 *
 *  -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f
 *
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParDifConv( HYPRE_Int                  argc,
                 char                *argv[],
                 HYPRE_Int                  arg_index,
                 HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   double              cx, cy, cz;
   double              ax, ay, az;
   double              hinx,hiny,hinz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   hinx = 1.0/(nx+1);
   hiny = 1.0/(ny+1);
   hinz = 1.0/(nz+1);

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;

   ax = 1.0;
   ay = 1.0;
   az = 1.0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-a") == 0 )
      {
         arg_index++;
         ax = atof(argv[arg_index++]);
         ay = atof(argv[arg_index++]);
         az = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  Convection-Diffusion: \n");
      hypre_printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");  
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 7);

   values[1] = -cx/(hinx*hinx);
   values[2] = -cy/(hiny*hiny);
   values[3] = -cz/(hinz*hinz);
   values[4] = -cx/(hinx*hinx) + ax/hinx;
   values[5] = -cy/(hiny*hiny) + ay/hiny;
   values[6] = -cz/(hinz*hinz) + az/hinz;

   values[0] = 0.0;
   if (nx > 1)
   {
      values[0] += 2.0*cx/(hinx*hinx) - 1.0*ax/hinx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy/(hiny*hiny) - 1.0*ay/hiny;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz/(hinz*hinz) - 1.0*az/hinz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateDifConv(hypre_MPI_COMM_WORLD,
                               nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from one file on Proc. 0. Expects matrix to be in
 * CSR format. Distributes matrix across processors giving each about
 * the same number of rows.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParFromOneFile( HYPRE_Int                  argc,
                     char                *argv[],
                     HYPRE_Int                  arg_index,
                     HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   HYPRE_ParCSRMatrix  A;
   HYPRE_CSRMatrix  A_CSR = NULL;

   HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix 
       *-----------------------------------------------------------*/
 
      A_CSR = HYPRE_CSRMatrixRead(filename);
   }
   HYPRE_CSRMatrixToParCSRMatrix(hypre_MPI_COMM_WORLD, A_CSR, NULL, NULL, &A);

   *A_ptr = A;

   if (myid == 0) HYPRE_CSRMatrixDestroy(A_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors 
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildRhsParFromOneFile( HYPRE_Int                  argc,
                        char                *argv[],
                        HYPRE_Int                  arg_index,
                        HYPRE_Int		    *partitioning,
                        HYPRE_ParVector     *b_ptr     )
{
   char           *filename;

   HYPRE_ParVector b;
   HYPRE_Vector    b_CSR;

   HYPRE_Int             myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  Rhs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix 
       *-----------------------------------------------------------*/
 
      b_CSR = HYPRE_VectorRead(filename);
   }
   HYPRE_VectorToParVector(hypre_MPI_COMM_WORLD, b_CSR, partitioning,&b); 

   *b_ptr = b;

   HYPRE_VectorDestroy(b_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian9pt( HYPRE_Int                  argc,
                      char                *argv[],
                      HYPRE_Int                  arg_index,
                      HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny;
   HYPRE_Int                 P, Q;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;

   P  = 1;
   Q  = num_procs;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  Laplacian 9pt:\n");
      hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      hypre_printf("    (Px, Py) = (%d, %d)\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p)/P;

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 2);

   values[1] = -1.0;

   values[0] = 0.0;
   if (nx > 1)
   {
      values[0] += 2.0;
   }
   if (ny > 1)
   {
      values[0] += 2.0;
   }
   if (nx > 1 && ny > 1)
   {
      values[0] += 4.0;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(hypre_MPI_COMM_WORLD,
                                  nx, ny, P, Q, p, q, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}
/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian27pt( HYPRE_Int                  argc,
                       char                *argv[],
                       HYPRE_Int                  arg_index,
                       HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   double             *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  Laplacian_27pt:\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 2);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
	values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
	values[0] = 2.0;
   values[1] = -1.0;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(hypre_MPI_COMM_WORLD,
                               nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}
