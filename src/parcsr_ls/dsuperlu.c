/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include <math.h>

#ifdef HYPRE_USING_DSUPERLU
#include "dsuperlu.h"

#include <math.h>
#include "superlu_ddefs.h"
/*
#ifndef hypre_DSLU_DATA_HEADER
#define hypre_DSLU_DATA_HEADER

typedef struct
{
   HYPRE_BigInt global_num_rows;
   SuperMatrix A_dslu;
   HYPRE_Real *berr;
   dLUstruct_t dslu_data_LU;
   SuperLUStat_t dslu_data_stat;
   superlu_dist_options_t dslu_options;
   gridinfo_t dslu_data_grid;
   dScalePermstruct_t dslu_ScalePermstruct;
   dSOLVEstruct_t dslu_solve;
}
hypre_DSLUData;

#endif
*/
HYPRE_Int
hypre_SLUDistSetup(HYPRE_Solver       *solver,
                   hypre_ParCSRMatrix *A,
                   HYPRE_Int           print_level)
{
   /* Par Data Structure variables */
   HYPRE_BigInt       global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   MPI_Comm           comm            = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix   *A_local;

   HYPRE_Int          pcols = 1;
   HYPRE_Int          prows = 1;
   hypre_DSLUData    *dslu_data = NULL;
   HYPRE_Int          info = 0;
   HYPRE_Int          nrhs = 0;

   HYPRE_Int          num_rows;
   HYPRE_Int          num_procs, my_id;
   HYPRE_Int          i;

   /* SuperLU_Dist variables. Note it uses "int_t" to denote integer types */
   int_t             *slu_rowptr;
   int_t             *slu_colidx;
   hypre_double      *slu_data;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* destroy solver if already setup */
   //   if (solver != NULL) { hypre_SLUDistDestroy(solver); }
   /* allocate memory for new solver */
   dslu_data = hypre_CTAlloc(hypre_DSLUData, 1, HYPRE_MEMORY_HOST);

   /* Merge diag and offd into one matrix (global ids) */
   A_local = hypre_MergeDiagAndOffd(A);

#if defined(HYPRE_USING_GPU)
   if (hypre_GetActualMemLocation(hypre_CSRMatrixMemoryLocation(A_local)) != hypre_MEMORY_HOST)
   {
      hypre_CSRMatrixMigrate(A_local, HYPRE_MEMORY_HOST);
   }
#endif
   num_rows = hypre_CSRMatrixNumRows(A_local);

   /* SuperLU uses int_t to denote its integer type. Hence, the conversion/checks below: */
   if (sizeof(int_t) != sizeof(HYPRE_Int))
   {
      slu_rowptr = hypre_CTAlloc(int_t, (num_rows + 1), hypre_CSRMatrixMemoryLocation(A_local));
      for (i = 0; i < num_rows + 1; i++)
      {
         slu_rowptr[i] = (int_t) hypre_CSRMatrixI(A_local)[i];
      }
   }
   else
   {
      slu_rowptr = (int_t*) hypre_CSRMatrixI(A_local);
   }

   if (sizeof(int_t) != sizeof(HYPRE_BigInt))
   {
      slu_colidx = hypre_CTAlloc(int_t, hypre_CSRMatrixNumNonzeros(A_local),
                                 hypre_CSRMatrixMemoryLocation(A_local));
      for (i = 0; i < hypre_CSRMatrixNumNonzeros(A_local); i++)
      {
         slu_colidx[i] = (int_t) hypre_CSRMatrixBigJ(A_local)[i];
      }
   }
   else
   {
      slu_colidx = (int_t*) hypre_CSRMatrixBigJ(A_local);
   }

   /* SuperLU uses dbl to denote its floating point type. Hence, the conversion/checks below: */
   if (sizeof(hypre_double) != sizeof(HYPRE_Complex))
   {
      slu_data = hypre_CTAlloc(hypre_double, hypre_CSRMatrixNumNonzeros(A_local),
                               hypre_CSRMatrixMemoryLocation(A_local));
      for (i = 0; i < hypre_CSRMatrixNumNonzeros(A_local); i++)
      {
         slu_data[i] = (hypre_double) hypre_CSRMatrixData(A_local)[i];
      }
   }
   else
   {
      slu_data = (hypre_double*) hypre_CSRMatrixData(A_local);
   }

   /* Now convert hypre matrix to a SuperMatrix */
   dCreate_CompRowLoc_Matrix_dist(
      &(dslu_data->A_dslu),
      (int_t) global_num_rows,
      (int_t) global_num_rows,
      (int_t) hypre_CSRMatrixNumNonzeros(A_local),
      (int_t) num_rows,
      (int_t) hypre_ParCSRMatrixFirstRowIndex(A),
      slu_data,
      slu_colidx,
      slu_rowptr,
      SLU_NR_loc, SLU_D, SLU_GE);

   /* DOK: SuperLU frees assigned data, so set them to null before
      calling hypre_CSRMatrixdestroy on A_local to avoid memory errors. */
   if ((void*) slu_rowptr == (void*) hypre_CSRMatrixI(A_local))
   {
      hypre_CSRMatrixI(A_local) = NULL;
   }
   if ((void*) slu_colidx == (void*) hypre_CSRMatrixBigJ(A_local))
   {
      hypre_CSRMatrixBigJ(A_local) = NULL;
   }
   if ((void*) slu_data == (void*) hypre_CSRMatrixData(A_local))
   {
      hypre_CSRMatrixData(A_local) = NULL;
   }
   hypre_CSRMatrixDestroy(A_local);

   /* Create process grid */
   while (prows * pcols <= num_procs) { ++prows; }
   --prows;
   pcols = num_procs / prows;
   while (prows * pcols != num_procs)
   {
      prows -= 1;
      pcols = num_procs / prows;
   }
   //hypre_printf(" prows %d pcols %d\n", prows, pcols);

   superlu_gridinit(comm, prows, pcols, &(dslu_data->dslu_data_grid));

   set_default_options_dist(&(dslu_data->dslu_options));

   dslu_data->dslu_options.Fact = DOFACT;
   if (print_level == 0 || print_level == 2) { dslu_data->dslu_options.PrintStat = NO; }
   /*dslu_data->dslu_options.IterRefine = SLU_DOUBLE;
   dslu_data->dslu_options.ColPerm = MMD_AT_PLUS_A;
   dslu_data->dslu_options.DiagPivotThresh = 1.0;
   dslu_data->dslu_options.ReplaceTinyPivot = NO; */

   dScalePermstructInit(global_num_rows, global_num_rows, &(dslu_data->dslu_ScalePermstruct));

   dLUstructInit(global_num_rows, &(dslu_data->dslu_data_LU));

   PStatInit(&(dslu_data->dslu_data_stat));

   dslu_data->global_num_rows = global_num_rows;

   dslu_data->berr = hypre_CTAlloc(HYPRE_Real, 1, HYPRE_MEMORY_HOST);
   dslu_data->berr[0] = 0.0;

   pdgssvx(&(dslu_data->dslu_options), &(dslu_data->A_dslu),
           &(dslu_data->dslu_ScalePermstruct), NULL, num_rows, nrhs,
           &(dslu_data->dslu_data_grid), &(dslu_data->dslu_data_LU),
           &(dslu_data->dslu_solve), dslu_data->berr, &(dslu_data->dslu_data_stat), &info);

   dslu_data->dslu_options.Fact = FACTORED;
   *solver = (HYPRE_Solver) dslu_data;

   return hypre_error_flag;
}

HYPRE_Int
hypre_SLUDistSolve(void            *solver,
                   hypre_ParVector *b,
                   hypre_ParVector *x)
{
   hypre_DSLUData  *dslu_data = (hypre_DSLUData *) solver;
   HYPRE_Int        info = 0;
   HYPRE_Real      *x_data;
   hypre_ParVector *x_host = NULL;
   HYPRE_Int        size = hypre_VectorSize(hypre_ParVectorLocalVector(x));
   HYPRE_Int        nrhs = 1;
   HYPRE_Int        i;

   hypre_double    *slu_data;

   hypre_ParVectorCopy(b, x);

#if defined(HYPRE_USING_GPU)
   if (hypre_GetActualMemLocation(hypre_ParVectorMemoryLocation(x)) != hypre_MEMORY_HOST)
   {
      x_host = hypre_ParVectorCloneDeep_v2(x, HYPRE_MEMORY_HOST);
      x_data = hypre_VectorData(hypre_ParVectorLocalVector(x_host));
   }
   else
#endif
   {
      x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   }

   /* SuperLU uses sbl to denote its floating point type. Hence, the conversion/checks below: */
   if (sizeof(hypre_double) != sizeof(HYPRE_Complex))
   {
      slu_data = hypre_CTAlloc(hypre_double, size, HYPRE_MEMORY_HOST);
      for (i = 0; i < size; i++)
      {
         slu_data[i] = (hypre_double) x_data[i];
      }
   }
   else
   {
      slu_data = (hypre_double*) x_data;
   }

   pdgssvx(&(dslu_data->dslu_options),
           &(dslu_data->A_dslu),
           &(dslu_data->dslu_ScalePermstruct),
           slu_data,
           (int_t) size,
           (int_t) nrhs,
           &(dslu_data->dslu_data_grid),
           &(dslu_data->dslu_data_LU),
           &(dslu_data->dslu_solve),
           dslu_data->berr,
           &(dslu_data->dslu_data_stat),
           &info);

   /* Free memory */
   if ((void*) slu_data != (void*) x_data)
   {
      hypre_TFree(slu_data, HYPRE_MEMORY_HOST);
   }

#if defined(HYPRE_USING_GPU)
   if (x_host)
   {
      hypre_ParVectorCopy(x_host, x);
      hypre_ParVectorDestroy(x_host);
   }
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_SLUDistDestroy(void* solver)
{
   hypre_DSLUData *dslu_data = (hypre_DSLUData *) solver;

   PStatFree(&(dslu_data->dslu_data_stat));
   Destroy_CompRowLoc_Matrix_dist(&(dslu_data->A_dslu));
   dScalePermstructFree(&(dslu_data->dslu_ScalePermstruct));
   dDestroy_LU(dslu_data->global_num_rows,
               &(dslu_data->dslu_data_grid),
               &(dslu_data->dslu_data_LU));
   dLUstructFree(&(dslu_data->dslu_data_LU));
   if (dslu_data->dslu_options.SolveInitialized)
   {
      dSolveFinalize(&(dslu_data->dslu_options), &(dslu_data->dslu_solve));
   }
   superlu_gridexit(&(dslu_data->dslu_data_grid));
   hypre_TFree(dslu_data->berr, HYPRE_MEMORY_HOST);
   hypre_TFree(dslu_data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

#endif
