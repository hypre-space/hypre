/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

#ifdef HYPRE_USING_DSUPERLU
#include "dsuperlu.h"

/*--------------------------------------------------------------------------
 * hypre_SLUDistCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SLUDistCreate(void)
{
   hypre_DSLUData *dslu_data;
   hypre_Solver   *base;

   dslu_data = hypre_CTAlloc(hypre_DSLUData, 1, HYPRE_MEMORY_HOST);
   base      = (hypre_Solver *) dslu_data;

   /* Initialize base solver function pointers first */
   hypre_SolverSetup(base)   = NULL;
   hypre_SolverSolve(base)   = NULL;
   hypre_SolverDestroy(base) = NULL;

   /* Set base solver function pointers */
   hypre_SolverSetup(base)   = (HYPRE_PtrToSolverFcn)  hypre_SLUDistSetup;
   hypre_SolverSolve(base)   = (HYPRE_PtrToSolverFcn)  hypre_SLUDistSolve;
   hypre_SolverDestroy(base) = (HYPRE_PtrToDestroyFcn) hypre_SLUDistDestroy;

   /* Initialize all fields */
   hypre_DSLUDataGlobalNumRows(dslu_data) = 0;
   hypre_DSLUDataA(dslu_data) = NULL;
   hypre_DSLUDataBerr(dslu_data) = NULL;
   hypre_DSLUDataLU(dslu_data) = NULL;
   hypre_DSLUDataStat(dslu_data) = NULL;
   hypre_DSLUDataOptions(dslu_data) = NULL;
   hypre_DSLUDataGrid(dslu_data) = NULL;
   hypre_DSLUDataScalePermstruct(dslu_data) = NULL;
   hypre_DSLUDataSolve(dslu_data) = NULL;
   hypre_DSLUDataPrintLevel(dslu_data) = 0;

   return (void*) dslu_data;
}

/*--------------------------------------------------------------------------
 * hypre_SLUDistSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SLUDistSetPrintLevel(void      *solver,
                           HYPRE_Int  print_level)
{
   hypre_DSLUData *dslu_data = (hypre_DSLUData *) solver;

   if (dslu_data)
   {
      hypre_DSLUDataPrintLevel(dslu_data) = print_level;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SLUDistSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SLUDistSetup(void               *solver,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *b,
                   hypre_ParVector    *x)
{
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);

   /* Par Data Structure variables */
   HYPRE_BigInt       global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   MPI_Comm           comm            = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix   *A_local;

   HYPRE_Int          pcols = 1;
   HYPRE_Int          prows = 1;
   hypre_DSLUData    *dslu_data = (hypre_DSLUData *) solver;
   HYPRE_Int          nrhs = 0;

   HYPRE_Int          num_rows;
   HYPRE_Int          num_procs, my_id;
   HYPRE_Int          i;

   /* SuperLU_Dist variables. Note it uses "int_t" to denote integer types */
   hypre_int          slu_info = 0;
   int_t             *slu_rowptr;
   int_t             *slu_colidx;
   hypre_double      *slu_data;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

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
   hypre_DSLUDataA(dslu_data) = hypre_CTAlloc(SuperMatrix, 1, HYPRE_MEMORY_HOST);
   dCreate_CompRowLoc_Matrix_dist(
      hypre_DSLUDataA(dslu_data),
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

   /* Allocate SuperLU structures */
   hypre_DSLUDataGrid(dslu_data) = hypre_CTAlloc(gridinfo_t, 1, HYPRE_MEMORY_HOST);
   hypre_DSLUDataOptions(dslu_data) = hypre_CTAlloc(superlu_dist_options_t, 1, HYPRE_MEMORY_HOST);
   hypre_DSLUDataScalePermstruct(dslu_data) = hypre_CTAlloc(dScalePermstruct_t, 1, HYPRE_MEMORY_HOST);
   hypre_DSLUDataLU(dslu_data) = hypre_CTAlloc(dLUstruct_t, 1, HYPRE_MEMORY_HOST);
   hypre_DSLUDataStat(dslu_data) = hypre_CTAlloc(SuperLUStat_t, 1, HYPRE_MEMORY_HOST);
   hypre_DSLUDataSolve(dslu_data) = hypre_CTAlloc(dSOLVEstruct_t, 1, HYPRE_MEMORY_HOST);

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

   superlu_gridinit(comm, prows, pcols, hypre_DSLUDataGrid(dslu_data));

   set_default_options_dist(hypre_DSLUDataOptions(dslu_data));

   hypre_DSLUDataOptions(dslu_data)->Fact = DOFACT;
   if (hypre_DSLUDataPrintLevel(dslu_data) == 0 || hypre_DSLUDataPrintLevel(dslu_data) == 2)
   {
      hypre_DSLUDataOptions(dslu_data)->PrintStat = NO;
   }
   /*hypre_DSLUDataOptions(dslu_data)->IterRefine = SLU_DOUBLE;
   hypre_DSLUDataOptions(dslu_data)->ColPerm = MMD_AT_PLUS_A;
   hypre_DSLUDataOptions(dslu_data)->DiagPivotThresh = 1.0;
   hypre_DSLUDataOptions(dslu_data)->ReplaceTinyPivot = NO; */

   dScalePermstructInit(global_num_rows, global_num_rows, hypre_DSLUDataScalePermstruct(dslu_data));

   dLUstructInit(global_num_rows, hypre_DSLUDataLU(dslu_data));

   PStatInit(hypre_DSLUDataStat(dslu_data));

   hypre_DSLUDataGlobalNumRows(dslu_data) = global_num_rows;

   hypre_DSLUDataBerr(dslu_data) = hypre_CTAlloc(HYPRE_Real, 1, HYPRE_MEMORY_HOST);

   pdgssvx(hypre_DSLUDataOptions(dslu_data),
           hypre_DSLUDataA(dslu_data),
           hypre_DSLUDataScalePermstruct(dslu_data),
           NULL, num_rows, nrhs,
           hypre_DSLUDataGrid(dslu_data),
           hypre_DSLUDataLU(dslu_data),
           hypre_DSLUDataSolve(dslu_data),
           hypre_DSLUDataBerr(dslu_data),
           hypre_DSLUDataStat(dslu_data),
           &slu_info);

   hypre_DSLUDataOptions(dslu_data)->Fact = FACTORED;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SLUDistSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SLUDistSolve(void               *solver,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *b,
                   hypre_ParVector    *x)
{
   HYPRE_UNUSED_VAR(A);

   hypre_DSLUData  *dslu_data = (hypre_DSLUData *) solver;
   HYPRE_Real      *x_data;
   HYPRE_Int        size = hypre_VectorSize(hypre_ParVectorLocalVector(x));
   HYPRE_Int        nrhs = 1;
   HYPRE_Int        i;

   hypre_int        slu_info;
   hypre_double    *slu_data;

   hypre_ParVectorCopy(b, x);

#if defined(HYPRE_USING_GPU)
   hypre_ParVector *x_host = NULL;

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

   pdgssvx(hypre_DSLUDataOptions(dslu_data),
           hypre_DSLUDataA(dslu_data),
           hypre_DSLUDataScalePermstruct(dslu_data),
           slu_data,
           (int_t) size,
           (int_t) nrhs,
           hypre_DSLUDataGrid(dslu_data),
           hypre_DSLUDataLU(dslu_data),
           hypre_DSLUDataSolve(dslu_data),
           hypre_DSLUDataBerr(dslu_data),
           hypre_DSLUDataStat(dslu_data),
           &slu_info);

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

/*--------------------------------------------------------------------------
 * hypre_SLUDistDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SLUDistDestroy(void *solver)
{
   hypre_DSLUData *dslu_data = (hypre_DSLUData *) solver;

   if (dslu_data)
   {
      if (hypre_DSLUDataStat(dslu_data))
      {
         PStatFree(hypre_DSLUDataStat(dslu_data));
      }
      if (hypre_DSLUDataA(dslu_data))
      {
         Destroy_CompRowLoc_Matrix_dist(hypre_DSLUDataA(dslu_data));
      }
      if (hypre_DSLUDataScalePermstruct(dslu_data))
      {
         dScalePermstructFree(hypre_DSLUDataScalePermstruct(dslu_data));
      }
      if (hypre_DSLUDataLU(dslu_data) && hypre_DSLUDataGrid(dslu_data))
      {
         dDestroy_LU(hypre_DSLUDataGlobalNumRows(dslu_data),
                     hypre_DSLUDataGrid(dslu_data),
                     hypre_DSLUDataLU(dslu_data));
         dLUstructFree(hypre_DSLUDataLU(dslu_data));
      }
      if (hypre_DSLUDataOptions(dslu_data) && hypre_DSLUDataOptions(dslu_data)->SolveInitialized)
      {
         dSolveFinalize(hypre_DSLUDataOptions(dslu_data), hypre_DSLUDataSolve(dslu_data));
      }
      if (hypre_DSLUDataGrid(dslu_data))
      {
         superlu_gridexit(hypre_DSLUDataGrid(dslu_data));
      }
      hypre_TFree(hypre_DSLUDataA(dslu_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_DSLUDataGrid(dslu_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_DSLUDataOptions(dslu_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_DSLUDataScalePermstruct(dslu_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_DSLUDataLU(dslu_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_DSLUDataStat(dslu_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_DSLUDataSolve(dslu_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_DSLUDataBerr(dslu_data), HYPRE_MEMORY_HOST);
      hypre_TFree(dslu_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

#endif
