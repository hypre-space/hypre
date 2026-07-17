/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Schwarz functions - Unified interface for domain-based and overlapping Schwarz
 *
 * Domain-based variants (0-4): Schwarz with LU factorization
 * Overlapping variants (10+): True overlapping Schwarz with various
 *                             local solvers (ILU(k), ILUT, AMG, SuperLU)
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "schwarz.h"
#include "protos.h"
#include "parcsr_mv/par_csr_overlap.h"
#include "parcsr_mv/protos.h"

static HYPRE_Int hypre_SchwarzOverlapVariantIsRAS(HYPRE_Int variant);

/*==========================================================================
 * Internal functions for Overlapping Schwarz implementation
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SchwarzOverlapWriteSolverParams
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SchwarzOverlapWriteSolverParams(hypre_SchwarzData *schwarz_data)
{
   HYPRE_Int local_solver_type = hypre_SchwarzDataLocalSolverType(schwarz_data);
   HYPRE_Int variant = hypre_SchwarzDataVariant(schwarz_data);
   const char *solver_name;

   switch (local_solver_type)
   {
      case HYPRE_SCHWARZ_LOCAL_SOLVER_ILUK:
         solver_name = "ILU(k)";
         break;
      case HYPRE_SCHWARZ_LOCAL_SOLVER_ILUT:
         solver_name = "ILUT";
         break;
      case HYPRE_SCHWARZ_LOCAL_SOLVER_AMG:
         solver_name = "AMG";
         break;
      case HYPRE_SCHWARZ_LOCAL_SOLVER_SUPERLU:
         solver_name = "SuperLU_dist";
         break;
      default:
         solver_name = "Unknown";
   }

   hypre_printf("\n\n Overlapping Schwarz Preconditioner Parameters:\n\n");
   hypre_printf("     Variant:                   %s\n",
                hypre_SchwarzOverlapVariantIsRAS(variant) ?
                "RAS (Restricted Additive Schwarz)" : "AS (Additive Schwarz)");
   hypre_printf("     Overlap Order:             %d\n",
                hypre_SchwarzDataOverlap(schwarz_data));
   hypre_printf("     Local Solver Type:         %s\n", solver_name);

   /* Print ILU-specific parameters */
   if (local_solver_type == HYPRE_SCHWARZ_LOCAL_SOLVER_ILUK)
   {
      hypre_printf("     ILU(k) Level of Fill:      %d\n",
                   hypre_SchwarzDataILUKLevelOfFill(schwarz_data));
   }
   else if (local_solver_type == HYPRE_SCHWARZ_LOCAL_SOLVER_ILUT)
   {
      hypre_printf("     ILUT Max NNZ/Row:          %d\n",
                   hypre_SchwarzDataILUTMaxNnzRow(schwarz_data));
      hypre_printf("     ILUT Drop Tolerance:       %e\n",
                   hypre_SchwarzDataILUTDroptol(schwarz_data));
   }

   hypre_printf("     Relaxation Weight:         %e\n",
                hypre_SchwarzDataRelaxWeight(schwarz_data));
   hypre_printf("     Max Iterations:            %d\n",
                hypre_SchwarzDataMaxIter(schwarz_data));
   hypre_printf("     Tolerance:                 %e\n",
                hypre_SchwarzDataTol(schwarz_data));
   hypre_printf("\n");

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SchwarzOverlapVariantFromLocalSolver
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SchwarzOverlapVariantFromLocalSolver(HYPRE_Int local_solver_type,
                                           HYPRE_Int use_ras)
{
   switch (local_solver_type)
   {
      case HYPRE_SCHWARZ_LOCAL_SOLVER_ILUK:
         return use_ras ? HYPRE_SCHWARZ_VARIANT_RAS_ILUK : HYPRE_SCHWARZ_VARIANT_AS_ILUK;

      case HYPRE_SCHWARZ_LOCAL_SOLVER_ILUT:
         return use_ras ? HYPRE_SCHWARZ_VARIANT_RAS_ILUT : HYPRE_SCHWARZ_VARIANT_AS_ILUT;

      case HYPRE_SCHWARZ_LOCAL_SOLVER_AMG:
         return use_ras ? HYPRE_SCHWARZ_VARIANT_RAS_AMG : HYPRE_SCHWARZ_VARIANT_AS_AMG;

      case HYPRE_SCHWARZ_LOCAL_SOLVER_SUPERLU:
         return use_ras ? HYPRE_SCHWARZ_VARIANT_RAS_SUPERLU : HYPRE_SCHWARZ_VARIANT_AS_SUPERLU;

      default:
         return -1;
   }
}

/*--------------------------------------------------------------------------
 * hypre_SchwarzOverlapVariantIsValid
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SchwarzOverlapVariantIsValid(HYPRE_Int variant)
{
   switch (variant)
   {
      case HYPRE_SCHWARZ_VARIANT_RAS_ILUK:
      case HYPRE_SCHWARZ_VARIANT_AS_ILUK:
      case HYPRE_SCHWARZ_VARIANT_RAS_ILUT:
      case HYPRE_SCHWARZ_VARIANT_AS_ILUT:
      case HYPRE_SCHWARZ_VARIANT_RAS_AMG:
      case HYPRE_SCHWARZ_VARIANT_AS_AMG:
      case HYPRE_SCHWARZ_VARIANT_RAS_SUPERLU:
      case HYPRE_SCHWARZ_VARIANT_AS_SUPERLU:
         return 1;

      default:
         return 0;
   }
}

/*--------------------------------------------------------------------------
 * hypre_SchwarzVariantIsValid
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SchwarzVariantIsValid(HYPRE_Int variant)
{
   return (variant >= HYPRE_SCHWARZ_VARIANT_MP &&
           variant <= HYPRE_SCHWARZ_VARIANT_MP_FW) ||
          hypre_SchwarzOverlapVariantIsValid(variant);
}

/*--------------------------------------------------------------------------
 * hypre_SchwarzLocalSolverFromOverlapVariant
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SchwarzLocalSolverFromOverlapVariant(HYPRE_Int variant)
{
   switch (variant)
   {
      case HYPRE_SCHWARZ_VARIANT_RAS_ILUK:
      case HYPRE_SCHWARZ_VARIANT_AS_ILUK:
         return HYPRE_SCHWARZ_LOCAL_SOLVER_ILUK;

      case HYPRE_SCHWARZ_VARIANT_RAS_ILUT:
      case HYPRE_SCHWARZ_VARIANT_AS_ILUT:
         return HYPRE_SCHWARZ_LOCAL_SOLVER_ILUT;

      case HYPRE_SCHWARZ_VARIANT_RAS_AMG:
      case HYPRE_SCHWARZ_VARIANT_AS_AMG:
         return HYPRE_SCHWARZ_LOCAL_SOLVER_AMG;

      case HYPRE_SCHWARZ_VARIANT_RAS_SUPERLU:
      case HYPRE_SCHWARZ_VARIANT_AS_SUPERLU:
         return HYPRE_SCHWARZ_LOCAL_SOLVER_SUPERLU;

      default:
         return -1;
   }
}

/*--------------------------------------------------------------------------
 * hypre_SchwarzOverlapVariantIsRAS
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SchwarzOverlapVariantIsRAS(HYPRE_Int variant)
{
   switch (variant)
   {
      case HYPRE_SCHWARZ_VARIANT_RAS_ILUK:
      case HYPRE_SCHWARZ_VARIANT_RAS_ILUT:
      case HYPRE_SCHWARZ_VARIANT_RAS_AMG:
      case HYPRE_SCHWARZ_VARIANT_RAS_SUPERLU:
         return 1;

      default:
         return 0;
   }
}

/*--------------------------------------------------------------------------
 * hypre_SchwarzOverlapDataDestroy
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SchwarzOverlapDataDestroy(hypre_SchwarzData *schwarz_data)
{
   if (!schwarz_data)
   {
      return hypre_error_flag;
   }

   if (hypre_SchwarzDataLocalSolverOwner(schwarz_data) &&
       hypre_SchwarzDataLocalSolverDestroy(schwarz_data) &&
       hypre_SchwarzDataLocalSolver(schwarz_data))
   {
      hypre_SchwarzDataLocalSolverDestroy(schwarz_data)(
         hypre_SchwarzDataLocalSolver(schwarz_data));
   }
   hypre_SchwarzDataLocalSolver(schwarz_data) = NULL;
   hypre_SchwarzDataLocalSolverSetup(schwarz_data) = NULL;
   hypre_SchwarzDataLocalSolverSolve(schwarz_data) = NULL;
   hypre_SchwarzDataLocalSolverDestroy(schwarz_data) = NULL;

   if (hypre_SchwarzDataALocalParCSR(schwarz_data))
   {
      hypre_ParCSRMatrix *A_local_parcsr = hypre_SchwarzDataALocalParCSR(schwarz_data);

      hypre_ParCSRMatrixDestroy(A_local_parcsr);
      hypre_SchwarzDataALocalParCSR(schwarz_data) = NULL;
   }

   hypre_SchwarzDataNumColsLocal(schwarz_data) = 0;

   hypre_TFree(hypre_SchwarzDataRowToColMap(schwarz_data), HYPRE_MEMORY_HOST);
   hypre_SchwarzDataRowToColMap(schwarz_data) = NULL;

   hypre_OverlapDataDestroy(hypre_SchwarzDataOverlapData(schwarz_data));
   hypre_SchwarzDataOverlapData(schwarz_data) = NULL;

   hypre_ParVectorDestroy(hypre_SchwarzDataVtemp(schwarz_data));
   hypre_SchwarzDataVtemp(schwarz_data) = NULL;

   hypre_ParVectorDestroy(hypre_SchwarzDataFLocalPar(schwarz_data));
   hypre_SchwarzDataFLocalPar(schwarz_data) = NULL;

   hypre_ParVectorDestroy(hypre_SchwarzDataULocalPar(schwarz_data));
   hypre_SchwarzDataULocalPar(schwarz_data) = NULL;

   hypre_TFree(hypre_SchwarzDataResNorms(schwarz_data), HYPRE_MEMORY_HOST);
   hypre_SchwarzDataResNorms(schwarz_data) = NULL;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SchwarzOverlapSetup
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SchwarzOverlapSetup(hypre_SchwarzData       *schwarz_data,
                          hypre_ParCSRMatrix       *A,
                          hypre_ParVector          *f,
                          hypre_ParVector          *u)
{
   MPI_Comm              comm;
   HYPRE_Int             num_procs, my_id;
   HYPRE_Int             overlap_order;
   HYPRE_Int             local_solver_type;
   HYPRE_Int             print_level;

   hypre_OverlapData    *overlap_data = NULL;
   hypre_CSRMatrix      *A_local = NULL;
   HYPRE_BigInt         *col_map = NULL;
   HYPRE_Int             num_cols_local = 0;
   HYPRE_Int             num_extended_rows;

   /* Statistics variables */
   HYPRE_BigInt          global_num_rows;
   HYPRE_Int             num_overlap_rows;
   HYPRE_Int             local_nnz;
   HYPRE_Real            local_size_real, avg_local_size, max_local_size, min_local_size;
   HYPRE_Real            avg_overlap, max_overlap, min_overlap;
   HYPRE_Real            avg_nnz, max_nnz, min_nnz;
   HYPRE_Real            load_imbalance;

   HYPRE_Int             i;

   HYPRE_UNUSED_VAR(f);
   HYPRE_UNUSED_VAR(u);

   if (!schwarz_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (hypre_GetActualMemLocation(hypre_ParCSRMatrixMemoryLocation(A)) == hypre_MEMORY_DEVICE)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Overlapping Schwarz setup requires a host-resident ParCSR matrix");
      return hypre_error_flag;
   }

   /* Get MPI info */
   comm = hypre_ParCSRMatrixComm(A);
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Get parameters */
   overlap_order = hypre_SchwarzDataOverlap(schwarz_data);
   print_level = hypre_SchwarzDataPrintLevel(schwarz_data);

   if (overlap_order < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzOverlapDataDestroy(schwarz_data);
   if (hypre_error_flag)
   {
      return hypre_error_flag;
   }

   /* Determine local solver type from variant if variant is set,
    * otherwise use the local_solver_type field */
   {
      HYPRE_Int variant = hypre_SchwarzDataVariant(schwarz_data);
      if (HYPRE_SCHWARZ_IS_OVERLAPPING(variant))
      {
         local_solver_type = hypre_SchwarzLocalSolverFromOverlapVariant(variant);
         if (local_solver_type < 0)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid overlapping Schwarz variant");
            return hypre_error_flag;
         }
         /* Update the local_solver_type field to match the variant */
         hypre_SchwarzDataLocalSolverType(schwarz_data) = local_solver_type;
      }
      else
      {
         local_solver_type = hypre_SchwarzDataLocalSolverType(schwarz_data);
      }
   }

   /* Store matrix reference */
   hypre_SchwarzDataA(schwarz_data) = A;

   /*------------------------------------------------------------------
    * Step 1: Compute overlap regions
    *------------------------------------------------------------------*/

   /* Ensure communication package exists */
   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }

   hypre_ParCSRMatrixComputeOverlap(A, overlap_order, &overlap_data);
   if (hypre_error_flag)
   {
      return hypre_error_flag;
   }
   hypre_SchwarzDataOverlapData(schwarz_data) = overlap_data;

   /*------------------------------------------------------------------
    * Step 2: Fetch overlap rows from neighboring processors
    *------------------------------------------------------------------*/

   hypre_ParCSRMatrixGetExternalMatrix(A, overlap_data);
   if (hypre_error_flag)
   {
      hypre_SchwarzOverlapDataDestroy(schwarz_data);
      return hypre_error_flag;
   }

   /*------------------------------------------------------------------
    * Step 3: Extract local restricted matrix
    *------------------------------------------------------------------*/

   hypre_ParCSRMatrixCreateExtendedMatrix(A, overlap_data, &A_local, &col_map, &num_cols_local);
   if (hypre_error_flag)
   {
      hypre_SchwarzOverlapDataDestroy(schwarz_data);
      return hypre_error_flag;
   }

   hypre_SchwarzDataNumColsLocal(schwarz_data) = num_cols_local;

   num_extended_rows = hypre_OverlapDataNumExtendedRows(overlap_data);

   /*------------------------------------------------------------------
    * Step 4: Build row-to-column mapping for solution restriction
    *------------------------------------------------------------------*/

   /* Map extended rows to column indices in the local matrix */
   {
      HYPRE_Int *row_to_col_map = hypre_TAlloc(HYPRE_Int, num_extended_rows, HYPRE_MEMORY_HOST);

      for (i = 0; i < num_extended_rows; i++)
      {
         row_to_col_map[i] = i;
      }

      hypre_SchwarzDataRowToColMap(schwarz_data) = row_to_col_map;
   }

   /*------------------------------------------------------------------
    * Step 5: Create local ParCSR matrix wrapper for local solver
    *------------------------------------------------------------------*/

   {
      hypre_ParCSRMatrix *A_local_parcsr;
      HYPRE_BigInt row_starts[2] = {0, num_extended_rows};
      HYPRE_BigInt col_starts[2] = {0, num_cols_local};

      A_local_parcsr = hypre_ParCSRMatrixCreate(hypre_MPI_COMM_SELF,
                                                num_extended_rows,
                                                num_cols_local,
                                                row_starts,
                                                col_starts,
                                                0,
                                                hypre_CSRMatrixNumNonzeros(A_local),
                                                0);
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(A_local_parcsr));
      hypre_ParCSRMatrixDiag(A_local_parcsr) = A_local;
      hypre_CSRMatrixInitialize(hypre_ParCSRMatrixOffd(A_local_parcsr));
      hypre_ParCSRMatrixNumNonzeros(A_local_parcsr) = hypre_CSRMatrixNumNonzeros(A_local);
      hypre_ParCSRMatrixDNumNonzeros(A_local_parcsr) =
         (hypre_double) hypre_CSRMatrixNumNonzeros(A_local);

      hypre_SchwarzDataALocalParCSR(schwarz_data) = A_local_parcsr;
      hypre_TFree(col_map, HYPRE_MEMORY_HOST);
      col_map = NULL;
   }

   /*------------------------------------------------------------------
    * Step 6: Setup local solver
    *------------------------------------------------------------------*/

   switch (local_solver_type)
   {
      case HYPRE_SCHWARZ_LOCAL_SOLVER_ILUK:
      {
         HYPRE_Solver ilu_solver = (HYPRE_Solver) hypre_ILUCreate();

         hypre_ILUSetType(ilu_solver, 0);     /* BJ with ILU(k) */
         hypre_ILUSetLevelOfFill(ilu_solver, hypre_SchwarzDataILUKLevelOfFill(schwarz_data));
         hypre_ILUSetMaxIter(ilu_solver, 1);
         hypre_ILUSetTol(ilu_solver, 0.0);
         if (!my_id)
         {
            hypre_ILUSetPrintLevel(ilu_solver, print_level);
         }

         hypre_ILUSetup(ilu_solver,
                        hypre_SchwarzDataALocalParCSR(schwarz_data),
                        NULL, NULL);
         if (hypre_error_flag)
         {
            hypre_SchwarzDataLocalSolver(schwarz_data) = ilu_solver;
            hypre_SchwarzDataLocalSolverDestroy(schwarz_data) = (HYPRE_PtrToDestroyFcn)
                                                                hypre_ILUDestroy;
            hypre_SchwarzDataLocalSolverOwner(schwarz_data) = 1;
            hypre_SchwarzOverlapDataDestroy(schwarz_data);
            return hypre_error_flag;
         }

         hypre_SchwarzDataLocalSolver(schwarz_data) = ilu_solver;
         hypre_SchwarzDataLocalSolverSetup(schwarz_data) = (HYPRE_PtrToSolverFcn)
                                                           hypre_ILUSetup;
         hypre_SchwarzDataLocalSolverSolve(schwarz_data) = (HYPRE_PtrToSolverFcn)
                                                           hypre_ILUSolve;
         hypre_SchwarzDataLocalSolverDestroy(schwarz_data) = (HYPRE_PtrToDestroyFcn)
                                                             hypre_ILUDestroy;
         hypre_SchwarzDataLocalSolverOwner(schwarz_data) = 1;

         break;
      }

      case HYPRE_SCHWARZ_LOCAL_SOLVER_ILUT:
      {
         HYPRE_Solver ilu_solver = (HYPRE_Solver) hypre_ILUCreate();

         hypre_ILUSetType(ilu_solver, 1);     /* BJ with ILUT */
         hypre_ILUSetMaxNnzPerRow(ilu_solver, hypre_SchwarzDataILUTMaxNnzRow(schwarz_data));
         hypre_ILUSetDropThreshold(ilu_solver, hypre_SchwarzDataILUTDroptol(schwarz_data));
         hypre_ILUSetMaxIter(ilu_solver, 1);
         hypre_ILUSetTol(ilu_solver, 0.0);
         if (!my_id)
         {
            hypre_ILUSetPrintLevel(ilu_solver, print_level);
         }

         hypre_ILUSetup(ilu_solver,
                        hypre_SchwarzDataALocalParCSR(schwarz_data),
                        NULL, NULL);
         if (hypre_error_flag)
         {
            hypre_SchwarzDataLocalSolver(schwarz_data) = ilu_solver;
            hypre_SchwarzDataLocalSolverDestroy(schwarz_data) = (HYPRE_PtrToDestroyFcn)
                                                                hypre_ILUDestroy;
            hypre_SchwarzDataLocalSolverOwner(schwarz_data) = 1;
            hypre_SchwarzOverlapDataDestroy(schwarz_data);
            return hypre_error_flag;
         }

         hypre_SchwarzDataLocalSolver(schwarz_data) = ilu_solver;
         hypre_SchwarzDataLocalSolverSetup(schwarz_data) = (HYPRE_PtrToSolverFcn)
                                                           hypre_ILUSetup;
         hypre_SchwarzDataLocalSolverSolve(schwarz_data) = (HYPRE_PtrToSolverFcn)
                                                           hypre_ILUSolve;
         hypre_SchwarzDataLocalSolverDestroy(schwarz_data) = (HYPRE_PtrToDestroyFcn)
                                                             hypre_ILUDestroy;
         hypre_SchwarzDataLocalSolverOwner(schwarz_data) = 1;

         break;
      }

      case HYPRE_SCHWARZ_LOCAL_SOLVER_AMG:
      {
         HYPRE_Solver amg_solver = (HYPRE_Solver) hypre_BoomerAMGCreate();

         hypre_BoomerAMGSetMaxIter(amg_solver, 1);
         hypre_BoomerAMGSetTol(amg_solver, 0.0);
         hypre_BoomerAMGSetNumSweeps(amg_solver, 1);
         if (!my_id)
         {
            hypre_BoomerAMGSetPrintLevel(amg_solver, print_level);
         }

         hypre_BoomerAMGSetup(amg_solver,
                              hypre_SchwarzDataALocalParCSR(schwarz_data),
                              NULL, NULL);
         if (hypre_error_flag)
         {
            hypre_SchwarzDataLocalSolver(schwarz_data) = amg_solver;
            hypre_SchwarzDataLocalSolverDestroy(schwarz_data) = (HYPRE_PtrToDestroyFcn)
                                                                hypre_BoomerAMGDestroy;
            hypre_SchwarzDataLocalSolverOwner(schwarz_data) = 1;
            hypre_SchwarzOverlapDataDestroy(schwarz_data);
            return hypre_error_flag;
         }

         hypre_SchwarzDataLocalSolver(schwarz_data) = amg_solver;
         hypre_SchwarzDataLocalSolverSetup(schwarz_data) = (HYPRE_PtrToSolverFcn)
                                                           hypre_BoomerAMGSetup;
         hypre_SchwarzDataLocalSolverSolve(schwarz_data) = (HYPRE_PtrToSolverFcn)
                                                           hypre_BoomerAMGSolve;
         hypre_SchwarzDataLocalSolverDestroy(schwarz_data) = (HYPRE_PtrToDestroyFcn)
                                                             hypre_BoomerAMGDestroy;
         hypre_SchwarzDataLocalSolverOwner(schwarz_data) = 1;

         break;
      }

      case HYPRE_SCHWARZ_LOCAL_SOLVER_SUPERLU:
      {
#ifdef HYPRE_USING_DSUPERLU
         HYPRE_Solver slu_solver = hypre_SLUDistCreate();

         hypre_SLUDistSetPrintLevel(slu_solver, print_level);
         hypre_SLUDistSetup(slu_solver,
                            hypre_SchwarzDataALocalParCSR(schwarz_data),
                            NULL, NULL);
         if (hypre_error_flag)
         {
            hypre_SchwarzDataLocalSolver(schwarz_data) = slu_solver;
            hypre_SchwarzDataLocalSolverDestroy(schwarz_data) = (HYPRE_PtrToDestroyFcn)
                                                                hypre_SLUDistDestroy;
            hypre_SchwarzDataLocalSolverOwner(schwarz_data) = 1;
            hypre_SchwarzOverlapDataDestroy(schwarz_data);
            return hypre_error_flag;
         }

         hypre_SchwarzDataLocalSolver(schwarz_data) = slu_solver;
         hypre_SchwarzDataLocalSolverSetup(schwarz_data) = NULL;  /* Already set up */
         hypre_SchwarzDataLocalSolverSolve(schwarz_data) = (HYPRE_PtrToSolverFcn)
                                                           hypre_SLUDistSolve;
         hypre_SchwarzDataLocalSolverDestroy(schwarz_data) = (HYPRE_PtrToDestroyFcn)
                                                             hypre_SLUDistDestroy;
         hypre_SchwarzDataLocalSolverOwner(schwarz_data) = 1;
#else
         if (my_id == 0)
         {
            hypre_printf("Schwarz Error: SuperLU_dist not available. "
                         "Build hypre with -DHYPRE_ENABLE_DSUPERLU=ON\n");
         }
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "SuperLU_dist local solver not available");
         hypre_SchwarzOverlapDataDestroy(schwarz_data);
         return hypre_error_flag;
#endif
         break;
      }

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown local solver type");
         hypre_SchwarzOverlapDataDestroy(schwarz_data);
         return hypre_error_flag;
   }

   /*------------------------------------------------------------------
    * Step 7: Create work vectors
    *------------------------------------------------------------------*/

   /* Global temporary vector */
   hypre_SchwarzDataVtemp(schwarz_data) = hypre_ParVectorCreate(comm,
                                                                hypre_ParCSRMatrixGlobalNumRows(A),
                                                                hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(hypre_SchwarzDataVtemp(schwarz_data));

   /* Local ParVector wrappers for local solver */
   {
      HYPRE_BigInt f_partitioning[2] = {0, num_extended_rows};
      HYPRE_BigInt u_partitioning[2] = {0, num_cols_local};

      hypre_SchwarzDataFLocalPar(schwarz_data) =
         hypre_ParVectorCreate(hypre_MPI_COMM_SELF, num_extended_rows, f_partitioning);
      hypre_SchwarzDataULocalPar(schwarz_data) =
         hypre_ParVectorCreate(hypre_MPI_COMM_SELF, num_cols_local, u_partitioning);

      hypre_ParVectorInitialize(hypre_SchwarzDataFLocalPar(schwarz_data));
      hypre_ParVectorInitialize(hypre_SchwarzDataULocalPar(schwarz_data));
   }

   /* Allocate res_norms if logging */
   if (hypre_SchwarzDataLogging(schwarz_data))
   {
      hypre_SchwarzDataResNorms(schwarz_data) =
         hypre_TAlloc(HYPRE_Real, hypre_SchwarzDataMaxIter(schwarz_data) + 1,
                      HYPRE_MEMORY_HOST);
   }

   /*------------------------------------------------------------------
    * Step 8: Compute and print setup statistics
    *------------------------------------------------------------------*/

   if (print_level > 0)
   {
      if (!my_id)
      {
         hypre_SchwarzOverlapWriteSolverParams(schwarz_data);
      }

      /* Local statistics */
      global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
      num_overlap_rows = hypre_OverlapDataNumOverlapRows(overlap_data);
      local_nnz = hypre_CSRMatrixNumNonzeros(A_local);
      local_size_real = (HYPRE_Real) num_extended_rows;

      hypre_MatrixStats *local_matrix_stats = hypre_MatrixStatsCreate();
      hypre_MatrixStats *global_matrix_stats = hypre_MatrixStatsCreate();
      hypre_ParCSRMatrixStatsComputeLocal(hypre_SchwarzDataALocalParCSR(schwarz_data),
                                          local_matrix_stats);
      hypre_MatrixStatsReduce(local_matrix_stats, global_matrix_stats, comm);

      HYPRE_Real stdev_local, stdev_overlap, stdev_nnz;

      {
         HYPRE_Real overlap_real = (HYPRE_Real) num_overlap_rows;
         HYPRE_Real nnz_real = (HYPRE_Real) local_nnz;
         HYPRE_Real sum_in[6], sum_out[6];
         HYPRE_Real max_in[3], max_out[3];
         HYPRE_Real min_in[3], min_out[3];
         HYPRE_Real var;

         sum_in[0] = local_size_real;
         sum_in[1] = local_size_real * local_size_real;
         sum_in[2] = overlap_real;
         sum_in[3] = overlap_real * overlap_real;
         sum_in[4] = nnz_real;
         sum_in[5] = nnz_real * nnz_real;
         max_in[0] = min_in[0] = local_size_real;
         max_in[1] = min_in[1] = overlap_real;
         max_in[2] = min_in[2] = nnz_real;

         hypre_MPI_Allreduce(sum_in, sum_out, 6, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
         hypre_MPI_Allreduce(max_in, max_out, 3, HYPRE_MPI_REAL, hypre_MPI_MAX, comm);
         hypre_MPI_Allreduce(min_in, min_out, 3, HYPRE_MPI_REAL, hypre_MPI_MIN, comm);

         avg_local_size = sum_out[0] / (HYPRE_Real) num_procs;
         max_local_size = max_out[0];
         min_local_size = min_out[0];
         var = sum_out[1] / (HYPRE_Real) num_procs - avg_local_size * avg_local_size;
         stdev_local = (var > 0.0) ? hypre_sqrt(var) : 0.0;

         avg_overlap = sum_out[2] / (HYPRE_Real) num_procs;
         max_overlap = max_out[1];
         min_overlap = min_out[1];
         var = sum_out[3] / (HYPRE_Real) num_procs - avg_overlap * avg_overlap;
         stdev_overlap = (var > 0.0) ? hypre_sqrt(var) : 0.0;

         avg_nnz = sum_out[4] / (HYPRE_Real) num_procs;
         max_nnz = max_out[2];
         min_nnz = min_out[2];
         var = sum_out[5] / (HYPRE_Real) num_procs - avg_nnz * avg_nnz;
         stdev_nnz = (var > 0.0) ? hypre_sqrt(var) : 0.0;
      }

      /* Load imbalance */
      load_imbalance = (avg_local_size > 0.0) ? max_local_size / avg_local_size : 1.0;

      if (my_id == 0)
      {
         hypre_printf("\n Schwarz Setup Info: N=%b, P=%d, delta=%d, imbalance=%.3f\n\n",
                      global_num_rows, num_procs, overlap_order, load_imbalance);

         hypre_printf("  %-28s  %10s  %10s  %10s  %10s\n", "Subdomain Statistics", "Min", "Max", "Avg",
                      "Stdev");
         hypre_printf("  %-28s  %10s  %10s  %10s  %10s\n", "---------------------------", "----------",
                      "----------", "----------", "----------");
         hypre_printf("  %-28s  %10.0f  %10.0f  %10.1f  %10.1f\n", "Local rows",
                      min_local_size, max_local_size, avg_local_size, stdev_local);
         hypre_printf("  %-28s  %10.0f  %10.0f  %10.1f  %10.1f\n", "Ghost rows",
                      min_overlap, max_overlap, avg_overlap, stdev_overlap);
         hypre_printf("  %-28s  %10.0f  %10.0f  %10.1f  %10.1f\n", "NNZ",
                      min_nnz, max_nnz, avg_nnz, stdev_nnz);
         hypre_printf("  %-28s  %10d  %10d  %10.2f  %10.2f\n", "Entries/row",
                      hypre_MatrixStatsNnzrowMin(global_matrix_stats),
                      hypre_MatrixStatsNnzrowMax(global_matrix_stats),
                      hypre_MatrixStatsNnzrowAvg(global_matrix_stats),
                      hypre_MatrixStatsNnzrowStDev(global_matrix_stats));
         hypre_printf("  %-28s  %10.2e  %10.2e  %10.2e  %10.2e\n", "Row sum (abs)",
                      hypre_MatrixStatsAbsrowsumMin(global_matrix_stats),
                      hypre_MatrixStatsAbsrowsumMax(global_matrix_stats),
                      hypre_MatrixStatsAbsrowsumAvg(global_matrix_stats),
                      hypre_MatrixStatsAbsrowsumStDev(global_matrix_stats));
         hypre_printf("\n");
      }

      /* Clean up matrix stats */
      hypre_MatrixStatsDestroy(local_matrix_stats);
      hypre_MatrixStatsDestroy(global_matrix_stats);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SchwarzOverlapSolve
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SchwarzOverlapSolve(hypre_SchwarzData       *schwarz_data,
                          hypre_ParCSRMatrix       *A,
                          hypre_ParVector          *f,
                          hypre_ParVector          *u)
{
   MPI_Comm              comm;
   HYPRE_Int             num_procs, my_id;
   HYPRE_Int             variant;
   HYPRE_Int             max_iter;
   HYPRE_Real            tol;
   HYPRE_Real            relax_weight;
   HYPRE_Int             print_level;
   HYPRE_Int             logging;

   hypre_OverlapData    *overlap_data;
   HYPRE_Int             num_cols_local;
   HYPRE_Int            *row_to_col_map;
   HYPRE_Int             num_extended_rows;
   HYPRE_Int             num_local_rows;

   hypre_ParVector      *Vtemp;
   hypre_ParVector      *f_local_par;
   hypre_ParVector      *u_local_par;
   hypre_Vector         *f_local_vec;
   hypre_Vector         *u_local_vec;

   HYPRE_Real           *u_data;
   HYPRE_Real           *f_local_data;
   HYPRE_Real           *u_local_data;
   HYPRE_Real           *vtemp_data;

   HYPRE_Int            *row_is_owned;
   HYPRE_BigInt         *extended_rows;
   HYPRE_BigInt          first_row;

   hypre_ParCSRCommPkg  *overlap_comm_pkg;
   hypre_ParCSRCommHandle *comm_handle = NULL;
   HYPRE_Int             num_sends = 0;
   HYPRE_Int             num_recvs = 0;
   HYPRE_Int            *send_map_starts = NULL;
   HYPRE_Int            *send_map_elmts = NULL;
   HYPRE_Int            *recv_vec_starts = NULL;
   HYPRE_Int             num_send_vals = 0;
   HYPRE_Int             num_recv_vals = 0;
   HYPRE_Real           *restrict_send_buf = NULL;
   HYPRE_Real           *restrict_recv_buf = NULL;
   HYPRE_Int            *as_multiplicity = NULL;
   HYPRE_Int             as_num_send_vals = 0;
   HYPRE_Int             as_num_recv_vals = 0;
   HYPRE_Real           *as_send_buf = NULL;
   HYPRE_Real           *as_recv_buf = NULL;

   HYPRE_Real            res_norm = 0.0;
   HYPRE_Int             iter;
   HYPRE_Int             i;

   if (!schwarz_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* Get MPI info */
   comm = hypre_ParVectorComm(f);
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Get parameters */
   variant = hypre_SchwarzDataVariant(schwarz_data);
   max_iter = hypre_SchwarzDataMaxIter(schwarz_data);
   tol = hypre_SchwarzDataTol(schwarz_data);
   relax_weight = hypre_SchwarzDataRelaxWeight(schwarz_data);
   print_level = hypre_SchwarzDataPrintLevel(schwarz_data);
   logging = hypre_SchwarzDataLogging(schwarz_data);

   if (!hypre_SchwarzOverlapVariantIsValid(variant))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid overlapping Schwarz variant");
      return hypre_error_flag;
   }

   if (hypre_GetActualMemLocation(hypre_ParCSRMatrixMemoryLocation(A)) == hypre_MEMORY_DEVICE ||
       hypre_GetActualMemLocation(hypre_ParVectorMemoryLocation(f)) == hypre_MEMORY_DEVICE ||
       hypre_GetActualMemLocation(hypre_ParVectorMemoryLocation(u)) == hypre_MEMORY_DEVICE)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Overlapping Schwarz solve requires host-resident matrix and vectors");
      return hypre_error_flag;
   }

   /* Get data structures */
   overlap_data = hypre_SchwarzDataOverlapData(schwarz_data);
   num_cols_local = hypre_SchwarzDataNumColsLocal(schwarz_data);
   row_to_col_map = hypre_SchwarzDataRowToColMap(schwarz_data);

   num_extended_rows = hypre_OverlapDataNumExtendedRows(overlap_data);
   num_local_rows = hypre_OverlapDataNumLocalRows(overlap_data);
   row_is_owned = hypre_OverlapDataRowIsOwned(overlap_data);
   extended_rows = hypre_OverlapDataExtendedRowIndices(overlap_data);
   first_row = hypre_OverlapDataFirstRowIndex(overlap_data);

   overlap_comm_pkg = hypre_OverlapDataOverlapCommPkg(overlap_data);

   Vtemp = hypre_SchwarzDataVtemp(schwarz_data);
   f_local_par = hypre_SchwarzDataFLocalPar(schwarz_data);
   u_local_par = hypre_SchwarzDataULocalPar(schwarz_data);
   f_local_vec = hypre_ParVectorLocalVector(f_local_par);
   u_local_vec = hypre_ParVectorLocalVector(u_local_par);

   u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   f_local_data = hypre_VectorData(f_local_vec);
   u_local_data = hypre_VectorData(u_local_vec);
   vtemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   HYPRE_Real            init_res_norm = 0.0;
   HYPRE_Real            rel_res_norm = 1.0;
   HYPRE_Real            rhs_norm = 0.0;
   HYPRE_Real            old_res_norm;
   HYPRE_Real            conv_factor = 0.0;

   /* Initialize solution to zero (for preconditioner use) */
   hypre_ParVectorSetConstantValues(u, 0.0);

   /*------------------------------------------------------------------
    * Print solve header (only when max_iter > 1, i.e., standalone use)
    *------------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1 && max_iter > 1)
   {
      hypre_printf("\n\n SCHWARZ SOLVER SOLUTION INFO:\n");
   }

   /*------------------------------------------------------------------
    * Main iteration loop
    *------------------------------------------------------------------*/

   if (overlap_comm_pkg && num_procs > 1)
   {
      num_sends = hypre_ParCSRCommPkgNumSends(overlap_comm_pkg);
      num_recvs = hypre_ParCSRCommPkgNumRecvs(overlap_comm_pkg);
      send_map_starts = hypre_ParCSRCommPkgSendMapStarts(overlap_comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(overlap_comm_pkg);
      recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(overlap_comm_pkg);

      num_send_vals = send_map_starts[num_sends];
      num_recv_vals = recv_vec_starts[num_recvs];
      restrict_send_buf = hypre_TAlloc(HYPRE_Real, num_send_vals, HYPRE_MEMORY_HOST);
      restrict_recv_buf = hypre_TAlloc(HYPRE_Real, num_recv_vals, HYPRE_MEMORY_HOST);
   }

   if (!hypre_SchwarzOverlapVariantIsRAS(variant))
   {
      as_multiplicity = hypre_TAlloc(HYPRE_Int, num_local_rows, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_local_rows; i++)
      {
         as_multiplicity[i] = 1;
      }

      if (overlap_comm_pkg && num_procs > 1)
      {
         as_num_send_vals = num_recv_vals;
         as_num_recv_vals = num_send_vals;
         as_send_buf = hypre_TAlloc(HYPRE_Real, as_num_send_vals, HYPRE_MEMORY_HOST);
         as_recv_buf = hypre_TAlloc(HYPRE_Real, as_num_recv_vals, HYPRE_MEMORY_HOST);

         for (i = 0; i < as_num_recv_vals; i++)
         {
            as_multiplicity[send_map_elmts[i]]++;
         }
      }
   }

   for (iter = 0; iter < max_iter; iter++)
   {
      /*---------------------------------------------------------------
       * Step 1: Compute residual r = f - A*u (stored in Vtemp)
       *         For first iteration with u=0, r = f
       *---------------------------------------------------------------*/

      if (iter == 0)
      {
         /* r = f */
         hypre_ParVectorCopy(f, Vtemp);
      }
      else
      {
         /* r = f - A*u */
         hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, u, 1.0, f, Vtemp);
      }

      /* Compute residual norm if needed (only when iterating or logging) */
      if ((tol > 0.0 || logging || print_level > 1) && max_iter > 1)
      {
         old_res_norm = res_norm;
         res_norm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));

         if (iter == 0)
         {
            init_res_norm = res_norm;
            rhs_norm = hypre_sqrt(hypre_ParVectorInnerProd(f, f));
            if (rhs_norm > HYPRE_REAL_EPSILON)
            {
               rel_res_norm = res_norm / rhs_norm;
            }
            else
            {
               rel_res_norm = res_norm;
            }

            if (my_id == 0 && print_level > 1)
            {
               hypre_printf("                                            relative\n");
               hypre_printf("               residual        factor       residual\n");
               hypre_printf("               --------        ------       --------\n");
               hypre_printf("    Initial    %e                 %e\n", init_res_norm, rel_res_norm);
            }
         }
         else
         {
            if (old_res_norm > HYPRE_REAL_EPSILON)
            {
               conv_factor = res_norm / old_res_norm;
            }
            else
            {
               conv_factor = res_norm;
            }

            if (rhs_norm > HYPRE_REAL_EPSILON)
            {
               rel_res_norm = res_norm / rhs_norm;
            }
            else
            {
               rel_res_norm = res_norm;
            }
         }

         if (logging)
         {
            hypre_SchwarzDataResNorms(schwarz_data)[iter] = rel_res_norm;
         }

         if (print_level > 1 && my_id == 0 && iter > 0)
         {
            hypre_printf("    Schwarz %2d   %e    %f     %e\n", iter, res_norm, conv_factor, rel_res_norm);
         }

         if (rel_res_norm < tol && iter > 0)
         {
            break;
         }
      }

      /*---------------------------------------------------------------
       * Step 2: Restrict residual to local extended domain
       *---------------------------------------------------------------*/

      /* Initialize local RHS to zero */
      hypre_SeqVectorSetConstantValues(f_local_vec, 0.0);

      /* Copy owned rows from Vtemp to f_local */
      for (i = 0; i < num_extended_rows; i++)
      {
         if (row_is_owned[i])
         {
            HYPRE_Int local_row = (HYPRE_Int)(extended_rows[i] - first_row);
            f_local_data[i] = vtemp_data[local_row];
         }
      }

      if (overlap_comm_pkg && num_procs > 1)
      {
         for (i = 0; i < num_send_vals; i++)
         {
            HYPRE_Int idx = send_map_elmts[i];
            restrict_send_buf[i] = vtemp_data[idx];
         }

         comm_handle = hypre_ParCSRCommHandleCreate(1, overlap_comm_pkg, restrict_send_buf,
                                                    restrict_recv_buf);
         hypre_ParCSRCommHandleDestroy(comm_handle);

         {
            HYPRE_Int ext_idx = 0;
            for (i = 0; i < num_extended_rows; i++)
            {
               if (!row_is_owned[i])
               {
                  f_local_data[i] = restrict_recv_buf[ext_idx++];
               }
            }
         }
      }

      /*---------------------------------------------------------------
       * Step 3: Solve local system: A_local * u_local = f_local
       *---------------------------------------------------------------*/

      /* Initialize local solution to zero */
      hypre_SeqVectorSetConstantValues(u_local_vec, 0.0);

      /* Call local solver */
      if (hypre_SchwarzDataLocalSolverSolve(schwarz_data))
      {
         hypre_SchwarzDataLocalSolverSolve(schwarz_data)(
            hypre_SchwarzDataLocalSolver(schwarz_data),
            (HYPRE_Matrix) hypre_SchwarzDataALocalParCSR(schwarz_data),
            (HYPRE_Vector) f_local_par,
            (HYPRE_Vector) u_local_par);
      }

      /*---------------------------------------------------------------
       * Step 4: Prolongate local solution back to global
       *---------------------------------------------------------------*/

      if (hypre_SchwarzOverlapVariantIsRAS(variant))
      {
         /* RAS: Only update owned rows */
         for (i = 0; i < num_extended_rows; i++)
         {
            if (row_is_owned[i])
            {
               HYPRE_Int local_row = (HYPRE_Int)(extended_rows[i] - first_row);
               HYPRE_Int col_idx = row_to_col_map[i];
               hypre_assert(col_idx >= 0 && col_idx < num_cols_local);
               u_data[local_row] += relax_weight * u_local_data[col_idx];
            }
         }
      }
      else /* HYPRE_SCHWARZ_OVERLAP_VARIANT_AS */
      {
         /* AS: average local corrections for owned rows with corrections
          * returned from all overlapping subdomains. */
         for (i = 0; i < num_extended_rows; i++)
         {
            if (row_is_owned[i])
            {
               HYPRE_Int local_row = (HYPRE_Int)(extended_rows[i] - first_row);
               HYPRE_Int col_idx = row_to_col_map[i];
               hypre_assert(col_idx >= 0 && col_idx < num_cols_local);
               u_data[local_row] += relax_weight * u_local_data[col_idx] /
                                    (HYPRE_Real) as_multiplicity[local_row];
            }
         }

         if (overlap_comm_pkg && num_procs > 1)
         {
            HYPRE_Int   ext_idx = 0;

            for (i = 0; i < num_extended_rows; i++)
            {
               if (!row_is_owned[i])
               {
                  HYPRE_Int col_idx = row_to_col_map[i];
                  as_send_buf[ext_idx++] = (col_idx >= 0 && col_idx < num_cols_local) ?
                                           relax_weight * u_local_data[col_idx] : 0.0;
               }
            }

            comm_handle = hypre_ParCSRCommHandleCreate(2, overlap_comm_pkg, as_send_buf,
                                                       as_recv_buf);
            hypre_ParCSRCommHandleDestroy(comm_handle);

            for (i = 0; i < as_num_recv_vals; i++)
            {
               HYPRE_Int local_row = send_map_elmts[i];
               u_data[local_row] += as_recv_buf[i] / (HYPRE_Real) as_multiplicity[local_row];
            }
         }
      }
   }

   hypre_TFree(restrict_send_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(restrict_recv_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(as_send_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(as_recv_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(as_multiplicity, HYPRE_MEMORY_HOST);

   /* Record final statistics */
   hypre_SchwarzDataNumIterations(schwarz_data) = iter;
   hypre_SchwarzDataFinalResNorm(schwarz_data) = rel_res_norm;

   /*------------------------------------------------------------------
    * Print closing statistics (only when used as standalone solver)
    *------------------------------------------------------------------*/

   if (print_level > 1 && max_iter > 1)
   {
      /* Compute average convergence factor */
      if (iter > 0 && init_res_norm > HYPRE_REAL_EPSILON)
      {
         conv_factor = hypre_pow((res_norm / init_res_norm), (1.0 / (HYPRE_Real) iter));
      }
      else
      {
         conv_factor = 1.0;
      }

      if (my_id == 0)
      {
         if (tol > 0.0 && rel_res_norm >= tol && iter == max_iter)
         {
            hypre_printf("\n\n==============================================");
            hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
            hypre_printf("      within the allowed %d iterations\n", max_iter);
            hypre_printf("==============================================");
         }
         hypre_printf("\n\n Average Convergence Factor = %f\n\n", conv_factor);
      }
   }

   return hypre_error_flag;
}

/*==========================================================================
 * Public Schwarz functions
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SchwarzCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SchwarzCreate( void )
{
   hypre_SchwarzData *schwarz_data;

   HYPRE_Int      variant;
   HYPRE_Int      domain_type;
   HYPRE_Int      overlap;
   HYPRE_Int      num_functions;
   HYPRE_Int      use_nonsymm;
   HYPRE_Real     relax_weight;

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/

   /* setup params */
   variant = 0;  /* multiplicative Schwarz */
   overlap = 1;  /* minimal overlap */
   domain_type = 2; /* domains generated by agglomeration */
   num_functions = 1;
   use_nonsymm = 0;
   relax_weight = 1.0;

   schwarz_data = hypre_CTAlloc(hypre_SchwarzData, 1, HYPRE_MEMORY_HOST);

   hypre_SchwarzSetVariant(schwarz_data, variant);
   hypre_SchwarzSetDomainType(schwarz_data, domain_type);
   hypre_SchwarzSetOverlap(schwarz_data, overlap);
   hypre_SchwarzSetNumFunctions(schwarz_data, num_functions);
   hypre_SchwarzSetNonSymm(schwarz_data, use_nonsymm);
   hypre_SchwarzSetRelaxWeight(schwarz_data, relax_weight);

   /* Domain-based Schwarz-specific data */
   hypre_SchwarzDataDomainStructure(schwarz_data) = NULL;
   hypre_SchwarzDataABoundary(schwarz_data) = NULL;
   hypre_SchwarzDataScale(schwarz_data) = NULL;
   hypre_SchwarzDataVtemp(schwarz_data) = NULL;
   hypre_SchwarzDataDofFunc(schwarz_data) = NULL;
   hypre_SchwarzDataPivots(schwarz_data) = NULL;

   /* Overlapping Schwarz-specific defaults */
   hypre_SchwarzDataLocalSolverType(schwarz_data) = 0;   /* ILU(k) */
   hypre_SchwarzDataILUKLevelOfFill(schwarz_data) = 0;
   hypre_SchwarzDataILUTMaxNnzRow(schwarz_data) = 1000;
   hypre_SchwarzDataILUTDroptol(schwarz_data) = 1.0e-2;
   hypre_SchwarzDataMaxIter(schwarz_data) = 1;
   hypre_SchwarzDataTol(schwarz_data) = 0.0;
   hypre_SchwarzDataPrintLevel(schwarz_data) = 0;
   hypre_SchwarzDataLogging(schwarz_data) = 0;

   /* Initialize overlapping Schwarz setup-time data to NULL */
   hypre_SchwarzDataA(schwarz_data) = NULL;
   hypre_SchwarzDataOverlapData(schwarz_data) = NULL;
   hypre_SchwarzDataNumColsLocal(schwarz_data) = 0;
   hypre_SchwarzDataRowToColMap(schwarz_data) = NULL;
   hypre_SchwarzDataLocalSolver(schwarz_data) = NULL;
   hypre_SchwarzDataLocalSolverSetup(schwarz_data) = NULL;
   hypre_SchwarzDataLocalSolverSolve(schwarz_data) = NULL;
   hypre_SchwarzDataLocalSolverDestroy(schwarz_data) = NULL;
   hypre_SchwarzDataLocalSolverOwner(schwarz_data) = 1;
   hypre_SchwarzDataALocalParCSR(schwarz_data) = NULL;
   hypre_SchwarzDataFLocalPar(schwarz_data) = NULL;
   hypre_SchwarzDataULocalPar(schwarz_data) = NULL;
   hypre_SchwarzDataNumIterations(schwarz_data) = 0;
   hypre_SchwarzDataFinalResNorm(schwarz_data) = 0.0;
   hypre_SchwarzDataResNorms(schwarz_data) = NULL;
   hypre_SchwarzDataCFSolveWarningIssued(schwarz_data) = 0;

   return (void *) schwarz_data;
}

/*--------------------------------------------------------------------------
 * hypre_SchwarzDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzDestroy( void *data )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;
   HYPRE_Int variant;

   if (!schwarz_data)
   {
      return hypre_error_flag;
   }

   variant = hypre_SchwarzDataVariant(schwarz_data);
   hypre_SchwarzOverlapDataDestroy(schwarz_data);

   if (!HYPRE_SCHWARZ_IS_OVERLAPPING(variant))
   {
      /* Destroy domain-based Schwarz data */
      if (hypre_SchwarzDataScale(schwarz_data))
      {
         hypre_TFree(hypre_SchwarzDataScale(schwarz_data), HYPRE_MEMORY_HOST);
      }
      if (hypre_SchwarzDataDofFunc(schwarz_data))
      {
         hypre_TFree(hypre_SchwarzDataDofFunc(schwarz_data), HYPRE_MEMORY_HOST);
      }
      hypre_CSRMatrixDestroy(hypre_SchwarzDataDomainStructure(schwarz_data));
      if (variant == 3)
      {
         hypre_CSRMatrixDestroy(hypre_SchwarzDataABoundary(schwarz_data));
      }
      hypre_ParVectorDestroy(hypre_SchwarzDataVtemp(schwarz_data));

      if (hypre_SchwarzDataPivots(schwarz_data))
      {
         hypre_TFree(hypre_SchwarzDataPivots(schwarz_data), HYPRE_MEMORY_HOST);
      }
   }

   hypre_TFree(schwarz_data, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SchwarzSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetup(void               *schwarz_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *f,
                   hypre_ParVector    *u)
{
   hypre_SchwarzData   *schwarz_data = (hypre_SchwarzData*) schwarz_vdata;
   HYPRE_Int variant = hypre_SchwarzDataVariant(schwarz_data);

   /* Dispatch to overlapping Schwarz if using new variant */
   if (HYPRE_SCHWARZ_IS_OVERLAPPING(variant))
   {
      /* Call overlapping Schwarz setup directly with unified structure */
      return hypre_SchwarzOverlapSetup(schwarz_data, A, f, u);
   }

   /* Original domain-based Schwarz setup code */
   {
      HYPRE_Int *dof_func;
      HYPRE_Real *scale;
      hypre_CSRMatrix *domain_structure;
      hypre_CSRMatrix *A_boundary;
      hypre_ParVector *Vtemp;

      HYPRE_Int *pivots = NULL;

      HYPRE_Int domain_type = hypre_SchwarzDataDomainType(schwarz_data);
      HYPRE_Int overlap = hypre_SchwarzDataOverlap(schwarz_data);
      HYPRE_Int num_functions = hypre_SchwarzDataNumFunctions(schwarz_data);
      HYPRE_Real relax_weight = hypre_SchwarzDataRelaxWeight(schwarz_data);
      HYPRE_Int use_nonsymm = hypre_SchwarzDataUseNonSymm(schwarz_data);

      HYPRE_UNUSED_VAR(f);
      HYPRE_UNUSED_VAR(u);

      dof_func = hypre_SchwarzDataDofFunc(schwarz_data);

      Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                    hypre_ParCSRMatrixGlobalNumRows(A),
                                    hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(Vtemp);
      hypre_SchwarzDataVtemp(schwarz_data) = Vtemp;

      if (variant > 1)
      {
         hypre_ParAMGCreateDomainDof(A,
                                     domain_type, overlap,
                                     num_functions, dof_func,
                                     &domain_structure, &pivots, use_nonsymm);

         if (domain_structure)
         {
            if (variant == 2)
            {
               hypre_ParGenerateScale(A, domain_structure, relax_weight,
                                      &scale);
               hypre_SchwarzDataScale(schwarz_data) = scale;
            }
            else
            {
               hypre_ParGenerateHybridScale(A, domain_structure, &A_boundary, &scale);
               hypre_SchwarzDataScale(schwarz_data) = scale;
               if (hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)))
               {
                  hypre_SchwarzDataABoundary(schwarz_data) = A_boundary;
               }
               else
               {
                  hypre_SchwarzDataABoundary(schwarz_data) = NULL;
               }
            }
         }
      }
      else
      {
         hypre_AMGCreateDomainDof(hypre_ParCSRMatrixDiag(A),
                                  domain_type, overlap,
                                  num_functions, dof_func,
                                  &domain_structure, &pivots, use_nonsymm);
         if (domain_structure)
         {
            if (variant == 1)
            {
               hypre_GenerateScale(domain_structure,
                                   hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A)),
                                   relax_weight, &scale);
               hypre_SchwarzDataScale(schwarz_data) = scale;
            }
         }
      }

      hypre_SchwarzDataDomainStructure(schwarz_data) = domain_structure;
      hypre_SchwarzDataPivots(schwarz_data) = pivots;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_SchwarzSolve
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSolve(void               *schwarz_vdata,
                   hypre_ParCSRMatrix * A,
                   hypre_ParVector    * f,
                   hypre_ParVector    * u)
{
   hypre_SchwarzData   *schwarz_data = (hypre_SchwarzData*) schwarz_vdata;
   HYPRE_Int variant = hypre_SchwarzDataVariant(schwarz_data);

   /* Dispatch to overlapping Schwarz if using new variant */
   if (HYPRE_SCHWARZ_IS_OVERLAPPING(variant))
   {
      /* Call overlapping Schwarz solve directly with unified structure */
      return hypre_SchwarzOverlapSolve(schwarz_data, A, f, u);
   }

   /* Original domain-based Schwarz solve code */
   {
      hypre_CSRMatrix *domain_structure =
         hypre_SchwarzDataDomainStructure(schwarz_data);
      hypre_CSRMatrix *A_boundary = hypre_SchwarzDataABoundary(schwarz_data);
      HYPRE_Real *scale = hypre_SchwarzDataScale(schwarz_data);
      hypre_ParVector *Vtemp = hypre_SchwarzDataVtemp(schwarz_data);
      HYPRE_Real relax_wt = hypre_SchwarzDataRelaxWeight(schwarz_data);
      HYPRE_Int use_nonsymm = hypre_SchwarzDataUseNonSymm(schwarz_data);

      HYPRE_Int *pivots = hypre_SchwarzDataPivots(schwarz_data);

      if (domain_structure)
      {
         if (variant == 2)
         {
            hypre_ParAdSchwarzSolve(A, f, domain_structure, scale, u, Vtemp, pivots, use_nonsymm);
         }
         else if (variant == 3)
         {
            hypre_ParMPSchwarzSolve(A, A_boundary, f, domain_structure, u,
                                    relax_wt, scale, Vtemp, pivots, use_nonsymm);
         }
         else if (variant == 1)
         {
            hypre_AdSchwarzSolve(A, f, domain_structure, scale, u, Vtemp, pivots, use_nonsymm);
         }
         else if (variant == 4)
         {
            hypre_MPSchwarzFWSolve(A, hypre_ParVectorLocalVector(f),
                                   domain_structure, u, relax_wt,
                                   hypre_ParVectorLocalVector(Vtemp), pivots, use_nonsymm);
         }
         else
         {
            hypre_MPSchwarzSolve(A, hypre_ParVectorLocalVector(f),
                                 domain_structure, u, relax_wt,
                                 hypre_ParVectorLocalVector(Vtemp), pivots, use_nonsymm);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_SchwarzCFSolve
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzCFSolve(void               *schwarz_vdata,
                     hypre_ParCSRMatrix * A,
                     hypre_ParVector    * f,
                     hypre_ParVector    * u,
                     HYPRE_Int * CF_marker,
                     HYPRE_Int rlx_pt)
{
   hypre_SchwarzData   *schwarz_data = (hypre_SchwarzData*) schwarz_vdata;
   HYPRE_Int variant = hypre_SchwarzDataVariant(schwarz_data);

   /* CF relaxation not supported for overlapping Schwarz - fall back to regular solve */
   if (HYPRE_SCHWARZ_IS_OVERLAPPING(variant))
   {
      if (!hypre_SchwarzDataCFSolveWarningIssued(schwarz_data))
      {
         HYPRE_Int my_id;

         hypre_MPI_Comm_rank(hypre_ParCSRMatrixComm(A), &my_id);
         if (my_id == 0)
         {
            hypre_printf("Warning: hypre_SchwarzCFSolve does not support CF relaxation "
                         "for overlapping Schwarz; using regular Schwarz solve\n");
         }
         hypre_SchwarzDataCFSolveWarningIssued(schwarz_data) = 1;
      }
      HYPRE_UNUSED_VAR(CF_marker);
      HYPRE_UNUSED_VAR(rlx_pt);
      return hypre_SchwarzSolve(schwarz_vdata, A, f, u);
   }

   /* Original domain-based Schwarz CF solve code */
   {
      hypre_CSRMatrix *domain_structure =
         hypre_SchwarzDataDomainStructure(schwarz_data);
      HYPRE_Real *scale = hypre_SchwarzDataScale(schwarz_data);
      hypre_ParVector *Vtemp = hypre_SchwarzDataVtemp(schwarz_data);
      HYPRE_Real relax_wt = hypre_SchwarzDataRelaxWeight(schwarz_data);

      HYPRE_Int use_nonsymm = hypre_SchwarzDataUseNonSymm(schwarz_data);

      HYPRE_Int *pivots = hypre_SchwarzDataPivots(schwarz_data);

      if (variant == 1)
      {
         hypre_AdSchwarzCFSolve(A, f, domain_structure, scale, u, Vtemp,
                                CF_marker, rlx_pt, pivots, use_nonsymm);
      }
      else if (variant == 4)
      {
         hypre_MPSchwarzCFFWSolve(A, hypre_ParVectorLocalVector(f),
                                  domain_structure, u, relax_wt,
                                  hypre_ParVectorLocalVector(Vtemp),
                                  CF_marker, rlx_pt, pivots, use_nonsymm);
      }
      else
      {
         hypre_MPSchwarzCFSolve(A, hypre_ParVectorLocalVector(f),
                                domain_structure, u, relax_wt,
                                hypre_ParVectorLocalVector(Vtemp),
                                CF_marker, rlx_pt, pivots, use_nonsymm);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the variant for the Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetVariant( void *data, HYPRE_Int variant )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   hypre_SchwarzDataVariant(schwarz_data) = variant;

   if (!hypre_SchwarzVariantIsValid(variant))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid Schwarz variant");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the domain type for the Domain-based (old) Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetDomainType( void *data, HYPRE_Int domain_type )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   if (domain_type < 0 || domain_type > 2)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzDataDomainType(schwarz_data) = domain_type;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the overlap order for the Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetOverlap( void *data, HYPRE_Int overlap )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   if (overlap < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzDataOverlap(schwarz_data) = overlap;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the number of functions for the Domain-based (old) Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetNumFunctions( void *data, HYPRE_Int num_functions )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   hypre_SchwarzDataNumFunctions(schwarz_data) = num_functions;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the non-symmetry flag for the Domain-based (old) Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetNonSymm( void *data, HYPRE_Int value )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   hypre_SchwarzDataUseNonSymm(schwarz_data) = value;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the relaxation weight for the Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetRelaxWeight( void *data, HYPRE_Real relax_weight )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   if (relax_weight < 0.0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzDataRelaxWeight(schwarz_data) = relax_weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the domain structure for the Domain-based (old) Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetDomainStructure( void *data, hypre_CSRMatrix * domain_structure )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   hypre_SchwarzDataDomainStructure(schwarz_data) = domain_structure;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the scale coefficient for the Domain-based (old) Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetScale( void *data, HYPRE_Real * scale)
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   hypre_SchwarzDataScale(schwarz_data) = scale;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Rescale the scale coefficient for the Domain-based (old) Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzReScale( void *data, HYPRE_Int size, HYPRE_Real value)
{
   HYPRE_Int i;
   HYPRE_Real *scale;
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   scale = hypre_SchwarzDataScale(schwarz_data);
   for (i = 0; i < size; i++)
   {
      scale[i] *= value;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the degree of freedom function for the Domain-based (old) Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetDofFunc( void *data, HYPRE_Int * dof_func)
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   hypre_SchwarzDataDofFunc(schwarz_data) = dof_func;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the local (subdomain) solver type for the Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetLocalSolverType( void *data, HYPRE_Int local_solver_type )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;
   HYPRE_Int           variant;
   HYPRE_Int           overlap_variant;

   overlap_variant = hypre_SchwarzOverlapVariantFromLocalSolver(local_solver_type, 1);
   if (overlap_variant < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzDataLocalSolverType(schwarz_data) = local_solver_type;
   variant = hypre_SchwarzDataVariant(schwarz_data);

   if (variant == HYPRE_SCHWARZ_VARIANT_MP)
   {
      hypre_SchwarzDataVariant(schwarz_data) = overlap_variant;
   }
   else if (HYPRE_SCHWARZ_IS_OVERLAPPING(variant))
   {
      hypre_SchwarzDataVariant(schwarz_data) =
         hypre_SchwarzOverlapVariantFromLocalSolver(local_solver_type,
                                                    hypre_SchwarzOverlapVariantIsRAS(variant));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the level of fill for ILU(k) local solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetILUKLevelOfFill( void *data, HYPRE_Int level_of_fill )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   if (level_of_fill < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzDataILUKLevelOfFill(schwarz_data) = level_of_fill;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the maximum nonzeros per row for ILUT local solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetILUTMaxNnzPerRow( void *data, HYPRE_Int max_nnz_row )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   if (max_nnz_row <= 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzDataILUTMaxNnzRow(schwarz_data) = max_nnz_row;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the drop tolerance for ILUT local solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetILUTDroptol( void *data, HYPRE_Real droptol )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   if (droptol < 0.0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzDataILUTDroptol(schwarz_data) = droptol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the maximum number of iterations for the Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetMaxIter( void *data, HYPRE_Int max_iter )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   if (max_iter < 0 || max_iter == HYPRE_INT_MAX)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzDataMaxIter(schwarz_data) = max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the convergence tolerance for the Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetTol( void *data, HYPRE_Real tol )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   if (tol < 0.0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzDataTol(schwarz_data) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the print level for the Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetPrintLevel( void *data, HYPRE_Int print_level )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   if (print_level < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzDataPrintLevel(schwarz_data) = print_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set the logging level for the Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzSetLogging( void *data, HYPRE_Int logging )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;

   if (logging < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_SchwarzDataLogging(schwarz_data) = logging;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
   * Getter functions for statistics
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzGetNumIterations( void *data, HYPRE_Int * num_iterations )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;
   HYPRE_Int variant = hypre_SchwarzDataVariant(schwarz_data);

   if (HYPRE_SCHWARZ_IS_OVERLAPPING(variant))
   {
      *num_iterations = hypre_SchwarzDataNumIterations(schwarz_data);
      return hypre_error_flag;
   }

   /* Domain-based Schwarz is always 1 iteration */
   *num_iterations = 1;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Get the final residual norm for the Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SchwarzGetFinalResidualNorm( void *data, HYPRE_Real * norm )
{
   hypre_SchwarzData  *schwarz_data = (hypre_SchwarzData*) data;
   HYPRE_Int variant = hypre_SchwarzDataVariant(schwarz_data);

   if (HYPRE_SCHWARZ_IS_OVERLAPPING(variant))
   {
      *norm = hypre_SchwarzDataFinalResNorm(schwarz_data);
      return hypre_error_flag;
   }

   /* Domain-based Schwarz doesn't track residual norm */
   *norm = 0.0;
   return hypre_error_flag;
}
