/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_mv.h"

/*==========================================================================
 * SStructPMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixComputeRowSum
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixComputeRowSum( hypre_SStructPMatrix  *pA,
                                   HYPRE_Int              type,
                                   hypre_SStructPVector  *prowsum )
{
   HYPRE_Int nvars = hypre_SStructPMatrixNVars(pA);

   hypre_StructMatrix  *sA;
   hypre_StructVector  *sv;
   HYPRE_Int            vi, vj;

   for (vi = 0; vi < nvars; vi++)
   {
      sv = hypre_SStructPVectorSVector(prowsum, vi);
      for (vj = 0; vj < nvars; vj++)
      {
         sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
         hypre_StructMatrixComputeRowSum(sA, type, sv);
      }
   }

   return hypre_error_flag;
}

/*==========================================================================
 * SStructMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixComputeRowSum
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixComputeRowSum( hypre_SStructMatrix  *A,
                                  HYPRE_Int             type,
                                  hypre_SStructVector **rowsum_ptr )
{
   MPI_Comm               comm  = hypre_SStructMatrixComm(A);
   hypre_ParCSRMatrix    *par_A = hypre_SStructMatrixParCSRMatrix(A);
   hypre_CSRMatrix       *A_diag = hypre_ParCSRMatrixDiag(par_A);
   hypre_CSRMatrix       *A_offd = hypre_ParCSRMatrixOffd(par_A);
   hypre_SStructGraph    *graph = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid     *grid  = hypre_SStructGraphGrid(graph);
   HYPRE_Int              object_type = hypre_SStructMatrixObjectType(A);
   HYPRE_Int              nparts = hypre_SStructMatrixNParts(A);
   hypre_SStructVector   *rowsum;

   hypre_SStructPMatrix  *pA;
   hypre_SStructPVector  *pv;
   HYPRE_Complex         *data;
   HYPRE_Int              part;

   if (*rowsum_ptr)
   {
      rowsum = *rowsum_ptr;
   }
   else
   {
      HYPRE_SStructVectorCreate(comm, grid, &rowsum);
      HYPRE_SStructVectorInitialize(rowsum);
      HYPRE_SStructVectorAssemble(rowsum);
   }
   data = hypre_SStructVectorData(rowsum);

   if ((object_type == HYPRE_SSTRUCT) || (object_type == HYPRE_STRUCT))
   {
      /* do S-matrix computations */
      for (part = 0; part < nparts; part++)
      {
         pA = hypre_SStructMatrixPMatrix(A, part);
         pv = hypre_SStructVectorPVector(rowsum, part);

         hypre_SStructPMatrixComputeRowSum(pA, type, pv);
      }

      if (object_type == HYPRE_SSTRUCT)
      {
         hypre_CSRMatrixComputeRowSum(A_diag, NULL, NULL, data, type, 1.0, "add");
         hypre_CSRMatrixComputeRowSum(A_offd, NULL, NULL, data, type, 1.0, "add");
      }
   }
   else
   {
      hypre_CSRMatrixComputeRowSum(A_diag, NULL, NULL, data, type, 1.0, "set");
      hypre_CSRMatrixComputeRowSum(A_offd, NULL, NULL, data, type, 1.0, "add");
   }

   *rowsum_ptr = rowsum;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixComputeL1Norms
 *
 * Compute the l1 norms of the rows of a given matrix, depending on
 * the option parameter:
 *
 * option 1 = Compute the l1 norm of the rows
 * option 2 = Compute the l1 norm of the (processor) off-diagonal
 *            part of the rows plus the diagonal of A
 * option 3 = Compute the l2 norm^2 of the rows
 * option 4 = Truncated version of option 2 based on Remark 6.2 in "Multigrid
 *            Smoothers for Ultra-Parallel Computing"
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixComputeL1Norms( hypre_SStructMatrix  *A,
                                   HYPRE_Int             option,
                                   hypre_SStructVector **l1_norms_ptr )
{
   MPI_Comm               comm  = hypre_SStructMatrixComm(A);
   hypre_SStructGraph    *graph = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid     *grid  = hypre_SStructGraphGrid(graph);
   hypre_SStructVector   *l1_norms;

   /* Create l1_norms vector */
   HYPRE_SStructVectorCreate(comm, grid, &l1_norms);
   HYPRE_SStructVectorInitialize(l1_norms);
   HYPRE_SStructVectorAssemble(l1_norms);

   /* Compute l1_norms */
   if (option == 1)
   {
      hypre_SStructMatrixComputeRowSum(A, 1, &l1_norms);
   }
   else if (option == 2)
   {
   }
   else if (option == 3)
   {
   }
   else if (option == 4)
   {
   }
   else
   {
   }

   *l1_norms_ptr = l1_norms;

   return hypre_error_flag;
}
