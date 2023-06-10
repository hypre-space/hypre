/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_utilities.h"

#if defined(HYPRE_USING_GPU)

//-----------------------------------------------------------------------
HYPRE_Int
hypre_BoomerAMGCreate2ndSDevice( hypre_ParCSRMatrix  *S,
                                 HYPRE_Int           *CF_marker,
                                 HYPRE_Int            num_paths,
                                 HYPRE_BigInt        *coarse_row_starts,
                                 hypre_ParCSRMatrix **S2_ptr)
{
   HYPRE_Int           S_nr_local = hypre_ParCSRMatrixNumRows(S);
   hypre_CSRMatrix    *S_diag     = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix    *S_offd     = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int           S_diag_nnz = hypre_CSRMatrixNumNonzeros(S_diag);
   HYPRE_Int           S_offd_nnz = hypre_CSRMatrixNumNonzeros(S_offd);
   hypre_CSRMatrix    *Id, *SI_diag;
   hypre_ParCSRMatrix *S_XC, *S_CX, *S2;
   HYPRE_Int          *new_end;
   HYPRE_Complex       coeff = 2.0;

   /*
   MPI_Comm comm = hypre_ParCSRMatrixComm(S);
   HYPRE_Int num_proc, myid;
   hypre_MPI_Comm_size(comm, &num_proc);
   hypre_MPI_Comm_rank(comm, &myid);
   */

   /* 1. Create new matrix with added diagonal */
   hypre_GpuProfilingPushRange("Setup");

   /* give S data arrays */
   hypre_CSRMatrixData(S_diag) = hypre_TAlloc(HYPRE_Complex, S_diag_nnz, HYPRE_MEMORY_DEVICE );
   hypreDevice_ComplexFilln( hypre_CSRMatrixData(S_diag),
                             S_diag_nnz,
                             1.0 );

   hypre_CSRMatrixData(S_offd) = hypre_TAlloc(HYPRE_Complex, S_offd_nnz, HYPRE_MEMORY_DEVICE );
   hypreDevice_ComplexFilln( hypre_CSRMatrixData(S_offd),
                             S_offd_nnz,
                             1.0 );

   if (!hypre_ParCSRMatrixCommPkg(S))
   {
      hypre_MatvecCommPkgCreate(S);
   }

   /* S(C, :) and S(:, C) */
   hypre_ParCSRMatrixGenerate1DCFDevice(S, CF_marker, coarse_row_starts, NULL, &S_CX, &S_XC);

   hypre_assert(S_nr_local == hypre_ParCSRMatrixNumCols(S_CX));

   /* add coeff*I to S_CX */
   Id = hypre_CSRMatrixCreate( hypre_ParCSRMatrixNumRows(S_CX),
                               hypre_ParCSRMatrixNumCols(S_CX),
                               hypre_ParCSRMatrixNumRows(S_CX) );

   hypre_CSRMatrixInitialize_v2(Id, 0, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   hypreSycl_sequence( hypre_CSRMatrixI(Id),
                       hypre_CSRMatrixI(Id) + hypre_ParCSRMatrixNumRows(S_CX) + 1,
                       0 );

   oneapi::dpl::counting_iterator<HYPRE_Int> count(0);
   new_end = hypreSycl_copy_if( count,
                                count + hypre_ParCSRMatrixNumCols(S_CX),
                                CF_marker,
                                hypre_CSRMatrixJ(Id),
                                is_nonnegative<HYPRE_Int>()  );
#else
   HYPRE_THRUST_CALL( sequence,
                      hypre_CSRMatrixI(Id),
                      hypre_CSRMatrixI(Id) + hypre_ParCSRMatrixNumRows(S_CX) + 1,
                      0  );

   new_end = HYPRE_THRUST_CALL( copy_if,
                                thrust::make_counting_iterator(0),
                                thrust::make_counting_iterator(hypre_ParCSRMatrixNumCols(S_CX)),
                                CF_marker,
                                hypre_CSRMatrixJ(Id),
                                is_nonnegative<HYPRE_Int>()  );
#endif

   hypre_assert(new_end - hypre_CSRMatrixJ(Id) == hypre_ParCSRMatrixNumRows(S_CX));

   hypreDevice_ComplexFilln( hypre_CSRMatrixData(Id),
                             hypre_ParCSRMatrixNumRows(S_CX),
                             coeff );

   SI_diag = hypre_CSRMatrixAddDevice(1.0, hypre_ParCSRMatrixDiag(S_CX), 1.0, Id);

   hypre_CSRMatrixDestroy(Id);

   /* global nnz has changed, but we do not care about it */
   /*
   hypre_ParCSRMatrixSetNumNonzeros(S_CX);
   hypre_ParCSRMatrixDNumNonzeros(S_CX) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(S_CX);
   */

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(S_CX));
   hypre_ParCSRMatrixDiag(S_CX) = SI_diag;

   hypre_GpuProfilingPopRange();

   /* 2. Perform matrix-matrix multiplication */
   hypre_GpuProfilingPushRange("Matrix-matrix mult");

   S2 = hypre_ParCSRMatMatDevice(S_CX, S_XC);

   hypre_ParCSRMatrixDestroy(S_CX);
   hypre_ParCSRMatrixDestroy(S_XC);

   hypre_GpuProfilingPopRange();

   // Clean up matrix before returning it.
   if (num_paths == 2)
   {
      // If num_paths = 2, prune elements < 2.
      hypre_ParCSRMatrixDropSmallEntries(S2, 1.5, 0);
   }

   hypre_TFree(hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(S2)), HYPRE_MEMORY_DEVICE);
   hypre_TFree(hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(S2)), HYPRE_MEMORY_DEVICE);

   hypre_CSRMatrixRemoveDiagonalDevice(hypre_ParCSRMatrixDiag(S2));

   /* global nnz has changed, but we do not care about it */

   hypre_MatvecCommPkgCreate(S2);

   *S2_ptr = S2;

   return 0;
}

#endif /* #if defined(HYPRE_USING_GPU) */
