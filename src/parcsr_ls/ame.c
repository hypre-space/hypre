/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"
#include "float.h"
#include "ams.h"
#include "temp_multivector.h"
#include "lobpcg.h"
#include "ame.h"
#include "_hypre_utilities.hpp"

/*--------------------------------------------------------------------------
 * hypre_AMECreate
 *
 * Allocate the AMS eigensolver structure.
 *--------------------------------------------------------------------------*/

void * hypre_AMECreate(void)
{
   hypre_AMEData *ame_data;

   ame_data = hypre_CTAlloc(hypre_AMEData,  1, HYPRE_MEMORY_HOST);

   /* Default parameters */

   ame_data -> block_size = 1;  /* compute 1 eigenvector */
   ame_data -> pcg_maxit = 20;  /* perform at most 20 PCG-AMG */
   ame_data -> maxit = 100;     /* perform at most 100 iterations */
   ame_data -> atol = 1e-6;     /* absolute convergence tolerance */
   ame_data -> rtol = 1e-6;     /* relative convergence tolerance */
   ame_data -> print_level = 1; /* print max residual norm at each step */

   /* These will be computed during setup */

   ame_data -> eigenvectors = NULL;
   ame_data -> eigenvalues  = NULL;
   ame_data -> interpreter  = NULL;
   ame_data -> G            = NULL;
   ame_data -> A_G          = NULL;
   ame_data -> B1_G         = NULL;
   ame_data -> B2_G         = NULL;
   ame_data -> t1           = NULL;
   ame_data -> t2           = NULL;
   ame_data -> t3           = NULL;

   /* The rest of the fields are initialized using the Set functions */

   ame_data -> precond      = NULL;
   ame_data -> M            = NULL;

   return (void *) ame_data;
}

/*--------------------------------------------------------------------------
 * hypre_AMEDestroy
 *
 * Deallocate the AMS eigensolver structure. If hypre_AMEGetEigenvectors()
 * has been called, the eigenvalue/vector data will not be destroyed.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMEDestroy(void *esolver)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   hypre_AMSData *ams_data;
   mv_InterfaceInterpreter* interpreter;
   mv_MultiVectorPtr eigenvectors;

   if (!ame_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   ams_data = ame_data -> precond;
   interpreter = (mv_InterfaceInterpreter*) ame_data -> interpreter;
   eigenvectors = (mv_MultiVectorPtr) ame_data -> eigenvectors;
   if (!ams_data || !interpreter || !eigenvectors)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (ame_data -> G)
   {
      hypre_ParCSRMatrixDestroy(ame_data -> G);
   }
   if (ame_data -> A_G)
   {
      hypre_ParCSRMatrixDestroy(ame_data -> A_G);
   }
   if (ame_data -> B1_G)
   {
      HYPRE_BoomerAMGDestroy(ame_data -> B1_G);
   }
   if (ame_data -> B2_G)
   {
      HYPRE_ParCSRPCGDestroy(ame_data -> B2_G);
   }

   if (ame_data -> eigenvalues)
   {
      hypre_TFree(ame_data -> eigenvalues, HYPRE_MEMORY_HOST);
   }
   if (eigenvectors)
   {
      mv_MultiVectorDestroy(eigenvectors);
   }

   if (interpreter)
   {
      hypre_TFree(interpreter, HYPRE_MEMORY_HOST);
   }

   if (ams_data ->  beta_is_zero)
   {
      if (ame_data -> t1)
      {
         hypre_ParVectorDestroy(ame_data -> t1);
      }
      if (ame_data -> t2)
      {
         hypre_ParVectorDestroy(ame_data -> t2);
      }
   }

   if (ame_data)
   {
      hypre_TFree(ame_data, HYPRE_MEMORY_HOST);
   }

   /* Fields initialized using the Set functions are not destroyed */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMESetAMSSolver
 *
 * Sets the AMS solver to be used as a preconditioner in the eigensolver.
 * This function should be called before hypre_AMESetup()!
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMESetAMSSolver(void *esolver,
                                void *ams_solver)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   ame_data -> precond = (hypre_AMSData*) ams_solver;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMESetMassMatrix
 *
 * Sets the edge mass matrix, which appear on the rhs of the eigenproblem.
 * This function should be called before hypre_AMESetup()!
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMESetMassMatrix(void *esolver,
                                 hypre_ParCSRMatrix *M)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   ame_data -> M = M;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMESetBlockSize
 *
 * Sets the block size -- the number of eigenvalues/eigenvectors to be
 * computed. This function should be called before hypre_AMESetup()!
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMESetBlockSize(void *esolver,
                                HYPRE_Int block_size)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   ame_data -> block_size = block_size;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMESetMaxIter
 *
 * Set the maximum number of iterations. The default value is 100.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMESetMaxIter(void *esolver,
                              HYPRE_Int maxit)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   ame_data -> maxit = maxit;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMESetMaxPCGIter
 *
 * Set the maximum number of iterations. The default value is 20.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMESetMaxPCGIter(void *esolver,
                                 HYPRE_Int maxit)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   ame_data -> pcg_maxit = maxit;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMESetTol
 *
 * Set the absolute convergence tolerance. The default value is 1e-6.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMESetTol(void *esolver,
                          HYPRE_Real tol)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   ame_data -> atol = tol;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMESetRTol
 *
 * Set the relative convergence tolerance. The default value is 1e-6.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMESetRTol(void *esolver,
                           HYPRE_Real tol)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   ame_data -> rtol = tol;
   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_AMESetPrintLevel
 *
 * Control how much information is printed during the solution iterations.
 * The default values is 1.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMESetPrintLevel(void *esolver,
                                 HYPRE_Int print_level)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   ame_data -> print_level = print_level;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMESetup
 *
 * Construct an eigensolver based on existing AMS solver. The number of
 * desired (minimal nonzero) eigenvectors is set by hypre_AMESetBlockSize().
 *
 * The following functions need to be called before hypre_AMSSetup():
 * - hypre_AMESetAMSSolver()
 * - hypre_AMESetMassMatrix()
 *--------------------------------------------------------------------------*/
#if defined(HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_GtEliminateBoundary( hypre_DeviceItem    &item,
                                    HYPRE_Int      nrows,
                                    HYPRE_Int     *Gt_diag_i,
                                    HYPRE_Int     *Gt_diag_j,
                                    HYPRE_Complex *Gt_diag_data,
                                    HYPRE_Int     *Gt_offd_i,
                                    HYPRE_Int     *Gt_offd_j,
                                    HYPRE_Complex *Gt_offd_data,
                                    HYPRE_Int     *edge_bc,
                                    HYPRE_Int     *edge_bc_offd)
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p1 = 0, q1, p2 = 0, q2 = 0;
   bool nonempty_offd = Gt_offd_j != NULL;
   bool bdr = false;

   if (lane < 2)
   {
      p1 = read_only_load(Gt_diag_i + row_i + lane);
      if (nonempty_offd)
      {
         p2 = read_only_load(Gt_offd_i + row_i + lane);
      }
   }

   q1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p1, 1);
   p1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p1, 0);
   if (nonempty_offd)
   {
      q2 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p2, 1);
      p2 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p2, 0);
   }

   for (HYPRE_Int j = p1 + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q1);
        j += HYPRE_WARP_SIZE)
   {
      const hypre_int k = j < q1 && read_only_load(&edge_bc[read_only_load(&Gt_diag_j[j])]) != 0;
      if ( warp_any_sync(item, HYPRE_WARP_FULL_MASK, k) )
      {
         bdr = true;
         break;
      }
   }

   if (!bdr)
   {
      for (HYPRE_Int j = p2 + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q2);
           j += HYPRE_WARP_SIZE)
      {
         const hypre_int k = j < q2 && read_only_load(&edge_bc_offd[read_only_load(&Gt_offd_j[j])]) != 0;
         if ( warp_any_sync(item, HYPRE_WARP_FULL_MASK, k) )
         {
            bdr = true;
            break;
         }
      }
   }

   if (bdr)
   {
      for (HYPRE_Int j = p1 + lane; j < q1; j += HYPRE_WARP_SIZE)
      {
         Gt_diag_data[j] = 0.0;
      }
      for (HYPRE_Int j = p2 + lane; j < q2; j += HYPRE_WARP_SIZE)
      {
         Gt_offd_data[j] = 0.0;
      }
   }
}
#endif

HYPRE_Int hypre_AMESetup(void *esolver)
{
   HYPRE_Int ne, *edge_bc;

   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   hypre_AMSData *ams_data = ame_data -> precond;

   if (ams_data -> beta_is_zero)
   {
      ame_data -> t1 = hypre_ParVectorInDomainOf(ams_data -> G);
      ame_data -> t2 = hypre_ParVectorInDomainOf(ams_data -> G);
   }
   else
   {
      ame_data -> t1 = ams_data -> r1;
      ame_data -> t2 = ams_data -> g1;
   }
   ame_data -> t3 = ams_data -> r0;

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(ams_data -> A);
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memory_location);
#endif

   /* Eliminate boundary conditions in G = [Gii, Gib; 0, Gbb], i.e.,
      compute [Gii, 0; 0, 0] */
   {
      HYPRE_Int i, j, k, nv;
      HYPRE_Int *offd_edge_bc;

      hypre_ParCSRMatrix *Gt;

      nv = hypre_ParCSRMatrixNumCols(ams_data -> G);
      ne = hypre_ParCSRMatrixNumRows(ams_data -> G);

      edge_bc = hypre_CTAlloc(HYPRE_Int, ne, memory_location);

      /* Find boundary (eliminated) edges */
      {
         hypre_CSRMatrix *Ad = hypre_ParCSRMatrixDiag(ams_data -> A);
         HYPRE_Int *AdI = hypre_CSRMatrixI(Ad);
         HYPRE_Int *AdJ = hypre_CSRMatrixJ(Ad);
         HYPRE_Real *AdA = hypre_CSRMatrixData(Ad);
         hypre_CSRMatrix *Ao = hypre_ParCSRMatrixOffd(ams_data -> A);
         HYPRE_Int *AoI = hypre_CSRMatrixI(Ao);
         HYPRE_Real *AoA = hypre_CSRMatrixData(Ao);

         /* A row (edge) is boundary if its off-diag l1 norm is less than eps */
         HYPRE_Real eps = DBL_EPSILON * 1e+4;

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            HYPRE_Real *l1norm_arr = hypre_TAlloc(HYPRE_Real, ne, memory_location);
            hypre_CSRMatrixExtractDiagonalDevice(Ad, l1norm_arr, 1);
#if defined(HYPRE_USING_SYCL)
            HYPRE_ONEDPL_CALL( std::transform,
                               l1norm_arr,
                               l1norm_arr + ne,
                               l1norm_arr,
                               std::negate<HYPRE_Real>() );
#else
            HYPRE_THRUST_CALL( transform,
                               l1norm_arr,
                               l1norm_arr + ne,
                               l1norm_arr,
                               thrust::negate<HYPRE_Real>() );
#endif
            hypre_CSRMatrixComputeRowSumDevice(Ad, NULL, NULL, l1norm_arr, 1, 1.0, "add");
            if (AoA)
            {
               hypre_CSRMatrixComputeRowSumDevice(Ao, NULL, NULL, l1norm_arr, 1, 1.0, "add");
            }
#if defined(HYPRE_USING_SYCL)
            hypreSycl_transform_if( edge_bc,
                                    edge_bc + ne,
                                    l1norm_arr,
                                    edge_bc,
            [] (const auto & x) {return 1;},
            less_than<HYPRE_Real>(eps) );
#else
            HYPRE_THRUST_CALL( replace_if,
                               edge_bc,
                               edge_bc + ne,
                               l1norm_arr,
                               less_than<HYPRE_Real>(eps),
                               1 );
#endif
            hypre_TFree(l1norm_arr, memory_location);
         }
         else
#endif
         {
            HYPRE_Real l1_norm;
            for (i = 0; i < ne; i++)
            {
               l1_norm = 0.0;
               for (j = AdI[i]; j < AdI[i + 1]; j++)
                  if (AdJ[j] != i)
                  {
                     l1_norm += hypre_abs(AdA[j]);
                  }
               if (AoI)
                  for (j = AoI[i]; j < AoI[i + 1]; j++)
                  {
                     l1_norm += hypre_abs(AoA[j]);
                  }
               if (l1_norm < eps)
               {
                  edge_bc[i] = 1;
               }
            }
         }
      }

      hypre_ParCSRMatrixTranspose(ams_data->G, &Gt, 1);

      hypre_assert( hypre_ParCSRMatrixMemoryLocation(ams_data->G) == memory_location);

      /* Use a Matvec communication to find which of the edges
         connected to local vertices are on the boundary */
      {
         hypre_ParCSRCommHandle *comm_handle;
         hypre_ParCSRCommPkg *comm_pkg;
         HYPRE_Int num_sends, *int_buf_data;
         HYPRE_Int index, start;

         offd_edge_bc = hypre_TAlloc(HYPRE_Int, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(Gt)),
                                     memory_location);

         hypre_MatvecCommPkgCreate(Gt);
         comm_pkg = hypre_ParCSRMatrixCommPkg(Gt);

         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
         int_buf_data = hypre_TAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                     memory_location );

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

#if defined(HYPRE_USING_SYCL)
            hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                              hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                                                                num_sends),
                              edge_bc,
                              int_buf_data );
#else
            HYPRE_THRUST_CALL( gather,
                               hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                               hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                     num_sends),
                               edge_bc,
                               int_buf_data );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
            /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
            if (hypre_GetGpuAwareMPI())
            {
               hypre_ForceSyncComputeStream(hypre_handle());
            }
#endif
         }
         else
#endif
         {
            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
               {
                  k = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
                  int_buf_data[index++] = edge_bc[k];
               }
            }
         }

         comm_handle = hypre_ParCSRCommHandleCreate_v2(11, comm_pkg,
                                                       memory_location, int_buf_data,
                                                       memory_location, offd_edge_bc);
         hypre_ParCSRCommHandleDestroy(comm_handle);
         hypre_TFree(int_buf_data, memory_location);
      }

      /* Eliminate boundary vertex entries in G^t */
      {
         hypre_CSRMatrix *Gtd = hypre_ParCSRMatrixDiag(Gt);
         HYPRE_Int *GtdI = hypre_CSRMatrixI(Gtd);
         HYPRE_Int *GtdJ = hypre_CSRMatrixJ(Gtd);
         HYPRE_Real *GtdA = hypre_CSRMatrixData(Gtd);
         hypre_CSRMatrix *Gto = hypre_ParCSRMatrixOffd(Gt);
         HYPRE_Int *GtoI = hypre_CSRMatrixI(Gto);
         HYPRE_Int *GtoJ = hypre_CSRMatrixJ(Gto);
         HYPRE_Real *GtoA = hypre_CSRMatrixData(Gto);

         HYPRE_Int bdr;

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(nv, "warp", bDim);
            HYPRE_GPU_LAUNCH( hypreGPUKernel_GtEliminateBoundary, gDim, bDim,
                              nv, GtdI, GtdJ, GtdA, GtoI, GtoJ, GtoA, edge_bc, offd_edge_bc );
         }
         else
#endif
         {
            for (i = 0; i < nv; i++)
            {
               bdr = 0;
               /* A vertex is boundary if it belongs to a boundary edge */
               for (j = GtdI[i]; j < GtdI[i + 1]; j++)
                  if (edge_bc[GtdJ[j]]) { bdr = 1; break; }
               if (!bdr && GtoI)
                  for (j = GtoI[i]; j < GtoI[i + 1]; j++)
                     if (offd_edge_bc[GtoJ[j]]) { bdr = 1; break; }

               if (bdr)
               {
                  for (j = GtdI[i]; j < GtdI[i + 1]; j++)
                     /* if (!edge_bc[GtdJ[j]]) */
                  {
                     GtdA[j] = 0.0;
                  }
                  if (GtoI)
                     for (j = GtoI[i]; j < GtoI[i + 1]; j++)
                        /* if (!offd_edge_bc[GtoJ[j]]) */
                     {
                        GtoA[j] = 0.0;
                     }
               }
            }
         }
      }

      hypre_ParCSRMatrixTranspose(Gt, &ame_data -> G, 1);

      hypre_ParCSRMatrixDestroy(Gt);
      hypre_TFree(offd_edge_bc, memory_location);
   }

   /* Compute G^t M G */
   {
      if (!hypre_ParCSRMatrixCommPkg(ame_data -> G))
      {
         hypre_MatvecCommPkgCreate(ame_data -> G);
      }

      if (!hypre_ParCSRMatrixCommPkg(ame_data -> M))
      {
         hypre_MatvecCommPkgCreate(ame_data -> M);
      }

#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         ame_data -> A_G = hypre_ParCSRMatrixRAPKT(ame_data -> G, ame_data -> M, ame_data -> G, 1);
      }
      else
#endif
      {
         hypre_BoomerAMGBuildCoarseOperator(ame_data -> G,
                                            ame_data -> M,
                                            ame_data -> G,
                                            &ame_data -> A_G);
      }

      hypre_ParCSRMatrixFixZeroRows(ame_data -> A_G);
   }

   /* Create AMG preconditioner and PCG-AMG solver for G^tMG */
   {
      HYPRE_BoomerAMGCreate(&ame_data -> B1_G);
      HYPRE_BoomerAMGSetCoarsenType(ame_data -> B1_G, ams_data -> B_G_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ame_data -> B1_G, ams_data -> B_G_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ame_data -> B1_G, ams_data -> B_G_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ame_data -> B1_G, 1);
      HYPRE_BoomerAMGSetMaxLevels(ame_data -> B1_G, 25);
      HYPRE_BoomerAMGSetTol(ame_data -> B1_G, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ame_data -> B1_G, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ame_data -> B1_G, ams_data -> B_G_theta);
      /* don't use exact solve on the coarsest level (matrix may be singular) */
      HYPRE_BoomerAMGSetCycleRelaxType(ame_data -> B1_G,
                                       ams_data -> B_G_relax_type,
                                       3);

      HYPRE_ParCSRPCGCreate(hypre_ParCSRMatrixComm(ame_data->A_G),
                            &ame_data -> B2_G);
      HYPRE_PCGSetPrintLevel(ame_data -> B2_G, 0);
      HYPRE_PCGSetTol(ame_data -> B2_G, 1e-12);
      HYPRE_PCGSetMaxIter(ame_data -> B2_G, ame_data -> pcg_maxit);

      HYPRE_PCGSetPrecond(ame_data -> B2_G,
                          (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                          (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                          ame_data -> B1_G);

      HYPRE_ParCSRPCGSetup(ame_data -> B2_G,
                           (HYPRE_ParCSRMatrix)ame_data->A_G,
                           (HYPRE_ParVector)ame_data->t1,
                           (HYPRE_ParVector)ame_data->t2);
   }

   /* Setup LOBPCG */
   {
      HYPRE_Int seed = 75;
      mv_InterfaceInterpreter* interpreter;
      mv_MultiVectorPtr eigenvectors;

      ame_data -> interpreter = hypre_CTAlloc(mv_InterfaceInterpreter, 1, HYPRE_MEMORY_HOST);
      interpreter = (mv_InterfaceInterpreter*) ame_data -> interpreter;
      HYPRE_ParCSRSetupInterpreter(interpreter);

      ame_data -> eigenvalues = hypre_CTAlloc(HYPRE_Real,  ame_data -> block_size, HYPRE_MEMORY_HOST);

      ame_data -> eigenvectors =
         mv_MultiVectorCreateFromSampleVector(interpreter,
                                              ame_data -> block_size,
                                              ame_data -> t3);
      eigenvectors = (mv_MultiVectorPtr) ame_data -> eigenvectors;

      mv_MultiVectorSetRandom (eigenvectors, seed);

      /* Make the initial vectors discretely divergence free */
      {
         HYPRE_Int i, j;
         HYPRE_Real *data;

         mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
         HYPRE_ParVector *v = (HYPRE_ParVector*)(tmp -> vector);
         hypre_ParVector *vi;

         for (i = 0; i < ame_data -> block_size; i++)
         {
            vi = (hypre_ParVector*) v[i];
            data = hypre_VectorData(hypre_ParVectorLocalVector(vi));
#if defined(HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_DEVICE)
            {
#if defined(HYPRE_USING_SYCL)
               hypreSycl_transform_if( data,
                                       data + ne,
                                       edge_bc,
                                       data,
               [] (const auto & x) {return 0.0;},
               [] (const auto & x) {return x;} );
#else
               HYPRE_THRUST_CALL( replace_if,
                                  data,
                                  data + ne,
                                  edge_bc,
                                  thrust::identity<HYPRE_Int>(),
                                  0.0 );
#endif
            }
            else
#endif
            {
               for (j = 0; j < ne; j++)
                  if (edge_bc[j])
                  {
                     data[j] = 0.0;
                  }
            }
            hypre_AMEDiscrDivFreeComponent(esolver, vi);
         }
      }
   }

   hypre_TFree(edge_bc, memory_location);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSDiscrDivFreeComponent
 *
 * Remove the component of b in the range of G, i.e., compute
 *              b = (I - G (G^t M G)^{-1} G^t M) b
 * This way b will be orthogonal to gradients of linear functions.
 * The problem with G^t M G is solved only approximately by PCG-AMG.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMEDiscrDivFreeComponent(void *esolver, hypre_ParVector *b)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;

   /* t3 = M b */
   hypre_ParCSRMatrixMatvec(1.0, ame_data -> M, b, 0.0, ame_data -> t3);

   /* t1 = G^t t3 */
   hypre_ParCSRMatrixMatvecT(1.0, ame_data -> G, ame_data -> t3, 0.0, ame_data -> t1);

   /* (G^t M G) t2 = t1 */
   hypre_ParVectorSetConstantValues(ame_data -> t2, 0.0);
   HYPRE_ParCSRPCGSolve(ame_data -> B2_G,
                        (HYPRE_ParCSRMatrix)ame_data -> A_G,
                        (HYPRE_ParVector)ame_data -> t1,
                        (HYPRE_ParVector)ame_data -> t2);

   /* b = b - G t2 */
   hypre_ParCSRMatrixMatvec(-1.0, ame_data -> G, ame_data -> t2, 1.0, b);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMEOperatorA and hypre_AMEMultiOperatorA
 *
 * The stiffness matrix considered as an operator on (multi)vectors.
 *--------------------------------------------------------------------------*/

void hypre_AMEOperatorA(void *data, void* x, void* y)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) data;
   hypre_AMSData *ams_data = ame_data -> precond;
   hypre_ParCSRMatrixMatvec(1.0, ams_data -> A, (hypre_ParVector*)x,
                            0.0, (hypre_ParVector*)y);
}

void hypre_AMEMultiOperatorA(void *data, void* x, void* y)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) data;
   mv_InterfaceInterpreter*
   interpreter = (mv_InterfaceInterpreter*) ame_data -> interpreter;
   interpreter -> Eval(hypre_AMEOperatorA, data, x, y);
}

/*--------------------------------------------------------------------------
 * hypre_AMEOperatorM and hypre_AMEMultiOperatorM
 *
 * The mass matrix considered as an operator on (multi)vectors.
 *--------------------------------------------------------------------------*/

void hypre_AMEOperatorM(void *data, void* x, void* y)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) data;
   hypre_ParCSRMatrixMatvec(1.0, ame_data -> M, (hypre_ParVector*)x,
                            0.0, (hypre_ParVector*)y);
}

void hypre_AMEMultiOperatorM(void *data, void* x, void* y)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) data;
   mv_InterfaceInterpreter*
   interpreter = (mv_InterfaceInterpreter*) ame_data -> interpreter;
   interpreter -> Eval(hypre_AMEOperatorM, data, x, y);
}

/*--------------------------------------------------------------------------
 * hypre_AMEOperatorB and hypre_AMEMultiOperatorB
 *
 * The AMS method considered as an operator on (multi)vectors.
 * Make sure that the result is discr. div. free.
 *--------------------------------------------------------------------------*/

void hypre_AMEOperatorB(void *data, void* x, void* y)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) data;
   hypre_AMSData *ams_data = ame_data -> precond;

   hypre_ParVectorSetConstantValues((hypre_ParVector*)y, 0.0);
   hypre_AMSSolve(ame_data -> precond, ams_data -> A, (hypre_ParVector*) x, (hypre_ParVector*) y);

   hypre_AMEDiscrDivFreeComponent(data, (hypre_ParVector *)y);
}

void hypre_AMEMultiOperatorB(void *data, void* x, void* y)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) data;
   mv_InterfaceInterpreter*
   interpreter = (mv_InterfaceInterpreter*) ame_data -> interpreter;
   interpreter -> Eval(hypre_AMEOperatorB, data, x, y);
}

/*--------------------------------------------------------------------------
 * hypre_AMESolve
 *
 * Solve the eigensystem A u = lambda M u, G^t u = 0 using a subspace
 * version of LOBPCG (i.e. we iterate in the discr. div. free space).
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMESolve(void *esolver)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;

   HYPRE_Int nit;
   lobpcg_BLASLAPACKFunctions blap_fn;
   lobpcg_Tolerance lobpcg_tol;
   HYPRE_Real *residuals;

   blap_fn.dsygv  = hypre_dsygv;
   blap_fn.dpotrf = hypre_dpotrf;
   lobpcg_tol.relative = ame_data -> rtol;
   lobpcg_tol.absolute = ame_data -> atol;
   residuals = hypre_TAlloc(HYPRE_Real,  ame_data -> block_size, HYPRE_MEMORY_HOST);

   lobpcg_solve((mv_MultiVectorPtr) ame_data -> eigenvectors,
                esolver, hypre_AMEMultiOperatorA,
                esolver, hypre_AMEMultiOperatorM,
                esolver, hypre_AMEMultiOperatorB,
                NULL, blap_fn, lobpcg_tol, ame_data -> maxit,
                ame_data -> print_level, &nit,
                ame_data -> eigenvalues,
                NULL, ame_data -> block_size,
                residuals,
                NULL, ame_data -> block_size);

   hypre_TFree(residuals, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMEGetEigenvectors
 *
 * Return a pointer to the computed eigenvectors.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMEGetEigenvectors(void *esolver,
                                   HYPRE_ParVector **eigenvectors_ptr)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   mv_MultiVectorPtr
   eigenvectors = (mv_MultiVectorPtr) ame_data -> eigenvectors;
   mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);

   *eigenvectors_ptr = (HYPRE_ParVector*)(tmp -> vector);
   tmp -> vector = NULL;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMEGetEigenvalues
 *
 * Return a pointer to the computed eigenvalues.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMEGetEigenvalues(void *esolver,
                                  HYPRE_Real **eigenvalues_ptr)
{
   hypre_AMEData *ame_data = (hypre_AMEData *) esolver;
   *eigenvalues_ptr = ame_data -> eigenvalues;
   ame_data -> eigenvalues = NULL;
   return hypre_error_flag;
}
