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
#include "ads.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)
#if defined(HYPRE_USING_SYCL)
SYCL_EXTERNAL
#endif
__global__ void hypreGPUKernel_AMSComputePi_copy1(hypre_DeviceItem &item, HYPRE_Int nnz,
                                                  HYPRE_Int dim,
                                                  HYPRE_Int *j_in,
                                                  HYPRE_Int *j_out);
#if defined(HYPRE_USING_SYCL)
SYCL_EXTERNAL
#endif
__global__ void hypreGPUKernel_AMSComputePi_copy2(hypre_DeviceItem &item, HYPRE_Int nrows,
                                                  HYPRE_Int dim,
                                                  HYPRE_Int *i_in,
                                                  HYPRE_Real *data_in, HYPRE_Real *Gx_data, HYPRE_Real *Gy_data, HYPRE_Real *Gz_data,
                                                  HYPRE_Real *data_out);
#if defined(HYPRE_USING_SYCL)
SYCL_EXTERNAL
#endif
__global__ void hypreGPUKernel_AMSComputePixyz_copy(hypre_DeviceItem &item, HYPRE_Int nrows,
                                                    HYPRE_Int dim,
                                                    HYPRE_Int *i_in, HYPRE_Real *data_in, HYPRE_Real *Gx_data, HYPRE_Real *Gy_data, HYPRE_Real *Gz_data,
                                                    HYPRE_Real *data_x_out, HYPRE_Real *data_y_out, HYPRE_Real *data_z_out );
#endif

/*--------------------------------------------------------------------------
 * hypre_ADSCreate
 *
 * Allocate the ADS solver structure.
 *--------------------------------------------------------------------------*/

void * hypre_ADSCreate(void)
{
   hypre_ADSData *ads_data;

   ads_data = hypre_CTAlloc(hypre_ADSData, 1, HYPRE_MEMORY_HOST);

   /* Default parameters */

   ads_data -> maxit = 20;             /* perform at most 20 iterations */
   ads_data -> tol = 1e-6;             /* convergence tolerance */
   ads_data -> print_level = 1;        /* print residual norm at each step */
   ads_data -> cycle_type = 1;         /* a 3-level multiplicative solver */
   ads_data -> A_relax_type = 2;       /* offd-l1-scaled GS */
   ads_data -> A_relax_times = 1;      /* one relaxation sweep */
   ads_data -> A_relax_weight = 1.0;   /* damping parameter */
   ads_data -> A_omega = 1.0;          /* SSOR coefficient */
   ads_data -> A_cheby_order = 2;      /* Cheby: order (1 -4 are vaild) */
   ads_data -> A_cheby_fraction = 0.3; /* Cheby: fraction of spectrum to smooth */

   ads_data -> B_C_cycle_type = 11;    /* a 5-level multiplicative solver */
   ads_data -> B_C_coarsen_type = 10;  /* HMIS coarsening */
   ads_data -> B_C_agg_levels = 1;     /* Levels of aggressive coarsening */
   ads_data -> B_C_relax_type = 3;     /* hybrid G-S/Jacobi */
   ads_data -> B_C_theta = 0.25;       /* strength threshold */
   ads_data -> B_C_interp_type = 0;    /* interpolation type */
   ads_data -> B_C_Pmax = 0;           /* max nonzero elements in interp. rows */
   ads_data -> B_Pi_coarsen_type = 10; /* HMIS coarsening */
   ads_data -> B_Pi_agg_levels = 1;    /* Levels of aggressive coarsening */
   ads_data -> B_Pi_relax_type = 3;    /* hybrid G-S/Jacobi */
   ads_data -> B_Pi_theta = 0.25;      /* strength threshold */
   ads_data -> B_Pi_interp_type = 0;   /* interpolation type */
   ads_data -> B_Pi_Pmax = 0;          /* max nonzero elements in interp. rows */

   /* The rest of the fields are initialized using the Set functions */

   ads_data -> A     = NULL;
   ads_data -> C     = NULL;
   ads_data -> A_C   = NULL;
   ads_data -> B_C   = 0;
   ads_data -> Pi    = NULL;
   ads_data -> A_Pi  = NULL;
   ads_data -> B_Pi  = 0;
   ads_data -> Pix    = NULL;
   ads_data -> Piy    = NULL;
   ads_data -> Piz    = NULL;
   ads_data -> A_Pix  = NULL;
   ads_data -> A_Piy  = NULL;
   ads_data -> A_Piz  = NULL;
   ads_data -> B_Pix  = 0;
   ads_data -> B_Piy  = 0;
   ads_data -> B_Piz  = 0;
   ads_data -> G     = NULL;
   ads_data -> x     = NULL;
   ads_data -> y     = NULL;
   ads_data -> z     = NULL;
   ads_data -> zz  = NULL;

   ads_data -> r0  = NULL;
   ads_data -> g0  = NULL;
   ads_data -> r1  = NULL;
   ads_data -> g1  = NULL;
   ads_data -> r2  = NULL;
   ads_data -> g2  = NULL;

   ads_data -> A_l1_norms = NULL;
   ads_data -> A_max_eig_est = 0;
   ads_data -> A_min_eig_est = 0;

   ads_data -> owns_Pi = 1;
   ads_data -> ND_Pi   = NULL;
   ads_data -> ND_Pix  = NULL;
   ads_data -> ND_Piy  = NULL;
   ads_data -> ND_Piz  = NULL;

   return (void *) ads_data;
}

/*--------------------------------------------------------------------------
 * hypre_ADSDestroy
 *
 * Deallocate the ADS solver structure. Note that the input data (given
 * through the Set functions) is not destroyed.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSDestroy(void *solver)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;

   if (!ads_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (ads_data -> A_C)
   {
      hypre_ParCSRMatrixDestroy(ads_data -> A_C);
   }
   if (ads_data -> B_C)
   {
      HYPRE_AMSDestroy(ads_data -> B_C);
   }

   if (ads_data -> owns_Pi && ads_data -> Pi)
   {
      hypre_ParCSRMatrixDestroy(ads_data -> Pi);
   }
   if (ads_data -> A_Pi)
   {
      hypre_ParCSRMatrixDestroy(ads_data -> A_Pi);
   }
   if (ads_data -> B_Pi)
   {
      HYPRE_BoomerAMGDestroy(ads_data -> B_Pi);
   }

   if (ads_data -> owns_Pi && ads_data -> Pix)
   {
      hypre_ParCSRMatrixDestroy(ads_data -> Pix);
   }
   if (ads_data -> A_Pix)
   {
      hypre_ParCSRMatrixDestroy(ads_data -> A_Pix);
   }
   if (ads_data -> B_Pix)
   {
      HYPRE_BoomerAMGDestroy(ads_data -> B_Pix);
   }
   if (ads_data -> owns_Pi && ads_data -> Piy)
   {
      hypre_ParCSRMatrixDestroy(ads_data -> Piy);
   }
   if (ads_data -> A_Piy)
   {
      hypre_ParCSRMatrixDestroy(ads_data -> A_Piy);
   }
   if (ads_data -> B_Piy)
   {
      HYPRE_BoomerAMGDestroy(ads_data -> B_Piy);
   }
   if (ads_data -> owns_Pi && ads_data -> Piz)
   {
      hypre_ParCSRMatrixDestroy(ads_data -> Piz);
   }
   if (ads_data -> A_Piz)
   {
      hypre_ParCSRMatrixDestroy(ads_data -> A_Piz);
   }
   if (ads_data -> B_Piz)
   {
      HYPRE_BoomerAMGDestroy(ads_data -> B_Piz);
   }

   if (ads_data -> r0)
   {
      hypre_ParVectorDestroy(ads_data -> r0);
   }
   if (ads_data -> g0)
   {
      hypre_ParVectorDestroy(ads_data -> g0);
   }
   if (ads_data -> r1)
   {
      hypre_ParVectorDestroy(ads_data -> r1);
   }
   if (ads_data -> g1)
   {
      hypre_ParVectorDestroy(ads_data -> g1);
   }
   if (ads_data -> r2)
   {
      hypre_ParVectorDestroy(ads_data -> r2);
   }
   if (ads_data -> g2)
   {
      hypre_ParVectorDestroy(ads_data -> g2);
   }
   if (ads_data -> zz)
   {
      hypre_ParVectorDestroy(ads_data -> zz);
   }

   hypre_SeqVectorDestroy(ads_data -> A_l1_norms);

   /* C, G, x, y and z are not destroyed */

   if (ads_data)
   {
      hypre_TFree(ads_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetDiscreteCurl
 *
 * Set the discrete curl matrix C.
 * This function should be called before hypre_ADSSetup()!
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetDiscreteCurl(void *solver,
                                   hypre_ParCSRMatrix *C)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> C = C;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetDiscreteGradient
 *
 * Set the discrete gradient matrix G.
 * This function should be called before hypre_ADSSetup()!
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetDiscreteGradient(void *solver,
                                       hypre_ParCSRMatrix *G)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> G = G;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetCoordinateVectors
 *
 * Set the x, y and z coordinates of the vertices in the mesh.
 * This function should be called before hypre_ADSSetup()!
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetCoordinateVectors(void *solver,
                                        hypre_ParVector *x,
                                        hypre_ParVector *y,
                                        hypre_ParVector *z)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> x = x;
   ads_data -> y = y;
   ads_data -> z = z;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetInterpolations
 *
 * Set the (components of) the Raviart-Thomas (RT_Pi) and the Nedelec (ND_Pi)
 * interpolation matrices.
 *
 * This function is generally intended to be used only for high-order H(div)
 * discretizations (in the lowest order case, these matrices are constructed
 * internally in ADS from the discreet gradient and curl matrices and the
 * coordinates of the vertices), though it can also be used in the lowest-order
 * case or for other types of discretizations.
 *
 * By definition, RT_Pi and ND_Pi are the matrix representations of the linear
 * operators that interpolate (high-order) vector nodal finite elements into the
 * (high-order) Raviart-Thomas and Nedelec spaces. The component matrices are
 * defined in both cases as Pix phi = Pi (phi,0,0) and similarly for Piy and
 * Piz. Note that all these operators depend on the choice of the basis and
 * degrees of freedom in the high-order spaces.
 *
 * The column numbering of RT_Pi and ND_Pi should be node-based, i.e. the x/y/z
 * components of the first node (vertex or high-order dof) should be listed
 * first, followed by the x/y/z components of the second node and so on (see the
 * documentation of HYPRE_BoomerAMGSetDofFunc).
 *
 * If used, this function should be called before hypre_ADSSetup() and there is
 * no need to provide the vertex coordinates. Furthermore, only one of the sets
 * {RT_Pi} and {RT_Pix,RT_Piy,RT_Piz} needs to be specified (though it is OK to
 * provide both).  If RT_Pix is NULL, then scalar Pi-based ADS cycles, i.e.
 * those with cycle_type > 10, will be unavailable. Similarly, ADS cycles based
 * on monolithic Pi (cycle_type < 10) require that RT_Pi is not NULL. The same
 * restrictions hold for the sets {ND_Pi} and {ND_Pix,ND_Piy,ND_Piz} -- only one
 * of them needs to be specified, and the availability of each enables different
 * AMS cycle type options for the subspace solve.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetInterpolations(void *solver,
                                     hypre_ParCSRMatrix *RT_Pi,
                                     hypre_ParCSRMatrix *RT_Pix,
                                     hypre_ParCSRMatrix *RT_Piy,
                                     hypre_ParCSRMatrix *RT_Piz,
                                     hypre_ParCSRMatrix *ND_Pi,
                                     hypre_ParCSRMatrix *ND_Pix,
                                     hypre_ParCSRMatrix *ND_Piy,
                                     hypre_ParCSRMatrix *ND_Piz)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> Pi = RT_Pi;
   ads_data -> Pix = RT_Pix;
   ads_data -> Piy = RT_Piy;
   ads_data -> Piz = RT_Piz;
   ads_data -> ND_Pi = ND_Pi;
   ads_data -> ND_Pix = ND_Pix;
   ads_data -> ND_Piy = ND_Piy;
   ads_data -> ND_Piz = ND_Piz;
   ads_data -> owns_Pi = 0;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetMaxIter
 *
 * Set the maximum number of iterations in the auxiliary-space method.
 * The default value is 20. To use the ADS solver as a preconditioner,
 * set maxit to 1, tol to 0.0 and print_level to 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetMaxIter(void *solver,
                              HYPRE_Int maxit)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> maxit = maxit;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetTol
 *
 * Set the convergence tolerance (if the method is used as a solver).
 * The default value is 1e-6.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetTol(void *solver,
                          HYPRE_Real tol)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> tol = tol;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetCycleType
 *
 * Choose which three-level solver to use. Possible values are:
 *
 *   1 = 3-level multipl. solver (01210)      <-- small solution time
 *   2 = 3-level additive solver (0+1+2)
 *   3 = 3-level multipl. solver (02120)
 *   4 = 3-level additive solver (010+2)
 *   5 = 3-level multipl. solver (0102010)    <-- small solution time
 *   6 = 3-level additive solver (1+020)
 *   7 = 3-level multipl. solver (0201020)    <-- small number of iterations
 *   8 = 3-level additive solver (0(1+2)0)    <-- small solution time
 *   9 = 3-level multipl. solver (01210) with discrete divergence
 *  11 = 5-level multipl. solver (013454310)  <-- small solution time, memory
 *  12 = 5-level additive solver (0+1+3+4+5)
 *  13 = 5-level multipl. solver (034515430)  <-- small solution time, memory
 *  14 = 5-level additive solver (01(3+4+5)10)
 *
 * The default value is 1.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetCycleType(void *solver,
                                HYPRE_Int cycle_type)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> cycle_type = cycle_type;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetPrintLevel
 *
 * Control how much information is printed during the solution iterations.
 * The defaut values is 1 (print residual norm at each step).
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetPrintLevel(void *solver,
                                 HYPRE_Int print_level)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> print_level = print_level;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetSmoothingOptions
 *
 * Set relaxation parameters for A. Default values: 2, 1, 1.0, 1.0.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetSmoothingOptions(void *solver,
                                       HYPRE_Int A_relax_type,
                                       HYPRE_Int A_relax_times,
                                       HYPRE_Real A_relax_weight,
                                       HYPRE_Real A_omega)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> A_relax_type = A_relax_type;
   ads_data -> A_relax_times = A_relax_times;
   ads_data -> A_relax_weight = A_relax_weight;
   ads_data -> A_omega = A_omega;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetChebySmoothingOptions
 *
 * Set parameters for Chebyshev relaxation. Default values: 2, 0.3.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetChebySmoothingOptions(void *solver,
                                            HYPRE_Int A_cheby_order,
                                            HYPRE_Real A_cheby_fraction)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> A_cheby_order =  A_cheby_order;
   ads_data -> A_cheby_fraction =  A_cheby_fraction;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetAMSOptions
 *
 * Set AMS parameters for B_C. Default values: 11, 10, 1, 3, 0.25, 0, 0.
 *
 * Note that B_C_cycle_type should be greater than 10, unless the high-order
 * interface of hypre_ADSSetInterpolations is being used!
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetAMSOptions(void *solver,
                                 HYPRE_Int B_C_cycle_type,
                                 HYPRE_Int B_C_coarsen_type,
                                 HYPRE_Int B_C_agg_levels,
                                 HYPRE_Int B_C_relax_type,
                                 HYPRE_Real B_C_theta,
                                 HYPRE_Int B_C_interp_type,
                                 HYPRE_Int B_C_Pmax)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> B_C_cycle_type = B_C_cycle_type;
   ads_data -> B_C_coarsen_type = B_C_coarsen_type;
   ads_data -> B_C_agg_levels = B_C_agg_levels;
   ads_data -> B_C_relax_type = B_C_relax_type;
   ads_data -> B_C_theta = B_C_theta;
   ads_data -> B_C_interp_type = B_C_interp_type;
   ads_data -> B_C_Pmax = B_C_Pmax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetAMGOptions
 *
 * Set AMG parameters for B_Pi. Default values: 10, 1, 3, 0.25, 0, 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSetAMGOptions(void *solver,
                                 HYPRE_Int B_Pi_coarsen_type,
                                 HYPRE_Int B_Pi_agg_levels,
                                 HYPRE_Int B_Pi_relax_type,
                                 HYPRE_Real B_Pi_theta,
                                 HYPRE_Int B_Pi_interp_type,
                                 HYPRE_Int B_Pi_Pmax)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   ads_data -> B_Pi_coarsen_type = B_Pi_coarsen_type;
   ads_data -> B_Pi_agg_levels = B_Pi_agg_levels;
   ads_data -> B_Pi_relax_type = B_Pi_relax_type;
   ads_data -> B_Pi_theta = B_Pi_theta;
   ads_data -> B_Pi_interp_type = B_Pi_interp_type;
   ads_data -> B_Pi_Pmax = B_Pi_Pmax;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSComputePi
 *
 * Construct the Pi interpolation matrix, which maps the space of vector
 * linear finite elements to the space of face finite elements.
 *
 * The construction is based on the fact that Pi = [Pi_x, Pi_y, Pi_z], where
 * each block has the same sparsity structure as C*G, with entries that can be
 * computed from the vectors RT100, RT010 and RT001.
 *
 * We assume a constant number of vertices per face (no prisms or pyramids).
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSComputePi(hypre_ParCSRMatrix *A,
                             hypre_ParCSRMatrix *C,
                             hypre_ParCSRMatrix *G,
                             hypre_ParVector *x,
                             hypre_ParVector *y,
                             hypre_ParVector *z,
                             hypre_ParCSRMatrix *PiNDx,
                             hypre_ParCSRMatrix *PiNDy,
                             hypre_ParCSRMatrix *PiNDz,
                             hypre_ParCSRMatrix **Pi_ptr)
{
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
#else
   HYPRE_UNUSED_VAR(A);
#endif

   hypre_ParCSRMatrix *Pi;

   /* Compute the representations of the coordinate vectors, RT100, RT010 and
      RT001, in the Raviart-Thomas space, by observing that the RT coordinates
      of (1,0,0) = -curl (0,z,0) are given by C*PiNDy*z, etc. (We ignore the
      minus sign since it is irrelevant for the coarse-grid correction.) */
   hypre_ParVector *RT100, *RT010, *RT001;
   {
      hypre_ParVector *PiNDlin = hypre_ParVectorInRangeOf(PiNDx);

      RT100 = hypre_ParVectorInRangeOf(C);
      hypre_ParCSRMatrixMatvec(1.0, PiNDy, z, 0.0, PiNDlin);
      hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT100);
      RT010 = hypre_ParVectorInRangeOf(C);
      hypre_ParCSRMatrixMatvec(1.0, PiNDz, x, 0.0, PiNDlin);
      hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT010);
      RT001 = hypre_ParVectorInRangeOf(C);
      hypre_ParCSRMatrixMatvec(1.0, PiNDx, y, 0.0, PiNDlin);
      hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT001);

      hypre_ParVectorDestroy(PiNDlin);
   }

   /* Compute Pi = [Pi_x, Pi_y, Pi_z] */
   {
      HYPRE_Int i, j, d;

      HYPRE_Real *RT100_data = hypre_VectorData(hypre_ParVectorLocalVector(RT100));
      HYPRE_Real *RT010_data = hypre_VectorData(hypre_ParVectorLocalVector(RT010));
      HYPRE_Real *RT001_data = hypre_VectorData(hypre_ParVectorLocalVector(RT001));

      /* Each component of Pi has the sparsity pattern of the topological
         face-to-vertex matrix. */
      hypre_ParCSRMatrix *F2V;
#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         F2V = hypre_ParCSRMatMat(C, G);
      }
      else
#endif
      {
         F2V = hypre_ParMatmul(C, G);
      }

      /* Create the parallel interpolation matrix */
      {
         MPI_Comm comm = hypre_ParCSRMatrixComm(F2V);
         HYPRE_BigInt global_num_rows = hypre_ParCSRMatrixGlobalNumRows(F2V);
         HYPRE_BigInt global_num_cols = 3 * hypre_ParCSRMatrixGlobalNumCols(F2V);
         HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(F2V);
         HYPRE_BigInt *col_starts;
         HYPRE_Int col_starts_size;
         HYPRE_Int num_cols_offd = 3 * hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(F2V));
         HYPRE_Int num_nonzeros_diag = 3 * hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(F2V));
         HYPRE_Int num_nonzeros_offd = 3 * hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(F2V));
         HYPRE_BigInt *col_starts_F2V = hypre_ParCSRMatrixColStarts(F2V);
         col_starts_size = 2;
         col_starts = hypre_TAlloc(HYPRE_BigInt, col_starts_size, HYPRE_MEMORY_HOST);
         for (i = 0; i < col_starts_size; i++)
         {
            col_starts[i] = 3 * col_starts_F2V[i];
         }

         Pi = hypre_ParCSRMatrixCreate(comm,
                                       global_num_rows,
                                       global_num_cols,
                                       row_starts,
                                       col_starts,
                                       num_cols_offd,
                                       num_nonzeros_diag,
                                       num_nonzeros_offd);

         hypre_ParCSRMatrixOwnsData(Pi) = 1;
         hypre_ParCSRMatrixInitialize(Pi);
      }

      /* Fill-in the diagonal part */
      {
         hypre_CSRMatrix *F2V_diag = hypre_ParCSRMatrixDiag(F2V);
         HYPRE_Int *F2V_diag_I = hypre_CSRMatrixI(F2V_diag);
         HYPRE_Int *F2V_diag_J = hypre_CSRMatrixJ(F2V_diag);

         HYPRE_Int F2V_diag_nrows = hypre_CSRMatrixNumRows(F2V_diag);
         HYPRE_Int F2V_diag_nnz = hypre_CSRMatrixNumNonzeros(F2V_diag);

         hypre_CSRMatrix *Pi_diag = hypre_ParCSRMatrixDiag(Pi);
         HYPRE_Int *Pi_diag_I = hypre_CSRMatrixI(Pi_diag);
         HYPRE_Int *Pi_diag_J = hypre_CSRMatrixJ(Pi_diag);
         HYPRE_Real *Pi_diag_data = hypre_CSRMatrixData(Pi_diag);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypreDevice_IntScalen( F2V_diag_I, F2V_diag_nrows + 1, Pi_diag_I, 3);

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(F2V_diag_nnz, "thread", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              F2V_diag_nnz, 3, F2V_diag_J, Pi_diag_J );

            gDim = hypre_GetDefaultDeviceGridDimension(F2V_diag_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy2, gDim, bDim,
                              F2V_diag_nrows, 3, F2V_diag_I, NULL, RT100_data, RT010_data, RT001_data,
                              Pi_diag_data );
         }
         else
#endif
         {
            for (i = 0; i < F2V_diag_nrows + 1; i++)
            {
               Pi_diag_I[i] = 3 * F2V_diag_I[i];
            }

            for (i = 0; i < F2V_diag_nnz; i++)
               for (d = 0; d < 3; d++)
               {
                  Pi_diag_J[3 * i + d] = 3 * F2V_diag_J[i] + d;
               }

            for (i = 0; i < F2V_diag_nrows; i++)
               for (j = F2V_diag_I[i]; j < F2V_diag_I[i + 1]; j++)
               {
                  *Pi_diag_data++ = RT100_data[i];
                  *Pi_diag_data++ = RT010_data[i];
                  *Pi_diag_data++ = RT001_data[i];
               }
         }
      }

      /* Fill-in the off-diagonal part */
      {
         hypre_CSRMatrix *F2V_offd = hypre_ParCSRMatrixOffd(F2V);
         HYPRE_Int *F2V_offd_I = hypre_CSRMatrixI(F2V_offd);
         HYPRE_Int *F2V_offd_J = hypre_CSRMatrixJ(F2V_offd);

         HYPRE_Int F2V_offd_nrows = hypre_CSRMatrixNumRows(F2V_offd);
         HYPRE_Int F2V_offd_ncols = hypre_CSRMatrixNumCols(F2V_offd);
         HYPRE_Int F2V_offd_nnz = hypre_CSRMatrixNumNonzeros(F2V_offd);

         hypre_CSRMatrix *Pi_offd = hypre_ParCSRMatrixOffd(Pi);
         HYPRE_Int *Pi_offd_I = hypre_CSRMatrixI(Pi_offd);
         HYPRE_Int *Pi_offd_J = hypre_CSRMatrixJ(Pi_offd);
         HYPRE_Real *Pi_offd_data = hypre_CSRMatrixData(Pi_offd);

         HYPRE_BigInt *F2V_cmap = hypre_ParCSRMatrixColMapOffd(F2V);
         HYPRE_BigInt *Pi_cmap = hypre_ParCSRMatrixColMapOffd(Pi);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            if (F2V_offd_ncols)
            {
               hypreDevice_IntScalen( F2V_offd_I, F2V_offd_nrows + 1, Pi_offd_I, 3 );
            }

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(F2V_offd_nnz, "thread", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              F2V_offd_nnz, 3, F2V_offd_J, Pi_offd_J );

            gDim = hypre_GetDefaultDeviceGridDimension(F2V_offd_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy2, gDim, bDim,
                              F2V_offd_nrows, 3, F2V_offd_I, NULL, RT100_data, RT010_data, RT001_data,
                              Pi_offd_data );
         }
         else
#endif
         {
            if (F2V_offd_ncols)
               for (i = 0; i < F2V_offd_nrows + 1; i++)
               {
                  Pi_offd_I[i] = 3 * F2V_offd_I[i];
               }

            for (i = 0; i < F2V_offd_nnz; i++)
               for (d = 0; d < 3; d++)
               {
                  Pi_offd_J[3 * i + d] = 3 * F2V_offd_J[i] + d;
               }

            for (i = 0; i < F2V_offd_nrows; i++)
               for (j = F2V_offd_I[i]; j < F2V_offd_I[i + 1]; j++)
               {
                  *Pi_offd_data++ = RT100_data[i];
                  *Pi_offd_data++ = RT010_data[i];
                  *Pi_offd_data++ = RT001_data[i];
               }
         }

         for (i = 0; i < F2V_offd_ncols; i++)
            for (d = 0; d < 3; d++)
            {
               Pi_cmap[3 * i + d] = 3 * F2V_cmap[i] + (HYPRE_BigInt)d;
            }
      }

      hypre_ParCSRMatrixDestroy(F2V);
   }

   hypre_ParVectorDestroy(RT100);
   hypre_ParVectorDestroy(RT010);
   hypre_ParVectorDestroy(RT001);

   *Pi_ptr = Pi;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSComputePixyz
 *
 * Construct the components Pix, Piy, Piz of the interpolation matrix Pi, which
 * maps the space of vector linear finite elements to the space of face finite
 * elements.
 *
 * The construction is based on the fact that each component has the same
 * sparsity structure as the matrix C*G, with entries that can be computed from
 * the vectors RT100, RT010 and RT001.
 *
 * We assume a constant number of vertices per face (no prisms or pyramids).
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSComputePixyz(hypre_ParCSRMatrix *A,
                                hypre_ParCSRMatrix *C,
                                hypre_ParCSRMatrix *G,
                                hypre_ParVector *x,
                                hypre_ParVector *y,
                                hypre_ParVector *z,
                                hypre_ParCSRMatrix *PiNDx,
                                hypre_ParCSRMatrix *PiNDy,
                                hypre_ParCSRMatrix *PiNDz,
                                hypre_ParCSRMatrix **Pix_ptr,
                                hypre_ParCSRMatrix **Piy_ptr,
                                hypre_ParCSRMatrix **Piz_ptr)
{
   hypre_ParCSRMatrix *Pix, *Piy, *Piz;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
#else
   HYPRE_UNUSED_VAR(A);
#endif

   /* Compute the representations of the coordinate vectors, RT100, RT010 and
      RT001, in the Raviart-Thomas space, by observing that the RT coordinates
      of (1,0,0) = -curl (0,z,0) are given by C*PiNDy*z, etc. (We ignore the
      minus sign since it is irrelevant for the coarse-grid correction.) */
   hypre_ParVector *RT100, *RT010, *RT001;
   {
      hypre_ParVector *PiNDlin = hypre_ParVectorInRangeOf(PiNDx);

      RT100 = hypre_ParVectorInRangeOf(C);
      hypre_ParCSRMatrixMatvec(1.0, PiNDy, z, 0.0, PiNDlin);
      hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT100);
      RT010 = hypre_ParVectorInRangeOf(C);
      hypre_ParCSRMatrixMatvec(1.0, PiNDz, x, 0.0, PiNDlin);
      hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT010);
      RT001 = hypre_ParVectorInRangeOf(C);
      hypre_ParCSRMatrixMatvec(1.0, PiNDx, y, 0.0, PiNDlin);
      hypre_ParCSRMatrixMatvec(1.0, C, PiNDlin, 0.0, RT001);

      hypre_ParVectorDestroy(PiNDlin);
   }

   /* Compute Pix, Piy, Piz */
   {
      HYPRE_Int i, j;

      HYPRE_Real *RT100_data = hypre_VectorData(hypre_ParVectorLocalVector(RT100));
      HYPRE_Real *RT010_data = hypre_VectorData(hypre_ParVectorLocalVector(RT010));
      HYPRE_Real *RT001_data = hypre_VectorData(hypre_ParVectorLocalVector(RT001));

      /* Each component of Pi has the sparsity pattern of the topological
         face-to-vertex matrix. */
      hypre_ParCSRMatrix *F2V;

#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         F2V = hypre_ParCSRMatMat(C, G);
      }
      else
#endif
      {
         F2V = hypre_ParMatmul(C, G);
      }

      /* Create the components of the parallel interpolation matrix */
      {
         MPI_Comm comm = hypre_ParCSRMatrixComm(F2V);
         HYPRE_BigInt global_num_rows = hypre_ParCSRMatrixGlobalNumRows(F2V);
         HYPRE_BigInt global_num_cols = hypre_ParCSRMatrixGlobalNumCols(F2V);
         HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(F2V);
         HYPRE_BigInt *col_starts = hypre_ParCSRMatrixColStarts(F2V);
         HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(F2V));
         HYPRE_Int num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(F2V));
         HYPRE_Int num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(F2V));

         Pix = hypre_ParCSRMatrixCreate(comm,
                                        global_num_rows,
                                        global_num_cols,
                                        row_starts,
                                        col_starts,
                                        num_cols_offd,
                                        num_nonzeros_diag,
                                        num_nonzeros_offd);
         hypre_ParCSRMatrixOwnsData(Pix) = 1;
         hypre_ParCSRMatrixInitialize(Pix);

         Piy = hypre_ParCSRMatrixCreate(comm,
                                        global_num_rows,
                                        global_num_cols,
                                        row_starts,
                                        col_starts,
                                        num_cols_offd,
                                        num_nonzeros_diag,
                                        num_nonzeros_offd);
         hypre_ParCSRMatrixOwnsData(Piy) = 1;
         hypre_ParCSRMatrixInitialize(Piy);

         Piz = hypre_ParCSRMatrixCreate(comm,
                                        global_num_rows,
                                        global_num_cols,
                                        row_starts,
                                        col_starts,
                                        num_cols_offd,
                                        num_nonzeros_diag,
                                        num_nonzeros_offd);
         hypre_ParCSRMatrixOwnsData(Piz) = 1;
         hypre_ParCSRMatrixInitialize(Piz);
      }

      /* Fill-in the diagonal part */
      {
         hypre_CSRMatrix *F2V_diag = hypre_ParCSRMatrixDiag(F2V);
         HYPRE_Int *F2V_diag_I = hypre_CSRMatrixI(F2V_diag);
         HYPRE_Int *F2V_diag_J = hypre_CSRMatrixJ(F2V_diag);

         HYPRE_Int F2V_diag_nrows = hypre_CSRMatrixNumRows(F2V_diag);
         HYPRE_Int F2V_diag_nnz = hypre_CSRMatrixNumNonzeros(F2V_diag);

         hypre_CSRMatrix *Pix_diag = hypre_ParCSRMatrixDiag(Pix);
         HYPRE_Int *Pix_diag_I = hypre_CSRMatrixI(Pix_diag);
         HYPRE_Int *Pix_diag_J = hypre_CSRMatrixJ(Pix_diag);
         HYPRE_Real *Pix_diag_data = hypre_CSRMatrixData(Pix_diag);

         hypre_CSRMatrix *Piy_diag = hypre_ParCSRMatrixDiag(Piy);
         HYPRE_Int *Piy_diag_I = hypre_CSRMatrixI(Piy_diag);
         HYPRE_Int *Piy_diag_J = hypre_CSRMatrixJ(Piy_diag);
         HYPRE_Real *Piy_diag_data = hypre_CSRMatrixData(Piy_diag);

         hypre_CSRMatrix *Piz_diag = hypre_ParCSRMatrixDiag(Piz);
         HYPRE_Int *Piz_diag_I = hypre_CSRMatrixI(Piz_diag);
         HYPRE_Int *Piz_diag_J = hypre_CSRMatrixJ(Piz_diag);
         HYPRE_Real *Piz_diag_data = hypre_CSRMatrixData(Piz_diag);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
#if defined(HYPRE_USING_SYCL)
            HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(F2V_diag_I, F2V_diag_I, F2V_diag_I),
                               F2V_diag_nrows + 1,
                               oneapi::dpl::make_zip_iterator(Pix_diag_I, Piy_diag_I, Piz_diag_I) );

            HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(F2V_diag_J, F2V_diag_J, F2V_diag_J),
                               F2V_diag_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_diag_J, Piy_diag_J, Piz_diag_J) );
#else
            HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(F2V_diag_I, F2V_diag_I, F2V_diag_I)),
                               F2V_diag_nrows + 1,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_I, Piy_diag_I, Piz_diag_I)) );

            HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(F2V_diag_J, F2V_diag_J, F2V_diag_J)),
                               F2V_diag_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_J, Piy_diag_J, Piz_diag_J)) );
#endif

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(F2V_diag_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              F2V_diag_nrows, 3, F2V_diag_I, NULL, RT100_data, RT010_data, RT001_data,
                              Pix_diag_data, Piy_diag_data, Piz_diag_data );
         }
         else
#endif
         {
            for (i = 0; i < F2V_diag_nrows + 1; i++)
            {
               Pix_diag_I[i] = F2V_diag_I[i];
               Piy_diag_I[i] = F2V_diag_I[i];
               Piz_diag_I[i] = F2V_diag_I[i];
            }

            for (i = 0; i < F2V_diag_nnz; i++)
            {
               Pix_diag_J[i] = F2V_diag_J[i];
               Piy_diag_J[i] = F2V_diag_J[i];
               Piz_diag_J[i] = F2V_diag_J[i];
            }

            for (i = 0; i < F2V_diag_nrows; i++)
               for (j = F2V_diag_I[i]; j < F2V_diag_I[i + 1]; j++)
               {
                  *Pix_diag_data++ = RT100_data[i];
                  *Piy_diag_data++ = RT010_data[i];
                  *Piz_diag_data++ = RT001_data[i];
               }
         }
      }

      /* Fill-in the off-diagonal part */
      {
         hypre_CSRMatrix *F2V_offd = hypre_ParCSRMatrixOffd(F2V);
         HYPRE_Int *F2V_offd_I = hypre_CSRMatrixI(F2V_offd);
         HYPRE_Int *F2V_offd_J = hypre_CSRMatrixJ(F2V_offd);

         HYPRE_Int F2V_offd_nrows = hypre_CSRMatrixNumRows(F2V_offd);
         HYPRE_Int F2V_offd_ncols = hypre_CSRMatrixNumCols(F2V_offd);
         HYPRE_Int F2V_offd_nnz = hypre_CSRMatrixNumNonzeros(F2V_offd);

         hypre_CSRMatrix *Pix_offd = hypre_ParCSRMatrixOffd(Pix);
         HYPRE_Int *Pix_offd_I = hypre_CSRMatrixI(Pix_offd);
         HYPRE_Int *Pix_offd_J = hypre_CSRMatrixJ(Pix_offd);
         HYPRE_Real *Pix_offd_data = hypre_CSRMatrixData(Pix_offd);

         hypre_CSRMatrix *Piy_offd = hypre_ParCSRMatrixOffd(Piy);
         HYPRE_Int *Piy_offd_I = hypre_CSRMatrixI(Piy_offd);
         HYPRE_Int *Piy_offd_J = hypre_CSRMatrixJ(Piy_offd);
         HYPRE_Real *Piy_offd_data = hypre_CSRMatrixData(Piy_offd);

         hypre_CSRMatrix *Piz_offd = hypre_ParCSRMatrixOffd(Piz);
         HYPRE_Int *Piz_offd_I = hypre_CSRMatrixI(Piz_offd);
         HYPRE_Int *Piz_offd_J = hypre_CSRMatrixJ(Piz_offd);
         HYPRE_Real *Piz_offd_data = hypre_CSRMatrixData(Piz_offd);

         HYPRE_BigInt *F2V_cmap = hypre_ParCSRMatrixColMapOffd(F2V);
         HYPRE_BigInt *Pix_cmap = hypre_ParCSRMatrixColMapOffd(Pix);
         HYPRE_BigInt *Piy_cmap = hypre_ParCSRMatrixColMapOffd(Piy);
         HYPRE_BigInt *Piz_cmap = hypre_ParCSRMatrixColMapOffd(Piz);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
#if defined(HYPRE_USING_SYCL)
            if (F2V_offd_ncols)
            {
               HYPRE_ONEDPL_CALL( std::copy_n,
                                  oneapi::dpl::make_zip_iterator(F2V_offd_I, F2V_offd_I, F2V_offd_I),
                                  F2V_offd_nrows + 1,
                                  oneapi::dpl::make_zip_iterator(Pix_offd_I, Piy_offd_I, Piz_offd_I) );
            }

            HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(F2V_offd_J, F2V_offd_J, F2V_offd_J),
                               F2V_offd_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_offd_J, Piy_offd_J, Piz_offd_J) );
#else
            if (F2V_offd_ncols)
            {
               HYPRE_THRUST_CALL( copy_n,
                                  thrust::make_zip_iterator(thrust::make_tuple(F2V_offd_I, F2V_offd_I, F2V_offd_I)),
                                  F2V_offd_nrows + 1,
                                  thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_I, Piy_offd_I, Piz_offd_I)) );
            }

            HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(F2V_offd_J, F2V_offd_J, F2V_offd_J)),
                               F2V_offd_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_J, Piy_offd_J, Piz_offd_J)) );
#endif

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(F2V_offd_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              F2V_offd_nrows, 3, F2V_offd_I, NULL, RT100_data, RT010_data, RT001_data,
                              Pix_offd_data, Piy_offd_data, Piz_offd_data );
         }
         else
#endif
         {
            if (F2V_offd_ncols)
               for (i = 0; i < F2V_offd_nrows + 1; i++)
               {
                  Pix_offd_I[i] = F2V_offd_I[i];
                  Piy_offd_I[i] = F2V_offd_I[i];
                  Piz_offd_I[i] = F2V_offd_I[i];
               }

            for (i = 0; i < F2V_offd_nnz; i++)
            {
               Pix_offd_J[i] = F2V_offd_J[i];
               Piy_offd_J[i] = F2V_offd_J[i];
               Piz_offd_J[i] = F2V_offd_J[i];
            }

            for (i = 0; i < F2V_offd_nrows; i++)
               for (j = F2V_offd_I[i]; j < F2V_offd_I[i + 1]; j++)
               {
                  *Pix_offd_data++ = RT100_data[i];
                  *Piy_offd_data++ = RT010_data[i];
                  *Piz_offd_data++ = RT001_data[i];
               }
         }

         for (i = 0; i < F2V_offd_ncols; i++)
         {
            Pix_cmap[i] = F2V_cmap[i];
            Piy_cmap[i] = F2V_cmap[i];
            Piz_cmap[i] = F2V_cmap[i];
         }
      }

      if (HYPRE_AssumedPartitionCheck())
      {
         hypre_ParCSRMatrixDestroy(F2V);
      }
      else
      {
         hypre_ParCSRBooleanMatrixDestroy((hypre_ParCSRBooleanMatrix*)F2V);
      }
   }

   hypre_ParVectorDestroy(RT100);
   hypre_ParVectorDestroy(RT010);
   hypre_ParVectorDestroy(RT001);

   *Pix_ptr = Pix;
   *Piy_ptr = Piy;
   *Piz_ptr = Piz;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSetup
 *
 * Construct the ADS solver components.
 *
 * The following functions need to be called before hypre_ADSSetup():
 * - hypre_ADSSetDiscreteCurl()
 * - hypre_ADSSetDiscreteGradient()
 * - hypre_ADSSetCoordinateVectors()
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ADSSetup(void *solver,
               hypre_ParCSRMatrix *A,
               hypre_ParVector *b,
               hypre_ParVector *x)
{
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   hypre_AMSData *ams_data;

   ads_data -> A = A;

   /* Make sure that the first entry in each row is the diagonal one. */
   /* hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(ads_data -> A)); */

   /* Compute the l1 norm of the rows of A */
   if (ads_data -> A_relax_type >= 1 && ads_data -> A_relax_type <= 4)
   {
      HYPRE_Real *l1_norm_data = NULL;

      hypre_ParCSRComputeL1Norms(ads_data -> A, ads_data -> A_relax_type, NULL, &l1_norm_data);

      ads_data -> A_l1_norms = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(ads_data -> A));
      hypre_VectorData(ads_data -> A_l1_norms) = l1_norm_data;
      hypre_SeqVectorInitialize_v2(ads_data -> A_l1_norms,
                                   hypre_ParCSRMatrixMemoryLocation(ads_data -> A));
   }

   /* Chebyshev? */
   if (ads_data -> A_relax_type == 16)
   {
      hypre_ParCSRMaxEigEstimateCG(ads_data->A, 1, 10,
                                   &ads_data->A_max_eig_est,
                                   &ads_data->A_min_eig_est);
   }

   /* Create the AMS solver on the range of C^T */
   {
      HYPRE_AMSCreate(&ads_data -> B_C);
      HYPRE_AMSSetDimension(ads_data -> B_C, 3);

      /* B_C is a preconditioner */
      HYPRE_AMSSetMaxIter(ads_data -> B_C, 1);
      HYPRE_AMSSetTol(ads_data -> B_C, 0.0);
      HYPRE_AMSSetPrintLevel(ads_data -> B_C, 0);

      HYPRE_AMSSetCycleType(ads_data -> B_C, ads_data -> B_C_cycle_type);
      HYPRE_AMSSetDiscreteGradient(ads_data -> B_C,
                                   (HYPRE_ParCSRMatrix) ads_data -> G);

      if (ads_data -> ND_Pi == NULL && ads_data -> ND_Pix == NULL)
      {
         if (ads_data -> B_C_cycle_type < 10)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "Unsupported AMS cycle type in ADS!");
         }
         HYPRE_AMSSetCoordinateVectors(ads_data -> B_C,
                                       (HYPRE_ParVector) ads_data -> x,
                                       (HYPRE_ParVector) ads_data -> y,
                                       (HYPRE_ParVector) ads_data -> z);
      }
      else
      {
         if ((ads_data -> B_C_cycle_type < 10 && ads_data -> ND_Pi == NULL) ||
             (ads_data -> B_C_cycle_type > 10 && ads_data -> ND_Pix == NULL))
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "Unsupported AMS cycle type in ADS!");
         }
         HYPRE_AMSSetInterpolations(ads_data -> B_C,
                                    (HYPRE_ParCSRMatrix) ads_data -> ND_Pi,
                                    (HYPRE_ParCSRMatrix) ads_data -> ND_Pix,
                                    (HYPRE_ParCSRMatrix) ads_data -> ND_Piy,
                                    (HYPRE_ParCSRMatrix) ads_data -> ND_Piz);
      }

      /* beta=0 in the subspace */
      HYPRE_AMSSetBetaPoissonMatrix(ads_data -> B_C, NULL);

      /* Reuse A's relaxation parameters for A_C */
      HYPRE_AMSSetSmoothingOptions(ads_data -> B_C,
                                   ads_data -> A_relax_type,
                                   ads_data -> A_relax_times,
                                   ads_data -> A_relax_weight,
                                   ads_data -> A_omega);

      HYPRE_AMSSetAlphaAMGOptions(ads_data -> B_C, ads_data -> B_C_coarsen_type,
                                  ads_data -> B_C_agg_levels, ads_data -> B_C_relax_type,
                                  ads_data -> B_C_theta, ads_data -> B_C_interp_type,
                                  ads_data -> B_C_Pmax);
      /* No need to call HYPRE_AMSSetBetaAMGOptions */

      /* Construct the coarse space matrix by RAP */
      if (!ads_data -> A_C)
      {
         if (!hypre_ParCSRMatrixCommPkg(ads_data -> C))
         {
            hypre_MatvecCommPkgCreate(ads_data -> C);
         }

         if (!hypre_ParCSRMatrixCommPkg(ads_data -> A))
         {
            hypre_MatvecCommPkgCreate(ads_data -> A);
         }

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            ads_data -> A_C = hypre_ParCSRMatrixRAPKT(ads_data -> C,
                                                      ads_data -> A,
                                                      ads_data -> C, 1);
         }
         else
#endif
         {
            hypre_BoomerAMGBuildCoarseOperator(ads_data -> C,
                                               ads_data -> A,
                                               ads_data -> C,
                                               &ads_data -> A_C);
         }

         /* Make sure that A_C has no zero rows (this can happen if beta is zero
            in part of the domain). */
         hypre_ParCSRMatrixFixZeroRows(ads_data -> A_C);
      }

      HYPRE_AMSSetup(ads_data -> B_C, (HYPRE_ParCSRMatrix)ads_data -> A_C, 0, 0);
   }

   ams_data = (hypre_AMSData *) ads_data -> B_C;

   if (ads_data -> Pi == NULL && ads_data -> Pix == NULL)
   {
      if (ads_data -> cycle_type > 10)
      {
         /* Construct Pi{x,y,z} instead of Pi = [Pix,Piy,Piz] */
         hypre_ADSComputePixyz(ads_data -> A,
                               ads_data -> C,
                               ads_data -> G,
                               ads_data -> x,
                               ads_data -> y,
                               ads_data -> z,
                               ams_data -> Pix,
                               ams_data -> Piy,
                               ams_data -> Piz,
                               &ads_data -> Pix,
                               &ads_data -> Piy,
                               &ads_data -> Piz);
      }
      else
      {
         /* Construct the Pi interpolation matrix */
         hypre_ADSComputePi(ads_data -> A,
                            ads_data -> C,
                            ads_data -> G,
                            ads_data -> x,
                            ads_data -> y,
                            ads_data -> z,
                            ams_data -> Pix,
                            ams_data -> Piy,
                            ams_data -> Piz,
                            &ads_data -> Pi);
      }
   }

   if (ads_data -> cycle_type > 10)
      /* Create the AMG solvers on the range of Pi{x,y,z}^T */
   {
      HYPRE_BoomerAMGCreate(&ads_data -> B_Pix);
      HYPRE_BoomerAMGSetCoarsenType(ads_data -> B_Pix, ads_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ads_data -> B_Pix, ads_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ads_data -> B_Pix, ads_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ads_data -> B_Pix, 1);
      HYPRE_BoomerAMGSetMaxLevels(ads_data -> B_Pix, 25);
      HYPRE_BoomerAMGSetTol(ads_data -> B_Pix, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ads_data -> B_Pix, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ads_data -> B_Pix, ads_data -> B_Pi_theta);
      HYPRE_BoomerAMGSetInterpType(ads_data -> B_Pix, ads_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ads_data -> B_Pix, ads_data -> B_Pi_Pmax);

      HYPRE_BoomerAMGCreate(&ads_data -> B_Piy);
      HYPRE_BoomerAMGSetCoarsenType(ads_data -> B_Piy, ads_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ads_data -> B_Piy, ads_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ads_data -> B_Piy, ads_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ads_data -> B_Piy, 1);
      HYPRE_BoomerAMGSetMaxLevels(ads_data -> B_Piy, 25);
      HYPRE_BoomerAMGSetTol(ads_data -> B_Piy, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ads_data -> B_Piy, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ads_data -> B_Piy, ads_data -> B_Pi_theta);
      HYPRE_BoomerAMGSetInterpType(ads_data -> B_Piy, ads_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ads_data -> B_Piy, ads_data -> B_Pi_Pmax);

      HYPRE_BoomerAMGCreate(&ads_data -> B_Piz);
      HYPRE_BoomerAMGSetCoarsenType(ads_data -> B_Piz, ads_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ads_data -> B_Piz, ads_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ads_data -> B_Piz, ads_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ads_data -> B_Piz, 1);
      HYPRE_BoomerAMGSetMaxLevels(ads_data -> B_Piz, 25);
      HYPRE_BoomerAMGSetTol(ads_data -> B_Piz, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ads_data -> B_Piz, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ads_data -> B_Piz, ads_data -> B_Pi_theta);
      HYPRE_BoomerAMGSetInterpType(ads_data -> B_Piz, ads_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ads_data -> B_Piz, ads_data -> B_Pi_Pmax);

      /* Don't use exact solve on the coarsest level (matrices may be singular) */
      HYPRE_BoomerAMGSetCycleRelaxType(ads_data -> B_Pix,
                                       ads_data -> B_Pi_relax_type, 3);
      HYPRE_BoomerAMGSetCycleRelaxType(ads_data -> B_Piy,
                                       ads_data -> B_Pi_relax_type, 3);
      HYPRE_BoomerAMGSetCycleRelaxType(ads_data -> B_Piz,
                                       ads_data -> B_Pi_relax_type, 3);

      /* Construct the coarse space matrices by RAP */
      if (!hypre_ParCSRMatrixCommPkg(ads_data -> Pix))
      {
         hypre_MatvecCommPkgCreate(ads_data -> Pix);
      }

#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         ads_data -> A_Pix = hypre_ParCSRMatrixRAPKT(ads_data -> Pix,
                                                     ads_data -> A,
                                                     ads_data -> Pix, 1);
      }
      else
#endif
      {
         hypre_BoomerAMGBuildCoarseOperator(ads_data -> Pix,
                                            ads_data -> A,
                                            ads_data -> Pix,
                                            &ads_data -> A_Pix);
      }

      HYPRE_BoomerAMGSetup(ads_data -> B_Pix,
                           (HYPRE_ParCSRMatrix)ads_data -> A_Pix,
                           NULL, NULL);

      if (!hypre_ParCSRMatrixCommPkg(ads_data -> Piy))
      {
         hypre_MatvecCommPkgCreate(ads_data -> Piy);
      }

#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         ads_data -> A_Piy = hypre_ParCSRMatrixRAPKT(ads_data -> Piy,
                                                     ads_data -> A,
                                                     ads_data -> Piy, 1);
      }
      else
#endif
      {
         hypre_BoomerAMGBuildCoarseOperator(ads_data -> Piy,
                                            ads_data -> A,
                                            ads_data -> Piy,
                                            &ads_data -> A_Piy);
      }

      HYPRE_BoomerAMGSetup(ads_data -> B_Piy,
                           (HYPRE_ParCSRMatrix)ads_data -> A_Piy,
                           NULL, NULL);

      if (!hypre_ParCSRMatrixCommPkg(ads_data -> Piz))
      {
         hypre_MatvecCommPkgCreate(ads_data -> Piz);
      }

#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         ads_data -> A_Piz = hypre_ParCSRMatrixRAPKT(ads_data -> Piz,
                                                     ads_data -> A,
                                                     ads_data -> Piz, 1);
      }
      else
#endif
      {
         hypre_BoomerAMGBuildCoarseOperator(ads_data -> Piz,
                                            ads_data -> A,
                                            ads_data -> Piz,
                                            &ads_data -> A_Piz);
      }

      HYPRE_BoomerAMGSetup(ads_data -> B_Piz,
                           (HYPRE_ParCSRMatrix)ads_data -> A_Piz,
                           NULL, NULL);
   }
   else
   {
      /* Create the AMG solver on the range of Pi^T */
      HYPRE_BoomerAMGCreate(&ads_data -> B_Pi);
      HYPRE_BoomerAMGSetCoarsenType(ads_data -> B_Pi, ads_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ads_data -> B_Pi, ads_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ads_data -> B_Pi, ads_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ads_data -> B_Pi, 1);
      HYPRE_BoomerAMGSetMaxLevels(ads_data -> B_Pi, 25);
      HYPRE_BoomerAMGSetTol(ads_data -> B_Pi, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ads_data -> B_Pi, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ads_data -> B_Pi, ads_data -> B_Pi_theta);
      HYPRE_BoomerAMGSetInterpType(ads_data -> B_Pi, ads_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ads_data -> B_Pi, ads_data -> B_Pi_Pmax);

      /* Don't use exact solve on the coarsest level (matrix may be singular) */
      HYPRE_BoomerAMGSetCycleRelaxType(ads_data -> B_Pi,
                                       ads_data -> B_Pi_relax_type,
                                       3);

      /* Construct the coarse space matrix by RAP and notify BoomerAMG that this
         is a 3 x 3 block system. */
      if (!ads_data -> A_Pi)
      {
         if (!hypre_ParCSRMatrixCommPkg(ads_data -> Pi))
         {
            hypre_MatvecCommPkgCreate(ads_data -> Pi);
         }

         if (!hypre_ParCSRMatrixCommPkg(ads_data -> A))
         {
            hypre_MatvecCommPkgCreate(ads_data -> A);
         }

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            ads_data -> A_Pi = hypre_ParCSRMatrixRAPKT(ads_data -> Pi,
                                                       ads_data -> A,
                                                       ads_data -> Pi, 1);
         }
         else
#endif
         {
            hypre_BoomerAMGBuildCoarseOperator(ads_data -> Pi,
                                               ads_data -> A,
                                               ads_data -> Pi,
                                               &ads_data -> A_Pi);
         }

         HYPRE_BoomerAMGSetNumFunctions(ads_data -> B_Pi, 3);
         /* HYPRE_BoomerAMGSetNodal(ads_data -> B_Pi, 1); */
      }

      HYPRE_BoomerAMGSetup(ads_data -> B_Pi,
                           (HYPRE_ParCSRMatrix)ads_data -> A_Pi,
                           NULL, NULL);
   }

   /* Allocate temporary vectors */
   ads_data -> r0 = hypre_ParVectorInRangeOf(ads_data -> A);
   ads_data -> g0 = hypre_ParVectorInRangeOf(ads_data -> A);
   if (ads_data -> A_C)
   {
      ads_data -> r1 = hypre_ParVectorInRangeOf(ads_data -> A_C);
      ads_data -> g1 = hypre_ParVectorInRangeOf(ads_data -> A_C);
   }
   if (ads_data -> cycle_type > 10)
   {
      ads_data -> r2 = hypre_ParVectorInDomainOf(ads_data -> Pix);
      ads_data -> g2 = hypre_ParVectorInDomainOf(ads_data -> Pix);
   }
   else
   {
      ads_data -> r2 = hypre_ParVectorInDomainOf(ads_data -> Pi);
      ads_data -> g2 = hypre_ParVectorInDomainOf(ads_data -> Pi);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSSolve
 *
 * Solve the system A x = b.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSSolve(void *solver,
                         hypre_ParCSRMatrix *A,
                         hypre_ParVector *b,
                         hypre_ParVector *x)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;

   HYPRE_Int   i, my_id = -1;
   HYPRE_Real  r0_norm  = 1.0;
   HYPRE_Real  r_norm   = 1.0;
   HYPRE_Real  b_norm   = 1.0;
   HYPRE_Real  relative_resid = 0, old_resid;

   char cycle[30];
   hypre_ParCSRMatrix *Ai[5], *Pi[5];
   HYPRE_Solver Bi[5];
   HYPRE_PtrToSolverFcn HBi[5];
   hypre_ParVector *ri[5], *gi[5];
   HYPRE_Int needZ = 0;

   hypre_ParVector *z = ads_data -> zz;

   Ai[0] = ads_data -> A_C;    Pi[0] = ads_data -> C;
   Ai[1] = ads_data -> A_Pi;   Pi[1] = ads_data -> Pi;
   Ai[2] = ads_data -> A_Pix;  Pi[2] = ads_data -> Pix;
   Ai[3] = ads_data -> A_Piy;  Pi[3] = ads_data -> Piy;
   Ai[4] = ads_data -> A_Piz;  Pi[4] = ads_data -> Piz;

   Bi[0] = ads_data -> B_C;    HBi[0] = (HYPRE_PtrToSolverFcn) hypre_AMSSolve;
   Bi[1] = ads_data -> B_Pi;   HBi[1] = (HYPRE_PtrToSolverFcn) hypre_BoomerAMGBlockSolve;
   Bi[2] = ads_data -> B_Pix;  HBi[2] = (HYPRE_PtrToSolverFcn) hypre_BoomerAMGSolve;
   Bi[3] = ads_data -> B_Piy;  HBi[3] = (HYPRE_PtrToSolverFcn) hypre_BoomerAMGSolve;
   Bi[4] = ads_data -> B_Piz;  HBi[4] = (HYPRE_PtrToSolverFcn) hypre_BoomerAMGSolve;

   ri[0] = ads_data -> r1;     gi[0] = ads_data -> g1;
   ri[1] = ads_data -> r2;     gi[1] = ads_data -> g2;
   ri[2] = ads_data -> r2;     gi[2] = ads_data -> g2;
   ri[3] = ads_data -> r2;     gi[3] = ads_data -> g2;
   ri[4] = ads_data -> r2;     gi[4] = ads_data -> g2;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   /* may need to create an additional temporary vector for relaxation */
#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_DEVICE)
   {
      needZ = ads_data -> A_relax_type == 2 ||
              ads_data -> A_relax_type == 4 ||
              ads_data -> A_relax_type == 16;
   }
   else
#endif
   {
      needZ = hypre_NumThreads() > 1 || ads_data -> A_relax_type == 16;
   }

   if (needZ && !z)
   {
      z = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(z);
      ads_data -> zz = z;
   }

   if (ads_data -> print_level > 0)
   {
      hypre_MPI_Comm_rank(hypre_ParCSRMatrixComm(A), &my_id);
   }

   switch (ads_data -> cycle_type)
   {
      case 1:
      default:
         hypre_sprintf(cycle, "%s", "01210");
         break;

      case 2:
         hypre_sprintf(cycle, "%s", "(0+1+2)");
         break;

      case 3:
         hypre_sprintf(cycle, "%s", "02120");
         break;

      case 4:
         hypre_sprintf(cycle, "%s", "(010+2)");
         break;

      case 5:
         hypre_sprintf(cycle, "%s", "0102010");
         break;

      case 6:
         hypre_sprintf(cycle, "%s", "(020+1)");
         break;

      case 7:
         hypre_sprintf(cycle, "%s", "0201020");
         break;

      case 8:
         hypre_sprintf(cycle, "%s", "0(+1+2)0");
         break;

      case 9:
         hypre_sprintf(cycle, "%s", "01210");
         break;

      case 11:
         hypre_sprintf(cycle, "%s", "013454310");
         break;

      case 12:
         hypre_sprintf(cycle, "%s", "(0+1+3+4+5)");
         break;

      case 13:
         hypre_sprintf(cycle, "%s", "034515430");
         break;

      case 14:
         hypre_sprintf(cycle, "%s", "01(+3+4+5)10");
         break;
   }

   for (i = 0; i < ads_data -> maxit; i++)
   {
      /* Compute initial residual norms */
      if (ads_data -> maxit > 1 && i == 0)
      {
         hypre_ParVectorCopy(b, ads_data -> r0);
         hypre_ParCSRMatrixMatvec(-1.0, ads_data -> A, x, 1.0, ads_data -> r0);
         r_norm = hypre_sqrt(hypre_ParVectorInnerProd(ads_data -> r0, ads_data -> r0));
         r0_norm = r_norm;
         b_norm = hypre_sqrt(hypre_ParVectorInnerProd(b, b));
         if (b_norm)
         {
            relative_resid = r_norm / b_norm;
         }
         else
         {
            relative_resid = r_norm;
         }
         if (my_id == 0 && ads_data -> print_level > 0)
         {
            hypre_printf("                                            relative\n");
            hypre_printf("               residual        factor       residual\n");
            hypre_printf("               --------        ------       --------\n");
            hypre_printf("    Initial    %e                 %e\n",
                         r_norm, relative_resid);
         }
      }

      /* Apply the preconditioner */
      hypre_ParCSRSubspacePrec(ads_data -> A,
                               ads_data -> A_relax_type,
                               ads_data -> A_relax_times,
                               ads_data -> A_l1_norms ? hypre_VectorData(ads_data -> A_l1_norms) : NULL,
                               ads_data -> A_relax_weight,
                               ads_data -> A_omega,
                               ads_data -> A_max_eig_est,
                               ads_data -> A_min_eig_est,
                               ads_data -> A_cheby_order,
                               ads_data -> A_cheby_fraction,
                               Ai, Bi, HBi, Pi, ri, gi,
                               b, x,
                               ads_data -> r0,
                               ads_data -> g0,
                               cycle,
                               z);

      /* Compute new residual norms */
      if (ads_data -> maxit > 1)
      {
         old_resid = r_norm;
         hypre_ParVectorCopy(b, ads_data -> r0);
         hypre_ParCSRMatrixMatvec(-1.0, ads_data -> A, x, 1.0, ads_data -> r0);
         r_norm = hypre_sqrt(hypre_ParVectorInnerProd(ads_data -> r0, ads_data -> r0));
         if (b_norm)
         {
            relative_resid = r_norm / b_norm;
         }
         else
         {
            relative_resid = r_norm;
         }
         if (my_id == 0 && ads_data -> print_level > 0)
            hypre_printf("    Cycle %2d   %e    %f     %e \n",
                         i + 1, r_norm, r_norm / old_resid, relative_resid);
      }

      if (relative_resid < ads_data -> tol)
      {
         i++;
         break;
      }
   }

   if (my_id == 0 && ads_data -> print_level > 0 && ads_data -> maxit > 1)
   {
      hypre_printf("\n\n Average Convergence Factor = %f\n\n",
                   hypre_pow((r_norm / r0_norm), (1.0 / (HYPRE_Real) i)));
   }

   ads_data -> num_iterations = i;
   ads_data -> rel_resid_norm = relative_resid;

   if (ads_data -> num_iterations == ads_data -> maxit && ads_data -> tol > 0.0)
   {
      hypre_error(HYPRE_ERROR_CONV);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSGetNumIterations
 *
 * Get the number of ADS iterations.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSGetNumIterations(void *solver,
                                    HYPRE_Int *num_iterations)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   *num_iterations = ads_data -> num_iterations;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ADSGetFinalRelativeResidualNorm
 *
 * Get the final relative residual norm in ADS.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ADSGetFinalRelativeResidualNorm(void *solver,
                                                HYPRE_Real *rel_resid_norm)
{
   hypre_ADSData *ads_data = (hypre_ADSData *) solver;
   *rel_resid_norm = ads_data -> rel_resid_norm;
   return hypre_error_flag;
}
