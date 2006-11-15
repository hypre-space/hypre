/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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




#include "headers.h"
#include "float.h"
#include "ams.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRRelax
 *
 * Relaxation on the ParCSR matrix A with right-hand side f and
 * initial guess u. Possible values for relax_type are:
 *
 * 1 = l1-scaled Jacobi
 * 2 = l1-scaled block Gauss-Seidel
 * x = BoomerAMG relaxation with relax_type = |x|
 *
 * The default value of relax_type is 2.
 *--------------------------------------------------------------------------*/

int hypre_ParCSRRelax(/* matrix to relax with */
                      hypre_ParCSRMatrix *A,
                      /* right-hand side */
                      hypre_ParVector *f,
                      /* relaxation type */
                      int relax_type,
                      /* number of sweeps */
                      int relax_times,
                      /* l1 norms of the rows of A */
                      double *l1_norms,
                      /* damping coefficient */
                      double relax_weight,
                      /* SOR parameter */
                      double omega,
                      /* initial/updated approximation */
                      hypre_ParVector *u,
                      /* temporary vector */
                      hypre_ParVector *v)
{
   int sweep;

   double *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   double *f_data = hypre_VectorData(hypre_ParVectorLocalVector(f));
   double *v_data = hypre_VectorData(hypre_ParVectorLocalVector(v));

   for (sweep = 0; sweep < relax_times; sweep++)
   {
      if (relax_type == 1) /* l1-scaled Jacobi */
      {
         int i, num_rows = hypre_ParCSRMatrixNumRows(A);

         hypre_ParVectorCopy(f,v);
         hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, v);

         /* u += D^{-1}(f - A u), where D_ii = ||A(i,:)||_1 */
         for (i = 0; i < num_rows; i++)
            u_data[i] += v_data[i] / l1_norms[i];
      }
      else if (relax_type == 2) /* offd-l1-scaled block Gauss-Seidel */
      {
         hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
         double *A_diag_data  = hypre_CSRMatrixData(A_diag);
         int *A_diag_I = hypre_CSRMatrixI(A_diag);
         int *A_diag_J = hypre_CSRMatrixJ(A_diag);

         hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
         int *A_offd_I = hypre_CSRMatrixI(A_offd);
         int *A_offd_J = hypre_CSRMatrixJ(A_offd);
         double *A_offd_data  = hypre_CSRMatrixData(A_offd);

         int i, j;
         int num_rows = hypre_CSRMatrixNumRows(A_diag);
         int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
         double *u_offd_data = hypre_TAlloc(double,num_cols_offd);

         double res;

         int num_procs;
         MPI_Comm_size( hypre_ParCSRMatrixComm(A), &num_procs);

         /* Copy off-diagonal values of u to the current processor */
         if (num_procs > 1)
         {
            hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
            int num_sends;
            double *u_buf_data;
            hypre_ParCSRCommHandle *comm_handle;

            int index = 0, start;

            if (!comm_pkg)
            {
               hypre_MatvecCommPkgCreate(A);
               comm_pkg = hypre_ParCSRMatrixCommPkg(A);
            }

            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
            u_buf_data = hypre_TAlloc(double,
                                      hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j++)
                  u_buf_data[index++] = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
            }
            comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,u_buf_data,u_offd_data);
            hypre_ParCSRCommHandleDestroy(comm_handle);

            hypre_TFree(u_buf_data);
         }

         /* Forward local GS pass */
         for (i = 0; i < num_rows; i++)
         {
            res = f_data[i];
            for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
               res -= A_diag_data[j] * u_data[A_diag_J[j]];
            if (num_cols_offd)
               for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
                  res -= A_offd_data[j] * u_offd_data[A_offd_J[j]];
            u_data[i] += res / l1_norms[i];
         }
         /* Backward local GS pass */
         for (i = num_rows-1; i > -1; i--)      {
            res = f_data[i];
            for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
               res -= A_diag_data[j] * u_data[A_diag_J[j]];
            if (num_cols_offd)
               for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
                  res -= A_offd_data[j] * u_offd_data[A_offd_J[j]];
            u_data[i] += res / l1_norms[i];
         }

         hypre_TFree(u_offd_data);
      }
      else /* call BoomerAMG relaxation */
      {
         hypre_BoomerAMGRelax(A, f, NULL, abs(relax_type), 0,
                              relax_weight, omega, u, v);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorInRangeOf
 *
 * Return a vector that belongs to the range of a given matrix.
 *--------------------------------------------------------------------------*/

hypre_ParVector *hypre_ParVectorInRangeOf(hypre_ParCSRMatrix *A)
{
   hypre_ParVector *x;

   x = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(x);
   hypre_ParVectorOwnsData(x) = 1;
   hypre_ParVectorOwnsPartitioning(x) = 0;

   return x;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorInDomainOf
 *
 * Return a vector that belongs to the domain of a given matrix.
 *--------------------------------------------------------------------------*/

hypre_ParVector *hypre_ParVectorInDomainOf(hypre_ParCSRMatrix *A)
{
   hypre_ParVector *x;

   x = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumCols(A),
                             hypre_ParCSRMatrixColStarts(A));
   hypre_ParVectorInitialize(x);
   hypre_ParVectorOwnsData(x) = 1;
   hypre_ParVectorOwnsPartitioning(x) = 0;

   return x;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorBlockSplit
 *
 * Extract the dim sub-vectors x_0,...,x_{dim-1} composing a parallel
 * block vector x. It is assumed that &x[i] = [x_0[i],...,x_{dim-1}[i]].
 *--------------------------------------------------------------------------*/

int hypre_ParVectorBlockSplit(hypre_ParVector *x,
                              hypre_ParVector *x_[3],
                              int dim)
{
   int i, d, size_;
   double *x_data, *x_data_[3];

   size_ = hypre_VectorSize(hypre_ParVectorLocalVector(x_[0]));

   x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   for (d = 0; d < dim; d++)
      x_data_[d] = hypre_VectorData(hypre_ParVectorLocalVector(x_[d]));

   for (i = 0; i < size_; i++)
      for (d = 0; d < dim; d++)
         x_data_[d][i] = x_data[dim*i+d];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorBlockGather
 *
 * Compose a parallel block vector x from dim given sub-vectors
 * x_0,...,x_{dim-1}, such that &x[i] = [x_0[i],...,x_{dim-1}[i]].
 *--------------------------------------------------------------------------*/

int hypre_ParVectorBlockGather(hypre_ParVector *x,
                               hypre_ParVector *x_[3],
                               int dim)
{
   int i, d, size_;
   double *x_data, *x_data_[3];

   size_ = hypre_VectorSize(hypre_ParVectorLocalVector(x_[0]));

   x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   for (d = 0; d < dim; d++)
      x_data_[d] = hypre_VectorData(hypre_ParVectorLocalVector(x_[d]));

   for (i = 0; i < size_; i++)
      for (d = 0; d < dim; d++)
         x_data[dim*i+d] = x_data_[d][i];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGBlockSolve
 *
 * Apply the block-diagonal solver diag(B) to the system diag(A) x = b.
 * Here B is a given BoomerAMG solver for A, while x and b are "block"
 * parallel vectors.
 *--------------------------------------------------------------------------*/

int hypre_BoomerAMGBlockSolve(void *B,
                              hypre_ParCSRMatrix *A,
                              hypre_ParVector *b,
                              hypre_ParVector *x)
{
   int d, dim;

   hypre_ParVector *b_[3];
   hypre_ParVector *x_[3];

   dim = hypre_ParVectorGlobalSize(x) / hypre_ParCSRMatrixGlobalNumRows(A);

   if (dim == 1)
   {
      hypre_BoomerAMGSolve(B, A, b, x);
      return hypre_error_flag;
   }

   for (d = 0; d < dim; d++)
   {
      b_[d] = hypre_ParVectorInRangeOf(A);
      x_[d] = hypre_ParVectorInRangeOf(A);
   }

   hypre_ParVectorBlockSplit(b, b_, dim);
   hypre_ParVectorBlockSplit(x, x_, dim);

   for (d = 0; d < dim; d++)
      hypre_BoomerAMGSolve(B, A, b_[d], x_[d]);

   hypre_ParVectorBlockGather(x, x_, dim);

   for (d = 0; d < dim; d++)
   {
      hypre_ParVectorDestroy(b_[d]);
      hypre_ParVectorDestroy(x_[d]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixFixZeroRows
 *
 * For every zero row in the matrix: set the diagonal element to 1.
 *--------------------------------------------------------------------------*/

int hypre_ParCSRMatrixFixZeroRows(hypre_ParCSRMatrix *A)
{
   int i, j;
   double l1_norm;
   int num_rows = hypre_ParCSRMatrixNumRows(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   int *A_diag_I = hypre_CSRMatrixI(A_diag);
   int *A_diag_J = hypre_CSRMatrixJ(A_diag);
   double *A_diag_data = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   int *A_offd_I = hypre_CSRMatrixI(A_offd);
   double *A_offd_data = hypre_CSRMatrixData(A_offd);
   int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   /* a row will be considered zero if its l1 norm is less than eps */
   double eps = DBL_EPSILON * 1e+4;

   for (i = 0; i < num_rows; i++)
   {
      l1_norm = 0.0;
      for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
         l1_norm += fabs(A_diag_data[j]);
      if (num_cols_offd)
         for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
            l1_norm += fabs(A_offd_data[j]);

      if (l1_norm < eps)
      {
        for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
            if (A_diag_J[j] == i)
               A_diag_data[j] = 1.0;
            else
               A_diag_data[j] = 0.0;
         if (num_cols_offd)
            for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
               A_offd_data[j] = 0.0;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRComputeL1Norms
 *
 * Compute the l1 norms of the rows of a given matrix, depending on
 * the option parameter:
 *
 * option 1 = Compute the l1 norm of the rows
 * option 2 = Compute the l1 norm of the (processor) off-diagonal
 *            part of the rows plus the diagonal of A
 *--------------------------------------------------------------------------*/

int hypre_ParCSRComputeL1Norms(hypre_ParCSRMatrix *A,
                               int option,
                               double **l1_norm_ptr)
{
   int i, j;
   int num_rows = hypre_ParCSRMatrixNumRows(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   int *A_diag_I = hypre_CSRMatrixI(A_diag);
   int *A_diag_J = hypre_CSRMatrixJ(A_diag);
   double *A_diag_data = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   int *A_offd_I = hypre_CSRMatrixI(A_offd);
   double *A_offd_data = hypre_CSRMatrixData(A_offd);
   int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   double *l1_norm = hypre_TAlloc(double, num_rows);

   for (i = 0; i < num_rows; i++)
   {
      if (option == 1)
      {
         /* Add the l1 norm of the diag part of the ith row */
         l1_norm[i] = 0.0;
         for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
            l1_norm[i] += fabs(A_diag_data[j]);
      }
      else if (option == 2)
      {
         /* Add the diag element of the ith row */
         for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
            if (A_diag_J[j] == i)
            {
               l1_norm[i] = A_diag_data[j];
               break;
            }
      }

      /* Add the l1 norm of the offd part of the ith row */
      if (num_cols_offd)
         for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
            l1_norm[i] += fabs(A_offd_data[j]);

      if (l1_norm[i] < DBL_EPSILON)
         hypre_error_in_arg(1);
   }

   *l1_norm_ptr = l1_norm;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetDiagRows
 *
 * For every row containing only a diagonal element: set it to d.
 *--------------------------------------------------------------------------*/

int hypre_ParCSRMatrixSetDiagRows(hypre_ParCSRMatrix *A, double d)
{
   int i, j;
   int num_rows = hypre_ParCSRMatrixNumRows(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   int *A_diag_I = hypre_CSRMatrixI(A_diag);
   int *A_diag_J = hypre_CSRMatrixJ(A_diag);
   double *A_diag_data = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   int *A_offd_I = hypre_CSRMatrixI(A_offd);
   int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   for (i = 0; i < num_rows; i++)
   {
      j = A_diag_I[i];
      if ((A_diag_I[i+1] == j+1) && (A_diag_J[j] == i) &&
          num_cols_offd && (A_offd_I[i+1] == A_offd_I[i]))
      {
         A_diag_data[j] = d;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSCreate
 *
 * Allocate the AMS solver structure.
 *--------------------------------------------------------------------------*/

void * hypre_AMSCreate()
{
   hypre_AMSData *ams_data;

   ams_data = hypre_CTAlloc(hypre_AMSData, 1);

   /* Default parameters */

   ams_data -> dim  = 3;               /* 3D problem */
   ams_data -> maxit = 20;             /* perform at most 20 iterations */
   ams_data -> tol = 1e-6;             /* convergence tolerance */
   ams_data -> print_level = 1;        /* print residual norm at each step */
   ams_data -> cycle_type = 1;         /* a 3-level multiplicative solver */
   ams_data -> A_relax_type = 2;       /* offd-l1-scaled GS */
   ams_data -> A_relax_times = 1;      /* one relaxation sweep */
   ams_data -> A_relax_weight = 1.0;   /* damping parameter */
   ams_data -> A_omega = 1.0;          /* SSOR coefficient */
   ams_data -> B_G_coarsen_type = 10;  /* HMIS coarsening */
   ams_data -> B_G_agg_levels = 1;     /* Levels of aggresive coarsening */
   ams_data -> B_G_relax_type = 3;     /* hybrid G-S/Jacobi */
   ams_data -> B_G_theta = 0.25;       /* strength threshold */
   ams_data -> B_Pi_coarsen_type = 10; /* HMIS coarsening */
   ams_data -> B_Pi_agg_levels = 1;    /* Levels of aggresive coarsening */
   ams_data -> B_Pi_relax_type = 3;    /* hybrid G-S/Jacobi */
   ams_data -> B_Pi_theta = 0.25;      /* strength threshold */
   ams_data -> beta_is_zero = 0;       /* the problem has a mass term */

   /* The rest of the fields are initialized using the Set functions */

   ams_data -> A    = NULL;
   ams_data -> G    = NULL;
   ams_data -> A_G  = NULL;
   ams_data -> Pi   = NULL;
   ams_data -> A_Pi = NULL;
   ams_data -> x    = NULL;
   ams_data -> y    = NULL;
   ams_data -> z    = NULL;
   ams_data -> Gx   = NULL;
   ams_data -> Gy   = NULL;
   ams_data -> Gz   = NULL;

   ams_data -> A_l1_norms = NULL;

   ams_data -> owns_A_G  = 0;
   ams_data -> owns_A_Pi = 0;

   return (void *) ams_data;
}

/*--------------------------------------------------------------------------
 * hypre_AMSDestroy
 *
 * Deallocate the AMS solver structure. Note that the input data (given
 * through the Set functions) is not destroyed.
 *--------------------------------------------------------------------------*/

int hypre_AMSDestroy(void *solver)
{
   hypre_AMSData *ams_data = solver;

   if (ams_data -> owns_A_G)
      if (ams_data -> A_G)
         hypre_ParCSRMatrixDestroy(ams_data -> A_G);
   if (!ams_data -> beta_is_zero)
      if (ams_data -> B_G)
         HYPRE_BoomerAMGDestroy(ams_data -> B_G);

   if (ams_data -> Pi)
      hypre_ParCSRMatrixDestroy(ams_data -> Pi);
   if (ams_data -> owns_A_Pi)
      if (ams_data -> A_Pi)
         hypre_ParCSRMatrixDestroy(ams_data -> A_Pi);
   if (ams_data -> B_Pi)
      HYPRE_BoomerAMGDestroy(ams_data -> B_Pi);

   if (ams_data -> r0)
      hypre_ParVectorDestroy(ams_data -> r0);
   if (ams_data -> g0)
      hypre_ParVectorDestroy(ams_data -> g0);
   if (!ams_data -> beta_is_zero)
   {
      if (ams_data -> r1)
         hypre_ParVectorDestroy(ams_data -> r1);
      if (ams_data -> g1)
         hypre_ParVectorDestroy(ams_data -> g1);
   }
   if (ams_data -> r2)
      hypre_ParVectorDestroy(ams_data -> r2);
   if (ams_data -> g2)
      hypre_ParVectorDestroy(ams_data -> g2);

   if (ams_data -> A_l1_norms)
      hypre_TFree(ams_data -> A_l1_norms);

   /* G, x, y ,z, Gx, Gy and Gz are not destroyed */

   if (ams_data)
      hypre_TFree(ams_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetDimension
 *
 * Set problem dimension (2 or 3). By default we assume dim = 3.
 *--------------------------------------------------------------------------*/

int hypre_AMSSetDimension(void *solver,
                          int dim)
{
   hypre_AMSData *ams_data = solver;

   if (dim != 2 && dim != 3)
      hypre_error_in_arg(2);

   ams_data -> dim = dim;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetDiscreteGradient
 *
 * Set the discrete gradient matrix G.
 * This function should be called before hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

int hypre_AMSSetDiscreteGradient(void *solver,
                                 hypre_ParCSRMatrix *G)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> G = G;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetCoordinateVectors
 *
 * Set the x, y and z coordinates of the vertices in the mesh.
 *
 * Either SetCoordinateVectors or SetEdgeConstantVectors should be
 * called before hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

int hypre_AMSSetCoordinateVectors(void *solver,
                                  hypre_ParVector *x,
                                  hypre_ParVector *y,
                                  hypre_ParVector *z)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> x = x;
   ams_data -> y = y;
   ams_data -> z = z;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetEdgeConstantVectors
 *
 * Set the vectors Gx, Gy and Gz which give the representations of
 * the constant vector fields (1,0,0), (0,1,0) and (0,0,1) in the
 * edge element basis.
 *
 * Either SetCoordinateVectors or SetEdgeConstantVectors should be
 * called before hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

int hypre_AMSSetEdgeConstantVectors(void *solver,
                                    hypre_ParVector *Gx,
                                    hypre_ParVector *Gy,
                                    hypre_ParVector *Gz)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> Gx = Gx;
   ams_data -> Gy = Gy;
   ams_data -> Gz = Gz;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetAlphaPoissonMatrix
 *
 * Set the matrix corresponding to the Poisson problem with coefficient
 * alpha (the curl-curl term coefficient in the Maxwell problem).
 *
 * If this function is called, the coarse space solver on the range
 * of Pi^T is a block-diagonal version of A_Pi. If this function is not
 * called, the coarse space solver on the range of Pi^T is constructed
 * as Pi^T A Pi in hypre_AMSSetup().
 *--------------------------------------------------------------------------*/

int hypre_AMSSetAlphaPoissonMatrix(void *solver,
                                   hypre_ParCSRMatrix *A_Pi)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> A_Pi = A_Pi;

   /* Penalize the eliminated degrees of freedom */
   hypre_ParCSRMatrixSetDiagRows(A_Pi, DBL_MAX);

   /* Make sure that the first entry in each row is the diagonal one. */
   /* hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A_Pi)); */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetBetaPoissonMatrix
 *
 * Set the matrix corresponding to the Poisson problem with coefficient
 * beta (the mass term coefficient in the Maxwell problem).
 *
 * This function call is optional - if not given, the Poisson matrix will
 * be computed in hypre_AMSSetup(). If the given matrix is NULL, we assume
 * that beta is 0 and use two-level (instead of three-level) methods.
 *--------------------------------------------------------------------------*/

int hypre_AMSSetBetaPoissonMatrix(void *solver,
                                  hypre_ParCSRMatrix *A_G)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> A_G = A_G;
   if (!A_G)
      ams_data -> beta_is_zero = 1;
   else
   {
      /* Penalize the eliminated degrees of freedom */
      hypre_ParCSRMatrixSetDiagRows(A_G, DBL_MAX);

      /* Make sure that the first entry in each row is the diagonal one. */
      /* hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A_G)); */
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetMaxIter
 *
 * Set the maximum number of iterations in the three-level method.
 * The default value is 20. To use the AMS solver as a preconditioner,
 * set maxit to 1, tol to 0.0 and print_level to 0.
 *--------------------------------------------------------------------------*/

int hypre_AMSSetMaxIter(void *solver,
                        int maxit)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> maxit = maxit;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetTol
 *
 * Set the convergence tolerance (if the method is used as a solver).
 * The default value is 1e-6.
 *--------------------------------------------------------------------------*/

int hypre_AMSSetTol(void *solver,
                    double tol)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> tol = tol;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetCycleType
 *
 * Choose which three-level solver to use. Possible values are:
 *
 *   1 = 3-level multipl. solver (01210)    <-- small solution time
 *   3 = 3-level multipl. solver (02120)
 *   5 = 3-level multipl. solver (0102010)  <-- small solution time
 *   7 = 3-level multipl. solver (0201020)  <-- small number of iterations
 *
 *   2 = 3-level additive solver (0+1+2)
 *   4 = 3-level additive solver (010+2)
 *   6 = 3-level additive solver (1+020)
 *   8 = 3-level additive solver (010+020)
 *
 *   0 = just the smoother (0)
 *
 * The default value is 1.
 *--------------------------------------------------------------------------*/

int hypre_AMSSetCycleType(void *solver,
                          int cycle_type)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> cycle_type = cycle_type;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetPrintLevel
 *
 * Control how much information is printed during the solution iterations.
 * The defaut values is 1 (print residual norm at each step).
 *--------------------------------------------------------------------------*/

int hypre_AMSSetPrintLevel(void *solver,
                           int print_level)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> print_level = print_level;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetSmoothingOptions
 *
 * Set relaxation parameters for A. Default values: 2, 1, 1.0, 1.0.
 *--------------------------------------------------------------------------*/

int hypre_AMSSetSmoothingOptions(void *solver,
                                 int A_relax_type,
                                 int A_relax_times,
                                 double A_relax_weight,
                                 double A_omega)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> A_relax_type = A_relax_type;
   ams_data -> A_relax_times = A_relax_times;
   ams_data -> A_relax_weight = A_relax_weight;
   ams_data -> A_omega = A_omega;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetAlphaAMGOptions
 *
 * Set AMG parameters for B_Pi. Default values: 10, 1, 3, 0.25.
 *--------------------------------------------------------------------------*/

int hypre_AMSSetAlphaAMGOptions(void *solver,
                                int B_Pi_coarsen_type,
                                int B_Pi_agg_levels,
                                int B_Pi_relax_type,
                                double B_Pi_theta)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> B_Pi_coarsen_type = B_Pi_coarsen_type;
   ams_data -> B_Pi_agg_levels = B_Pi_agg_levels;
   ams_data -> B_Pi_relax_type = B_Pi_relax_type;
   ams_data -> B_Pi_theta = B_Pi_theta;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetBetaAMGOptions
 *
 * Set AMG parameters for B_G. Default values: 10, 1, 3, 0.25.
 *--------------------------------------------------------------------------*/

int hypre_AMSSetBetaAMGOptions(void *solver,
                               int B_G_coarsen_type,
                               int B_G_agg_levels,
                               int B_G_relax_type,
                               double B_G_theta)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> B_G_coarsen_type = B_G_coarsen_type;
   ams_data -> B_G_agg_levels = B_G_agg_levels;
   ams_data -> B_G_relax_type = B_G_relax_type;
   ams_data -> B_G_theta = B_G_theta;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSComputePi
 *
 * Construct the Pi interpolation matrix, which maps the space of vector
 * linear finite elements to the space of edge finite elements.

 * The construction is based on the fact that Pi = [Pi_x, Pi_y, Pi_z],
 * where each block has the same sparsity structure as G, and the entries
 * can be computed from the vectors Gx, Gy, Gz.
 *--------------------------------------------------------------------------*/

int hypre_AMSComputePi(hypre_ParCSRMatrix *A,
                       hypre_ParCSRMatrix *G,
                       hypre_ParVector *x,
                       hypre_ParVector *y,
                       hypre_ParVector *z,
                       hypre_ParVector *Gx,
                       hypre_ParVector *Gy,
                       hypre_ParVector *Gz,
                       int dim,
                       hypre_ParCSRMatrix **Pi_ptr)
{
   int input_info = 0;

   hypre_ParCSRMatrix *Pi;

   if (x != NULL && y != NULL && (dim == 2 || z != NULL))
      input_info = 1;

   if (Gx != NULL && Gy != NULL && (dim == 2 || Gz != NULL))
      input_info = 2;

   if (!input_info)
      hypre_error_in_arg(3);

   /* If not given, compute Gx, Gy and Gz */
   if (input_info == 1)
   {
      Gx = hypre_ParVectorInRangeOf(G);
      hypre_ParCSRMatrixMatvec (1.0, G, x, 0.0, Gx);
      Gy = hypre_ParVectorInRangeOf(G);
      hypre_ParCSRMatrixMatvec (1.0, G, y, 0.0, Gy);
      if (dim == 3)
      {
         Gz = hypre_ParVectorInRangeOf(G);
         hypre_ParCSRMatrixMatvec (1.0, G, z, 0.0, Gz);
      }
   }

   /* Compute Pi = [Pi_x, Pi_y, Pi_z] */
   {
      int i, j, d;

      double *Gx_data, *Gy_data, *Gz_data;

      MPI_Comm comm = hypre_ParCSRMatrixComm(G);
      int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(G);
      int global_num_cols = dim*hypre_ParCSRMatrixGlobalNumCols(G);
      int *row_starts = hypre_ParCSRMatrixRowStarts(G);
      int col_starts_size, *col_starts;
      int num_cols_offd = dim*hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(G));
      int num_nonzeros_diag = dim*hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(G));
      int num_nonzeros_offd = dim*hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(G));
      int *col_starts_G = hypre_ParCSRMatrixColStarts(G);
#ifdef HYPRE_NO_GLOBAL_PARTITION
      col_starts_size = 2;
#else
      int num_procs;
      MPI_Comm_size(comm, &num_procs);
      col_starts_size = num_procs+1;
#endif
      col_starts = hypre_TAlloc(int,col_starts_size);
      for (i = 0; i < col_starts_size; i++)
         col_starts[i] = dim * col_starts_G[i];

      Pi = hypre_ParCSRMatrixCreate(comm,
                                    global_num_rows,
                                    global_num_cols,
                                    row_starts,
                                    col_starts,
                                    num_cols_offd,
                                    num_nonzeros_diag,
                                    num_nonzeros_offd);

      hypre_ParCSRMatrixOwnsData(Pi) = 1;
      hypre_ParCSRMatrixOwnsRowStarts(Pi) = 0;
      hypre_ParCSRMatrixOwnsColStarts(Pi) = 1;

      hypre_ParCSRMatrixInitialize(Pi);

      Gx_data = hypre_VectorData(hypre_ParVectorLocalVector(Gx));
      Gy_data = hypre_VectorData(hypre_ParVectorLocalVector(Gy));
      if (dim == 3)
         Gz_data = hypre_VectorData(hypre_ParVectorLocalVector(Gz));

      /* Fill-in the diagonal part */
      {
         hypre_CSRMatrix *G_diag = hypre_ParCSRMatrixDiag(G);
         int *G_diag_I = hypre_CSRMatrixI(G_diag);
         int *G_diag_J = hypre_CSRMatrixJ(G_diag);

         int G_diag_nrows = hypre_CSRMatrixNumRows(G_diag);
         int G_diag_nnz = hypre_CSRMatrixNumNonzeros(G_diag);

         hypre_CSRMatrix *Pi_diag = hypre_ParCSRMatrixDiag(Pi);
         int *Pi_diag_I = hypre_CSRMatrixI(Pi_diag);
         int *Pi_diag_J = hypre_CSRMatrixJ(Pi_diag);
         double *Pi_diag_data = hypre_CSRMatrixData(Pi_diag);

         for (i = 0; i < G_diag_nrows+1; i++)
            Pi_diag_I[i] = dim * G_diag_I[i];

         for (i = 0; i < G_diag_nnz; i++)
            for (d = 0; d < dim; d++)
               Pi_diag_J[dim*i+d] = dim*G_diag_J[i]+d;

         for (i = 0; i < G_diag_nrows; i++)
            for (j = G_diag_I[i]; j < G_diag_I[i+1]; j++)
            {
               *Pi_diag_data++ = 0.5 * Gx_data[i];
               *Pi_diag_data++ = 0.5 * Gy_data[i];
               if (dim == 3)
                  *Pi_diag_data++ = 0.5 * Gz_data[i];
            }
      }

      /* Fill-in the off-diagonal part */
      {
         hypre_CSRMatrix *G_offd = hypre_ParCSRMatrixOffd(G);
         int *G_offd_I = hypre_CSRMatrixI(G_offd);
         int *G_offd_J = hypre_CSRMatrixJ(G_offd);

         int G_offd_nrows = hypre_CSRMatrixNumRows(G_offd);
         int G_offd_ncols = hypre_CSRMatrixNumCols(G_offd);
         int G_offd_nnz = hypre_CSRMatrixNumNonzeros(G_offd);

         hypre_CSRMatrix *Pi_offd = hypre_ParCSRMatrixOffd(Pi);
         int *Pi_offd_I = hypre_CSRMatrixI(Pi_offd);
         int *Pi_offd_J = hypre_CSRMatrixJ(Pi_offd);
         double *Pi_offd_data = hypre_CSRMatrixData(Pi_offd);

         int *G_cmap = hypre_ParCSRMatrixColMapOffd(G);
         int *Pi_cmap = hypre_ParCSRMatrixColMapOffd(Pi);

         if (G_offd_ncols)
            for (i = 0; i < G_offd_nrows+1; i++)
               Pi_offd_I[i] = dim * G_offd_I[i];

         for (i = 0; i < G_offd_nnz; i++)
            for (d = 0; d < dim; d++)
               Pi_offd_J[dim*i+d] = dim*G_offd_J[i]+d;

         for (i = 0; i < G_offd_nrows; i++)
            for (j = G_offd_I[i]; j < G_offd_I[i+1]; j++)
            {
               *Pi_offd_data++ = 0.5 * Gx_data[i];
               *Pi_offd_data++ = 0.5 * Gy_data[i];
               if (dim == 3)
                  *Pi_offd_data++ = 0.5 * Gz_data[i];
            }

         for (i = 0; i < G_offd_ncols; i++)
            for (d = 0; d < dim; d++)
               Pi_cmap[dim*i+d] = dim*G_cmap[i]+d;
      }

   }

   if (input_info == 1)
   {
      hypre_ParVectorDestroy(Gx);
      hypre_ParVectorDestroy(Gy);
      if (dim == 3)
         hypre_ParVectorDestroy(Gz);
   }

   *Pi_ptr = Pi;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetup
 *
 * Construct the AMS solver components.
 *
 * The following functions need to be called before hypre_AMSSetup():
 * - hypre_AMSSetDimension() (if solving a 2D problem)
 * - hypre_AMSSetDiscreteGradient()
 * - hypre_AMSSetCoordinateVectors() or hypre_AMSSetEdgeConstantVectors
 *--------------------------------------------------------------------------*/

int hypre_AMSSetup(void *solver,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector *b,
                   hypre_ParVector *x)
{
   hypre_AMSData *ams_data = solver;

   ams_data -> A = A;

   /* Make sure that the first entry in each row is the diagonal one. */
   /* hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A)); */

   /* Compute the l1 norm of the rows of A */
   if (ams_data -> A_relax_type >= 1 && ams_data -> A_relax_type <= 2)
      hypre_ParCSRComputeL1Norms(A, ams_data -> A_relax_type,
                                 &ams_data -> A_l1_norms);

   /* Construct the Pi interpolation matrix */
   hypre_AMSComputePi(ams_data -> A,
                      ams_data -> G,
                      ams_data -> x,
                      ams_data -> y,
                      ams_data -> z,
                      ams_data -> Gx,
                      ams_data -> Gy,
                      ams_data -> Gz,
                      ams_data -> dim,
                      &ams_data -> Pi);

   /* Create the AMG solver on the range of G^T */
   if (!ams_data -> beta_is_zero)
   {
      HYPRE_BoomerAMGCreate(&ams_data -> B_G);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_G, ams_data -> B_G_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_G, ams_data -> B_G_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_G, ams_data -> B_G_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_G, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_G, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_G, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_G, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_G, ams_data -> B_G_theta);

      /* don't use exact solve on the coarsest level (matrix may be singular) */
      HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_G,
                                       ams_data -> B_G_relax_type,
                                       3);

      /* If not given, construct the coarse space matrix by RAP */
      if (!ams_data -> A_G)
      {
         if (!hypre_ParCSRMatrixCommPkg(ams_data -> G))
            hypre_MatvecCommPkgCreate(ams_data -> G);

         if (!hypre_ParCSRMatrixCommPkg(ams_data -> A))
            hypre_MatvecCommPkgCreate(ams_data -> A);

         hypre_BoomerAMGBuildCoarseOperator(ams_data -> G,
                                            ams_data -> A,
                                            ams_data -> G,
                                            &ams_data -> A_G);

         /* Make sure that A_G has no zero rows (this can happen
            if beta is zero in part of the domain). */
         /* hypre_ParCSRMatrixFixZeroRows(ams_data -> A_G); */

         ams_data -> owns_A_G = 1;
      }

      HYPRE_BoomerAMGSetup(ams_data -> B_G,
                           (HYPRE_ParCSRMatrix)ams_data -> A_G,
                           0, 0);
   }

   /* Create the AMG solver on the range of Pi^T */
   {
      HYPRE_BoomerAMGCreate(&ams_data -> B_Pi);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Pi, ams_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Pi, ams_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Pi, ams_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Pi, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pi, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_Pi, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Pi, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Pi, ams_data -> B_Pi_theta);

      /* don't use exact solve on the coarsest level (matrix may be singular) */
      HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_Pi,
                                       ams_data -> B_Pi_relax_type,
                                       3);

      /* If not given, construct the coarse space matrix by RAP and
         notify BoomerAMG that this is a dim x dim block system. */
      if (!ams_data -> A_Pi)
      {
         if (!hypre_ParCSRMatrixCommPkg(ams_data -> Pi))
            hypre_MatvecCommPkgCreate(ams_data -> Pi);

         hypre_BoomerAMGBuildCoarseOperator(ams_data -> Pi,
                                            ams_data -> A,
                                            ams_data -> Pi,
                                            &ams_data -> A_Pi);

         ams_data -> owns_A_Pi = 1;

         HYPRE_BoomerAMGSetNumFunctions(ams_data -> B_Pi, ams_data -> dim);
         /* HYPRE_BoomerAMGSetNodal(ams_data -> B_Pi, 1); */
      }

      HYPRE_BoomerAMGSetup(ams_data -> B_Pi,
                           (HYPRE_ParCSRMatrix)ams_data -> A_Pi,
                           0, 0);
   }

   /* Allocate temporary vectors */
   ams_data -> r0 = hypre_ParVectorInRangeOf(ams_data -> A);
   ams_data -> g0 = hypre_ParVectorInRangeOf(ams_data -> A);
   if (!ams_data -> beta_is_zero)
   {
      ams_data -> r1 = hypre_ParVectorInRangeOf(ams_data -> A_G);
      ams_data -> g1 = hypre_ParVectorInRangeOf(ams_data -> A_G);
   }
   ams_data -> r2 = hypre_ParVectorInDomainOf(ams_data -> Pi);
   ams_data -> g2 = hypre_ParVectorInDomainOf(ams_data -> Pi);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSolve
 *
 * Solve the system A x = b.
 *--------------------------------------------------------------------------*/

int hypre_AMSSolve(void *solver,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector *b,
                   hypre_ParVector *x)
{
   hypre_AMSData *ams_data = solver;
   int (*TwoLevelPrec)(hypre_ParCSRMatrix*,int,int,double*,double,double,hypre_ParCSRMatrix*,
                       HYPRE_Solver,hypre_ParCSRMatrix*,hypre_ParVector*,hypre_ParVector*,
                       hypre_ParVector*,hypre_ParVector*,hypre_ParVector*,hypre_ParVector*);
   int (*ThreeLevelPrec)(hypre_ParCSRMatrix*,int,int,double*,double,double,hypre_ParCSRMatrix*,
                         HYPRE_Solver,hypre_ParCSRMatrix*,hypre_ParCSRMatrix*,
                         HYPRE_Solver,hypre_ParCSRMatrix*,hypre_ParVector*,hypre_ParVector*,
                         hypre_ParVector*,hypre_ParVector*,hypre_ParVector*,hypre_ParVector*,
                         hypre_ParVector*,hypre_ParVector*,int);

   int i, my_id;
   double r0_norm, r_norm, b_norm, relative_resid = 0, old_resid;

   if (ams_data -> print_level > 0)
      MPI_Comm_rank(hypre_ParCSRMatrixComm(A), &my_id);

   if (ams_data -> cycle_type % 2 == 1)
   {
      TwoLevelPrec = hypre_TwoLevelParCSRMulPrec;
      ThreeLevelPrec = hypre_ThreeLevelParCSRMulPrec;
   }
   else if (ams_data -> cycle_type %2 == 0)
   {
      TwoLevelPrec = hypre_TwoLevelParCSRAddPrec;
      ThreeLevelPrec = hypre_ThreeLevelParCSRAddPrec;
   }

   for (i = 0; i < ams_data -> maxit; i++)
   {
      /* Compute initial residual norms */
      if (ams_data -> maxit > 1 && i == 0)
      {
         hypre_ParVectorCopy(b, ams_data -> r0);
         hypre_ParCSRMatrixMatvec(-1.0, ams_data -> A, x, 1.0, ams_data -> r0);
         r_norm = sqrt(hypre_ParVectorInnerProd(ams_data -> r0,ams_data -> r0));
         r0_norm = r_norm;
         b_norm = sqrt(hypre_ParVectorInnerProd(b, b));
         if (b_norm)
            relative_resid = r_norm / b_norm;
         else
            relative_resid = r_norm;
         if (my_id == 0 && ams_data -> print_level > 0)
         {
            printf("                                            relative\n");
            printf("               residual        factor       residual\n");
            printf("               --------        ------       --------\n");
            printf("    Initial    %e                 %e\n",
                   r_norm, relative_resid);
         }
      }

      /* Apply the preconditioner */
      if (ams_data -> beta_is_zero)
         (*TwoLevelPrec) (ams_data -> A,
                          ams_data -> A_relax_type,
                          ams_data -> A_relax_times,
                          ams_data -> A_l1_norms,
                          ams_data -> A_relax_weight,
                          ams_data -> A_omega,
                          ams_data -> A_Pi,
                          ams_data -> B_Pi,
                          ams_data -> Pi,
                          b, x,
                          ams_data -> r0,
                          ams_data -> r2,
                          ams_data -> g0,
                          ams_data -> g2);
      else
         (*ThreeLevelPrec) (ams_data -> A,
                            ams_data -> A_relax_type,
                            ams_data -> A_relax_times,
                            ams_data -> A_l1_norms,
                            ams_data -> A_relax_weight,
                            ams_data -> A_omega,
                            ams_data -> A_G,
                            ams_data -> B_G,
                            ams_data -> G,
                            ams_data -> A_Pi,
                            ams_data -> B_Pi,
                            ams_data -> Pi,
                            b, x,
                            ams_data -> r0,
                            ams_data -> r1,
                            ams_data -> r2,
                            ams_data -> g0,
                            ams_data -> g1,
                            ams_data -> g2,
                            ams_data -> cycle_type);

      /* Compute new residual norms */
      if (ams_data -> maxit > 1)
      {
         old_resid = r_norm;
         hypre_ParVectorCopy(b, ams_data -> r0);
         hypre_ParCSRMatrixMatvec(-1.0, ams_data -> A, x, 1.0, ams_data -> r0);
         r_norm = sqrt(hypre_ParVectorInnerProd(ams_data -> r0,ams_data -> r0));
         if (b_norm)
            relative_resid = r_norm / b_norm;
         else
            relative_resid = r_norm;
         if (my_id == 0 && ams_data -> print_level > 0)
            printf("    Cycle %2d   %e    %f     %e \n",
                   i+1, r_norm, r_norm / old_resid, relative_resid);
      }

      if (relative_resid < ams_data -> tol)
      {
         i++;
         break;
      }
   }

   if (my_id == 0 && ams_data -> print_level > 0 && ams_data -> maxit > 1)
      printf("\n\n Average Convergence Factor = %f\n\n",
             pow((r_norm/r0_norm),(1.0/(double) i)));

   ams_data -> num_iterations = i;
   ams_data -> rel_resid_norm = relative_resid;

   if (ams_data -> num_iterations == ams_data -> maxit && ams_data -> tol > 0.0)
      hypre_error(HYPRE_ERROR_CONV);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_TwoLevelParCSRMulPrec
 *
 * Two-level (symmetric) multiplicative preconditioner.
 * All operations are based on ParCSR matrices and BoomerAMG.
 *--------------------------------------------------------------------------*/

int hypre_TwoLevelParCSRMulPrec(/* fine space matrix */
                                hypre_ParCSRMatrix *A0,
                                /* relaxation parameters */
                                int A0_relax_type,
                                int A0_relax_times,
                                double *A0_l1_norms,
                                double A0_relax_weight,
                                double A0_omega,
                                /* coarse space matrix */
                                hypre_ParCSRMatrix *A1,
                                /* coarse space preconditioner */
                                HYPRE_Solver B1,
                                /* coarse-to-fine interpolation */
                                hypre_ParCSRMatrix *P1,
                                /* input */
                                hypre_ParVector *x,
                                /* input/output */
                                hypre_ParVector *y,
                                /* temporary vectors */
                                hypre_ParVector *r0,
                                hypre_ParVector *r1,
                                hypre_ParVector *g0,
                                hypre_ParVector *g1)
{
   /* pre-smooth: y += S (x - Ay) */
   hypre_ParCSRRelax(A0, x,
                     A0_relax_type,
                     A0_relax_times,
                     A0_l1_norms,
                     A0_relax_weight,
                     A0_omega,
                     y, r0);

   /* coarse grid correction: y += P B^{-1} P^t (x - Ay) */
   hypre_ParVectorCopy(x,r0);
   hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
   hypre_ParCSRMatrixMatvecT(1.0, P1, r0, 0.0, r1);
   hypre_ParVectorSetConstantValues(g1, 0.0);
   hypre_BoomerAMGBlockSolve((void *)B1, A1, r1, g1);
   hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, r0);
   hypre_ParVectorAxpy(1.0, r0, y);

   /* post-smooth: y += S (x - Ay) */
   hypre_ParCSRRelax(A0, x,
                     A0_relax_type,
                     A0_relax_times,
                     A0_l1_norms,
                     A0_relax_weight,
                     A0_omega,
                     y, r0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_TwoLevelParCSRAddPrec
 *
 * Two-level additive preconditioner.
 * All operations are based on ParCSR matrices and BoomerAMG.
 *--------------------------------------------------------------------------*/

int hypre_TwoLevelParCSRAddPrec(/* fine space matrix */
                                hypre_ParCSRMatrix *A0,
                                /* relaxation parameters */
                                int A0_relax_type,
                                int A0_relax_times,
                                double *A0_l1_norms,
                                double A0_relax_weight,
                                double A0_omega,
                                /* coarse space matrix */
                                hypre_ParCSRMatrix *A1,
                                /* coarse space preconditioner */
                                HYPRE_Solver B1,
                                /* coarse-to-fine interpolation */
                                hypre_ParCSRMatrix *P1,
                                /* input */
                                hypre_ParVector *x,
                                /* input/output */
                                hypre_ParVector *y,
                                /* temporary vectors */
                                hypre_ParVector *r0,
                                hypre_ParVector *r1,
                                hypre_ParVector *g0,
                                hypre_ParVector *g1)
{
   /* compute the residual: r0 = x - Ay */
   hypre_ParVectorCopy(x,r0);
   hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);

   /* add smoother correction: y += S r0 */
   hypre_ParCSRRelax(A0, x,
                     A0_relax_type,
                     A0_relax_times,
                     A0_l1_norms,
                     A0_relax_weight,
                     A0_omega,
                     y, g0);

   /* add coarse grid correction: y += P B^{-1} P^t r0 */
   hypre_ParCSRMatrixMatvecT(1.0, P1, r0, 0.0, r1);
   hypre_ParVectorSetConstantValues(g1, 0.0);
   hypre_BoomerAMGBlockSolve((void *)B1, A1, r1, g1);
   hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, r0);
   hypre_ParVectorAxpy(1.0, r0, y);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ThreeLevelParCSRMulPrec
 *
 * Three-level (symmetric) multiplicative preconditioner.
 * All operations are based on ParCSR matrices and (block) BoomerAMG.
 *--------------------------------------------------------------------------*/

int hypre_ThreeLevelParCSRMulPrec(/* fine space matrix */
                                  hypre_ParCSRMatrix *A0,
                                  /* relaxation parameters */
                                  int A0_relax_type,
                                  int A0_relax_times,
                                  double *A0_l1_norms,
                                  double A0_relax_weight,
                                  double A0_omega,
                                  /* coarse space matrix */
                                  hypre_ParCSRMatrix *A1,
                                  /* coarse space preconditioner */
                                  HYPRE_Solver B1,
                                  /* coarse-to-fine interpolation */
                                  hypre_ParCSRMatrix *P1,
                                  /* second coarse space matrix */
                                  hypre_ParCSRMatrix *A2,
                                  /* second coarse space preconditioner */
                                  HYPRE_Solver B2,
                                  /* second coarse-to-fine interpolation */
                                  hypre_ParCSRMatrix *P2,
                                  /* input */
                                  hypre_ParVector *x,
                                  /* input/output */
                                  hypre_ParVector *y,
                                  /* temporary vectors */
                                  hypre_ParVector *r0,
                                  hypre_ParVector *r1,
                                  hypre_ParVector *r2,
                                  hypre_ParVector *g0,
                                  hypre_ParVector *g1,
                                  hypre_ParVector *g2,
                                  int cycle_type)
{
   if (cycle_type == 1)
   {
      /* pre-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);

      /* first coarse grid correction: y += P1 B1^{-1} P1^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P1, r0, 0.0, r1);
      hypre_ParVectorSetConstantValues(g1, 0.0);
      hypre_BoomerAMGSolve((void *)B1, A1, r1, g1);
      hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* second coarse grid correction: y += P2 B2^{-1} P2^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P2, r0, 0.0, r2);
      hypre_ParVectorSetConstantValues(g2, 0.0);
      hypre_BoomerAMGBlockSolve((void *)B2, A2, r2, g2);
      hypre_ParCSRMatrixMatvec(1.0, P2, g2, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* first coarse grid correction: y += P1 B1^{-1} P1^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P1, r0, 0.0, r1);
      hypre_ParVectorSetConstantValues(g1, 0.0);
      hypre_BoomerAMGSolve((void *)B1, A1, r1, g1);
      hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* post-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);
   }
   else if (cycle_type == 3)
   {
      /* pre-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);

      /* second coarse grid correction: y += P2 B2^{-1} P2^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P2, r0, 0.0, r2);
      hypre_ParVectorSetConstantValues(g2, 0.0);
      hypre_BoomerAMGBlockSolve((void *)B2, A2, r2, g2);
      hypre_ParCSRMatrixMatvec(1.0, P2, g2, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* first coarse grid correction: y += P1 B1^{-1} P1^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P1, r0, 0.0, r1);
      hypre_ParVectorSetConstantValues(g1, 0.0);
      hypre_BoomerAMGSolve((void *)B1, A1, r1, g1);
      hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* second coarse grid correction: y += P2 B2^{-1} P2^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P2, r0, 0.0, r2);
      hypre_ParVectorSetConstantValues(g2, 0.0);
      hypre_BoomerAMGBlockSolve((void *)B2, A2, r2, g2);
      hypre_ParCSRMatrixMatvec(1.0, P2, g2, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* post-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);
   }
   else if (cycle_type == 5)
   {
      /* pre-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);

      /* first coarse grid correction: y += P1 B1^{-1} P1^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P1, r0, 0.0, r1);
      hypre_ParVectorSetConstantValues(g1, 0.0);
      hypre_BoomerAMGSolve((void *)B1, A1, r1, g1);
      hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* extra smoothing: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);

      /* second coarse grid correction: y += P2 B2^{-1} P2^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P2, r0, 0.0, r2);
      hypre_ParVectorSetConstantValues(g2, 0.0);
      hypre_BoomerAMGBlockSolve((void *)B2, A2, r2, g2);
      hypre_ParCSRMatrixMatvec(1.0, P2, g2, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* extra smoothing: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);

      /* first coarse grid correction: y += P1 B1^{-1} P1^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P1, r0, 0.0, r1);
      hypre_ParVectorSetConstantValues(g1, 0.0);
      hypre_BoomerAMGSolve((void *)B1, A1, r1, g1);
      hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* post-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);
   }
   else if (cycle_type == 7)
   {
      /* pre-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);

      /* second coarse grid correction: y += P2 B2^{-1} P2^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P2, r0, 0.0, r2);
      hypre_ParVectorSetConstantValues(g2, 0.0);
      hypre_BoomerAMGBlockSolve((void *)B2, A2, r2, g2);
      hypre_ParCSRMatrixMatvec(1.0, P2, g2, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* extra smoothing: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);

      /* first coarse grid correction: y += P1 B1^{-1} P1^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P1, r0, 0.0, r1);
      hypre_ParVectorSetConstantValues(g1, 0.0);
      hypre_BoomerAMGSolve((void *)B1, A1, r1, g1);
      hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* extra smoothing: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);

      /* second coarse grid correction: y += P2 B2^{-1} P2^t (x - Ay) */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      hypre_ParCSRMatrixMatvecT(1.0, P2, r0, 0.0, r2);
      hypre_ParVectorSetConstantValues(g2, 0.0);
      hypre_BoomerAMGBlockSolve((void *)B2, A2, r2, g2);
      hypre_ParCSRMatrixMatvec(1.0, P2, g2, 0.0, r0);
      hypre_ParVectorAxpy(1.0, r0, y);

      /* post-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, r0);
   }
   else
      hypre_error_in_arg(19);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ThreeLevelParCSRAddPrec
 *
 * Three-level additive preconditioner.
 * All operations are based on ParCSR matrices and (block) BoomerAMG.
 *--------------------------------------------------------------------------*/

int hypre_ThreeLevelParCSRAddPrec(/* fine space matrix */
                                  hypre_ParCSRMatrix *A0,
                                  /* relaxation parameters */
                                  int A0_relax_type,
                                  int A0_relax_times,
                                  double *A0_l1_norms,
                                  double A0_relax_weight,
                                  double A0_omega,
                                  /* first coarse space matrix */
                                  hypre_ParCSRMatrix *A1,
                                  /* first coarse space preconditioner */
                                  HYPRE_Solver B1,
                                  /* first coarse-to-fine interpolation */
                                  hypre_ParCSRMatrix *P1,
                                  /* second coarse space matrix */
                                  hypre_ParCSRMatrix *A2,
                                  /* second coarse space preconditioner */
                                  HYPRE_Solver B2,
                                  /* second coarse-to-fine interpolation */
                                  hypre_ParCSRMatrix *P2,
                                  /* input */
                                  hypre_ParVector *x,
                                  /* input/output */
                                  hypre_ParVector *y,
                                  /* temporary vectors */
                                  hypre_ParVector *r0,
                                  hypre_ParVector *r1,
                                  hypre_ParVector *r2,
                                  hypre_ParVector *g0,
                                  hypre_ParVector *g1,
                                  hypre_ParVector *g2,
                                  int cycle_type)
{
   if (cycle_type == 0)
   {
      /* apply smoother: y += S r0 */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, g0);
   }
   else if (cycle_type == 2)
   {
      /* compute the residual: r0 = x - Ay */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);

      /* add smoother correction: y += S r0 */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, g0);

      /* add first coarse grid correction: y += P1 B1^{-1} P1^t r0 */
      hypre_ParCSRMatrixMatvecT(1.0, P1, r0, 0.0, r1);
      hypre_ParVectorSetConstantValues(g1, 0.0);
      hypre_BoomerAMGSolve((void *)B1, A1, r1, g1);
      hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, g0);
      hypre_ParVectorAxpy(1.0, g0, y);

      /* add second coarse grid correction: y += P2 B2^{-1} P2^t r0 */
      hypre_ParCSRMatrixMatvecT(1.0, P2, r0, 0.0, r2);
      hypre_ParVectorSetConstantValues(g2, 0.0);
      hypre_BoomerAMGBlockSolve((void *)B2, A2, r2, g2);
      hypre_ParCSRMatrixMatvec(1.0, P2, g2, 0.0, g0);
      hypre_ParVectorAxpy(1.0, g0, y);
   }
   else if (cycle_type == 4)
   {
      /* compute the residual: r0 = x - Ay */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);

      /* pre-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, g0);

      /* first coarse grid correction: y += P1 B1^{-1} P1^t (x - Ay) */
      hypre_ParVectorCopy(x,g0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, g0);
      hypre_ParCSRMatrixMatvecT(1.0, P1, g0, 0.0, r1);
      hypre_ParVectorSetConstantValues(g1, 0.0);
      hypre_BoomerAMGSolve((void *)B1, A1, r1, g1);
      hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, g0);
      hypre_ParVectorAxpy(1.0, g0, y);

      /* post-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, g0);

      /* add second coarse grid correction: y += P2 B2^{-1} P2^t r0 */
      hypre_ParCSRMatrixMatvecT(1.0, P2, r0, 0.0, r2);
      hypre_ParVectorSetConstantValues(g2, 0.0);
      hypre_BoomerAMGBlockSolve((void *)B2, A2, r2, g2);
      hypre_ParCSRMatrixMatvec(1.0, P2, g2, 0.0, g0);
      hypre_ParVectorAxpy(1.0, g0, y);
   }
   else if (cycle_type == 6)
   {
      /* compute the residual: r0 = x - Ay */
      hypre_ParVectorCopy(x,r0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);

      /* pre-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, g0);

      /* second coarse grid correction: y += P2 B2^{-1} P2^t (x - Ay) */
      hypre_ParVectorCopy(x,g0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, g0);
      hypre_ParCSRMatrixMatvecT(1.0, P2, g0, 0.0, r2);
      hypre_ParVectorSetConstantValues(g2, 0.0);
      hypre_BoomerAMGBlockSolve((void *)B2, A2, r2, g2);
      hypre_ParCSRMatrixMatvec(1.0, P2, g2, 0.0, g0);
      hypre_ParVectorAxpy(1.0, g0, y);

      /* post-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, g0);

      /* add first coarse grid correction: y += P1 B1^{-1} P1^t r0 */
      hypre_ParCSRMatrixMatvecT(1.0, P1, r0, 0.0, r1);
      hypre_ParVectorSetConstantValues(g1, 0.0);
      hypre_BoomerAMGSolve((void *)B1, A1, r1, g1);
      hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, g0);
      hypre_ParVectorAxpy(1.0, g0, y);
   }
   else if (cycle_type == 8)
   {
      /* r0 = y */
      hypre_ParVectorCopy(y,r0);

      /* pre-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, g0);

      /* first coarse grid correction: y += P1 B1^{-1} P1^t (x - Ay) */
      hypre_ParVectorCopy(x,g0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, g0);
      hypre_ParCSRMatrixMatvecT(1.0, P1, g0, 0.0, r1);
      hypre_ParVectorSetConstantValues(g1, 0.0);
      hypre_BoomerAMGSolve((void *)B1, A1, r1, g1);
      hypre_ParCSRMatrixMatvec(1.0, P1, g1, 0.0, g0);
      hypre_ParVectorAxpy(1.0, g0, y);

      /* post-smooth: y += S (x - Ay) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        y, g0);

      /* pre-smooth: r0 += S (x - A r0) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        r0, g0);

      /* second coarse grid correction: r0 += P2 B2^{-1} P2^t (x - A r0) */
      hypre_ParVectorCopy(x,g0);
      hypre_ParCSRMatrixMatvec(-1.0, A0, r0, 1.0, g0);
      hypre_ParCSRMatrixMatvecT(1.0, P2, g0, 0.0, r2);
      hypre_ParVectorSetConstantValues(g2, 0.0);
      hypre_BoomerAMGBlockSolve((void *)B2, A2, r2, g2);
      hypre_ParCSRMatrixMatvec(1.0, P2, g2, 0.0, g0);
      hypre_ParVectorAxpy(1.0, g0, r0);

      /* post-smooth: r0 += S (x - A r0) */
      hypre_ParCSRRelax(A0, x,
                        A0_relax_type,
                        A0_relax_times,
                        A0_l1_norms,
                        A0_relax_weight,
                        A0_omega,
                        r0, g0);

      /* y += r0 */
      hypre_ParVectorAxpy(1.0, r0, y);
   }
   else
      hypre_error_in_arg(19);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSGetNumIterations
 *
 * Get the number of AMS iterations.
 *--------------------------------------------------------------------------*/

int hypre_AMSGetNumIterations(void *solver,
                              int *num_iterations)
{
   hypre_AMSData *ams_data = solver;
   *num_iterations = ams_data -> num_iterations;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSGetFinalRelativeResidualNorm
 *
 * Get the final relative residual norm in AMS.
 *--------------------------------------------------------------------------*/

int hypre_AMSGetFinalRelativeResidualNorm(void *solver,
                                          double *rel_resid_norm)
{
   hypre_AMSData *ams_data = solver;
   *rel_resid_norm = ams_data -> rel_resid_norm;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSConstructDiscreteGradient
 *
 * Construct and return the discrete gradient matrix G, based on:
 * - a matrix on the egdes (e.g. the stiffness matrix A)
 * - a vector on the vertices (e.g. the x coordinates)
 * - the array edge_vertex, which lists the global indexes of the
 *   vertices of the local edges.
 *
 * We assume that edge_vertex lists the edge vertices consecutively,
 * and that the orientation of edge i depends only on the sign of
 * edge_vertex[2*i+1] - edge_vertex[2*i].
 *
 * Warning: G steals the (row) partitionings of A and x_coord. This may
 * break some code, but is necessery since the user is responsible for
 * destroying the output matrix.
 *--------------------------------------------------------------------------*/

int hypre_AMSConstructDiscreteGradient(hypre_ParCSRMatrix *A,
                                       hypre_ParVector *x_coord,
                                       int *edge_vertex,
                                       hypre_ParCSRMatrix **G_ptr)
{
   hypre_ParCSRMatrix *G;

   int nedges, vxstart, vxend, nvert;

   nedges = hypre_ParCSRMatrixNumRows(A);

   vxstart = hypre_ParVectorFirstIndex(x_coord);
   vxend = hypre_ParVectorLastIndex(x_coord);
   nvert = vxend - vxstart + 1;

   /* Construct the local part of G based on edge_vertex and the edge
      and vertex partitionings from A and x_coord */
   {
      int i, *I = hypre_CTAlloc(int, nedges+1);
      double *data = hypre_CTAlloc(double, 2*nedges);
      hypre_CSRMatrix *local = hypre_CSRMatrixCreate (nedges,
                                                      hypre_ParVectorGlobalSize(x_coord),
                                                      2*nedges);

      for (i = 0; i <= nedges; i++)
         I[i] = 2*i;

      /* Assume that the edge orientation is based on the vertex indexes */
      for (i = 0; i < 2*nedges; i+=2)
      {
         if (edge_vertex[i] < edge_vertex[i+1])
         {
            data[i]   = -1.0;
            data[i+1] =  1.0;
         }
         else
         {
            data[i]   =  1.0;
            data[i+1] = -1.0;
         }
      }

      hypre_CSRMatrixI(local) = I;
      hypre_CSRMatrixJ(local) = edge_vertex;
      hypre_CSRMatrixData(local) = data;

      hypre_CSRMatrixRownnz(local) = NULL;
      hypre_CSRMatrixOwnsData(local) = 1;
      hypre_CSRMatrixNumRownnz(local) = nedges;

      G = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                   hypre_ParCSRMatrixGlobalNumRows(A),
                                   hypre_ParVectorGlobalSize(x_coord),
                                   hypre_ParCSRMatrixRowStarts(A),
                                   hypre_ParVectorPartitioning(x_coord),
                                   0, 0, 0);

      hypre_ParCSRMatrixOwnsRowStarts(A) = 0;
      hypre_ParVectorOwnsPartitioning(x_coord) = 0;
      hypre_ParCSRMatrixOwnsRowStarts(G) = 1;
      hypre_ParCSRMatrixOwnsColStarts(G) = 1;

      GenerateDiagAndOffd(local, G,
                          hypre_ParVectorFirstIndex(x_coord),
                          hypre_ParVectorLastIndex(x_coord));

      hypre_CSRMatrixJ(local) = NULL;
      hypre_CSRMatrixDestroy(local);
   }

   *G_ptr = G;

   return hypre_error_flag;
}
