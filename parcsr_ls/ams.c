/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
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
 * 2 = l1-scaled block Gauss-Seidel/SSOR
 * 3 = Kaczmarz
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
                      /* damping coefficient (usually <= 1) */
                      double relax_weight,
                      /* SOR parameter (usually in (0,2) */
                      double omega,
                      /* initial/updated approximation */
                      hypre_ParVector *u,
                      /* temporary vector */
                      hypre_ParVector *v,
                      /* temporary vector */
                      hypre_ParVector *z )
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
         hypre_ParCSRMatrixMatvec(-relax_weight, A, u, relax_weight, v);

         /* u += w D^{-1}(f - A u), where D_ii = ||A(i,:)||_1 */
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

         if (relax_weight == 1.0 && omega == 1.0) /* symmetric Gauss-Seidel */
         {
            /* Forward local pass */
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
            /* Backward local pass */
            for (i = num_rows-1; i > -1; i--)
            {
               res = f_data[i];
               for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
                  res -= A_diag_data[j] * u_data[A_diag_J[j]];
               if (num_cols_offd)
                  for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
                     res -= A_offd_data[j] * u_offd_data[A_offd_J[j]];
               u_data[i] += res / l1_norms[i];
            }
         }
         else if (relax_weight == 1.0) /* SSOR */
         {
            /* Forward local pass */
            for (i = 0; i < num_rows; i++)
            {
               res = f_data[i];
               for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
                  res -= A_diag_data[j] * u_data[A_diag_J[j]];
               if (num_cols_offd)
                  for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
                     res -= A_offd_data[j] * u_offd_data[A_offd_J[j]];
               u_data[i] += omega * res / l1_norms[i];
            }
            /* Backward local pass */
            for (i = num_rows-1; i > -1; i--)
            {
               res = f_data[i];
               for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
                  res -= A_diag_data[j] * u_data[A_diag_J[j]];
               if (num_cols_offd)
                  for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
                     res -= A_offd_data[j] * u_offd_data[A_offd_J[j]];
               u_data[i] += omega * res / l1_norms[i];
            }
         }
         else /* scaled SSOR */
         {
            double dif;
            double c1 = omega * relax_weight;
            double c2 = omega * (1.0 - relax_weight);

            /* Forward local pass (save initial guess in v_data) */
            for (i = 0; i < num_rows; i++)
            {
               dif = 0.0;
               v_data[i] = u_data[i];
               res = f_data[i];
               for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
               {
                  res -= A_diag_data[j] * u_data[A_diag_J[j]];
                  if (A_diag_J[j] < i)
                     dif += A_diag_data[j] * (v_data[A_diag_J[j]] - u_data[A_diag_J[j]]);
               }
               if (num_cols_offd)
                  for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
                     res -= A_offd_data[j] * u_offd_data[A_offd_J[j]];
               u_data[i] += (c1 * res + c2 * dif) / l1_norms[i];
            }
            /* Backward local pass */
            for (i = num_rows-1; i > -1; i--)
            {
               dif = 0.0;
               res = f_data[i];
               for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
               {
                  res -= A_diag_data[j] * u_data[A_diag_J[j]];
                  if (A_diag_J[j] > i)
                     dif += A_diag_data[j] * (v_data[A_diag_J[j]] - u_data[A_diag_J[j]]);
               }
               if (num_cols_offd)
                  for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
                     res -= A_offd_data[j] * u_offd_data[A_offd_J[j]];
               u_data[i] += (c1 * res + c2 * dif) / l1_norms[i];
            }
         }

         hypre_TFree(u_offd_data);
      }
      else if (relax_type == 3) /* Kaczmarz */
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

         /* Forward local pass */
         for (i = 0; i < num_rows; i++)
         {
            res = f_data[i];
            for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
               res -= A_diag_data[j] * u_data[A_diag_J[j]];
            if (num_cols_offd)
               for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
                  res -= A_offd_data[j] * u_offd_data[A_offd_J[j]];
            res /= l1_norms[i];
            for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
               u_data[A_diag_J[j]] += omega * res * A_diag_data[j];
         }

         /* Backward local pass */
         for (i = num_rows-1; i > -1; i--)
         {
            res = f_data[i];
            for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
               res -= A_diag_data[j] * u_data[A_diag_J[j]];
            if (num_cols_offd)
               for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
                  res -= A_offd_data[j] * u_offd_data[A_offd_J[j]];
            res /= l1_norms[i];
            for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
               u_data[A_diag_J[j]] += omega * res * A_diag_data[j];
         }

         hypre_TFree(u_offd_data);
      }
      else /* call BoomerAMG relaxation */
      {
         hypre_BoomerAMGRelax(A, f, NULL, abs(relax_type), 0,
                              relax_weight, omega, u, v, z);
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
 * option 3 = Compute the l2 norm^2 of the rows
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
      else if (option == 3)
      {
         l1_norm[i] = 0.0;
         for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
            l1_norm[i] += A_diag_data[j] * A_diag_data[j];
         if (num_cols_offd)
            for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
               l1_norm[i] += A_offd_data[j] * A_offd_data[j];
         continue;
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
          (!num_cols_offd || (A_offd_I[i+1] == A_offd_I[i])))
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
   ams_data -> B_G_agg_levels = 1;     /* Levels of aggressive coarsening */
   ams_data -> B_G_relax_type = 3;     /* hybrid G-S/Jacobi */
   ams_data -> B_G_theta = 0.25;       /* strength threshold */
   ams_data -> B_G_interp_type = 0;    /* interpolation type */
   ams_data -> B_G_Pmax = 0;           /* max nonzero elements in interp. rows */
   ams_data -> B_Pi_coarsen_type = 10; /* HMIS coarsening */
   ams_data -> B_Pi_agg_levels = 1;    /* Levels of aggressive coarsening */
   ams_data -> B_Pi_relax_type = 3;    /* hybrid G-S/Jacobi */
   ams_data -> B_Pi_theta = 0.25;      /* strength threshold */
   ams_data -> B_Pi_interp_type = 0;   /* interpolation type */
   ams_data -> B_Pi_Pmax = 0;          /* max nonzero elements in interp. rows */
   ams_data -> beta_is_zero = 0;       /* the problem has a mass term */

   /* The rest of the fields are initialized using the Set functions */

   ams_data -> A    = NULL;
   ams_data -> G    = NULL;
   ams_data -> A_G  = NULL;
   ams_data -> B_G  = 0;
   ams_data -> Pi   = NULL;
   ams_data -> A_Pi = NULL;
   ams_data -> B_Pi = 0;
   ams_data -> x    = NULL;
   ams_data -> y    = NULL;
   ams_data -> z    = NULL;
   ams_data -> Gx   = NULL;
   ams_data -> Gy   = NULL;
   ams_data -> Gz   = NULL;

   ams_data -> r0  = NULL;
   ams_data -> g0  = NULL;
   ams_data -> r1  = NULL;
   ams_data -> g1  = NULL;
   ams_data -> r2  = NULL;
   ams_data -> g2  = NULL;

   ams_data -> Pix    = NULL;
   ams_data -> Piy    = NULL;
   ams_data -> Piz    = NULL;
   ams_data -> A_Pix  = NULL;
   ams_data -> A_Piy  = NULL;
   ams_data -> A_Piz  = NULL;
   ams_data -> B_Pix  = 0;
   ams_data -> B_Piy  = 0;
   ams_data -> B_Piz  = 0;

   ams_data -> interior_nodes       = NULL;
   ams_data -> G0                   = NULL;
   ams_data -> A_G0                 = NULL;
   ams_data -> B_G0                 = 0;
   ams_data -> projection_frequency = 5;

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

   if (ams_data -> Pix)
      hypre_ParCSRMatrixDestroy(ams_data -> Pix);
   if (ams_data -> A_Pix)
      hypre_ParCSRMatrixDestroy(ams_data -> A_Pix);
   if (ams_data -> B_Pix)
      HYPRE_BoomerAMGDestroy(ams_data -> B_Pix);
   if (ams_data -> Piy)
      hypre_ParCSRMatrixDestroy(ams_data -> Piy);
   if (ams_data -> A_Piy)
      hypre_ParCSRMatrixDestroy(ams_data -> A_Piy);
   if (ams_data -> B_Piy)
      HYPRE_BoomerAMGDestroy(ams_data -> B_Piy);
   if (ams_data -> Piz)
      hypre_ParCSRMatrixDestroy(ams_data -> Piz);
   if (ams_data -> A_Piz)
      hypre_ParCSRMatrixDestroy(ams_data -> A_Piz);
   if (ams_data -> B_Piz)
      HYPRE_BoomerAMGDestroy(ams_data -> B_Piz);

   if (ams_data -> r0)
      hypre_ParVectorDestroy(ams_data -> r0);
   if (ams_data -> g0)
      hypre_ParVectorDestroy(ams_data -> g0);
   if (ams_data -> r1)
      hypre_ParVectorDestroy(ams_data -> r1);
   if (ams_data -> g1)
      hypre_ParVectorDestroy(ams_data -> g1);
   if (ams_data -> r2)
      hypre_ParVectorDestroy(ams_data -> r2);
   if (ams_data -> g2)
      hypre_ParVectorDestroy(ams_data -> g2);

   if (ams_data -> G0)
      hypre_ParCSRMatrixDestroy(ams_data -> A);
   if (ams_data -> G0)
      hypre_ParCSRMatrixDestroy(ams_data -> G0);
   if (ams_data -> A_G0)
      hypre_ParCSRMatrixDestroy(ams_data -> A_G0);
   if (ams_data -> B_G0)
      HYPRE_BoomerAMGDestroy(ams_data -> B_G0);

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
 * hypre_AMSSetInteriorNodes
 *
 * Set the list of nodes which are interior to the zero-conductivity region.
 *
 * Should be called before hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

int hypre_AMSSetInteriorNodes(void *solver,
                              hypre_ParVector *interior_nodes)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> interior_nodes = interior_nodes;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetProjectionFrequency
 *
 * How often to project the r.h.s. onto the compatible sub-space Ker(G0^T),
 * when iterating with the solver.
 *
 * The default value is every 5th iteration.
 *--------------------------------------------------------------------------*/

int hypre_AMSSetProjectionFrequency(void *solver,
                                    int projection_frequency)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> projection_frequency = projection_frequency;
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
 *  20 = 2-level multipl. solver (0[12]0)
 *
 *   0 = a Hiptmair-like smoother (010)
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
 * Set AMG parameters for B_Pi. Default values: 10, 1, 3, 0.25, 0, 0.
 *--------------------------------------------------------------------------*/

int hypre_AMSSetAlphaAMGOptions(void *solver,
                                int B_Pi_coarsen_type,
                                int B_Pi_agg_levels,
                                int B_Pi_relax_type,
                                double B_Pi_theta,
                                int B_Pi_interp_type,
                                int B_Pi_Pmax)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> B_Pi_coarsen_type = B_Pi_coarsen_type;
   ams_data -> B_Pi_agg_levels = B_Pi_agg_levels;
   ams_data -> B_Pi_relax_type = B_Pi_relax_type;
   ams_data -> B_Pi_theta = B_Pi_theta;
   ams_data -> B_Pi_interp_type = B_Pi_interp_type;
   ams_data -> B_Pi_Pmax = B_Pi_Pmax;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetBetaAMGOptions
 *
 * Set AMG parameters for B_G. Default values: 10, 1, 3, 0.25, 0, 0.
 *--------------------------------------------------------------------------*/

int hypre_AMSSetBetaAMGOptions(void *solver,
                               int B_G_coarsen_type,
                               int B_G_agg_levels,
                               int B_G_relax_type,
                               double B_G_theta,
                               int B_G_interp_type,
                               int B_G_Pmax)
{
   hypre_AMSData *ams_data = solver;
   ams_data -> B_G_coarsen_type = B_G_coarsen_type;
   ams_data -> B_G_agg_levels = B_G_agg_levels;
   ams_data -> B_G_relax_type = B_G_relax_type;
   ams_data -> B_G_theta = B_G_theta;
   ams_data -> B_G_interp_type = B_G_interp_type;
   ams_data -> B_G_Pmax = B_G_Pmax;
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
 * hypre_AMSComputePixyz
 *
 * Construct the components Pix, Piy, Piz of the interpolation matrix Pi,
 * which maps the space of vector linear finite elements to the space of
 * edge finite elements.
 *
 * The construction is based on the fact that each component has the same
 * sparsity structure as G, and the entries can be computed from the vectors
 * Gx, Gy, Gz.
 *--------------------------------------------------------------------------*/

int hypre_AMSComputePixyz(hypre_ParCSRMatrix *A,
                          hypre_ParCSRMatrix *G,
                          hypre_ParVector *x,
                          hypre_ParVector *y,
                          hypre_ParVector *z,
                          hypre_ParVector *Gx,
                          hypre_ParVector *Gy,
                          hypre_ParVector *Gz,
                          int dim,
                          hypre_ParCSRMatrix **Pix_ptr,
                          hypre_ParCSRMatrix **Piy_ptr,
                          hypre_ParCSRMatrix **Piz_ptr)
{
   int input_info = 0;

   hypre_ParCSRMatrix *Pix, *Piy, *Piz;

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

   /* Compute Pix, Piy, Piz  */
   {
      int i, j;

      double *Gx_data, *Gy_data, *Gz_data;

      MPI_Comm comm = hypre_ParCSRMatrixComm(G);
      int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(G);
      int global_num_cols = hypre_ParCSRMatrixGlobalNumCols(G);
      int *row_starts = hypre_ParCSRMatrixRowStarts(G);
      int *col_starts = hypre_ParCSRMatrixColStarts(G);
      int num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(G));
      int num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(G));
      int num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(G));

      Pix = hypre_ParCSRMatrixCreate(comm,
                                     global_num_rows,
                                     global_num_cols,
                                     row_starts,
                                     col_starts,
                                     num_cols_offd,
                                     num_nonzeros_diag,
                                     num_nonzeros_offd);
      hypre_ParCSRMatrixOwnsData(Pix) = 1;
      hypre_ParCSRMatrixOwnsRowStarts(Pix) = 0;
      hypre_ParCSRMatrixOwnsColStarts(Pix) = 0;
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
      hypre_ParCSRMatrixOwnsRowStarts(Piy) = 0;
      hypre_ParCSRMatrixOwnsColStarts(Piy) = 0;
      hypre_ParCSRMatrixInitialize(Piy);

      if (dim == 3)
      {
         Piz = hypre_ParCSRMatrixCreate(comm,
                                        global_num_rows,
                                        global_num_cols,
                                        row_starts,
                                        col_starts,
                                        num_cols_offd,
                                        num_nonzeros_diag,
                                        num_nonzeros_offd);
         hypre_ParCSRMatrixOwnsData(Piz) = 1;
         hypre_ParCSRMatrixOwnsRowStarts(Piz) = 0;
         hypre_ParCSRMatrixOwnsColStarts(Piz) = 0;
         hypre_ParCSRMatrixInitialize(Piz);
      }

      Gx_data = hypre_VectorData(hypre_ParVectorLocalVector(Gx));
      Gy_data = hypre_VectorData(hypre_ParVectorLocalVector(Gy));
      if (dim == 3)
         Gz_data = hypre_VectorData(hypre_ParVectorLocalVector(Gz));

      /* Fill-in the diagonal part */
      if (dim == 3)
      {
         hypre_CSRMatrix *G_diag = hypre_ParCSRMatrixDiag(G);
         int *G_diag_I = hypre_CSRMatrixI(G_diag);
         int *G_diag_J = hypre_CSRMatrixJ(G_diag);

         int G_diag_nrows = hypre_CSRMatrixNumRows(G_diag);
         int G_diag_nnz = hypre_CSRMatrixNumNonzeros(G_diag);

         hypre_CSRMatrix *Pix_diag = hypre_ParCSRMatrixDiag(Pix);
         int *Pix_diag_I = hypre_CSRMatrixI(Pix_diag);
         int *Pix_diag_J = hypre_CSRMatrixJ(Pix_diag);
         double *Pix_diag_data = hypre_CSRMatrixData(Pix_diag);

         hypre_CSRMatrix *Piy_diag = hypre_ParCSRMatrixDiag(Piy);
         int *Piy_diag_I = hypre_CSRMatrixI(Piy_diag);
         int *Piy_diag_J = hypre_CSRMatrixJ(Piy_diag);
         double *Piy_diag_data = hypre_CSRMatrixData(Piy_diag);

         hypre_CSRMatrix *Piz_diag = hypre_ParCSRMatrixDiag(Piz);
         int *Piz_diag_I = hypre_CSRMatrixI(Piz_diag);
         int *Piz_diag_J = hypre_CSRMatrixJ(Piz_diag);
         double *Piz_diag_data = hypre_CSRMatrixData(Piz_diag);

         for (i = 0; i < G_diag_nrows+1; i++)
         {
            Pix_diag_I[i] = G_diag_I[i];
            Piy_diag_I[i] = G_diag_I[i];
            Piz_diag_I[i] = G_diag_I[i];
         }

         for (i = 0; i < G_diag_nnz; i++)
         {
            Pix_diag_J[i] = G_diag_J[i];
            Piy_diag_J[i] = G_diag_J[i];
            Piz_diag_J[i] = G_diag_J[i];
         }

         for (i = 0; i < G_diag_nrows; i++)
            for (j = G_diag_I[i]; j < G_diag_I[i+1]; j++)
            {
               *Pix_diag_data++ = 0.5 * Gx_data[i];
               *Piy_diag_data++ = 0.5 * Gy_data[i];
               *Piz_diag_data++ = 0.5 * Gz_data[i];
            }
      }
      else
      {
         hypre_CSRMatrix *G_diag = hypre_ParCSRMatrixDiag(G);
         int *G_diag_I = hypre_CSRMatrixI(G_diag);
         int *G_diag_J = hypre_CSRMatrixJ(G_diag);

         int G_diag_nrows = hypre_CSRMatrixNumRows(G_diag);
         int G_diag_nnz = hypre_CSRMatrixNumNonzeros(G_diag);

         hypre_CSRMatrix *Pix_diag = hypre_ParCSRMatrixDiag(Pix);
         int *Pix_diag_I = hypre_CSRMatrixI(Pix_diag);
         int *Pix_diag_J = hypre_CSRMatrixJ(Pix_diag);
         double *Pix_diag_data = hypre_CSRMatrixData(Pix_diag);

         hypre_CSRMatrix *Piy_diag = hypre_ParCSRMatrixDiag(Piy);
         int *Piy_diag_I = hypre_CSRMatrixI(Piy_diag);
         int *Piy_diag_J = hypre_CSRMatrixJ(Piy_diag);
         double *Piy_diag_data = hypre_CSRMatrixData(Piy_diag);

         for (i = 0; i < G_diag_nrows+1; i++)
         {
            Pix_diag_I[i] = G_diag_I[i];
            Piy_diag_I[i] = G_diag_I[i];
         }

         for (i = 0; i < G_diag_nnz; i++)
         {
            Pix_diag_J[i] = G_diag_J[i];
            Piy_diag_J[i] = G_diag_J[i];
         }

         for (i = 0; i < G_diag_nrows; i++)
            for (j = G_diag_I[i]; j < G_diag_I[i+1]; j++)
            {
               *Pix_diag_data++ = 0.5 * Gx_data[i];
               *Piy_diag_data++ = 0.5 * Gy_data[i];
            }
      }


      /* Fill-in the off-diagonal part */
      if (dim == 3)
      {
         hypre_CSRMatrix *G_offd = hypre_ParCSRMatrixOffd(G);
         int *G_offd_I = hypre_CSRMatrixI(G_offd);
         int *G_offd_J = hypre_CSRMatrixJ(G_offd);

         int G_offd_nrows = hypre_CSRMatrixNumRows(G_offd);
         int G_offd_ncols = hypre_CSRMatrixNumCols(G_offd);
         int G_offd_nnz = hypre_CSRMatrixNumNonzeros(G_offd);

         hypre_CSRMatrix *Pix_offd = hypre_ParCSRMatrixOffd(Pix);
         int *Pix_offd_I = hypre_CSRMatrixI(Pix_offd);
         int *Pix_offd_J = hypre_CSRMatrixJ(Pix_offd);
         double *Pix_offd_data = hypre_CSRMatrixData(Pix_offd);

         hypre_CSRMatrix *Piy_offd = hypre_ParCSRMatrixOffd(Piy);
         int *Piy_offd_I = hypre_CSRMatrixI(Piy_offd);
         int *Piy_offd_J = hypre_CSRMatrixJ(Piy_offd);
         double *Piy_offd_data = hypre_CSRMatrixData(Piy_offd);

         hypre_CSRMatrix *Piz_offd = hypre_ParCSRMatrixOffd(Piz);
         int *Piz_offd_I = hypre_CSRMatrixI(Piz_offd);
         int *Piz_offd_J = hypre_CSRMatrixJ(Piz_offd);
         double *Piz_offd_data = hypre_CSRMatrixData(Piz_offd);

         int *G_cmap = hypre_ParCSRMatrixColMapOffd(G);
         int *Pix_cmap = hypre_ParCSRMatrixColMapOffd(Pix);
         int *Piy_cmap = hypre_ParCSRMatrixColMapOffd(Piy);
         int *Piz_cmap = hypre_ParCSRMatrixColMapOffd(Piz);

         if (G_offd_ncols)
            for (i = 0; i < G_offd_nrows+1; i++)
            {
               Pix_offd_I[i] = G_offd_I[i];
               Piy_offd_I[i] = G_offd_I[i];
               Piz_offd_I[i] = G_offd_I[i];
            }

         for (i = 0; i < G_offd_nnz; i++)
         {
            Pix_offd_J[i] = G_offd_J[i];
            Piy_offd_J[i] = G_offd_J[i];
            Piz_offd_J[i] = G_offd_J[i];
         }

         for (i = 0; i < G_offd_nrows; i++)
            for (j = G_offd_I[i]; j < G_offd_I[i+1]; j++)
            {
               *Pix_offd_data++ = 0.5 * Gx_data[i];
               *Piy_offd_data++ = 0.5 * Gy_data[i];
               *Piz_offd_data++ = 0.5 * Gz_data[i];
            }

         for (i = 0; i < G_offd_ncols; i++)
         {
            Pix_cmap[i] = G_cmap[i];
            Piy_cmap[i] = G_cmap[i];
            Piz_cmap[i] = G_cmap[i];
         }
      }
      else
      {
         hypre_CSRMatrix *G_offd = hypre_ParCSRMatrixOffd(G);
         int *G_offd_I = hypre_CSRMatrixI(G_offd);
         int *G_offd_J = hypre_CSRMatrixJ(G_offd);

         int G_offd_nrows = hypre_CSRMatrixNumRows(G_offd);
         int G_offd_ncols = hypre_CSRMatrixNumCols(G_offd);
         int G_offd_nnz = hypre_CSRMatrixNumNonzeros(G_offd);

         hypre_CSRMatrix *Pix_offd = hypre_ParCSRMatrixOffd(Pix);
         int *Pix_offd_I = hypre_CSRMatrixI(Pix_offd);
         int *Pix_offd_J = hypre_CSRMatrixJ(Pix_offd);
         double *Pix_offd_data = hypre_CSRMatrixData(Pix_offd);

         hypre_CSRMatrix *Piy_offd = hypre_ParCSRMatrixOffd(Piy);
         int *Piy_offd_I = hypre_CSRMatrixI(Piy_offd);
         int *Piy_offd_J = hypre_CSRMatrixJ(Piy_offd);
         double *Piy_offd_data = hypre_CSRMatrixData(Piy_offd);

         int *G_cmap = hypre_ParCSRMatrixColMapOffd(G);
         int *Pix_cmap = hypre_ParCSRMatrixColMapOffd(Pix);
         int *Piy_cmap = hypre_ParCSRMatrixColMapOffd(Piy);

         if (G_offd_ncols)
            for (i = 0; i < G_offd_nrows+1; i++)
            {
               Pix_offd_I[i] = G_offd_I[i];
               Piy_offd_I[i] = G_offd_I[i];
            }

         for (i = 0; i < G_offd_nnz; i++)
         {
            Pix_offd_J[i] = G_offd_J[i];
            Piy_offd_J[i] = G_offd_J[i];
         }

         for (i = 0; i < G_offd_nrows; i++)
            for (j = G_offd_I[i]; j < G_offd_I[i+1]; j++)
            {
               *Pix_offd_data++ = 0.5 * Gx_data[i];
               *Piy_offd_data++ = 0.5 * Gy_data[i];
            }

         for (i = 0; i < G_offd_ncols; i++)
         {
            Pix_cmap[i] = G_cmap[i];
            Piy_cmap[i] = G_cmap[i];
         }
      }
   }

   if (input_info == 1)
   {
      hypre_ParVectorDestroy(Gx);
      hypre_ParVectorDestroy(Gy);
      if (dim == 3)
         hypre_ParVectorDestroy(Gz);
   }

   *Pix_ptr = Pix;
   *Piy_ptr = Piy;
   if (dim == 3)
      *Piz_ptr = Piz;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSComputeGPi
 *
 * Construct the matrix [G,Pi] which can be considered an interpolation
 * matrix from S_h^4 (4 copies of the scalar linear finite element space)
 * to the edge finite elements space.
 *--------------------------------------------------------------------------*/

int hypre_AMSComputeGPi(hypre_ParCSRMatrix *A,
                        hypre_ParCSRMatrix *G,
                        hypre_ParVector *x,
                        hypre_ParVector *y,
                        hypre_ParVector *z,
                        hypre_ParVector *Gx,
                        hypre_ParVector *Gy,
                        hypre_ParVector *Gz,
                        int dim,
                        hypre_ParCSRMatrix **GPi_ptr)
{
   int input_info = 0;

   hypre_ParCSRMatrix *GPi;

   if (x != NULL && y != NULL && (dim == 2 || z != NULL))
      input_info = 1;

   if (Gx != NULL && Gy != NULL && (dim == 2 || Gz != NULL))
      input_info = 2;

   if (!input_info)
      hypre_error_in_arg(3);

   /* Take into account G */
   dim++;

   /* If not given, compute Gx, Gy and Gz */
   if (input_info == 1)
   {
      Gx = hypre_ParVectorInRangeOf(G);
      hypre_ParCSRMatrixMatvec (1.0, G, x, 0.0, Gx);
      Gy = hypre_ParVectorInRangeOf(G);
      hypre_ParCSRMatrixMatvec (1.0, G, y, 0.0, Gy);
      if (dim == 4)
      {
         Gz = hypre_ParVectorInRangeOf(G);
         hypre_ParCSRMatrixMatvec (1.0, G, z, 0.0, Gz);
      }
   }

   /* Compute GPi = [Pi_x, Pi_y, Pi_z, G] */
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

      GPi = hypre_ParCSRMatrixCreate(comm,
                                     global_num_rows,
                                     global_num_cols,
                                     row_starts,
                                     col_starts,
                                     num_cols_offd,
                                     num_nonzeros_diag,
                                     num_nonzeros_offd);

      hypre_ParCSRMatrixOwnsData(GPi) = 1;
      hypre_ParCSRMatrixOwnsRowStarts(GPi) = 0;
      hypre_ParCSRMatrixOwnsColStarts(GPi) = 1;

      hypre_ParCSRMatrixInitialize(GPi);

      Gx_data = hypre_VectorData(hypre_ParVectorLocalVector(Gx));
      Gy_data = hypre_VectorData(hypre_ParVectorLocalVector(Gy));
      if (dim == 4)
         Gz_data = hypre_VectorData(hypre_ParVectorLocalVector(Gz));

      /* Fill-in the diagonal part */
      {
         hypre_CSRMatrix *G_diag = hypre_ParCSRMatrixDiag(G);
         int *G_diag_I = hypre_CSRMatrixI(G_diag);
         int *G_diag_J = hypre_CSRMatrixJ(G_diag);
         double *G_diag_data = hypre_CSRMatrixData(G_diag);

         int G_diag_nrows = hypre_CSRMatrixNumRows(G_diag);
         int G_diag_nnz = hypre_CSRMatrixNumNonzeros(G_diag);

         hypre_CSRMatrix *GPi_diag = hypre_ParCSRMatrixDiag(GPi);
         int *GPi_diag_I = hypre_CSRMatrixI(GPi_diag);
         int *GPi_diag_J = hypre_CSRMatrixJ(GPi_diag);
         double *GPi_diag_data = hypre_CSRMatrixData(GPi_diag);

         for (i = 0; i < G_diag_nrows+1; i++)
            GPi_diag_I[i] = dim * G_diag_I[i];

         for (i = 0; i < G_diag_nnz; i++)
            for (d = 0; d < dim; d++)
               GPi_diag_J[dim*i+d] = dim*G_diag_J[i]+d;

         for (i = 0; i < G_diag_nrows; i++)
            for (j = G_diag_I[i]; j < G_diag_I[i+1]; j++)
            {
               *GPi_diag_data++ = G_diag_data[j];
               *GPi_diag_data++ = 0.5 * Gx_data[i];
               *GPi_diag_data++ = 0.5 * Gy_data[i];
               if (dim == 4)
                  *GPi_diag_data++ = 0.5 * Gz_data[i];
            }
      }

      /* Fill-in the off-diagonal part */
      {
         hypre_CSRMatrix *G_offd = hypre_ParCSRMatrixOffd(G);
         int *G_offd_I = hypre_CSRMatrixI(G_offd);
         int *G_offd_J = hypre_CSRMatrixJ(G_offd);
         double *G_offd_data = hypre_CSRMatrixData(G_offd);

         int G_offd_nrows = hypre_CSRMatrixNumRows(G_offd);
         int G_offd_ncols = hypre_CSRMatrixNumCols(G_offd);
         int G_offd_nnz = hypre_CSRMatrixNumNonzeros(G_offd);

         hypre_CSRMatrix *GPi_offd = hypre_ParCSRMatrixOffd(GPi);
         int *GPi_offd_I = hypre_CSRMatrixI(GPi_offd);
         int *GPi_offd_J = hypre_CSRMatrixJ(GPi_offd);
         double *GPi_offd_data = hypre_CSRMatrixData(GPi_offd);

         int *G_cmap = hypre_ParCSRMatrixColMapOffd(G);
         int *GPi_cmap = hypre_ParCSRMatrixColMapOffd(GPi);

         if (G_offd_ncols)
            for (i = 0; i < G_offd_nrows+1; i++)
               GPi_offd_I[i] = dim * G_offd_I[i];

         for (i = 0; i < G_offd_nnz; i++)
            for (d = 0; d < dim; d++)
               GPi_offd_J[dim*i+d] = dim*G_offd_J[i]+d;

         for (i = 0; i < G_offd_nrows; i++)
            for (j = G_offd_I[i]; j < G_offd_I[i+1]; j++)
            {
               *GPi_offd_data++ = G_offd_data[j];
               *GPi_offd_data++ = 0.5 * Gx_data[i];
               *GPi_offd_data++ = 0.5 * Gy_data[i];
               if (dim == 4)
                  *GPi_offd_data++ = 0.5 * Gz_data[i];
            }

         for (i = 0; i < G_offd_ncols; i++)
            for (d = 0; d < dim; d++)
               GPi_cmap[dim*i+d] = dim*G_cmap[i]+d;
      }

   }

   if (input_info == 1)
   {
      hypre_ParVectorDestroy(Gx);
      hypre_ParVectorDestroy(Gy);
      if (dim == 4)
         hypre_ParVectorDestroy(Gz);
   }

   *GPi_ptr = GPi;

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

   /* Modifications for problems with zero-conductivity regions */
   if (ams_data -> interior_nodes)
   {
      hypre_ParCSRMatrix *G0t, *Aorig = A;

      /* Make sure that multiple Setup()+Solve() give identical results */
      ams_data -> solve_counter = 0;

      /* Construct the discrete gradient matrix for the zero-conductivity region
         by eliminating the zero-conductivity nodes from G^t */
      hypre_ParCSRMatrixTranspose(ams_data -> G, &G0t, 1);
      {
         int i, j;
         int nv = hypre_ParCSRMatrixNumCols(ams_data -> G);
         hypre_CSRMatrix *G0td = hypre_ParCSRMatrixDiag(G0t);
         int *G0tdI = hypre_CSRMatrixI(G0td);
         double *G0tdA = hypre_CSRMatrixData(G0td);
         hypre_CSRMatrix *G0to = hypre_ParCSRMatrixOffd(G0t);
         int *G0toI = hypre_CSRMatrixI(G0to);
         double *G0toA = hypre_CSRMatrixData(G0to);
         double *interior_nodes_data=hypre_VectorData(
            hypre_ParVectorLocalVector((hypre_ParVector*) ams_data -> interior_nodes));

         for (i = 0; i < nv; i++)
         {
            if (interior_nodes_data[i] != 1)
            {
               for (j = G0tdI[i]; j < G0tdI[i+1]; j++)
                  G0tdA[j] = 0.0;
               if (G0toI)
                  for (j = G0toI[i]; j < G0toI[i+1]; j++)
                     G0toA[j] = 0.0;
            }
         }
      }
      hypre_ParCSRMatrixTranspose(G0t, & ams_data -> G0, 1);

      /* Construct the subspace matrix A_G0 = G0^T G0 */
      ams_data -> A_G0 = hypre_ParMatmul(G0t, ams_data -> G0);
      hypre_ParCSRMatrixFixZeroRows(ams_data -> A_G0);

      /* Create AMG solver for A_G0 */
      HYPRE_BoomerAMGCreate(&ams_data -> B_G0);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_G0, ams_data -> B_G_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_G0, ams_data -> B_G_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_G0, ams_data -> B_G_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_G0, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_G0, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_G0, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_G0, 3); /* use just a few V-cycles */
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_G0, ams_data -> B_G_theta);
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_G0, ams_data -> B_G_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_G0, ams_data -> B_G_Pmax);
      HYPRE_BoomerAMGSetup(ams_data -> B_G0,
                           (HYPRE_ParCSRMatrix)ams_data -> A_G0,
                           0, 0);

      /* Construct the preconditioner for ams_data->A = A + G0 G0^T.
         NOTE: this can be optimized significantly by taking into account that
         the sparsity pattern of A is subset of the sparsity pattern of G0 G0^T */
      {
         hypre_ParCSRMatrix *A = hypre_ParMatmul(ams_data -> G0, G0t);
         hypre_ParCSRMatrix *B = Aorig;
         hypre_ParCSRMatrix **C_ptr = &ams_data -> A;

         hypre_ParCSRMatrix *C;
         hypre_CSRMatrix *A_local, *B_local, *C_local, *C_tmp;

         MPI_Comm comm = hypre_ParCSRMatrixComm(A);
         int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
         int global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
         int *row_starts = hypre_ParCSRMatrixRowStarts(A);
         int *col_starts = hypre_ParCSRMatrixColStarts(A);
         int A_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
         int A_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(A));
         int A_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(A));
         int B_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(B));
         int B_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(B));
         int B_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(B));

         A_local = hypre_MergeDiagAndOffd(A);
         B_local = hypre_MergeDiagAndOffd(B);
         /* scale (penalize) G0 G0^T before adding it to the matrix */
         {
            int i, nnz = hypre_CSRMatrixNumNonzeros(A_local);
            double *data = hypre_CSRMatrixData(A_local);
            double *dataB = hypre_CSRMatrixData(B_local);
            double factor = dataB[0]*1e-8; /* assume dataB[0] = A11 */
            for (i = 0; i < nnz; i++) data[i] *= factor;
         }
         C_tmp = hypre_CSRMatrixAdd(A_local, B_local);
         C_local = hypre_CSRMatrixDeleteZeros(C_tmp,0.0);
	 if (C_local)
	   hypre_CSRMatrixDestroy(C_tmp);
	 else
	   C_local = C_tmp;

         C = hypre_ParCSRMatrixCreate (comm,
                                       global_num_rows,
                                       global_num_cols,
                                       row_starts,
                                       col_starts,
                                       A_num_cols_offd + B_num_cols_offd,
                                       A_num_nonzeros_diag + B_num_nonzeros_diag,
                                       A_num_nonzeros_offd + B_num_nonzeros_offd);
         GenerateDiagAndOffd(C_local, C,
                             hypre_ParCSRMatrixFirstColDiag(A),
                             hypre_ParCSRMatrixLastColDiag(A));
         hypre_ParCSRMatrixOwnsRowStarts(C) = 0;
         hypre_ParCSRMatrixOwnsColStarts(C) = 1;
         hypre_ParCSRMatrixOwnsColStarts(G0t) = 0;

         hypre_CSRMatrixDestroy(A_local);
         hypre_CSRMatrixDestroy(B_local);
         hypre_CSRMatrixDestroy(C_local);

         hypre_ParCSRMatrixDestroy(A);

         *C_ptr = C;
      }

      hypre_ParCSRMatrixDestroy(G0t);
   }

   /* Make sure that the first entry in each row is the diagonal one. */
   /* hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(ams_data -> A)); */

   /* Compute the l1 norm of the rows of A */
   if (ams_data -> A_relax_type >= 1 && ams_data -> A_relax_type <= 3)
      hypre_ParCSRComputeL1Norms(ams_data -> A, ams_data -> A_relax_type,
                                 &ams_data -> A_l1_norms);

   if (ams_data -> cycle_type == 20)
      /* Construct the combined interpolation matrix [G,Pi] */
      hypre_AMSComputeGPi(ams_data -> A,
                          ams_data -> G,
                          ams_data -> x,
                          ams_data -> y,
                          ams_data -> z,
                          ams_data -> Gx,
                          ams_data -> Gy,
                          ams_data -> Gz,
                          ams_data -> dim,
                          &ams_data -> Pi);
   else if (ams_data -> cycle_type > 10)
      /* Construct Pi{x,y,z} instead of Pi = [Pix,Piy,Piz] */
      hypre_AMSComputePixyz(ams_data -> A,
                            ams_data -> G,
                            ams_data -> x,
                            ams_data -> y,
                            ams_data -> z,
                            ams_data -> Gx,
                            ams_data -> Gy,
                            ams_data -> Gz,
                            ams_data -> dim,
                            &ams_data -> Pix,
                            &ams_data -> Piy,
                            &ams_data -> Piz);
   else
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
   if (!ams_data -> beta_is_zero && ams_data -> cycle_type != 20)
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
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_G, ams_data -> B_G_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_G, ams_data -> B_G_Pmax);

      if (ams_data -> cycle_type == 0)
         HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_G, 2);

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
         hypre_ParCSRMatrixFixZeroRows(ams_data -> A_G);

         ams_data -> owns_A_G = 1;
      }

      HYPRE_BoomerAMGSetup(ams_data -> B_G,
                           (HYPRE_ParCSRMatrix)ams_data -> A_G,
                           0, 0);
   }

   if (ams_data -> cycle_type > 10 && ams_data -> cycle_type != 20)
   /* Create the AMG solvers on the range of Pi{x,y,z}^T */
   {
      HYPRE_BoomerAMGCreate(&ams_data -> B_Pix);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Pix, ams_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Pix, ams_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Pix, ams_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Pix, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pix, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_Pix, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Pix, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Pix, ams_data -> B_Pi_theta);
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_Pix, ams_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Pix, ams_data -> B_Pi_Pmax);

      HYPRE_BoomerAMGCreate(&ams_data -> B_Piy);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Piy, ams_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Piy, ams_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Piy, ams_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Piy, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piy, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_Piy, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Piy, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Piy, ams_data -> B_Pi_theta);
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_Piy, ams_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Piy, ams_data -> B_Pi_Pmax);

      HYPRE_BoomerAMGCreate(&ams_data -> B_Piz);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Piz, ams_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Piz, ams_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Piz, ams_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Piz, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piz, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_Piz, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Piz, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Piz, ams_data -> B_Pi_theta);
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_Piz, ams_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Piz, ams_data -> B_Pi_Pmax);

      if (ams_data -> cycle_type == 0)
      {
         HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pix, 2);
         HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piy, 2);
         HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piz, 2);
      }

      /* Construct the coarse space matrices by RAP */
      if (!hypre_ParCSRMatrixCommPkg(ams_data -> Pix))
         hypre_MatvecCommPkgCreate(ams_data -> Pix);
      hypre_BoomerAMGBuildCoarseOperator(ams_data -> Pix,
                                         ams_data -> A,
                                         ams_data -> Pix,
                                         &ams_data -> A_Pix);
      hypre_ParCSRMatrixOwnsRowStarts(ams_data -> A_Pix) = 0;
      hypre_ParCSRMatrixOwnsColStarts(ams_data -> A_Pix) = 0;
      HYPRE_BoomerAMGSetup(ams_data -> B_Pix,
                           (HYPRE_ParCSRMatrix)ams_data -> A_Pix,
                           0, 0);

      if (!hypre_ParCSRMatrixCommPkg(ams_data -> Piy))
         hypre_MatvecCommPkgCreate(ams_data -> Piy);
      hypre_BoomerAMGBuildCoarseOperator(ams_data -> Piy,
                                         ams_data -> A,
                                         ams_data -> Piy,
                                         &ams_data -> A_Piy);
      hypre_ParCSRMatrixOwnsRowStarts(ams_data -> A_Piy) = 0;
      hypre_ParCSRMatrixOwnsColStarts(ams_data -> A_Piy) = 0;
      HYPRE_BoomerAMGSetup(ams_data -> B_Piy,
                           (HYPRE_ParCSRMatrix)ams_data -> A_Piy,
                           0, 0);

      if (ams_data -> Piz)
      {
         if (!hypre_ParCSRMatrixCommPkg(ams_data -> Piz))
            hypre_MatvecCommPkgCreate(ams_data -> Piz);
         hypre_BoomerAMGBuildCoarseOperator(ams_data -> Piz,
                                            ams_data -> A,
                                            ams_data -> Piz,
                                            &ams_data -> A_Piz);
         hypre_ParCSRMatrixOwnsRowStarts(ams_data -> A_Piz) = 0;
         hypre_ParCSRMatrixOwnsColStarts(ams_data -> A_Piz) = 0;
         HYPRE_BoomerAMGSetup(ams_data -> B_Piz,
                              (HYPRE_ParCSRMatrix)ams_data -> A_Piz,
                              0, 0);
      }
   }
   else
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
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_Pi, ams_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Pi, ams_data -> B_Pi_Pmax);

      if (ams_data -> cycle_type == 0)
         HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pi, 2);

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

         if (!hypre_ParCSRMatrixCommPkg(ams_data -> A))
            hypre_MatvecCommPkgCreate(ams_data -> A);

         if (ams_data -> cycle_type == 9)
         {
            /* Add a discrete divergence term to A before computing  Pi^t A Pi */
            {
               hypre_ParCSRMatrix *Gt, *GGt, *ApGGt;
               hypre_ParCSRMatrixTranspose(ams_data -> G, &Gt, 1);

               /* scale GGt by h^2 */
               {
                  double h2;
                  int i, j, k, ne;

                  hypre_CSRMatrix *Gt_diag = hypre_ParCSRMatrixDiag(Gt);
                  int Gt_num_rows = hypre_CSRMatrixNumRows(Gt_diag);
                  int *Gt_diag_I = hypre_CSRMatrixI(Gt_diag);
                  int *Gt_diag_J = hypre_CSRMatrixJ(Gt_diag);
                  double *Gt_diag_data = hypre_CSRMatrixData(Gt_diag);

                  hypre_CSRMatrix *Gt_offd = hypre_ParCSRMatrixOffd(Gt);
                  int *Gt_offd_I = hypre_CSRMatrixI(Gt_offd);
                  double *Gt_offd_data = hypre_CSRMatrixData(Gt_offd);

                  double *Gx_data = hypre_VectorData(hypre_ParVectorLocalVector(ams_data -> Gx));
                  double *Gy_data = hypre_VectorData(hypre_ParVectorLocalVector(ams_data -> Gy));
                  double *Gz_data = hypre_VectorData(hypre_ParVectorLocalVector(ams_data -> Gz));

                  for (i = 0; i < Gt_num_rows; i++)
                  {
                     /* determine the characteristic mesh size for vertex i */
                     h2 = 0.0;
                     ne = 0;
                     for (j = Gt_diag_I[i]; j < Gt_diag_I[i+1]; j++)
                     {
                        k = Gt_diag_J[j];
                        h2 += Gx_data[k]*Gx_data[k]+Gy_data[k]*Gy_data[k]+Gz_data[k]*Gz_data[k];
                        ne++;
                     }

                     if (ne != 0)
                     {
                        h2 /= ne;
                        for (j = Gt_diag_I[i]; j < Gt_diag_I[i+1]; j++)
                           Gt_diag_data[j] *= h2;
                        for (j = Gt_offd_I[i]; j < Gt_offd_I[i+1]; j++)
                           Gt_offd_data[j] *= h2;
                     }
                  }
               }

               GGt = hypre_ParMatmul(ams_data -> G, Gt);

               /* hypre_ParCSRMatrixAdd(GGt, A, &ams_data -> A); */
               {
                  hypre_ParCSRMatrix *A = GGt;
                  hypre_ParCSRMatrix *B = ams_data -> A;
                  hypre_ParCSRMatrix **C_ptr = &ApGGt;

                  hypre_ParCSRMatrix *C;
                  hypre_CSRMatrix *A_local, *B_local, *C_local;

                  MPI_Comm comm = hypre_ParCSRMatrixComm(A);
                  int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
                  int global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
                  int *row_starts = hypre_ParCSRMatrixRowStarts(A);
                  int *col_starts = hypre_ParCSRMatrixColStarts(A);
                  int A_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
                  int A_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(A));
                  int A_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(A));
                  int B_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(B));
                  int B_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(B));
                  int B_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(B));

                  A_local = hypre_MergeDiagAndOffd(A);
                  B_local = hypre_MergeDiagAndOffd(B);
                  C_local = hypre_CSRMatrixAdd(A_local, B_local);

                  C = hypre_ParCSRMatrixCreate (comm,
                                                global_num_rows,
                                                global_num_cols,
                                                row_starts,
                                                col_starts,
                                                A_num_cols_offd + B_num_cols_offd,
                                                A_num_nonzeros_diag + B_num_nonzeros_diag,
                                                A_num_nonzeros_offd + B_num_nonzeros_offd);
                  GenerateDiagAndOffd(C_local, C,
                                      hypre_ParCSRMatrixFirstColDiag(A),
                                      hypre_ParCSRMatrixLastColDiag(A));
                  hypre_ParCSRMatrixOwnsRowStarts(C) = 0;
                  hypre_ParCSRMatrixOwnsColStarts(C) = 0;

                  hypre_CSRMatrixDestroy(A_local);
                  hypre_CSRMatrixDestroy(B_local);
                  hypre_CSRMatrixDestroy(C_local);

                  *C_ptr = C;
               }

               hypre_BoomerAMGBuildCoarseOperator(ams_data -> Pi,
                                                  ApGGt,
                                                  ams_data -> Pi,
                                                  &ams_data -> A_Pi);
            }
         }
         else
         {
            hypre_BoomerAMGBuildCoarseOperator(ams_data -> Pi,
                                               ams_data -> A,
                                               ams_data -> Pi,
                                               &ams_data -> A_Pi);
         }

         ams_data -> owns_A_Pi = 1;

         if (ams_data -> cycle_type != 20)
            HYPRE_BoomerAMGSetNumFunctions(ams_data -> B_Pi, ams_data -> dim);
         else
            HYPRE_BoomerAMGSetNumFunctions(ams_data -> B_Pi, ams_data -> dim + 1);
         /* HYPRE_BoomerAMGSetNodal(ams_data -> B_Pi, 1); */
      }

      HYPRE_BoomerAMGSetup(ams_data -> B_Pi,
                           (HYPRE_ParCSRMatrix)ams_data -> A_Pi,
                           0, 0);
   }

   /* Allocate temporary vectors */
   ams_data -> r0 = hypre_ParVectorInRangeOf(ams_data -> A);
   ams_data -> g0 = hypre_ParVectorInRangeOf(ams_data -> A);
   if (ams_data -> A_G)
   {
      ams_data -> r1 = hypre_ParVectorInRangeOf(ams_data -> A_G);
      ams_data -> g1 = hypre_ParVectorInRangeOf(ams_data -> A_G);
   }
   if (ams_data -> r1 == NULL && ams_data -> A_Pix)
   {
      ams_data -> r1 = hypre_ParVectorInRangeOf(ams_data -> A_Pix);
      ams_data -> g1 = hypre_ParVectorInRangeOf(ams_data -> A_Pix);
   }
   if (ams_data -> Pi)
   {
      ams_data -> r2 = hypre_ParVectorInDomainOf(ams_data -> Pi);
      ams_data -> g2 = hypre_ParVectorInDomainOf(ams_data -> Pi);
   }

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

   int i, my_id = -1;
   double r0_norm, r_norm, b_norm, relative_resid = 0, old_resid;

   char cycle[30];
   hypre_ParCSRMatrix *Ai[5], *Pi[5];
   HYPRE_Solver Bi[5];
   hypre_ParVector *ri[5], *gi[5];

   hypre_ParVector *z = NULL;


   Ai[0] = ams_data -> A_G;   Bi[0] = ams_data -> B_G;   Pi[0] = ams_data -> G;
   Ai[1] = ams_data -> A_Pi;  Bi[1] = ams_data -> B_Pi;  Pi[1] = ams_data -> Pi;
   Ai[2] = ams_data -> A_Pix; Bi[2] = ams_data -> B_Pix; Pi[2] = ams_data -> Pix;
   Ai[3] = ams_data -> A_Piy; Bi[3] = ams_data -> B_Piy; Pi[3] = ams_data -> Piy;
   Ai[4] = ams_data -> A_Piz; Bi[4] = ams_data -> B_Piz; Pi[4] = ams_data -> Piz;

   ri[0] = ams_data -> r1;    gi[0] = ams_data -> g1;
   ri[1] = ams_data -> r2;    gi[1] = ams_data -> g2;
   ri[2] = ams_data -> r1;    gi[2] = ams_data -> g1;
   ri[3] = ams_data -> r1;    gi[3] = ams_data -> g1;
   ri[4] = ams_data -> r1;    gi[4] = ams_data -> g1;

   /* create an additional temporary vector for relaxation */
   if (hypre_NumThreads() > 1)
   {
      z = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(z);
      hypre_ParVectorSetPartitioningOwner(z,0);
   }
   
         
   if (ams_data -> print_level > 0)
      MPI_Comm_rank(hypre_ParCSRMatrixComm(A), &my_id);

   /* Compatible subspace projection for problems with zero-conductivity regions.
      Note that this modifies the input (r.h.s.) vector b! */
   if ( (ams_data -> B_G0) &&
        (++ams_data->solve_counter % ( ams_data -> projection_frequency ) == 0) )
   {
      /* b = (I - G0 (G0^t G0)^{-1} G0^T) b */
      /* printf("Projecting onto the compatible subspace...\n"); */
      hypre_ParCSRMatrixMatvecT(1.0, ams_data -> G0, b, 0.0, ams_data -> r1);
      hypre_ParVectorSetConstantValues(ams_data -> g1, 0.0);
      hypre_BoomerAMGSolve(ams_data -> B_G0, ams_data -> A_G0, ams_data -> r1, ams_data -> g1);
      hypre_ParCSRMatrixMatvec(1.0, ams_data -> G0, ams_data -> g1, 0.0, ams_data -> g0);
      hypre_ParVectorAxpy(-1.0, ams_data -> g0, b);
   }

   if (ams_data -> beta_is_zero)
   {
      switch (ams_data -> cycle_type)
      {
         case 0:
            sprintf(cycle,"%s","0");
            break;
         case 1:
         case 3:
         case 5:
         case 7:
         default:
            sprintf(cycle,"%s","020");
            break;
         case 2:
         case 4:
         case 6:
         case 8:
            sprintf(cycle,"%s","(0+2)");
            break;
         case 11:
         case 13:
            sprintf(cycle,"%s","0345430");
            break;
         case 12:
            sprintf(cycle,"%s","(0+3+4+5)");
            break;
         case 14:
            sprintf(cycle,"%s","0(+3+4+5)0");
            break;
      }
   }
   else
   {
      switch (ams_data -> cycle_type)
      {
         case 0:
            sprintf(cycle,"%s","010");
            break;
         case 1:
         default:
            sprintf(cycle,"%s","01210");
            break;
         case 2:
            sprintf(cycle,"%s","(0+1+2)");
            break;
         case 3:
            sprintf(cycle,"%s","02120");
            break;
         case 4:
            sprintf(cycle,"%s","(010+2)");
            break;
         case 5:
            sprintf(cycle,"%s","0102010");
            break;
         case 6:
            sprintf(cycle,"%s","(020+1)");
            break;
         case 7:
            sprintf(cycle,"%s","0201020");
            break;
         case 8:
            sprintf(cycle,"%s","0(+1+2)0");
            break;
         case 9:
            sprintf(cycle,"%s","01210");
            break;
         case 11:
            sprintf(cycle,"%s","013454310");
            break;
         case 12:
            sprintf(cycle,"%s","(0+1+3+4+5)");
            break;
         case 13:
            sprintf(cycle,"%s","034515430");
            break;
         case 14:
            sprintf(cycle,"%s","01(+3+4+5)10");
            break;
         case 20:
            sprintf(cycle,"%s","020");
            break;
      }
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
      hypre_ParCSRSubspacePrec(ams_data -> A,
                               ams_data -> A_relax_type,
                               ams_data -> A_relax_times,
                               ams_data -> A_l1_norms,
                               ams_data -> A_relax_weight,
                               ams_data -> A_omega,
                               Ai, Bi, Pi, ri, gi,
                               b, x,
                               ams_data -> r0,
                               ams_data -> g0,
                               cycle,
                               z);

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

   if (z)
      hypre_ParVectorDestroy(z);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRSubspacePrec
 *
 * General subspace preconditioner for A0 y = x, based on ParCSR storage.
 *
 * P[i] and A[i] are the interpolation and coarse grid matrices for
 * the (i+1)'th subspace. B[i] is an AMG solver for A[i]. r[i] and g[i]
 * are temporary vectors. A0_* are the fine grid smoothing parameters.
 *
 * The default mode is multiplicative, '+' changes the next correction
 * to additive, based on residual computed at '('.
 *--------------------------------------------------------------------------*/

int hypre_ParCSRSubspacePrec(/* fine space matrix */
                             hypre_ParCSRMatrix *A0,
                             /* relaxation parameters */
                             int A0_relax_type,
                             int A0_relax_times,
                             double *A0_l1_norms,
                             double A0_relax_weight,
                             double A0_omega,
                             /* subspace matrices */
                             hypre_ParCSRMatrix **A,
                             /* subspace preconditioners */
                             HYPRE_Solver *B,
                             /* subspace interpolations */
                             hypre_ParCSRMatrix **P,
                             /* temporary subspace vectors */
                             hypre_ParVector **r,
                             hypre_ParVector **g,
                             /* right-hand side */
                             hypre_ParVector *x,
                             /* current approximation */
                             hypre_ParVector *y,
                             /* current residual */
                             hypre_ParVector *r0,
                             /* temporary vector */
                             hypre_ParVector *g0,
                             char *cycle,
                             /* temporary vector */
                             hypre_ParVector *z)
{
   char *op;
   int use_saved_residual = 0;

   for (op = cycle; *op != '\0'; op++)
   {
      /* do nothing */
      if (*op == ')')
         continue;

      /* compute the residual: r = x - Ay */
      else if (*op == '(')
      {
         hypre_ParVectorCopy(x,r0);
         hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      }

      /* switch to additive correction */
      else if (*op == '+')
      {
         use_saved_residual = 1;
         continue;
      }

      /* smooth: y += S (x - Ay) */
      else if (*op == '0')
      {
         hypre_ParCSRRelax(A0, x,
                           A0_relax_type,
                           A0_relax_times,
                           A0_l1_norms,
                           A0_relax_weight,
                           A0_omega,
                           y, g0, z);
      }

      /* subspace correction: y += P B^{-1} P^t r */
      else
      {
         int i = *op - '1';
         if (i < 0)
            hypre_error_in_arg(16);

         /* skip empty subspaces */
         if (!A[i]) continue;

         /* compute the residual? */
         if (use_saved_residual)
         {
            use_saved_residual = 0;
            hypre_ParCSRMatrixMatvecT(1.0, P[i], r0, 0.0, r[i]);
         }
         else
         {
            hypre_ParVectorCopy(x,g0);
            hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, g0);
            hypre_ParCSRMatrixMatvecT(1.0, P[i], g0, 0.0, r[i]);
         }

         hypre_ParVectorSetConstantValues(g[i], 0.0);
         if (i == 1)
            hypre_BoomerAMGBlockSolve((void *)B[i], A[i], r[i], g[i]);
         else
            hypre_BoomerAMGSolve((void *)B[i], A[i], r[i], g[i]);
         hypre_ParCSRMatrixMatvec(1.0, P[i], g[i], 0.0, g0);
         hypre_ParVectorAxpy(1.0, g0, y);
      }
   }

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
 * and that the orientation of all edges is consistent. More specificaly:
 * If edge_orientation = 1, the edges are already oriented.
 * If edge_orientation = 2, the orientation of edge i depends only on the
 *                          sign of edge_vertex[2*i+1] - edge_vertex[2*i].
 *--------------------------------------------------------------------------*/

int hypre_AMSConstructDiscreteGradient(hypre_ParCSRMatrix *A,
                                       hypre_ParVector *x_coord,
                                       int *edge_vertex,
                                       int edge_orientation,
                                       hypre_ParCSRMatrix **G_ptr)
{
   hypre_ParCSRMatrix *G;

   int nedges;

   nedges = hypre_ParCSRMatrixNumRows(A);

   /* Construct the local part of G based on edge_vertex and the edge
      and vertex partitionings from A and x_coord */
   {
      int i, *I = hypre_CTAlloc(int, nedges+1);
      int part_size, *row_starts, *col_starts;
      double *data = hypre_CTAlloc(double, 2*nedges);
      hypre_CSRMatrix *local = hypre_CSRMatrixCreate (nedges,
                                                      hypre_ParVectorGlobalSize(x_coord),
                                                      2*nedges);

      for (i = 0; i <= nedges; i++)
         I[i] = 2*i;

      if (edge_orientation == 1)
      {
         /* Assume that the edges are already oriented */
         for (i = 0; i < 2*nedges; i+=2)
         {
            data[i]   = -1.0;
            data[i+1] =  1.0;
         }
      }
      else if (edge_orientation == 2)
      {
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
      }
      else
         hypre_error_in_arg(4);


      hypre_CSRMatrixI(local) = I;
      hypre_CSRMatrixJ(local) = edge_vertex;
      hypre_CSRMatrixData(local) = data;

      hypre_CSRMatrixRownnz(local) = NULL;
      hypre_CSRMatrixOwnsData(local) = 1;
      hypre_CSRMatrixNumRownnz(local) = nedges;

      /* Copy partitioning from A and x_coord (previously they were re-used) */
#ifdef HYPRE_NO_GLOBAL_PARTITION
      part_size = 2;
#else
      MPI_Comm_size(hypre_ParCSRMatrixComm(A), &part_size);
      part_size++;
#endif
      row_starts = hypre_TAlloc(int,part_size);
      col_starts = hypre_TAlloc(int,part_size);
      for (i = 0; i < part_size; i++)
      {
         row_starts[i] = hypre_ParCSRMatrixRowStarts(A)[i];
         col_starts[i] = hypre_ParVectorPartitioning(x_coord)[i];
      }

      /* Generate the discrete gradient matrix */
      G = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                   hypre_ParCSRMatrixGlobalNumRows(A),
                                   hypre_ParVectorGlobalSize(x_coord),
                                   row_starts, col_starts, 0, 0, 0);
      hypre_ParCSRMatrixOwnsRowStarts(G) = 1;
      hypre_ParCSRMatrixOwnsColStarts(G) = 1;

      GenerateDiagAndOffd(local, G,
                          hypre_ParVectorFirstIndex(x_coord),
                          hypre_ParVectorLastIndex(x_coord));


      /* Account for empty rows in G. These may appear when A includes only
         the interior (non-Dirichlet b.c.) edges. */
      {
         hypre_CSRMatrix *G_diag = hypre_ParCSRMatrixDiag(G);
         G_diag->num_cols = hypre_VectorSize(hypre_ParVectorLocalVector(x_coord));
      }

      /* Free the local matrix */
      hypre_CSRMatrixJ(local) = NULL;
      hypre_CSRMatrixDestroy(local);
   }

   *G_ptr = G;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSFEISetup
 *
 * Construct an AMS solver object based on the following data:
 *
 *    A              - the edge element stiffness matrix
 *    num_vert       - number of vertices (nodes) in the processor
 *    num_local_vert - number of vertices owned by the processor
 *    vert_number    - global indexes of the vertices in the processor
 *    vert_coord     - coordinates of the vertices in the processor
 *    num_edges      - number of edges owned by the processor
 *    edge_vertex    - the vertices of the edges owned by the processor.
 *                     Vertices are in local numbering (the same as in
 *                     vert_number), and edge orientation is always from
 *                     the first to the second vertex.
 *
 * Here we distinguish between vertices that belong to elements in the
 * current processor, and the subset of these vertices that is owned by
 * the processor.
 *
 * This function is written specifically for input from the FEI and should
 * be called before hypre_AMSSetup().
 *--------------------------------------------------------------------------*/

int hypre_AMSFEISetup(void *solver,
                      hypre_ParCSRMatrix *A,
                      hypre_ParVector *b,
                      hypre_ParVector *x,
                      int num_vert,
                      int num_local_vert,
                      int *vert_number,
                      double *vert_coord,
                      int num_edges,
                      int *edge_vertex)
{
   hypre_AMSData *ams_data = solver;

   int i, j;

   hypre_ParCSRMatrix *G;
   hypre_ParVector *x_coord, *y_coord, *z_coord;
   double *x_data, *y_data, *z_data;

   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   int *vert_part, num_global_vert;
   int vert_start, vert_end;

   /* Find the processor partitioning of the vertices */
#ifdef HYPRE_NO_GLOBAL_PARTITION
   vert_part = hypre_TAlloc(int,2);
   MPI_Scan(&num_local_vert, &vert_part[1], 1, MPI_INT, MPI_SUM, comm);
   vert_part[0] = vert_part[1] - num_local_vert;
   MPI_Allreduce(&num_local_vert, &num_global_vert, 1, MPI_INT, MPI_SUM, comm);
#else
   int num_procs;
   MPI_Comm_size(comm, &num_procs);
   vert_part = hypre_TAlloc(int,num_procs+1);
   MPI_Allgather(&num_local_vert, 1, MPI_INT, &vert_part[1], 1, MPI_INT, comm);
   vert_part[0] = 0;
   for (i = 0; i < num_procs; i++)
      vert_part[i+1] += vert_part[i];
   num_global_vert = vert_part[num_procs];
#endif

   /* Construct hypre parallel vectors for the vertex coordinates */
   x_coord = hypre_ParVectorCreate(comm, num_global_vert, vert_part);
   hypre_ParVectorInitialize(x_coord);
   hypre_ParVectorOwnsData(x_coord) = 1;
   hypre_ParVectorOwnsPartitioning(x_coord) = 0;
   x_data = hypre_VectorData(hypre_ParVectorLocalVector(x_coord));

   y_coord = hypre_ParVectorCreate(comm, num_global_vert, vert_part);
   hypre_ParVectorInitialize(y_coord);
   hypre_ParVectorOwnsData(y_coord) = 1;
   hypre_ParVectorOwnsPartitioning(y_coord) = 0;
   y_data = hypre_VectorData(hypre_ParVectorLocalVector(y_coord));

   z_coord = hypre_ParVectorCreate(comm, num_global_vert, vert_part);
   hypre_ParVectorInitialize(z_coord);
   hypre_ParVectorOwnsData(z_coord) = 1;
   hypre_ParVectorOwnsPartitioning(z_coord) = 0;
   z_data = hypre_VectorData(hypre_ParVectorLocalVector(z_coord));

   vert_start = hypre_ParVectorFirstIndex(x_coord);
   vert_end   = hypre_ParVectorLastIndex(x_coord);

   /* Save coordinates of locally owned vertices */
   for (i = 0; i < num_vert; i++)
   {
      if (vert_number[i] >= vert_start && vert_number[i] <= vert_end)
      {
         j = vert_number[i] - vert_start;
         x_data[j] = vert_coord[3*i];
         y_data[j] = vert_coord[3*i+1];
         z_data[j] = vert_coord[3*i+2];
      }
   }

   /* Change vertex numbers from local to global */
   for (i = 0; i < 2*num_edges; i++)
     edge_vertex[i] = vert_number[edge_vertex[i]];

   /* Construct the local part of G based on edge_vertex */
   {
      /* int num_edges = hypre_ParCSRMatrixNumRows(A); */
      int *I = hypre_CTAlloc(int, num_edges+1);
      double *data = hypre_CTAlloc(double, 2*num_edges);
      hypre_CSRMatrix *local = hypre_CSRMatrixCreate (num_edges,
                                                      num_global_vert,
                                                      2*num_edges);

      for (i = 0; i <= num_edges; i++)
         I[i] = 2*i;

      /* Assume that the edge orientation is based on the vertex indexes */
      for (i = 0; i < 2*num_edges; i+=2)
      {
         data[i]   =  1.0;
         data[i+1] = -1.0;
      }

      hypre_CSRMatrixI(local) = I;
      hypre_CSRMatrixJ(local) = edge_vertex;
      hypre_CSRMatrixData(local) = data;

      hypre_CSRMatrixRownnz(local) = NULL;
      hypre_CSRMatrixOwnsData(local) = 1;
      hypre_CSRMatrixNumRownnz(local) = num_edges;

      G = hypre_ParCSRMatrixCreate(comm,
                                   hypre_ParCSRMatrixGlobalNumRows(A),
                                   num_global_vert,
                                   hypre_ParCSRMatrixRowStarts(A),
                                   vert_part,
                                   0, 0, 0);
      hypre_ParCSRMatrixOwnsRowStarts(G) = 0;
      hypre_ParCSRMatrixOwnsColStarts(G) = 1;

      GenerateDiagAndOffd(local, G, vert_start, vert_end);

      hypre_CSRMatrixJ(local) = NULL;
      hypre_CSRMatrixDestroy(local);
   }

   ams_data -> G = G;

   ams_data -> x = x_coord;
   ams_data -> y = y_coord;
   ams_data -> z = z_coord;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSFEIDestroy
 *
 * Free the additional memory allocated in hypre_AMSFEISetup().
 *
 * This function is written specifically for input from the FEI and should
 * be called before hypre_AMSDestroy().
 *--------------------------------------------------------------------------*/

int hypre_AMSFEIDestroy(void *solver)
{
   hypre_AMSData *ams_data = solver;

   if (ams_data -> G)
      hypre_ParCSRMatrixDestroy(ams_data -> G);

   if (ams_data -> x)
      hypre_ParVectorDestroy(ams_data -> x);
   if (ams_data -> y)
      hypre_ParVectorDestroy(ams_data -> y);
   if (ams_data -> z)
      hypre_ParVectorDestroy(ams_data -> z);

   return hypre_error_flag;
}
