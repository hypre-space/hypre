/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.43 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Geometrically smooth interpolation multigrid
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "headers.h"
#include "par_amg.h"

#ifdef HYPRE_USING_ESSL
#include <essl.h>
#else
#include "fortran.h"
HYPRE_Int hypre_F90_NAME_LAPACK(dgels, DGELS)(char *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, double *, 
  HYPRE_Int *, double *, HYPRE_Int *, double *, HYPRE_Int *, HYPRE_Int *);
#endif

#ifndef ABS
#define ABS(x) ((x)>0 ? (x) : -(x))
#endif
#ifndef MAX
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif

static double mydnrm2(HYPRE_Int n, double *x)
{
    double temp = 0.;
    HYPRE_Int i;

    for (i=0; i<n; i++)
        temp = temp + x[i]*x[i];
    return sqrt(temp);
}

static void mydscal(HYPRE_Int n, double a, double *x)
{
    HYPRE_Int i;

    for (i=0; i<n; i++)
        x[i] = a * x[i];
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixClone
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixClone(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **Sp,
   HYPRE_Int copy_data)
{
   MPI_Comm            comm            = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int                *A_offd_i        = hypre_CSRMatrixI(A_offd);

   HYPRE_Int                *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int                 n               = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int                 num_cols_offd     = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int                 num_nonzeros_diag = A_diag_i[n];
   HYPRE_Int                 num_nonzeros_offd = A_offd_i[n];

   hypre_ParCSRMatrix *S;

   S = hypre_ParCSRMatrixCreate(comm, n, n, row_starts, row_starts,
       num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
   hypre_ParCSRMatrixSetRowStartsOwner(S,0);
   hypre_ParCSRMatrixInitialize(S); /* allocate memory */

   hypre_ParCSRMatrixCopy(A,S,copy_data);

   *Sp = S;

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixFillSmooth
 * - fill in smooth matrix
 * - this function will scale the smooth vectors
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixFillSmooth(HYPRE_Int nsamples, double *samples, 
  hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
  HYPRE_Int num_functions, HYPRE_Int *dof_func)
{
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   HYPRE_Int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);
   double             *S_diag_data     = hypre_CSRMatrixData(S_diag);
   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   HYPRE_Int                *S_offd_j        = hypre_CSRMatrixJ(S_offd);
   double             *S_offd_data     = hypre_CSRMatrixData(S_offd);
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);
   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   double             *A_offd_data     = hypre_CSRMatrixData(A_offd);
   HYPRE_Int                 n               = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int i, j, k, ii, index, start;
   HYPRE_Int num_cols_offd;
   HYPRE_Int num_sends;
   HYPRE_Int *dof_func_offd;
   HYPRE_Int *int_buf_data;
   double temp;
   double *p;
   double *p_offd;
   double *p_ptr;
   double *buf_data;
   double nm;
#if 0
   double mx = 0., my = 1.e+10;
#endif

   /* normalize each sample vector and divide by number of samples */
   for (k=0; k<nsamples; k++)
   {
       nm = mydnrm2(n, samples+k*n);
       nm = 1./nm/nsamples;
       mydscal(n, nm, samples+k*n);
   }

   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   buf_data = hypre_CTAlloc(double,hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   p_offd = hypre_CTAlloc(double, nsamples*num_cols_offd);
   p_ptr = p_offd;

   p = samples;
   for (k = 0; k < nsamples; k++)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                buf_data[index++]
                 = p[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }

      comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, buf_data,
        p_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      p = p+n;
      p_offd = p_offd+num_cols_offd;
   }

   hypre_TFree(buf_data);

   if (num_functions > 1)
   {
      dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd);
      int_buf_data = hypre_CTAlloc(HYPRE_Int,hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++]
                 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }

      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
        dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      hypre_TFree(int_buf_data);
   }

   for (i = 0; i < n; i++)
   {
       for (j = S_diag_i[i]+1; j < S_diag_i[i+1]; j++)
       {
           ii = S_diag_j[j];

           /* only interpolate between like functions */
           if (num_functions > 1 && dof_func[i] != dof_func[ii])
           {
               S_diag_data[j] = 0.;
               continue;
           }

           /* explicit zeros */
           if (A_diag_data[j] == 0.)
           {
               S_diag_data[j] = 0.;
               continue;
           }

           temp = 0.;
           p = samples;
           for (k=0; k<nsamples; k++)
           {
               temp = temp + ABS(p[i] - p[ii]);
               p = p + n;
           }

           /* explicit zeros in matrix may cause this */
           if (temp == 0.)
           {
               S_diag_data[j] = 0.;
               continue;
           }

           temp = 1./temp; /* reciprocal */
#if 0
           my = hypre_min(my,temp);
           mx = hypre_max(mx,temp);
#endif
           S_diag_data[j] = temp;
       }

       for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
       {
           ii = S_offd_j[j];

           /* only interpolate between like functions */
           if (num_functions > 1 && dof_func[i] != dof_func_offd[ii])
           {
               S_offd_data[j] = 0.;
               continue;
           }

           /* explicit zeros */
           if (A_offd_data[j] == 0.)
           {
               S_offd_data[j] = 0.;
               continue;
           }

           temp = 0.;
           p = samples;
           p_offd = p_ptr;
           for (k=0; k<nsamples; k++)
           {
               temp = temp + ABS(p[i] - p_offd[ii]);
               p = p + n;
               p_offd = p_offd + num_cols_offd;
           }

           /* explicit zeros in matrix may cause this */
           if (temp == 0.)
           {
               S_offd_data[j] = 0.;
               continue;
           }

           temp = 1./temp; /* reciprocal */
#if 0
           my = hypre_min(my,temp);
           mx = hypre_max(mx,temp);
#endif
           S_offd_data[j] = temp;
       }
   }

#if 0
      hypre_printf("MIN, MAX: %f %f\n", my, mx);
#endif

   hypre_TFree(p_ptr);
   if (num_functions > 1)
      hypre_TFree(dof_func_offd);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixChooseThresh
 *--------------------------------------------------------------------------*/

double
hypre_ParCSRMatrixChooseThresh(hypre_ParCSRMatrix *S)
{
   MPI_Comm            comm            = hypre_ParCSRMatrixComm(S);

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   HYPRE_Int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   double             *S_diag_data     = hypre_CSRMatrixData(S_diag);
   double             *S_offd_data     = hypre_CSRMatrixData(S_offd);
   HYPRE_Int                 n               = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int i, j;
   double mx, minimax = 1.e+10;
   double minmin;

   for (i=0; i<n; i++)
   {
      mx = 0.;
      for (j=S_diag_i[i]; j<S_diag_i[i+1]; j++)
         mx = hypre_max(mx, S_diag_data[j]);
      for (j=S_offd_i[i]; j<S_offd_i[i+1]; j++)
         mx = hypre_max(mx, S_offd_data[j]);

      if (mx != 0.)
         minimax = hypre_min(minimax, mx);
   }

   hypre_MPI_Allreduce(&minimax, &minmin, 1, hypre_MPI_DOUBLE, hypre_MPI_MIN, comm);

   return minmin;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixThreshold(hypre_ParCSRMatrix *A, double thresh)
{
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   double             *A_offd_data     = hypre_CSRMatrixData(A_offd);

   HYPRE_Int                 n               = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int                 num_nonzeros_diag = A_diag_i[n];
   HYPRE_Int                 num_nonzeros_offd = A_offd_i[n];

   HYPRE_Int                *S_diag_i;
   HYPRE_Int                *S_diag_j;
   double             *S_diag_data;
   HYPRE_Int                *S_offd_i;
   HYPRE_Int                *S_offd_j;
   double             *S_offd_data;

   HYPRE_Int count, i, jS, jA;

   /* first count the number of nonzeros we will need */
   count = 0;
   for (i=0; i<num_nonzeros_diag; i++)
       if (A_diag_data[i] >= thresh)
           count++;

   /* allocate vectors */
   S_diag_i = hypre_CTAlloc(HYPRE_Int, n+1);
   S_diag_j = hypre_CTAlloc(HYPRE_Int, count);
   S_diag_data = hypre_CTAlloc(double, count);

   jS = 0;
   for (i = 0; i < n; i++)
   {
      S_diag_i[i] = jS;
      for (jA = A_diag_i[i]; jA < A_diag_i[i+1]; jA++)
      {
         if (A_diag_data[jA] >= thresh)
         {
            S_diag_data[jS] = A_diag_data[jA];
            S_diag_j[jS] = A_diag_j[jA];
            jS++;
         }
      }
   }
   S_diag_i[n] = jS;
   hypre_CSRMatrixNumNonzeros(A_diag) = jS;

   /* free the vectors we don't need */
   hypre_TFree(A_diag_i);
   hypre_TFree(A_diag_j);
   hypre_TFree(A_diag_data);

   /* assign the new vectors */
   hypre_CSRMatrixI(A_diag) = S_diag_i;
   hypre_CSRMatrixJ(A_diag) = S_diag_j;
   hypre_CSRMatrixData(A_diag) = S_diag_data;

   /*
    * Offd part
    */

   /* first count the number of nonzeros we will need */
   count = 0;
   for (i=0; i<num_nonzeros_offd; i++)
       if (A_offd_data[i] >= thresh)
           count++;

   /* allocate vectors */
   S_offd_i = hypre_CTAlloc(HYPRE_Int, n+1);
   S_offd_j = hypre_CTAlloc(HYPRE_Int, count);
   S_offd_data = hypre_CTAlloc(double, count);

   jS = 0;
   for (i = 0; i < n; i++)
   {
      S_offd_i[i] = jS;
      for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
      {
         if (A_offd_data[jA] >= thresh)
         {
            S_offd_data[jS] = A_offd_data[jA];
            S_offd_j[jS] = A_offd_j[jA];
            jS++;
         }
      }
   }
   S_offd_i[n] = jS;
   hypre_CSRMatrixNumNonzeros(A_offd) = jS;

   /* free the vectors we don't need */
   hypre_TFree(A_offd_i);
   hypre_TFree(A_offd_j);
   hypre_TFree(A_offd_data);

   /* assign the new vectors */
   hypre_CSRMatrixI(A_offd) = S_offd_i;
   hypre_CSRMatrixJ(A_offd) = S_offd_j;
   hypre_CSRMatrixData(A_offd) = S_offd_data;

   return 0;
}

/*--------------------------------------------------------------------------
 * CreateSmoothVecs
 * - smoother depends on the level being used
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCreateSmoothVecs(void         *data,
                       hypre_ParCSRMatrix    *A,
                       HYPRE_Int                    num_sweeps,
                       HYPRE_Int                    level,
                       double               **SmoothVecs_p)
{
   hypre_ParAMGData  *amg_data = data;

   MPI_Comm             comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);

   hypre_ParVector *Zero;
   hypre_ParVector *Temp;
   hypre_ParVector *U;

   hypre_ParVector    *Qtemp = NULL;

   HYPRE_Int    i;
   HYPRE_Int    n = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int    n_local = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int   *starts = hypre_ParCSRMatrixRowStarts(A);

   HYPRE_Int sample;
   HYPRE_Int nsamples = hypre_ParAMGDataNumSamples(amg_data);
   HYPRE_Int ret;
   double *datax, *bp, *p;

   HYPRE_Int rlx_type;
   HYPRE_Int smooth_type;
   HYPRE_Int smooth_option = 0;
   HYPRE_Int smooth_num_levels;
   HYPRE_Solver *smoother;

   HYPRE_Int debug_flag = hypre_ParAMGDataDebugFlag(amg_data);
   HYPRE_Int num_threads;

   num_threads = hypre_NumThreads();

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);

        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   if (debug_flag >= 1)
      hypre_printf("Creating smooth dirs, %d sweeps, %d samples\n", num_sweeps, 
         nsamples);

   smooth_type = hypre_ParAMGDataSmoothType(amg_data);
   smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   if (smooth_num_levels > level)
   {
      smooth_option = smooth_type;
      smoother = hypre_ParAMGDataSmoother(amg_data);
      num_sweeps = hypre_ParAMGDataSmoothNumSweeps(amg_data);
   }
   rlx_type = hypre_ParAMGDataGridRelaxType(amg_data)[0];
   /* rlx_wt = hypre_ParAMGDataRelaxWeight(amg_data)[level]; */
   /* omega = hypre_ParAMGDataOmega(amg_data)[level]; */

   /* generate par vectors */

   Zero = hypre_ParVectorCreate(comm, n, starts);
   hypre_ParVectorSetPartitioningOwner(Zero,0);
   hypre_ParVectorInitialize(Zero);
   datax = hypre_VectorData(hypre_ParVectorLocalVector(Zero));
   for (i=0; i<n_local; i++)
       datax[i] = 0.;

   Temp = hypre_ParVectorCreate(comm, n, starts);
   hypre_ParVectorSetPartitioningOwner(Temp,0);
   hypre_ParVectorInitialize(Temp);
   datax = hypre_VectorData(hypre_ParVectorLocalVector(Temp));
   for (i=0; i<n_local; i++)
       datax[i] = 0.;

   U = hypre_ParVectorCreate(comm, n, starts);
   hypre_ParVectorSetPartitioningOwner(U,0);
   hypre_ParVectorInitialize(U);
   datax = hypre_VectorData(hypre_ParVectorLocalVector(U));


   if (num_threads > 1)
   {
      Qtemp = hypre_ParVectorCreate(comm, n, starts);
      hypre_ParVectorInitialize(Qtemp);
      hypre_ParVectorSetPartitioningOwner(Qtemp,0);
   }






   /* allocate space for the vectors */
   bp = hypre_CTAlloc(double, nsamples*n_local);
   p = bp;

   /* generate random vectors */
   for (sample=0; sample<nsamples; sample++)
   {
       for (i=0; i<n_local; i++)
           datax[i] = (rand()/(double)RAND_MAX) - .5;

       for (i=0; i<num_sweeps; i++)
       {
	   if (smooth_option == 6)
	   {
	      HYPRE_SchwarzSolve(smoother[level],
			(HYPRE_ParCSRMatrix) A, 
			(HYPRE_ParVector) Zero,
			(HYPRE_ParVector) U);
	   }
	   else
	   {
              ret = hypre_BoomerAMGRelax(A, Zero, NULL /*CFmarker*/,
                rlx_type , 0 /*rel pts*/, 1.0 /*weight*/, 
                                         1.0 /*omega*/, NULL, U, Temp, 
                                         Qtemp);
              hypre_assert(ret == 0);
	   }
       }

       /* copy out the solution */
       for (i=0; i<n_local; i++)
           *p++ = datax[i];
   }

   hypre_ParVectorDestroy(Zero);
   hypre_ParVectorDestroy(Temp);
   hypre_ParVectorDestroy(U);
   if (num_threads > 1)
      hypre_ParVectorDestroy(Qtemp);

   *SmoothVecs_p = bp;

   return 0;
}

/*--------------------------------------------------------------------------
 * CreateSmoothDirs replaces CreateS in AMG
 * - smoother depends on the level being used
 * - in this version, CreateSmoothVecs must be called prior to this function
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCreateSmoothDirs(void         *data,
                       hypre_ParCSRMatrix    *A,
                       double                *SmoothVecs,
                       double                 thresh,
                       HYPRE_Int                    num_functions, 
                       HYPRE_Int                   *dof_func,
                       hypre_ParCSRMatrix   **S_ptr)
{
   hypre_ParAMGData  *amg_data = data;
   hypre_ParCSRMatrix *S;
   double minimax;
   HYPRE_Int debug_flag = hypre_ParAMGDataDebugFlag(amg_data);

   hypre_ParCSRMatrixClone(A, &S, 0);

   /* Traverse S and fill in differences */
   hypre_ParCSRMatrixFillSmooth(
       hypre_ParAMGDataNumSamples(amg_data), SmoothVecs,
       S, A, num_functions, dof_func);

   minimax = hypre_ParCSRMatrixChooseThresh(S);
   if (debug_flag >= 1)
      hypre_printf("Minimax chosen: %f\n", minimax);

   /* Threshold and compress */
   hypre_ParCSRMatrixThreshold(S, thresh*minimax);

   *S_ptr = S;

   return 0;
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGNormalizeVecs
 *
 * Normalize the smooth vectors and also make the first vector the constant
 * vector
 *
 * inputs:
 * n   = length of smooth vectors
 * num = number of smooth vectors
 * V   = smooth vectors (array of length n*num), also an output
 * 
 * output:
 * V   = adjusted smooth vectors
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGNormalizeVecs(HYPRE_Int n, HYPRE_Int num, double *V)
{
   HYPRE_Int i, j;
   double nrm;

   /* change first vector to the constant vector */
   for (i=0; i<n; i++)
      V[i] = 1.0;

   for (j=0; j<num; j++)
   {
       nrm = mydnrm2(n, &V[j*n]);
       mydscal(n, 1./nrm, &V[j*n]);
   }

   return 0;
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGFitVectors
 *
 * Construct interpolation weights based on fitting smooth vectors
 *
 * inputs:
 * ip  = row number of row in P being processed (0-based)
 * n   = length of smooth vectors
 * num = number of smooth vectors
 * V   = smooth vectors (array of length n*num), also an output
 * nc  = number of coarse grid points
 * ind = indices of coarse grid points (0-based)
 * 
 * output:
 * val = interpolation weights for the coarse grid points
 * V   = smooth vectors; first one has been changed to constant vector;
 *       vectors have also been normalized; this is also an input
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGFitVectors(HYPRE_Int ip, HYPRE_Int n, HYPRE_Int num, const double *V, 
  HYPRE_Int nc, const HYPRE_Int *ind, double *val)
{
   double *a, *b;
   double *ap;
   HYPRE_Int i, j;
   double *work;
   HYPRE_Int    work_size;
   HYPRE_Int    info;
   HYPRE_Int  temp;

/*
   hypre_printf("Fit: row %d, n %d num %d, nc = %d ", ip, n, num, nc);
   for (i=0; i<nc; i++)
      hypre_printf("%d ", ind[i]);
   hypre_printf("\n");
*/

   if (nc == 0)
      return 0;

   work_size = 2000*64;
   work = hypre_CTAlloc(double, work_size);

   a = hypre_CTAlloc(double, num*nc);
   ap = a;

   for (j=0; j<nc; j++)
   {
      for (i=0; i<num; i++)
      {
          *ap = V[i*n+ind[j]];
	  ap++;
      }
   }

   temp = MAX(nc, num);
   b = hypre_CTAlloc(double, temp);
   for (i=0; i<num; i++)
      b[i] = V[i*n+ip];

#ifdef HYPRE_USING_ESSL
   dgells(0, a, num, b, num, val, nc, NULL, 1.e-12, num, nc, 1, 
      &info, work, work_size);
#else
   {
   char trans = 'N';
   HYPRE_Int  one   = 1;
   hypre_F90_NAME_LAPACK(dgels, DGELS)(&trans, &num, &nc, &one, a, &num,
      b, &temp, work, &work_size, &info);

   if (info != 0)
      hypre_printf("par_gsmg: dgels returned %d\n", info);

   /* copy solution into output vector */
   for (j=0; j<nc; j++)
      val[j] = b[j];
   }
#endif

   hypre_TFree(b);
   hypre_TFree(a);
   hypre_TFree(work);

   return info;
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildInterpLS
 *
 * Interpolation built from fitting smooth vectors
 * - sequential version only
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildInterpLS( hypre_ParCSRMatrix   *A,
                         HYPRE_Int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         HYPRE_Int                  *num_cpts_global,
                         HYPRE_Int                   num_functions,
                         HYPRE_Int                  *dof_func,
                         HYPRE_Int                   debug_flag,
                         double                trunc_factor,
                         HYPRE_Int                   num_smooth,
                         double               *SmoothVecs,
                         hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(S);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
/* double          *S_diag_data = hypre_CSRMatrixData(S_diag); */
   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
/* double          *S_offd_data = hypre_CSRMatrixData(S_offd);
   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd); */

   HYPRE_Int              num_cols_S_offd = hypre_CSRMatrixNumCols(S_offd);
/* HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(S); */

   hypre_ParCSRMatrix *P;
   HYPRE_Int		      *col_map_offd_P;

   HYPRE_Int             *CF_marker_offd;
   HYPRE_Int             *dof_func_offd = NULL;

   hypre_CSRMatrix *S_ext;
   
/* double          *S_ext_data;
   HYPRE_Int             *S_ext_i;
   HYPRE_Int             *S_ext_j; */

   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;   

   double          *P_diag_data;
   HYPRE_Int             *P_diag_i;
   HYPRE_Int             *P_diag_j;
   double          *P_offd_data;
   HYPRE_Int             *P_offd_i;
   HYPRE_Int             *P_offd_j;

   HYPRE_Int              P_diag_size, P_offd_size;
   
   HYPRE_Int             *P_marker;
/* HYPRE_Int             *P_marker_offd; */

   HYPRE_Int              jj_counter,jj_counter_offd;
   HYPRE_Int             *jj_count, *jj_count_offd;
/* HYPRE_Int              jj_begin_row,jj_begin_row_offd;
   HYPRE_Int              jj_end_row,jj_end_row_offd; */
   
   HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int              n_fine = hypre_CSRMatrixNumRows(S_diag);

   HYPRE_Int             *fine_to_coarse;
   HYPRE_Int             *fine_to_coarse_offd;
   HYPRE_Int             *coarse_counter;
   HYPRE_Int              coarse_shift;
   HYPRE_Int              total_global_cpts;
   HYPRE_Int              num_cols_P_offd,my_first_cpt;

   HYPRE_Int              i,i1;
   HYPRE_Int              j,jl,jj;
   HYPRE_Int              start;
   
   double           one  = 1.0;
   
   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads;
   HYPRE_Int              num_sends;
   HYPRE_Int              index;
   HYPRE_Int              ns, ne, size, rest;
   HYPRE_Int             *int_buf_data;

   double           wall_time;  /* for debugging instrumentation  */

   hypre_MPI_Comm_size(comm, &num_procs);   
   hypre_MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_S_offd);
   if (num_functions > 1 && num_cols_S_offd)
	dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_S_offd);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(S);
	comm_pkg = hypre_ParCSRMatrixCommPkg(S); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends));

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }
	
   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	CF_marker_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);   
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
	
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);   
   }

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of S
    *---------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S,S,1);
/*
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
      S_ext_data = hypre_CSRMatrixData(S_ext);
*/
   }
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d  Interp: Comm 2   Get S_ext =  %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int, num_threads);
   jj_count = hypre_CTAlloc(HYPRE_Int, num_threads);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int, num_threads);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] = -1;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
      
   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

/* RDF: this looks a little tricky, but doable */
#define HYPRE_SMP_PRIVATE i,j,i1,jj,ns,ne,size,rest
#include "../utilities/hypre_smp_forloop.h"
   for (j = 0; j < num_threads; j++)
   {
     size = n_fine/num_threads;
     rest = n_fine - size*num_threads;
     if (j < rest)
     {
        ns = j*size+j;
        ne = (j+1)*size+j+1;
     }
     else
     {
        ns = j*size+rest;
        ne = (j+1)*size+rest;
     }
     for (i = ns; i < ne; i++)
     {
      
      /*--------------------------------------------------------------------
       *  If i is a C-point, interpolation is the identity. Also set up
       *  mapping vector.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         jj_count[j]++;
         fine_to_coarse[i] = coarse_counter[j];
         coarse_counter[j]++;
      }
      
      /*--------------------------------------------------------------------
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i.
       *--------------------------------------------------------------------*/

      else
      {
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];           
            if (CF_marker[i1] >= 0)
            {
               jj_count[j]++;
            }
         }

         if (num_procs > 1)
         {
            /* removed */
         }
      }
    }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   for (i=0; i < num_threads-1; i++)
   {
      coarse_counter[i+1] += coarse_counter[i];
      jj_count[i+1] += jj_count[i];
      jj_count_offd[i+1] += jj_count_offd[i];
   }
   i = num_threads-1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(HYPRE_Int, n_fine+1);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int, P_diag_size);
   P_diag_data = hypre_CTAlloc(double, P_diag_size);

   P_diag_i[n_fine] = jj_counter; 


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(HYPRE_Int, n_fine+1);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int, P_offd_size);
   P_offd_data = hypre_CTAlloc(double, P_offd_size);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/ 

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   fine_to_coarse_offd = hypre_CTAlloc(HYPRE_Int, num_cols_S_offd); 

#define HYPRE_SMP_PRIVATE i,j,ns,ne,size,rest,coarse_shift
#include "../utilities/hypre_smp_forloop.h"
   for (j = 0; j < num_threads; j++)
   {
     coarse_shift = 0;
     if (j > 0) coarse_shift = coarse_counter[j-1];
     size = n_fine/num_threads;
     rest = n_fine - size*num_threads;
     if (j < rest)
     {
        ns = j*size+j;
        ne = (j+1)*size+j+1;
     }
     else
     {
        ns = j*size+rest;
        ne = (j+1)*size+rest;
     }
     for (i = ns; i < ne; i++)
	fine_to_coarse[i] += my_first_cpt+coarse_shift;
   }
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }
	
   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	fine_to_coarse_offd);  

   hypre_ParCSRCommHandleDestroy(comm_handle);   

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
    
#define HYPRE_SMP_PRIVATE i,j,jl,i1,jj,ns,ne,size,rest,P_marker,jj_counter,jj_counter_offd
#include "../utilities/hypre_smp_forloop.h"
   for (jl = 0; jl < num_threads; jl++)
   {
     size = n_fine/num_threads;
     rest = n_fine - size*num_threads;
     if (jl < rest)
     {
        ns = jl*size+jl;
        ne = (jl+1)*size+jl+1;
     }
     else
     {
        ns = jl*size+rest;
        ne = (jl+1)*size+rest;
     }
     jj_counter = 0;
     if (jl > 0) jj_counter = jj_count[jl-1];
     jj_counter_offd = 0;
     if (jl > 0) jj_counter_offd = jj_count_offd[jl-1];

     for (i = ns; i < ne; i++)
     {
             
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/
      
      if (CF_marker[i] >= 0)
      {
         P_diag_i[i] = jj_counter;
         P_diag_j[jj_counter]    = fine_to_coarse[i];
         P_diag_data[jj_counter] = one;
         jj_counter++;
      }
      
      /*--------------------------------------------------------------------
       *  If i is an F-point, build interpolation.
       *--------------------------------------------------------------------*/

      else
      {         
         HYPRE_Int kk;
         HYPRE_Int indices[1000]; /* kludge */

         /* Diagonal part of P */
         P_diag_i[i] = jj_counter;

         kk = 0;
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];   

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               P_diag_j[jj_counter]    = fine_to_coarse[i1];
               jj_counter++;
               indices[kk] = i1;
               kk++;
            }
         }

         hypre_BoomerAMGFitVectors(i, n_fine, num_smooth, SmoothVecs, 
            kk, indices, &P_diag_data[P_diag_i[i]]);

         /* Off-Diagonal part of P */
         /* undone */
      }
     }
   }
   P_diag_i[i] = jj_counter; /* check that this is in right place for threads */

   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(S),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(S),
                                num_cpts_global,
                                0,
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);
                                                                                
                                                                                
   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;
                                                                                
   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, 0);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int, P_offd_size);

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i=0; i < P_offd_size; i++)
	 P_marker[i] = P_offd_j[i];

      qsort0(P_marker, 0, P_offd_size-1);

      num_cols_P_offd = 1;
      index = P_marker[0];
      for (i=1; i < P_offd_size; i++)
      {
	if (P_marker[i] > index)
	{
 	  index = P_marker[i];
 	  P_marker[num_cols_P_offd++] = index;
  	}
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_Int,num_cols_P_offd);

      for (i=0; i < num_cols_P_offd; i++)
         col_map_offd_P[i] = P_marker[i];

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i=0; i < P_offd_size; i++)
	P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
					 P_offd_j[i],
					 num_cols_P_offd);
      hypre_TFree(P_marker); 
   }

   if (num_cols_P_offd)
   { 
   	hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
        hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   } 

   hypre_GetCommPkgRTFromCommPkgA(P,S,fine_to_coarse_offd);

   *P_ptr = P;

   hypre_TFree(CF_marker_offd);
   hypre_TFree(dof_func_offd);
   hypre_TFree(int_buf_data);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(fine_to_coarse_offd);
   hypre_TFree(coarse_counter);
   hypre_TFree(jj_count);
   hypre_TFree(jj_count_offd);

   if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);

   return(0);  

}
/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildInterpGSMG
 *
 * Difference with hypre_BoomerAMGBuildInterp is that S contains values
 * and is used to build interpolation weights.  Matrix A is not used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildInterpGSMG( hypre_ParCSRMatrix   *A,
                         HYPRE_Int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         HYPRE_Int                  *num_cpts_global,
                         HYPRE_Int                   num_functions,
                         HYPRE_Int                  *dof_func,
                         HYPRE_Int                   debug_flag,
                         double                trunc_factor,
                         hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(S);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   double          *S_diag_data = hypre_CSRMatrixData(S_diag);
   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
   double          *S_offd_data = hypre_CSRMatrixData(S_offd);
   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   HYPRE_Int              num_cols_S_offd = hypre_CSRMatrixNumCols(S_offd);
   HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(S);

   hypre_ParCSRMatrix *P;
   HYPRE_Int		      *col_map_offd_P;

   HYPRE_Int             *CF_marker_offd;
   HYPRE_Int             *dof_func_offd = NULL;

   hypre_CSRMatrix *S_ext;
   
   double          *S_ext_data;
   HYPRE_Int             *S_ext_i;
   HYPRE_Int             *S_ext_j;

   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;   

   double          *P_diag_data;
   HYPRE_Int             *P_diag_i;
   HYPRE_Int             *P_diag_j;
   double          *P_offd_data;
   HYPRE_Int             *P_offd_i;
   HYPRE_Int             *P_offd_j;

   HYPRE_Int              P_diag_size, P_offd_size;
   
   HYPRE_Int             *P_marker, *P_marker_offd;

   HYPRE_Int              jj_counter,jj_counter_offd;
   HYPRE_Int             *jj_count, *jj_count_offd;
   HYPRE_Int              jj_begin_row,jj_begin_row_offd;
   HYPRE_Int              jj_end_row,jj_end_row_offd;
   
   HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int              n_fine = hypre_CSRMatrixNumRows(S_diag);

   HYPRE_Int              strong_f_marker;

   HYPRE_Int             *fine_to_coarse;
   HYPRE_Int             *fine_to_coarse_offd;
   HYPRE_Int             *coarse_counter;
   HYPRE_Int              coarse_shift;
   HYPRE_Int              total_global_cpts;
   HYPRE_Int              num_cols_P_offd,my_first_cpt;

   HYPRE_Int              i,i1,i2;
   HYPRE_Int              j,jl,jj,jj1;
   HYPRE_Int              start;
   HYPRE_Int              c_num;
   
   double           sum;
   double           distribute;          
   
   double           zero = 0.0;
   double           one  = 1.0;
   
   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads;
   HYPRE_Int              num_sends;
   HYPRE_Int              index;
   HYPRE_Int              ns, ne, size, rest;
   HYPRE_Int             *int_buf_data;

   HYPRE_Int col_1 = hypre_ParCSRMatrixFirstRowIndex(S);
   HYPRE_Int local_numrows = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int col_n = col_1 + local_numrows;

   double           wall_time;  /* for debugging instrumentation  */

   hypre_MPI_Comm_size(comm, &num_procs);   
   hypre_MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   total_global_cpts = 0; /* we will set this later for the matrix in the setup */

   /* if (myid == (num_procs -1)) total_global_cpts = coarse_pts_global[1];
      hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_INT, num_procs-1, comm);*/
#else
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_S_offd);
   if (num_functions > 1 && num_cols_S_offd)
	dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_S_offd);

   if (!comm_pkg)
   {
	hypre_MatvecCommPkgCreate(S);
	comm_pkg = hypre_ParCSRMatrixCommPkg(S); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends));

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }
	
   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	CF_marker_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);   
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
	
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);   
   }

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of S
    *---------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S,S,1);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
      S_ext_data = hypre_CSRMatrixData(S_ext);
   }
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d  Interp: Comm 2   Get S_ext =  %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int, num_threads);
   jj_count = hypre_CTAlloc(HYPRE_Int, num_threads);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int, num_threads);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] = -1;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
      
   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

/* RDF: this looks a little tricky, but doable */
#define HYPRE_SMP_PRIVATE i,j,i1,jj,ns,ne,size,rest
#include "../utilities/hypre_smp_forloop.h"
   for (j = 0; j < num_threads; j++)
   {
     size = n_fine/num_threads;
     rest = n_fine - size*num_threads;
     if (j < rest)
     {
        ns = j*size+j;
        ne = (j+1)*size+j+1;
     }
     else
     {
        ns = j*size+rest;
        ne = (j+1)*size+rest;
     }
     for (i = ns; i < ne; i++)
     {
      
      /*--------------------------------------------------------------------
       *  If i is a C-point, interpolation is the identity. Also set up
       *  mapping vector.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         jj_count[j]++;
         fine_to_coarse[i] = coarse_counter[j];
         coarse_counter[j]++;
      }
      
      /*--------------------------------------------------------------------
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i.
       *--------------------------------------------------------------------*/

      else
      {
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];           
            if (CF_marker[i1] >= 0)
            {
               jj_count[j]++;
            }
         }

         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];           
               if (CF_marker_offd[i1] >= 0)
               {
                  jj_count_offd[j]++;
               }
            }
         }
      }
    }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   for (i=0; i < num_threads-1; i++)
   {
      coarse_counter[i+1] += coarse_counter[i];
      jj_count[i+1] += jj_count[i];
      jj_count_offd[i+1] += jj_count_offd[i];
   }
   i = num_threads-1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(HYPRE_Int, n_fine+1);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int, P_diag_size);
   P_diag_data = hypre_CTAlloc(double, P_diag_size);

   P_diag_i[n_fine] = jj_counter; 


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(HYPRE_Int, n_fine+1);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int, P_offd_size);
   P_offd_data = hypre_CTAlloc(double, P_offd_size);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/ 

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   fine_to_coarse_offd = hypre_CTAlloc(HYPRE_Int, num_cols_S_offd); 

#define HYPRE_SMP_PRIVATE i,j,ns,ne,size,rest,coarse_shift
#include "../utilities/hypre_smp_forloop.h"
   for (j = 0; j < num_threads; j++)
   {
     coarse_shift = 0;
     if (j > 0) coarse_shift = coarse_counter[j-1];
     size = n_fine/num_threads;
     rest = n_fine - size*num_threads;
     if (j < rest)
     {
        ns = j*size+j;
        ne = (j+1)*size+j+1;
     }
     else
     {
        ns = j*size+rest;
        ne = (j+1)*size+rest;
     }
     for (i = ns; i < ne; i++)
	fine_to_coarse[i] += my_first_cpt+coarse_shift;
   }
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }
	
   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	fine_to_coarse_offd);  

   hypre_ParCSRCommHandleDestroy(comm_handle);   

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
    
#define HYPRE_SMP_PRIVATE i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum,distribute,P_marker,P_marker_offd,strong_f_marker,jj_counter,jj_counter_offd,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd
#include "../utilities/hypre_smp_forloop.h"
   for (jl = 0; jl < num_threads; jl++)
   {
     size = n_fine/num_threads;
     rest = n_fine - size*num_threads;
     if (jl < rest)
     {
        ns = jl*size+jl;
        ne = (jl+1)*size+jl+1;
     }
     else
     {
        ns = jl*size+rest;
        ne = (jl+1)*size+rest;
     }
     jj_counter = 0;
     if (jl > 0) jj_counter = jj_count[jl-1];
     jj_counter_offd = 0;
     if (jl > 0) jj_counter_offd = jj_count_offd[jl-1];

     P_marker = hypre_CTAlloc(HYPRE_Int, n_fine);
     P_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_S_offd);

     for (i = 0; i < n_fine; i++)
     {      
        P_marker[i] = -1;
     }
     for (i = 0; i < num_cols_S_offd; i++)
     {      
        P_marker_offd[i] = -1;
     }
     strong_f_marker = -2;
 
     for (i = ns; i < ne; i++)
     {
             
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/
      
      if (CF_marker[i] >= 0)
      {
         P_diag_i[i] = jj_counter;
         P_diag_j[jj_counter]    = fine_to_coarse[i];
         P_diag_data[jj_counter] = one;
         jj_counter++;
      }
      
      /*--------------------------------------------------------------------
       *  If i is an F-point, build interpolation.
       *--------------------------------------------------------------------*/

      else
      {         
         /* Diagonal part of P */
         P_diag_i[i] = jj_counter;
         jj_begin_row = jj_counter;

         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];   

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               P_marker[i1] = jj_counter;
               P_diag_j[jj_counter]    = fine_to_coarse[i1];
               P_diag_data[jj_counter] = zero;
               jj_counter++;
            }

            /*--------------------------------------------------------------
             * If neighbor i1 is an F-point, mark it as a strong F-point
             * whose connection needs to be distributed.
             *--------------------------------------------------------------*/

            else
            {
               P_marker[i1] = strong_f_marker;
            }            
         }
         jj_end_row = jj_counter;

         /* Off-Diagonal part of P */
         P_offd_i[i] = jj_counter_offd;
         jj_begin_row_offd = jj_counter_offd;


         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];   

               /*-----------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_offd_j
                * and initialize interpolation weight to zero.
                *-----------------------------------------------------------*/

               if (CF_marker_offd[i1] >= 0)
               {
                  P_marker_offd[i1] = jj_counter_offd;
		  P_offd_j[jj_counter_offd]  = i1;
                  P_offd_data[jj_counter_offd] = zero;
                  jj_counter_offd++;
               }

               /*-----------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *-----------------------------------------------------------*/

               else
               {
                  P_marker_offd[i1] = strong_f_marker;
               }            
            }
         }
      
         jj_end_row_offd = jj_counter_offd;
         
         /* Loop over ith row of S.  First, the diagonal part of S */

         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];

            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

            if (P_marker[i1] >= jj_begin_row)
            {
               P_diag_data[P_marker[i1]] += S_diag_data[jj];
            }

            /*--------------------------------------------------------------
             * Case 2: neighbor i1 is an F-point and strongly influences i,
             * distribute a_{i,i1} to C-points that strongly infuence i.
             * Note: currently no distribution to the diagonal in this case.
             *--------------------------------------------------------------*/
            
            else if (P_marker[i1] == strong_f_marker)
            {
               sum = zero;
               
               /*-----------------------------------------------------------
                * Loop over row of S for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *-----------------------------------------------------------*/

               /* Diagonal block part of row i1 */
               for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1+1]; jj1++)
               {
                  i2 = S_diag_j[jj1];
                  if (P_marker[i2] >= jj_begin_row)
                     sum += S_diag_data[jj1];
               }

               /* Off-Diagonal block part of row i1 */ 
               if (num_procs > 1)
               {              
                  for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1+1]; jj1++)
                  {
                     i2 = S_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd)
                        sum += S_offd_data[jj1];
                  }
               } 

               if (sum != 0)
	       {
	       distribute = S_diag_data[jj] / sum;
 
               /*-----------------------------------------------------------
                * Loop over row of S for point i1 and do the distribution.
                *-----------------------------------------------------------*/

               /* Diagonal block part of row i1 */
               for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1+1]; jj1++)
               {
                  i2 = S_diag_j[jj1];
                  if (P_marker[i2] >= jj_begin_row)
                     P_diag_data[P_marker[i2]]
                                  += distribute * S_diag_data[jj1];
               }

               /* Off-Diagonal block part of row i1 */
               if (num_procs > 1)
               {
                  for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1+1]; jj1++)
                  {
                     i2 = S_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd)
                        P_offd_data[P_marker_offd[i2]]    
                                  += distribute * S_offd_data[jj1]; 
                  }
               }
               }
               else
               {
                  /* do nothing */
               }
            }
            
            /*--------------------------------------------------------------
             * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
             * into the diagonal.
             *--------------------------------------------------------------*/

            else
            {
               /* do nothing */
            } 

         }    
       

          /*----------------------------------------------------------------
           * Still looping over ith row of S. Next, loop over the 
           * off-diagonal part of S 
           *---------------------------------------------------------------*/

         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];

            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

               if (P_marker_offd[i1] >= jj_begin_row_offd)
               {
                  P_offd_data[P_marker_offd[i1]] += S_offd_data[jj];
               }

               /*------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and strongly influences i,
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.
                *-----------------------------------------------------------*/
            
               else if (P_marker_offd[i1] == strong_f_marker)
               {
                  sum = zero;
               
               /*---------------------------------------------------------
                * Loop over row of S_ext for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *---------------------------------------------------------*/

                  /* find row number */
                  c_num = S_offd_j[jj];

                  for (jj1 = S_ext_i[c_num]; jj1 < S_ext_i[c_num+1]; jj1++)
                  {
                     i2 = S_ext_j[jj1];
                                         
                     if (i2 >= col_1 && i2 < col_n)    
                     {                            
                        /* in the diagonal block */
                        if (P_marker[i2-col_1] >= jj_begin_row)
                           sum += S_ext_data[jj1];
                     }
                     else                       
                     {                          
                        /* in the off_diagonal block  */
                        j = hypre_BinarySearch(col_map_offd,i2,num_cols_S_offd);
                        if (j != -1)
                        { 
                          if (P_marker_offd[j] >= jj_begin_row_offd)
			      sum += S_ext_data[jj1];
                        }
 
                     }

                  }

                  if (sum != 0)
		  {
		  distribute = S_offd_data[jj] / sum;   
                  /*---------------------------------------------------------
                   * Loop over row of S_ext for point i1 and do 
                   * the distribution.
                   *--------------------------------------------------------*/

                  /* Diagonal block part of row i1 */
                          
                  for (jj1 = S_ext_i[c_num]; jj1 < S_ext_i[c_num+1]; jj1++)
                  {
                     i2 = S_ext_j[jj1];

                     if (i2 >= col_1 && i2 < col_n) /* in the diagonal block */           
                     {
                        if (P_marker[i2-col_1] >= jj_begin_row)
                           P_diag_data[P_marker[i2-col_1]]
                                     += distribute * S_ext_data[jj1];
                     }
                     else
                     {
                        /* check to see if it is in the off_diagonal block  */
                        j = hypre_BinarySearch(col_map_offd,i2,num_cols_S_offd);
                        if (j != -1)
                        { 
                           if (P_marker_offd[j] >= jj_begin_row_offd)
                                  P_offd_data[P_marker_offd[j]]
                                     += distribute * S_ext_data[jj1];
                        }
                     }
                  }
                  }
		  else
                  {
                     /* do nothing */
                  }
               }
            
               /*-----------------------------------------------------------
                * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                * into the diagonal.
                *-----------------------------------------------------------*/

               else
               {
                  /* do nothing */
               } 

            }
         }           

        /*-----------------------------------------------------------------
          * Set interpolation weight by dividing by the diagonal.
          *-----------------------------------------------------------------*/

         sum = 0.;
         for (jj = jj_begin_row; jj < jj_end_row; jj++)
            sum += P_diag_data[jj];
         for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            sum += P_offd_data[jj];

         for (jj = jj_begin_row; jj < jj_end_row; jj++)
            P_diag_data[jj] /= sum;
         for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            P_offd_data[jj] /= sum;

      }

      strong_f_marker--; 

      P_offd_i[i+1] = jj_counter_offd;
     }
     hypre_TFree(P_marker);
     hypre_TFree(P_marker_offd);
   }

   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(S),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(S),
                                num_cpts_global,
                                0,
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);
                                                                                
                                                                                
   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
                                                                                
   if (trunc_factor != 0.0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, 0);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int, P_offd_size);

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i=0; i < P_offd_size; i++)
	 P_marker[i] = P_offd_j[i];

      qsort0(P_marker, 0, P_offd_size-1);

      num_cols_P_offd = 1;
      index = P_marker[0];
      for (i=1; i < P_offd_size; i++)
      {
	if (P_marker[i] > index)
	{
 	  index = P_marker[i];
 	  P_marker[num_cols_P_offd++] = index;
  	}
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_Int,num_cols_P_offd);

      for (i=0; i < num_cols_P_offd; i++)
         col_map_offd_P[i] = P_marker[i];

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i=0; i < P_offd_size; i++)
	P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
					 P_offd_j[i],
					 num_cols_P_offd);
      hypre_TFree(P_marker); 
   }

   if (num_cols_P_offd)
   { 
   	hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
        hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   } 

   hypre_GetCommPkgRTFromCommPkgA(P,S,fine_to_coarse_offd);

   *P_ptr = P;

   hypre_TFree(CF_marker_offd);
   hypre_TFree(dof_func_offd);
   hypre_TFree(int_buf_data);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(fine_to_coarse_offd);
   hypre_TFree(coarse_counter);
   hypre_TFree(jj_count);
   hypre_TFree(jj_count_offd);

   if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);

   return(0);  

}
