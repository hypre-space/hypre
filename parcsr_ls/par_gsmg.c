/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

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

#include "fortran.h"
void hypre_F90_NAME_BLAS(dgels, DGELS)(char *, int *, int *, int *, double *, 
  int *, double *, int *, double *, int *, int *);

#ifndef ABS
#define ABS(x) ((x)>0 ? (x) : -(x))
#endif
#ifndef MAX
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif

static double dnrm2(int n, double *x)
{
    double temp = 0.;
    int i;

    for (i=0; i<n; i++)
        temp = temp + x[i]*x[i];
    return sqrt(temp);
}

static void dscal(int n, double a, double *x)
{
    int i;

    for (i=0; i<n; i++)
        x[i] = a * x[i];
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixClone
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixClone(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **Sp,
   int copy_data)
{
   MPI_Comm            comm            = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   int                *A_offd_i        = hypre_CSRMatrixI(A_offd);

   int                *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   int                 n               = hypre_CSRMatrixNumRows(A_diag);

   int                 num_cols_offd     = hypre_CSRMatrixNumCols(A_offd);
   int                 num_nonzeros_diag = A_diag_i[n];
   int                 num_nonzeros_offd = A_offd_i[n];

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

int
hypre_ParCSRMatrixFillSmooth(int nsamples, double *samples, 
  hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
  int num_functions, int *dof_func)
{
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);
   double             *S_diag_data     = hypre_CSRMatrixData(S_diag);
   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j        = hypre_CSRMatrixJ(S_offd);
   double             *S_offd_data     = hypre_CSRMatrixData(S_offd);
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);
   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   double             *A_offd_data     = hypre_CSRMatrixData(A_offd);
   int                 n               = hypre_CSRMatrixNumRows(S_diag);
   int i, j, k, ii, index, start;
   int num_cols_offd;
   int num_sends;
   int *dof_func_offd;
   int *int_buf_data;
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
       nm = dnrm2(n, samples+k*n);
       nm = 1./nm/nsamples;
       dscal(n, nm, samples+k*n);
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
      dof_func_offd = hypre_CTAlloc(int, num_cols_offd);
      int_buf_data = hypre_CTAlloc(int,hypre_ParCSRCommPkgSendMapStart(comm_pkg,
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
      printf("MIN, MAX: %f %f\n", my, mx);
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
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   double             *S_diag_data     = hypre_CSRMatrixData(S_diag);
   double             *S_offd_data     = hypre_CSRMatrixData(S_offd);
   int                 n               = hypre_CSRMatrixNumRows(S_diag);
   int i, j;
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

   MPI_Allreduce(&minimax, &minmin, 1, MPI_DOUBLE, MPI_MIN, comm);

   return minmin;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixThreshold
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixThreshold(hypre_ParCSRMatrix *A, double thresh)
{
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   double             *A_offd_data     = hypre_CSRMatrixData(A_offd);

   int                 n               = hypre_CSRMatrixNumRows(A_diag);

   int                 num_nonzeros_diag = A_diag_i[n];
   int                 num_nonzeros_offd = A_offd_i[n];

   int                *S_diag_i;
   int                *S_diag_j;
   double             *S_diag_data;
   int                *S_offd_i;
   int                *S_offd_j;
   double             *S_offd_data;

   int count, i, jS, jA;

   /* first count the number of nonzeros we will need */
   count = 0;
   for (i=0; i<num_nonzeros_diag; i++)
       if (A_diag_data[i] >= thresh)
           count++;

   /* allocate vectors */
   S_diag_i = hypre_CTAlloc(int, n+1);
   S_diag_j = hypre_CTAlloc(int, count);
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
   S_offd_i = hypre_CTAlloc(int, n+1);
   S_offd_j = hypre_CTAlloc(int, count);
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

int
hypre_BoomerAMGCreateSmoothVecs(void         *data,
                       hypre_ParCSRMatrix    *A,
                       int                    num_sweeps,
                       int                    level,
                       double               **SmoothVecs_p)
{
   hypre_ParAMGData  *amg_data = data;

   MPI_Comm             comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);

   hypre_ParVector *Zero;
   hypre_ParVector *Temp;
   hypre_ParVector *U;

   int    i;
   int    n = hypre_ParCSRMatrixGlobalNumRows(A);
   int    n_local = hypre_CSRMatrixNumRows(A_diag);
   int   *starts = hypre_ParCSRMatrixRowStarts(A);

   int sample;
   int nsamples = hypre_ParAMGDataNumSamples(amg_data);
   int ret;
   double *datax, *bp, *p;
   double rlx_wt, omega;

   int rlx_type;
   int *smooth_option;
   HYPRE_Solver *smoother;

   int debug_flag = hypre_ParAMGDataDebugFlag(amg_data);

   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   if (debug_flag >= 1)
      printf("Creating smooth dirs, %d sweeps, %d samples\n", num_sweeps, 
         nsamples);

   smooth_option = hypre_ParAMGDataSmoothOption(amg_data);
   if (smooth_option[0] > 0)
   {
      smoother = hypre_ParAMGDataSmoother(amg_data);
      if (smooth_option[level] != -1)
	 num_sweeps = hypre_ParAMGDataSmoothNumSweep(amg_data);
   }
   rlx_type = hypre_ParAMGDataGridRelaxType(amg_data)[0];
   rlx_wt = hypre_ParAMGDataRelaxWeight(amg_data)[level];
   omega = hypre_ParAMGDataOmega(amg_data)[level];

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
	   if (smooth_option[level] == 6)
	   {
	      HYPRE_SchwarzSolve(smoother[level],
			(HYPRE_ParCSRMatrix) A, 
			(HYPRE_ParVector) Zero,
			(HYPRE_ParVector) U);
	   }
	   else
	   {
              ret = hypre_BoomerAMGRelax(A, Zero, NULL /*CFmarker*/,
                rlx_type , 0 /*rel pts*/, rlx_wt /*weight*/, 
		omega /*omega*/, U, Temp);
              assert(ret == 0);
	   }
       }

       /* copy out the solution */
       for (i=0; i<n_local; i++)
           *p++ = datax[i];
   }

   hypre_ParVectorDestroy(Zero);
   hypre_ParVectorDestroy(Temp);
   hypre_ParVectorDestroy(U);

   *SmoothVecs_p = bp;

   return 0;
}

/*--------------------------------------------------------------------------
 * CreateSmoothDirs replaces CreateS in AMG
 * - smoother depends on the level being used
 * - in this version, CreateSmoothVecs must be called prior to this function
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGCreateSmoothDirs(void         *data,
                       hypre_ParCSRMatrix    *A,
                       double                *SmoothVecs,
                       double                 thresh,
                       int                    num_functions, 
                       int                   *dof_func,
                       hypre_ParCSRMatrix   **S_ptr)
{
   hypre_ParAMGData  *amg_data = data;
   hypre_ParCSRMatrix *S;
   double minimax;
   int debug_flag = hypre_ParAMGDataDebugFlag(amg_data);

   hypre_ParCSRMatrixClone(A, &S, 0);

   /* Traverse S and fill in differences */
   hypre_ParCSRMatrixFillSmooth(
       hypre_ParAMGDataNumSamples(amg_data), SmoothVecs,
       S, A, num_functions, dof_func);

   minimax = hypre_ParCSRMatrixChooseThresh(S);
   if (debug_flag >= 1)
      printf("Minimax chosen: %f\n", minimax);

   /* Threshold and compress */
   hypre_ParCSRMatrixThreshold(S, thresh*minimax);

   *S_ptr = S;

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
int
hypre_BoomerAMGFitVectors(int ip, int n, int num, double *V, 
  int nc, const int *ind, double *val)
{
   double *a, *b;
   double *ap;
   int i, j;
   double *work;
   int    work_size;
   int    info;
   char trans = 'T';
   int  one   = 1;
   int  temp;

/*
   printf("Fit: row %d, n %d num %d, nc = %d ", ip, n, num, nc);
   for (i=0; i<nc; i++)
      printf("%d ", ind[i]);
   printf("\n");
*/

   /* change first vector to the constant vector */
   for (i=0; i<n; i++)
      V[i] = 1.0;

   work_size = 2000*64;
   work = hypre_CTAlloc(double, work_size);

   a = hypre_CTAlloc(double, num*nc);
   ap = a;
   for (i=0; i<num; i++)
   {
      for (j=0; j<nc; j++)
      {
          *ap = V[i*n+ind[j]];
	  ap++;
      }
   }

   temp = MAX(nc, num);
   b = hypre_CTAlloc(double, temp);
   for (i=0; i<num; i++)
      b[i] = V[i*n+ip];

   hypre_F90_NAME_BLAS(dgels, DGELS)(&trans, &nc, &num, &one, a, &nc,
      b, &temp, work, &work_size, &info);

   if (info != 0)
      printf("par_gsmg: dgels returned %d\n", info);

   /* copy solution into output vector */
   for (j=0; j<nc; j++)
      val[j] = b[j];

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

int
hypre_BoomerAMGBuildInterpLS( hypre_ParCSRMatrix   *A,
                         int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         int                  *num_cpts_global,
                         int                   num_functions,
                         int                  *dof_func,
                         int                   debug_flag,
                         double                trunc_factor,
                         int                   num_smooth,
                         double               *SmoothVecs,
                         hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(S);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   double          *S_diag_data = hypre_CSRMatrixData(S_diag);
   int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
   double          *S_offd_data = hypre_CSRMatrixData(S_offd);
   int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   int              num_cols_S_offd = hypre_CSRMatrixNumCols(S_offd);
   int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(S);

   hypre_ParCSRMatrix *P;
   int		      *col_map_offd_P;

   int             *CF_marker_offd;
   int             *dof_func_offd = NULL;

   hypre_CSRMatrix *S_ext;
   
   double          *S_ext_data;
   int             *S_ext_i;
   int             *S_ext_j;

   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;   

   double          *P_diag_data;
   int             *P_diag_i;
   int             *P_diag_j;
   double          *P_offd_data;
   int             *P_offd_i;
   int             *P_offd_j;

   int              P_diag_size, P_offd_size;
   
   int             *P_marker, *P_marker_offd;

   int              jj_counter,jj_counter_offd;
   int             *jj_count, *jj_count_offd;
   int              jj_begin_row,jj_begin_row_offd;
   int              jj_end_row,jj_end_row_offd;
   
   int              start_indexing = 0; /* start indexing for P_data at 0 */

   int              n_fine = hypre_CSRMatrixNumRows(S_diag);

   int             *fine_to_coarse;
   int             *fine_to_coarse_offd;
   int             *coarse_counter;
   int              coarse_shift;
   int              total_global_cpts;
   int              num_cols_P_offd,my_first_cpt;

   int              i,i1;
   int              j,jl,jj;
   int              start;
   
   double           zero = 0.0;
   double           one  = 1.0;
   
   int              my_id;
   int              num_procs;
   int              num_threads;
   int              num_sends;
   int              index;
   int              ns, ne, size, rest;
   int             *int_buf_data;

   double           max_coef;
   double           row_sum, scale;
   int              next_open,now_checking,num_lost,start_j;
   int              next_open_offd,now_checking_offd,num_lost_offd;

   double           wall_time;  /* for debugging instrumentation  */

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   CF_marker_offd = hypre_CTAlloc(int, num_cols_S_offd);
   if (num_functions > 1 && num_cols_S_offd)
	dof_func_offd = hypre_CTAlloc(int, num_cols_S_offd);

   if (!comm_pkg)
   {
	hypre_MatvecCommPkgCreate(S);
	comm_pkg = hypre_ParCSRMatrixCommPkg(S); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
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
      printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
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
      printf("Proc = %d  Interp: Comm 2   Get S_ext =  %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(int, num_threads);
   jj_count = hypre_CTAlloc(int, num_threads);
   jj_count_offd = hypre_CTAlloc(int, num_threads);

   fine_to_coarse = hypre_CTAlloc(int, n_fine);
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

   P_diag_i    = hypre_CTAlloc(int, n_fine+1);
   P_diag_j    = hypre_CTAlloc(int, P_diag_size);
   P_diag_data = hypre_CTAlloc(double, P_diag_size);

   P_diag_i[n_fine] = jj_counter; 


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(int, n_fine+1);
   P_offd_j    = hypre_CTAlloc(int, P_offd_size);
   P_offd_data = hypre_CTAlloc(double, P_offd_size);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/ 

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   fine_to_coarse_offd = hypre_CTAlloc(int, num_cols_S_offd); 

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
      printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
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
    
#define HYPRE_SMP_PRIVATE i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum,distribute,P_marker,P_marker_offd,jj_counter,jj_counter_offd,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd
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
         int kk;
         int indices[1000];

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

         /* edmond */
         hypre_BoomerAMGFitVectors(i, n_fine, num_smooth, SmoothVecs, 
            kk,
            /*jj_end_row-jj_begin_row,*/
            indices,
            /*&P_diag_j[jj_begin_row],*/
            &P_diag_data[P_diag_i[i]]);

         /* Off-Diagonal part of P */
         /* undone */
      }
     }
   }
   P_diag_i[i] = jj_counter; /* check that this is in right place for threads */

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0)
   {
      next_open = 0;
      now_checking = 0;
      num_lost = 0;
      next_open_offd = 0;
      now_checking_offd = 0;
      num_lost_offd = 0;

      for (i = 0; i < n_fine; i++)
      {
       /*  if (CF_marker[i] < 0) */
         {
            max_coef = 0;
            for (j = P_diag_i[i]; j < P_diag_i[i+1]; j++)
               max_coef = (max_coef < fabs(P_diag_data[j])) ? 
				fabs(P_diag_data[j]) : max_coef;
            for (j = P_offd_i[i]; j < P_offd_i[i+1]; j++)
               max_coef = (max_coef < fabs(P_offd_data[j])) ? 
				fabs(P_offd_data[j]) : max_coef;
            max_coef *= trunc_factor;

            start_j = P_diag_i[i];
            P_diag_i[i] -= num_lost;
	    row_sum = 0;
	    scale = 0;
            for (j = start_j; j < P_diag_i[i+1]; j++)
            {
	       row_sum += P_diag_data[now_checking];
               if (fabs(P_diag_data[now_checking]) < max_coef)
               {
                  num_lost++;
                  now_checking++;
               }
               else
               {
		  scale += P_diag_data[now_checking];
                  P_diag_data[next_open] = P_diag_data[now_checking];
                  P_diag_j[next_open] = P_diag_j[now_checking];
                  now_checking++;
                  next_open++;
               }
            }

            start_j = P_offd_i[i];
            P_offd_i[i] -= num_lost_offd;

            for (j = start_j; j < P_offd_i[i+1]; j++)
            {
	       row_sum += P_offd_data[now_checking_offd];
               if (fabs(P_offd_data[now_checking_offd]) < max_coef)
               {
                  num_lost_offd++;
                  now_checking_offd++;
               }
               else
               {
		  scale += P_offd_data[now_checking_offd];
                  P_offd_data[next_open_offd] = P_offd_data[now_checking_offd];
                  P_offd_j[next_open_offd] = P_offd_j[now_checking_offd];
                  now_checking_offd++;
                  next_open_offd++;
               }
            }
	    /* normalize row of P */

	    if (scale != 0.)
	    {
	     if (scale != row_sum)
	     {
   	       scale = row_sum/scale;
   	       for (j = P_diag_i[i]; j < (P_diag_i[i+1]-num_lost); j++)
      	          P_diag_data[j] *= scale;
   	       for (j = P_offd_i[i]; j < (P_offd_i[i+1]-num_lost_offd); j++)
      	          P_offd_data[j] *= scale;
	     }
	    }
         }
      }
      P_diag_i[n_fine] -= num_lost;
      P_offd_i[n_fine] -= num_lost_offd;
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(int, P_offd_size);

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

      col_map_offd_P = hypre_CTAlloc(int,num_cols_P_offd);

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

   P = hypre_ParCSRMatrixCreate(comm, 
                                hypre_ParCSRMatrixGlobalNumRows(S), 
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(S),
                                num_cpts_global,
                                num_cols_P_offd, 
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data; 
   hypre_CSRMatrixI(P_diag) = P_diag_i; 
   hypre_CSRMatrixJ(P_diag) = P_diag_j; 
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0; 

   /*-------------------------------------------------------------------
    * The following block was originally in an 
    *
    *           if (num_cols_P_offd)
    *
    * block, which has been eliminated to ensure that the code 
    * runs on one processor.
    *
    *-------------------------------------------------------------------*/

   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixI(P_offd) = P_offd_i; 
   if (num_cols_P_offd)
   { 
	hypre_CSRMatrixData(P_offd) = P_offd_data; 
   	hypre_CSRMatrixJ(P_offd) = P_offd_j; 
   	hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
   } 
   hypre_ParCSRMatrixOffd(P) = P_offd;
   hypre_GetCommPkgRTFromCommPkgA(P,S);

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

/*
    for (i=0; i<n_fine; i++)
        for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
            printf("%d %d %f\n", i+1, P_diag_j[j]+1, P_diag_data[j]);
*/

   return(0);  

}
/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildInterpGSMG
 *
 * Difference with hypre_BoomerAMGBuildInterp is that S contains values
 * and is used to build interpolation weights.  Matrix A is not used.
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGBuildInterpGSMG( hypre_ParCSRMatrix   *A,
                         int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         int                  *num_cpts_global,
                         int                   num_functions,
                         int                  *dof_func,
                         int                   debug_flag,
                         double                trunc_factor,
                         hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(S);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   double          *S_diag_data = hypre_CSRMatrixData(S_diag);
   int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
   double          *S_offd_data = hypre_CSRMatrixData(S_offd);
   int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   int              num_cols_S_offd = hypre_CSRMatrixNumCols(S_offd);
   int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(S);

   hypre_ParCSRMatrix *P;
   int		      *col_map_offd_P;

   int             *CF_marker_offd;
   int             *dof_func_offd = NULL;

   hypre_CSRMatrix *S_ext;
   
   double          *S_ext_data;
   int             *S_ext_i;
   int             *S_ext_j;

   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;   

   double          *P_diag_data;
   int             *P_diag_i;
   int             *P_diag_j;
   double          *P_offd_data;
   int             *P_offd_i;
   int             *P_offd_j;

   int              P_diag_size, P_offd_size;
   
   int             *P_marker, *P_marker_offd;

   int              jj_counter,jj_counter_offd;
   int             *jj_count, *jj_count_offd;
   int              jj_begin_row,jj_begin_row_offd;
   int              jj_end_row,jj_end_row_offd;
   
   int              start_indexing = 0; /* start indexing for P_data at 0 */

   int              n_fine = hypre_CSRMatrixNumRows(S_diag);

   int              strong_f_marker;

   int             *fine_to_coarse;
   int             *fine_to_coarse_offd;
   int             *coarse_counter;
   int              coarse_shift;
   int              total_global_cpts;
   int              num_cols_P_offd,my_first_cpt;

   int              i,i1,i2;
   int              j,jl,jj,jj1;
   int              start;
   int              c_num;
   
   double           sum;
   double           distribute;          
   
   double           zero = 0.0;
   double           one  = 1.0;
   
   int              my_id;
   int              num_procs;
   int              num_threads;
   int              num_sends;
   int              index;
   int              ns, ne, size, rest;
   int             *int_buf_data;

   double           max_coef;
   double           row_sum, scale;
   int              next_open,now_checking,num_lost,start_j;
   int              next_open_offd,now_checking_offd,num_lost_offd;

   int col_1 = hypre_ParCSRMatrixFirstRowIndex(S);
   int local_numrows = hypre_CSRMatrixNumRows(S_diag);
   int col_n = col_1 + local_numrows;

   double           wall_time;  /* for debugging instrumentation  */

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   CF_marker_offd = hypre_CTAlloc(int, num_cols_S_offd);
   if (num_functions > 1 && num_cols_S_offd)
	dof_func_offd = hypre_CTAlloc(int, num_cols_S_offd);

   if (!comm_pkg)
   {
	hypre_MatvecCommPkgCreate(S);
	comm_pkg = hypre_ParCSRMatrixCommPkg(S); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
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
      printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
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
      printf("Proc = %d  Interp: Comm 2   Get S_ext =  %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(int, num_threads);
   jj_count = hypre_CTAlloc(int, num_threads);
   jj_count_offd = hypre_CTAlloc(int, num_threads);

   fine_to_coarse = hypre_CTAlloc(int, n_fine);
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

   P_diag_i    = hypre_CTAlloc(int, n_fine+1);
   P_diag_j    = hypre_CTAlloc(int, P_diag_size);
   P_diag_data = hypre_CTAlloc(double, P_diag_size);

   P_diag_i[n_fine] = jj_counter; 


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(int, n_fine+1);
   P_offd_j    = hypre_CTAlloc(int, P_offd_size);
   P_offd_data = hypre_CTAlloc(double, P_offd_size);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/ 

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   fine_to_coarse_offd = hypre_CTAlloc(int, num_cols_S_offd); 

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
      printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
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

     P_marker = hypre_CTAlloc(int, n_fine);
     P_marker_offd = hypre_CTAlloc(int, num_cols_S_offd);

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
                  P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];
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

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0)
   {
      next_open = 0;
      now_checking = 0;
      num_lost = 0;
      next_open_offd = 0;
      now_checking_offd = 0;
      num_lost_offd = 0;

      for (i = 0; i < n_fine; i++)
      {
       /*  if (CF_marker[i] < 0) */
         {
            max_coef = 0;
            for (j = P_diag_i[i]; j < P_diag_i[i+1]; j++)
               max_coef = (max_coef < fabs(P_diag_data[j])) ? 
				fabs(P_diag_data[j]) : max_coef;
            for (j = P_offd_i[i]; j < P_offd_i[i+1]; j++)
               max_coef = (max_coef < fabs(P_offd_data[j])) ? 
				fabs(P_offd_data[j]) : max_coef;
            max_coef *= trunc_factor;

            start_j = P_diag_i[i];
            P_diag_i[i] -= num_lost;
	    row_sum = 0;
	    scale = 0;
            for (j = start_j; j < P_diag_i[i+1]; j++)
            {
	       row_sum += P_diag_data[now_checking];
               if (fabs(P_diag_data[now_checking]) < max_coef)
               {
                  num_lost++;
                  now_checking++;
               }
               else
               {
		  scale += P_diag_data[now_checking];
                  P_diag_data[next_open] = P_diag_data[now_checking];
                  P_diag_j[next_open] = P_diag_j[now_checking];
                  now_checking++;
                  next_open++;
               }
            }

            start_j = P_offd_i[i];
            P_offd_i[i] -= num_lost_offd;

            for (j = start_j; j < P_offd_i[i+1]; j++)
            {
	       row_sum += P_offd_data[now_checking_offd];
               if (fabs(P_offd_data[now_checking_offd]) < max_coef)
               {
                  num_lost_offd++;
                  now_checking_offd++;
               }
               else
               {
		  scale += P_offd_data[now_checking_offd];
                  P_offd_data[next_open_offd] = P_offd_data[now_checking_offd];
                  P_offd_j[next_open_offd] = P_offd_j[now_checking_offd];
                  now_checking_offd++;
                  next_open_offd++;
               }
            }
	    /* normalize row of P */

	    if (scale != 0.)
	    {
	     if (scale != row_sum)
	     {
   	       scale = row_sum/scale;
   	       for (j = P_diag_i[i]; j < (P_diag_i[i+1]-num_lost); j++)
      	          P_diag_data[j] *= scale;
   	       for (j = P_offd_i[i]; j < (P_offd_i[i+1]-num_lost_offd); j++)
      	          P_offd_data[j] *= scale;
	     }
	    }
         }
      }
      P_diag_i[n_fine] -= num_lost;
      P_offd_i[n_fine] -= num_lost_offd;
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(int, P_offd_size);

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

      col_map_offd_P = hypre_CTAlloc(int,num_cols_P_offd);

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

   P = hypre_ParCSRMatrixCreate(comm, 
                                hypre_ParCSRMatrixGlobalNumRows(S), 
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(S),
                                num_cpts_global,
                                num_cols_P_offd, 
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data; 
   hypre_CSRMatrixI(P_diag) = P_diag_i; 
   hypre_CSRMatrixJ(P_diag) = P_diag_j; 
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0; 

   /*-------------------------------------------------------------------
    * The following block was originally in an 
    *
    *           if (num_cols_P_offd)
    *
    * block, which has been eliminated to ensure that the code 
    * runs on one processor.
    *
    *-------------------------------------------------------------------*/

   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixI(P_offd) = P_offd_i; 
   if (num_cols_P_offd)
   { 
	hypre_CSRMatrixData(P_offd) = P_offd_data; 
   	hypre_CSRMatrixJ(P_offd) = P_offd_j; 
   	hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
   } 
   hypre_ParCSRMatrixOffd(P) = P_offd;
   hypre_GetCommPkgRTFromCommPkgA(P,S);

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

/*
    for (i=0; i<n_fine; i++)
        for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
            printf("%d %d %f\n", i+1, P_diag_j[j]+1, P_diag_data[j]);
*/

   return(0);  

}
