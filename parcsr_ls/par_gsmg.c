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

/* labels are a mapping from local to global */
/* The mapping is determined using the CF marker */

/* These arrays will need to be freed in destroy routine */

int
hypre_BoomerAMGSetGSMG( void *data,
                            int   par )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   amg_data->gsmg = par;

   return (ierr);
}

/* local scaling */

int
hypre_ParCSRMatrixLocalScale(hypre_ParCSRMatrix *S)
{
   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   double             *S_diag_data     = hypre_CSRMatrixData(S_diag);
   int                 n               = hypre_CSRMatrixNumRows(S_diag);
   int i, j;
   double mx;

   for (i = 0; i < n; i++)
   {
       /* find the max in each row - assume diagonal is 0 */
       mx = 0.;
       for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
           mx = hypre_max(mx,S_diag_data[j]);

       /* now go through and divide by this max */
       for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
           S_diag_data[j] = S_diag_data[j] / mx;
   }

   return 0;
}


/*--------------------------------------------------------------------------
 * read_tgo
 *--------------------------------------------------------------------------*/

/* this function only needs to read coordinates */

/* old algorithm was to find a red-black coarsening based on the
   element edges (needs to build matrix based on elements) */

/* format of tgo file:
 * header is first six lines, followed by the coordinates.
 * second line contains: 1 nnodes nele ...
 * coordinates are: dummy dummy x y z dummy
 */

void read_tgo(void *data, double *x, double *y, double *z)
{
    hypre_ParAMGData   *amg_data = data;

    FILE *file;
    char line[101];
    int num_nodes, num_eles;
    int i, ret;
    char new_filename[255];

    sprintf(new_filename,"%s.tgo", amg_data->tgofilename);
    file = fopen(new_filename, "r");
    assert(file != NULL);

    fgets(line, 100, file);
    fgets(line, 100, file);

    ret = sscanf(line, "%*d %d %d", &num_nodes, &num_eles);
    assert(ret == 2);

    /* read four irrelevant lines */
    fgets(line, 100, file);
    fgets(line, 100, file);
    fgets(line, 100, file);
    fgets(line, 100, file);

    /* nodes and coordinates */
    for (i=0; i<num_nodes; i++)
    {
        fgets(line, 100, file);
        ret = sscanf(line, "%*d %*f %le %le %le %*f", x+i, y+i, z+i);
        assert(ret == 3);
    }
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixClone
 * - assumes one processor only - no offd part
 * - updated for multiple processors 7/15/02
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
 * - also assumes no offd part
 * - fill in smooth matrix
 *--------------------------------------------------------------------------*/
#define ABS(x) ((x)>0 ? (x) : -(x))

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

int
hypre_ParCSRMatrixFillSmooth(int nsamples, double *samples, 
  hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
  int num_functions, int *dof_func)
{
   MPI_Comm           comm = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(S);
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
   double *buf_data;
   double nm;
   double mx = 0., my = 1.e+10;

   /* normalize each sample vector and divide by number of samples */
   for (k=0; k<nsamples; k++)
   {
       nm = dnrm2(n, samples+k*n);
       nm = 1./nm/nsamples;
       dscal(n, nm, samples+k*n);
   }

   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_cols_offd = S_offd_i[n];
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   buf_data = hypre_CTAlloc(double,hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   p_offd = hypre_CTAlloc(double, nsamples*num_cols_offd);

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

#if 0
           if (amg_data->gsi_map1[i] % 3 != amg_data->gsi_map1[ii] % 3)
           {
               S_diag_data[j] = 0.; /* unlike variables */
               continue;
           }
#endif
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
my = hypre_min(my,temp);
mx = hypre_max(mx,temp);
           S_diag_data[j] = temp;
       }
       for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
       {
           ii = S_offd_j[j];

           /* only interpolate between like functions */
           if (num_functions > 1 && dof_func_offd[i] != dof_func_offd[ii])
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
my = hypre_min(my,temp);
mx = hypre_max(mx,temp);
           S_offd_data[j] = temp;
       }
   }
printf("MIN, MAX: %f %f\n", my, mx);

   hypre_TFree(p_offd);

   return 0;
}

/* prints the number of edges that change identity */
/* assumes that FillSmooth has already been run, and a threshold is known */

int
hypre_ParCSRMatrixFillSmoothIncrementally(int nsamples, 
  double *samples, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, double thresh)
{
   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);
   double             *S_diag_data     = hypre_CSRMatrixData(S_diag);
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);
   int                 n               = hypre_CSRMatrixNumRows(S_diag);
   int i, j, k, ii;
   double temp;
   double *p;
   double nm;

   double new, old;
   int count1, count2;

   /* unscale by multiplying by number of samples */
   for (k=0; k<nsamples; k++)
   {
       nm = (double) nsamples;
       dscal(n, nm, samples+k*n);
   }

   /* do first pass to fill S initially */
   p = samples;
   for (i = 0; i < n; i++)
   {
       for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
       {
           ii = S_diag_j[j];

           if (i == ii) /* diagonal element */
           {
               S_diag_data[j] = 0.; /* no diagonal element */
               continue;
           }

/* fixed: Aug 31,2001 explicit zeros in matrix need to be handled */
           if (A_diag_data[j] == 0.)
           {
               S_diag_data[j] = 0.;
               continue;
           }

           temp = ABS(p[i] - p[ii]);

#if 0
           /* explicit zeros in matrix may cause this */
           if (temp == 0.)
           {
               S_diag_data[j] = 0.;
               continue;
           }

           temp = 1./temp; /* reciprocal */
#endif
           S_diag_data[j] = temp;
       }
   }

   /* now do subsequent passes */
   for (k=1; k<nsamples; k++)
   {
     count1 = 0;
     count2 = 0;
     p = samples + k*n;

     for (i = 0; i < n; i++)
     {
       for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
       {
           ii = S_diag_j[j];

           if (i == ii) /* diagonal element */
           {
               S_diag_data[j] = 0.; /* no diagonal element */
               continue;
           }

/* fixed: Aug 31,2001 explicit zeros in matrix need to be handled */
           if (A_diag_data[j] == 0.)
           {
               S_diag_data[j] = 0.;
               continue;
           }

           temp = ABS(p[i] - p[ii]);

#if 0
           /* explicit zeros in matrix may cause this */
           if (temp == 0.)
           {
               S_diag_data[j] = 0.;
               continue;
           }

           old = 1. / S_diag_data[j];
           new = 1. / (k*old + temp) / (k+1);
           S_diag_data[j] = new;
#else
           old = S_diag_data[j];
           new = (k*old + temp) / (k+1);
           S_diag_data[j] = new;
#endif

           if (1./old < thresh && 1./new >= thresh)
	       count1++;

           if (1./old >= thresh && 1./new < thresh)
	       count2++;

       }
     }
     printf("%2d counts: %5d %5d %5d\n", k, count1, count2, count1+count2);
   }

     for (i = 0; i < n; i++)
       for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
           S_diag_data[j] = 1. / S_diag_data[j];

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
 * - also assumes no offd part
 * - creates a new matrix that only has structure (no values)
 * - creates into new temp vectors, then replaces the vectors in A
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
 * CreateSmoothDirs replaces CreateS
 * - smoother depends on the level being used
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGCreateSmoothDirs(void *datay,
                       hypre_ParCSRMatrix *A,
                       int                    num_sweeps,
                       double                 thresh,
/**** change ****/
                       int                    level,
/**** change ****/
                       int                    num_functions, 
                       int                   *dof_func,
                       hypre_ParCSRMatrix   **S_ptr)
{
   hypre_ParAMGData   *amg_data = datay;

   MPI_Comm            comm            = hypre_ParCSRMatrixComm(A);

   hypre_ParCSRMatrix *S;

   hypre_ParVector *Zero;
   hypre_ParVector *Temp;
   hypre_ParVector *U;

   int    i;
   int    n = hypre_ParCSRMatrixGlobalNumRows(A);
   int   *starts = hypre_ParCSRMatrixRowStarts(A);

   int sample;
   int nsamples = 20; /* hard-coded number of samples */
   int ret;
   double *datax, *bp, *p;
   double minimax;

/**** change ****/
   int rlx_type;
   int *smooth_option;
   HYPRE_Solver *smoother;
/**** change ****/

#if 0
    hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
    double          *A_diag_data = hypre_CSRMatrixData(A_diag);
    int             *A_diag_i = hypre_CSRMatrixI(A_diag);
    int             *A_diag_j = hypre_CSRMatrixJ(A_diag);
    int j;
    for (i=0; i<n; i++)
        for (j=A_diag_i[i]; j<A_diag_i[i+1]; j++)
            printf("%d %d %f\n", i+1, A_diag_j[j]+1, A_diag_data[j]);
#endif

printf("Creating smooth dirs, %d sweeps, %d samples\n", num_sweeps, nsamples);

/**** change ****/
   smooth_option = hypre_ParAMGDataSmoothOption(amg_data);
   if (smooth_option[0] > 0)
   {
      smoother = hypre_ParAMGDataSmoother(amg_data);
      if (smooth_option[level] != -1)
	 num_sweeps = hypre_ParAMGDataSmoothNumSweep(amg_data);
   }
   rlx_type = hypre_ParAMGDataGridRelaxType(amg_data)[0];
/**** change ****/

   if (amg_data->gsi_x == NULL)
   {
       amg_data->gsi_f2c = hypre_CTAlloc(int, n);

       amg_data->p_index = hypre_CTAlloc(int, n);
       for (i=0; i<n; i++)
           amg_data->p_index[i] = -1;

       amg_data->gsi_map2 = hypre_CTAlloc(int, n);
       amg_data->gsi_map1 = hypre_CTAlloc(int, n);
       for (i=0; i<n; i++)
           amg_data->gsi_map1[i] = i;

       amg_data->gsi_x = hypre_CTAlloc(double, n);
       amg_data->gsi_y = hypre_CTAlloc(double, n);
       amg_data->gsi_z = hypre_CTAlloc(double, n);

       if (amg_data->gsmg == 1 || amg_data->gsmg == 2)
           read_tgo(datay, amg_data->gsi_x, amg_data->gsi_y, amg_data->gsi_z);
   }

   /* generate par vectors */

   Zero = hypre_ParVectorCreate(comm, n, starts);
   hypre_ParVectorSetPartitioningOwner(Zero,0);
   hypre_ParVectorInitialize(Zero);
   datax = hypre_VectorData(hypre_ParVectorLocalVector(Zero));
   for (i=0; i<n; i++)
       datax[i] = 0.;

   Temp = hypre_ParVectorCreate(comm, n, starts);
   hypre_ParVectorSetPartitioningOwner(Temp,0);
   hypre_ParVectorInitialize(Temp);
   datax = hypre_VectorData(hypre_ParVectorLocalVector(Temp));
   for (i=0; i<n; i++)
       datax[i] = 0.;

   U = hypre_ParVectorCreate(comm, n, starts);
   hypre_ParVectorSetPartitioningOwner(U,0);
   hypre_ParVectorInitialize(U);
   datax = hypre_VectorData(hypre_ParVectorLocalVector(U));

   /* allocate space for the vectors */
   bp = hypre_CTAlloc(double, nsamples*n);
   p = bp;

   /* generate random vectors */
   for (sample=0; sample<nsamples; sample++)
   {
       for (i=0; i<n; i++)
           /*datax[i] = hypre_Rand() -.5;*//* or no sub */
           datax[i] = (rand()/(double)RAND_MAX) - .5;
           /*datax[i] = 2.*(rand()/(double)RAND_MAX) - 1.;*/

       for (i=0; i<num_sweeps; i++)
       {
/**** change ****/
	   if (smooth_option[level] == 6)
	   {
	      HYPRE_SchwarzSolve(smoother[level],
			(HYPRE_ParCSRMatrix) A, 
			(HYPRE_ParVector) Zero,
			(HYPRE_ParVector) U);
	   }
	   else
	   {
/**** change ****/
              ret = hypre_BoomerAMGRelax(A, Zero, NULL /*CFmarker*/,
               rlx_type , 0 /*rel pts*/, 1.0 /*weight*/, 1.0 /*omega*/,
		U, Temp);
              assert(ret == 0);
/**** change ****/
	   }
/**** change ****/
       }

       /* copy out the solution */
       for (i=0; i<n; i++)
           *p++ = datax[i];
   }

   hypre_ParVectorDestroy(Zero);
   hypre_ParVectorDestroy(Temp);
   hypre_ParVectorDestroy(U);

   hypre_ParCSRMatrixClone(A, &S, 0);

   /* Traverse S and fill in differences */
   hypre_ParCSRMatrixFillSmooth(nsamples, bp, S, A, num_functions,
       dof_func);

   /* hypre_TFree(bp);  moved below */

   /* copy if want to use values of S for interpolation */
   if (amg_data->gsmg == 4)
   {
       if (amg_data->Sfactors != NULL)
           hypre_ParCSRMatrixDestroy(amg_data->Sfactors);

       hypre_ParCSRMatrixClone(S, &amg_data->Sfactors, 1);
   }

  /* apply local scaling if thresh is negative */
  if (thresh >= 0.)
  {
   minimax = hypre_ParCSRMatrixChooseThresh(S);
   printf("Minimax chosen: %f\n", minimax);
/*
hypre_ParCSRMatrixFillSmoothIncrementally(nsamples, bp, S, A, thresh*minimax);
*/

  }
  else
  {
   thresh = -thresh;
   printf("APPLYING LOCAL SCALING\n");
   hypre_ParCSRMatrixLocalScale(S);
   minimax = 1.0;
  }

   hypre_TFree(bp);

   /* Threshold and compress */
   hypre_ParCSRMatrixThreshold(S, thresh*minimax);

   *S_ptr = S;

   return 0;
}

/*--------------------------------------------------------------------------
 * BuildInterpLinear needs to be called after BuildInterp for linear interp.
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGBuildInterpLinear(void *data,
                                 hypre_ParCSRMatrix   *P,
                         int                  *CF_marker)
{
    hypre_ParAMGData   *amg_data = data;

    int n, n_coarse, i, j, jj;
    double centerx, centery, centerz;
    double denom;
    double dist;

    /* extract the parts of P */
    hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
    double          *P_diag_data = hypre_CSRMatrixData(P_diag);
    int             *P_diag_i = hypre_CSRMatrixI(P_diag);
    int             *P_diag_j = hypre_CSRMatrixJ(P_diag);

    n = hypre_ParCSRMatrixGlobalNumRows(P);

    /* print the number of rows and columns */
    printf("InterpLinear rows cols %d %d\n", 
       hypre_ParCSRMatrixGlobalNumRows(P),
       hypre_ParCSRMatrixGlobalNumCols(P));

/*
    for (i=0; i<n; i++)
        if (CF_marker[i] >= 0)
            printf("%d ", i+1);
*/

    /* create map2 from map1 and CF_marker */
    /* at the end, move map2 to map1 */
    j = 0;
    for (i=0; i<n; i++)
        if (CF_marker[i] >= 0)
            amg_data->gsi_map2[j++] = amg_data->gsi_map1[i];
    n_coarse = j;

    for (i=0; i<n; i++)
    {
        if (CF_marker[i] >= 0)
            continue; /* do not modify this row */

        centerx = amg_data->gsi_x[amg_data->gsi_map1[i]];
        centery = amg_data->gsi_y[amg_data->gsi_map1[i]];
        centerz = amg_data->gsi_z[amg_data->gsi_map1[i]];

        denom = 0.;

        for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
        {
            jj = amg_data->gsi_map2[P_diag_j[j]];
            dist = 1. / sqrt( 
	      (centerx - amg_data->gsi_x[jj])* (centerx - amg_data->gsi_x[jj]) +
              (centery - amg_data->gsi_y[jj])* (centery - amg_data->gsi_y[jj]) +
              (centerz - amg_data->gsi_z[jj])* (centerz - amg_data->gsi_z[jj]));
            denom = denom + dist;

            assert(dist > 0.);

            P_diag_data[j] = dist;
        }

        for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
            P_diag_data[j] = P_diag_data[j] / denom;
    }

    /* copy map2 to map1 */
    for (i=0; i<n_coarse; i++)
        amg_data->gsi_map1[i] = amg_data->gsi_map2[i];

#if 0
    for (i=0; i<n; i++)
        for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
            printf("%d %d %f\n", i+1, P_diag_j[j]+1, P_diag_data[j]);
#endif

    return 0;
}

/*--------------------------------------------------------------------------
 * version with indirect interpolation
 *  - can comment out indirect part for direct interpolation only
 *--------------------------------------------------------------------------*/

#define DISTRECIP(i,j) \
1. / sqrt( (amg_data->gsi_x[i] - amg_data->gsi_x[j])*(amg_data->gsi_x[i] - amg_data->gsi_x[j]) \
         + (amg_data->gsi_y[i] - amg_data->gsi_y[j])*(amg_data->gsi_y[i] - amg_data->gsi_y[j]) \
         + (amg_data->gsi_z[i] - amg_data->gsi_z[j])*(amg_data->gsi_z[i] - amg_data->gsi_z[j]))

#define XDISTRECIP(i,j) \
1. / sqrt( (amg_data->gsi_x[i/3] - amg_data->gsi_x[j/3])*(amg_data->gsi_x[i/3] - amg_data->gsi_x[j/3]) \
         + (amg_data->gsi_y[i/3] - amg_data->gsi_y[j/3])*(amg_data->gsi_y[i/3] - amg_data->gsi_y[j/3]) \
         + (amg_data->gsi_z[i/3] - amg_data->gsi_z[j/3])*(amg_data->gsi_z[i/3] - amg_data->gsi_z[j/3]))

int is_strong(hypre_ParCSRMatrix *S ,int i, int j)
{
    hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
    int             *S_diag_i = hypre_CSRMatrixI(S_diag);
    int             *S_diag_j = hypre_CSRMatrixJ(S_diag);
    int k;

    for (k=S_diag_i[i]; k<S_diag_i[i+1]; k++)
        if (S_diag_j[k] == j)
            return 1;

    return 0;
}

double mat_entry(hypre_ParCSRMatrix *S ,int i, int j)
{
    hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
    int             *S_diag_i = hypre_CSRMatrixI(S_diag);
    int             *S_diag_j = hypre_CSRMatrixJ(S_diag);
    double          *S_diag_data = hypre_CSRMatrixData(S_diag);
    int k;

    for (k=S_diag_i[i]; k<S_diag_i[i+1]; k++)
        if (S_diag_j[k] == j)
            return S_diag_data[k];

    return 0.;
}

int
hypre_BoomerAMGBuildInterpLinearIndirect(void *data,
                         hypre_ParCSRMatrix   *A,
                         int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         hypre_ParCSRMatrix   *P)
{
    hypre_ParAMGData   *amg_data = data;

    int n, n_coarse, i, j, k, col, colcol;
    double denom;
    double dist;

    /* extract the parts of A */
    hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
    int             *A_diag_i = hypre_CSRMatrixI(A_diag);
    int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

    /* extract the parts of S */
    hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
    int             *S_diag_i = hypre_CSRMatrixI(S_diag);

    /* extract the parts of P */
    hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
    double          *P_diag_data = hypre_CSRMatrixData(P_diag);
    int             *P_diag_i = hypre_CSRMatrixI(P_diag);
    int             *P_diag_j = hypre_CSRMatrixJ(P_diag);

    n = hypre_ParCSRMatrixGlobalNumRows(P);

    /* mapping from coarse point to 1..n_coarse */
    j = 0;
    for (i=0; i<n; i++)
        amg_data->gsi_f2c[i] = (CF_marker[i] >= 0) ? j++ : 0;

    /* create map2 from map1 and CF_marker */
    /* at the end, move map2 to map1 */
    j = 0;
    for (i=0; i<n; i++)
        if (CF_marker[i] >= 0)
            amg_data->gsi_map2[j++] = amg_data->gsi_map1[i];
    n_coarse = j;

    /* set all entries in P to zero */
    for (i=0; i<n; i++)
        for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
            P_diag_data[j] = 0.;

    for (i=0; i<n; i++)
    {
        if (CF_marker[i] >= 0) /* coarse point */
        {
            /* verify that this row of P only has a single nonzero */
            assert(P_diag_i[i] + 1 == P_diag_i[i+1]);

            /* set the single nonzero */
            P_diag_data[P_diag_i[i]] = 1.0; 
        }

        else if (S_diag_i[i] == S_diag_i[i+1]) /* no strong connections */
            ; /* this row of P will be zero */

        else /* fine point */
        {
            /* p_index maps from P column index for a row to index into P */
            for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
                amg_data->p_index[P_diag_j[j]] = j;

            /* loop over entries in row i of A */
            for (j=A_diag_i[i]; j<A_diag_i[i+1]; j++)
            {
		col = A_diag_j[j];

                if (col == i)
                    ; /* ignore diagonal element */

                else if (is_strong(S,i,col)) /* strong connection */
                {
                    if (CF_marker[col] >= 0) /* fine-coarse connection */
                    {
                        dist = DISTRECIP(amg_data->gsi_map1[i], 
			  amg_data->gsi_map1[col]);

                        assert(dist > 0.); /* may fail if bad tgo file */

                        P_diag_data[amg_data->p_index[amg_data->gsi_f2c[col]]] 
			  += dist;
                    }
#if 1
                    else /* fine-fine connection between i and col */
                    {
                        dist = DISTRECIP(amg_data->gsi_map1[i], 
			  amg_data->gsi_map1[col]);

                        /* find common strongly connected c-points */
                        /* do this by examining rows i and col of P */
                        /* pattern of row i of P already scattered */

			denom = 0.;

                        for (k=P_diag_i[col]; k<P_diag_i[col+1]; k++)
                        {
			    colcol = P_diag_j[k];
                            if (amg_data->p_index[colcol] != -1)
                            {
                                denom += DISTRECIP(amg_data->gsi_map1[col], 
				  amg_data->gsi_map2[colcol]);
                            }
                        }

                        assert(denom > 0.); /* if fail, then no common point */

                        for (k=P_diag_i[col]; k<P_diag_i[col+1]; k++)
                        {
			    colcol = P_diag_j[k];
                            if (amg_data->p_index[colcol] != -1)
                            {
                                P_diag_data[amg_data->p_index[colcol]] += dist
                                  * DISTRECIP(amg_data->gsi_map1[col], 
				  amg_data->gsi_map2[colcol]) / denom;
                            }
                        }
                    }
#endif
                }
		else /* weak connection */
		{
	        }
            }

            /* zero out p_index */
            for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
                amg_data->p_index[P_diag_j[j]] = -1;

            /* scale */
	    denom = 0.;
            for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
	        denom += P_diag_data[j];

            for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
                P_diag_data[j] = P_diag_data[j] / denom;
        }
    }

#if 0
    for (i=0; i<n; i++)
        for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
	{
            printf("(%d %d) ", (amg_data->gsi_map1[i]+1) % 3, 
	      (amg_data->gsi_map2[P_diag_j[j]]+1) % 3);
            printf("%4d %4d %f", amg_data->gsi_map1[i]+1, 
	      amg_data->gsi_map2[P_diag_j[j]]+1, P_diag_data[j]);
            if (CF_marker[i] >= 0)
	        printf(" x\n");
	    else
	        printf("\n");
	}
#endif

    /* copy map2 to map1 */
    for (i=0; i<n_coarse; i++)
        amg_data->gsi_map1[i] = amg_data->gsi_map2[i];
#if 0
    for (i=0; i<n; i++)
        for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
            printf("%d %d %f\n", i+1, P_diag_j[j]+1, P_diag_data[j]);
#endif

    return 0;
}

/*--------------------------------------------------------------------------
 * version that uses smoothness factor
 *  - S matrix in this version contains the smoothness factors
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGBuildInterpWithSmoothnessFactor(void *data,
                         hypre_ParCSRMatrix   *A,
                         int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         hypre_ParCSRMatrix   *P)
{
    hypre_ParAMGData   *amg_data = data;

    int n, n_coarse, i, j, k, col, colcol;
    double denom;
    double dist;

    /* extract the parts of A */
    hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
    int             *A_diag_i = hypre_CSRMatrixI(A_diag);
    int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

    /* extract the parts of S */
    hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
    int             *S_diag_i = hypre_CSRMatrixI(S_diag);

    /* extract the parts of P */
    hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
    double          *P_diag_data = hypre_CSRMatrixData(P_diag);
    int             *P_diag_i = hypre_CSRMatrixI(P_diag);
    int             *P_diag_j = hypre_CSRMatrixJ(P_diag);

    int *map;

    n = hypre_ParCSRMatrixGlobalNumRows(P);

    /* mapping from coarse point to 1..n_coarse */
    j = 0;
    for (i=0; i<n; i++)
        amg_data->gsi_f2c[i] = (CF_marker[i] >= 0) ? j++ : 0;

    /* create map2 from map1 and CF_marker */
    /* at the end, move map2 to map1 */
    j = 0;
    for (i=0; i<n; i++)
        if (CF_marker[i] >= 0)
            amg_data->gsi_map2[j++] = amg_data->gsi_map1[i];
    n_coarse = j;

    /* new in this function */
    /* set up an inverse mapping for P, from col index to row index */
    map = hypre_CTAlloc(int, n_coarse);
    j = 0;
    for (i=0; i<n; i++)
	if (CF_marker[i] >= 0)
	    map[j++] = i;

    /* set all entries in P to zero */
    for (i=0; i<n; i++)
        for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
            P_diag_data[j] = 0.;

    for (i=0; i<n; i++)
    {
        if (CF_marker[i] >= 0) /* coarse point */
        {
            /* verify that this row of P only has a single nonzero */
            assert(P_diag_i[i] + 1 == P_diag_i[i+1]);

            /* set the single nonzero */
            P_diag_data[P_diag_i[i]] = 1.0; 
        }

        else if (S_diag_i[i] == S_diag_i[i+1]) /* no strong connections */
            ; /* this row of P will be zero */

        else /* fine point */
        {
            /* p_index maps from P column index for a row to index into P */
            for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
                amg_data->p_index[P_diag_j[j]] = j;

            /* loop over entries in row i of A */
            for (j=A_diag_i[i]; j<A_diag_i[i+1]; j++)
            {
		col = A_diag_j[j];

                if (col == i)
                    ; /* ignore diagonal element */

                else if (is_strong(S,i,col)) /* strong connection */
                {
                    if (CF_marker[col] >= 0) /* fine-coarse connection */
                    {
                        dist = mat_entry(amg_data->Sfactors, i, col);

			if (dist == 0.)
			   printf("dist zero %d %d\n", i, col);

                        assert(dist != 0.); /* may fail if bad tgo file */
                        assert(dist > 0.); /* may fail if bad tgo file */

                        P_diag_data[amg_data->p_index[amg_data->gsi_f2c[col]]] 
			  += dist;
                    }
#if 1
                    else /* fine-fine connection between i and col */
                    {
                        dist = mat_entry(amg_data->Sfactors, i, col);

                        /* find common strongly connected c-points */
                        /* do this by examining rows i and col of P */
                        /* pattern of row i of P already scattered */

			denom = 0.;

                        for (k=P_diag_i[col]; k<P_diag_i[col+1]; k++)
                        {
			    colcol = P_diag_j[k];
                            if (amg_data->p_index[colcol] != -1)
                            {
                                denom += mat_entry(amg_data->Sfactors, col, 
				  map[colcol]);
                            }
                        }

                        assert(denom > 0.); /* if fail, then no common point */

                        for (k=P_diag_i[col]; k<P_diag_i[col+1]; k++)
                        {
			    colcol = P_diag_j[k];
                            if (amg_data->p_index[colcol] != -1)
                            {
                                P_diag_data[amg_data->p_index[colcol]] += dist
                                  * mat_entry(amg_data->Sfactors, col, 
				  map[colcol]) / denom;
                            }
                        }
                    }
#endif
                }
		else /* weak connection */
		{
	        }
            }

            /* zero out p_index */
            for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
                amg_data->p_index[P_diag_j[j]] = -1;

            /* scale */
	    denom = 0.;
            for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
	        denom += P_diag_data[j];

            for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
                P_diag_data[j] = P_diag_data[j] / denom;
        }
    }

#if 0
    for (i=0; i<n; i++)
        for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
	{
            printf("(%d %d) ", (amg_data->gsi_map1[i]+1) % 3, 
	      (amg_data->gsi_map2[P_diag_j[j]]+1) % 3);
            printf("%4d %4d %f", amg_data->gsi_map1[i]+1, 
	      amg_data->gsi_map2[P_diag_j[j]]+1, P_diag_data[j]);
            if (CF_marker[i] >= 0)
	        printf(" x\n");
	    else
	        printf("\n");
	}
#endif

    /* copy map2 to map1 */
    for (i=0; i<n_coarse; i++)
        amg_data->gsi_map1[i] = amg_data->gsi_map2[i];

    /* new in this function */
    hypre_TFree(map);

#if 0
    for (i=0; i<n; i++)
        for (j=P_diag_i[i]; j<P_diag_i[i+1]; j++)
            printf("%d %d %f\n", i+1, P_diag_j[j]+1, P_diag_data[j]);
#endif

    return 0;
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildInterpGSMG
 *
 * Added 7/15/02
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

/*
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   double          *A_diag_data = hypre_CSRMatrixData(A_diag);
   int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
   double          *A_offd_data = hypre_CSRMatrixData(A_offd);
   int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   int             *A_offd_j = hypre_CSRMatrixJ(A_offd);
*/

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

   A = NULL;

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
