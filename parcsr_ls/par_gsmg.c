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
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixClone(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **Sp,
   int copy_data)
{
   MPI_Comm            comm            = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   int                *A_diag_i        = hypre_CSRMatrixI(A_diag);

   int                *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   int                 n               = hypre_CSRMatrixNumRows(A_diag);

   int                 num_nonzeros_diag = A_diag_i[n];

   hypre_ParCSRMatrix *S;

   S = hypre_ParCSRMatrixCreate(comm, n, n, row_starts, row_starts,
       0, num_nonzeros_diag, 0);
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
   double mx = 0., my = 1.e+10;

   /* normalize each sample vector and divide by number of samples */
   for (k=0; k<nsamples; k++)
   {
       nm = dnrm2(n, samples+k*n);
       nm = 1./nm/nsamples;
       dscal(n, nm, samples+k*n);
   }

   for (i = 0; i < n; i++)
   {
       for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
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
           if (i == ii) /* diagonal element */
           {
               S_diag_data[j] = 0.; /* no diagonal element */
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
my = hypre_min(my,temp);
mx = hypre_max(mx,temp);
           S_diag_data[j] = temp;
       }
   }
printf("MIN, MAX: %f %f\n", my, mx);

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
   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   double             *S_diag_data     = hypre_CSRMatrixData(S_diag);
   int                 n               = hypre_CSRMatrixNumRows(S_diag);
   int i, j;
   double mx, minimax = 1.e+10;

   for (i=0; i<n; i++)
   {
      mx = 0.;
      for (j=S_diag_i[i]; j<S_diag_i[i+1]; j++)
         mx = hypre_max(mx, S_diag_data[j]);

      if (mx != 0.)
         minimax = hypre_min(minimax, mx);
   }

   return minimax;
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

   int                 n               = hypre_CSRMatrixNumRows(A_diag);

   int                 num_nonzeros_diag = A_diag_i[n];

   int                *S_diag_i;
   int                *S_diag_j;

   int count, i, jS, jA;

   /* first count the number of nonzeros we will need */
   count = 0;
   for (i=0; i<num_nonzeros_diag; i++)
       if (A_diag_data[i] >= thresh)
           count++;

   /* allocate vectors */
   S_diag_i = hypre_CTAlloc(int, n+1);
   S_diag_j = hypre_CTAlloc(int, count);

   jS = 0;
   for (i = 0; i < n; i++)
   {
      S_diag_i[i] = jS;
      for (jA = A_diag_i[i]; jA < A_diag_i[i+1]; jA++)
      {
         if (A_diag_data[jA] >= thresh)
         {
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
   hypre_CSRMatrixData(A_diag) = NULL;

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
               rlx_type , 0 /*rel pts*/, 1.0 /*weight*/, U, Temp);
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
