/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Weighted Jacobi functions
 *
 *****************************************************************************/

#include "amg.h"
#include "wjacobi.h"


/*--------------------------------------------------------------------------
 * WJacobi
 *--------------------------------------------------------------------------*/

void   	 WJacobi(x, b, tol, data)
Vector 	*x;
Vector 	*b;
double 	 tol;
Data    *data;
{
   WJacobiData    *wjacobi_data = data;

   double          weight   = WJacobiDataWeight(wjacobi_data);
   int         	   max_iter = WJacobiDataMaxIter(wjacobi_data);

   Matrix         *A        = WJacobiDataA(wjacobi_data);
   Vector      	  *t        = WJacobiDataT(wjacobi_data);

   double         *a  = MatrixData(A);
   int            *ia = MatrixIA(A);
   int            *ja = MatrixJA(A);
   int             n  = MatrixSize(A);
	          
   double         *xp = VectorData(x);
   double         *bp = VectorData(b);
   double         *tp = VectorData(t);
	          
   int             i, j, jj;
   int             iter = 0;


   /*-----------------------------------------------------------------------
    * Start WJacobi
    *-----------------------------------------------------------------------*/

   while ((iter+1) <= max_iter)
   {
      iter++;

      CopyVector(b, t);

      for (i = 0; i < n; i++)
      {
	 for (jj = ia[i]; jj < ia[i+1]-1; jj++)
	 {
	    j = ja[jj]-1;
	    tp[i] -= a[jj] * xp[j];
	 }
	 tp[i] /= a[ia[i]-1];
      }

      if (weight != 1.0)
      {
	 ScaleVector((1.0 - weight), x);
	 Axpy(weight, t, x);
      }
      else
      {
	 CopyVector(t, x);
      }
   }
}

/*--------------------------------------------------------------------------
 * WJacobiSetup
 *--------------------------------------------------------------------------*/

void      WJacobiSetup(problem, data)
Problem  *problem;
Data     *data;
{
   WJacobiData  *wjacobi_data = data;

   double   *darray;
   int       size;


   WJacobiDataA(wjacobi_data) = ProblemA(problem);

   size = VectorSize(ProblemF(problem));
   darray = ctalloc(double, NDIMU(size));
   WJacobiDataT(wjacobi_data) = NewVector(darray, size);
}

/*--------------------------------------------------------------------------
 * ReadWJacobiParams
 *--------------------------------------------------------------------------*/

Data  *ReadWJacobiParams(fp)
FILE  *fp;
{
   WJacobiData  *data;

   double   weight;
   int      max_iter;


   fscanf(fp, "%le", &weight);
   fscanf(fp, "%d",  &max_iter);

   data = NewWJacobiData(weight, max_iter, GlobalsLogFileName);

   return data;
}

/*--------------------------------------------------------------------------
 * NewWJacobiData
 *--------------------------------------------------------------------------*/

Data   *NewWJacobiData(weight, max_iter, log_file_name)
double  weight;
int     max_iter;
char   *log_file_name;
{
   WJacobiData  *wjacobi_data;

   wjacobi_data = ctalloc(WJacobiData, 1);

   WJacobiDataWeight(wjacobi_data)      = weight;
   WJacobiDataMaxIter(wjacobi_data)     = max_iter;

   WJacobiDataLogFileName(wjacobi_data) = log_file_name;

   return (Data *)wjacobi_data;
}

/*--------------------------------------------------------------------------
 * FreeWJacobiData
 *--------------------------------------------------------------------------*/

void   FreeWJacobiData(data)
Data  *data;
{
   WJacobiData  *wjacobi_data = data;


   if (wjacobi_data)
   {
      FreeVector(WJacobiDataT(wjacobi_data));
      tfree(wjacobi_data);
   }
}

