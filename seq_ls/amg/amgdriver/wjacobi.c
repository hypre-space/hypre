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

#include "headers.h"
#include "wjacobi.h"


/*--------------------------------------------------------------------------
 * WJacobi
 *--------------------------------------------------------------------------*/

int    	 WJacobi(x, b, tol, data)
Vector 	*x;
Vector 	*b;
double 	 tol;
void    *data;
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

void      WJacobiSetup(A, data)
Matrix   *A;
void     *data;
{
   WJacobiData  *wjacobi_data = data;

   double   *darray;
   int       size;


   WJacobiDataA(wjacobi_data) = A;

   size = MatrixSize(A);
   darray = ctalloc(double, NDIMU(size));
   WJacobiDataT(wjacobi_data) = NewVector(darray, size);
}

/*--------------------------------------------------------------------------
 * NewWJacobiData
 *--------------------------------------------------------------------------*/

void     *NewWJacobiData(problem, solver, log_file_name)
Problem  *problem;
Solver   *solver;
char     *log_file_name;
{
   WJacobiData  *wjacobi_data;

   wjacobi_data = ctalloc(WJacobiData, 1);

   WJacobiDataWeight(wjacobi_data)      = SolverWJacobiWeight(solver);
   WJacobiDataMaxIter(wjacobi_data)     = SolverWJacobiMaxIter(solver);

   WJacobiDataLogFileName(wjacobi_data) = log_file_name;

   return (void *)wjacobi_data;
}

/*--------------------------------------------------------------------------
 * FreeWJacobiData
 *--------------------------------------------------------------------------*/

void   FreeWJacobiData(data)
void  *data;
{
   WJacobiData  *wjacobi_data = data;


   if (wjacobi_data)
   {
      FreeVector(WJacobiDataT(wjacobi_data));
      tfree(wjacobi_data);
   }
}

