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
 * Relaxation scheme
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * Relax
 *--------------------------------------------------------------------------*/

int    	 Relax(u, f, A, icg, vtype)
Vector 	u;
Vector 	f;
Matrix  A;
int    *icg;
int     vtype;
{
   double         *a  = MatrixData(A);
   int            *ia = MatrixIA(A);
   int            *ja = MatrixJA(A);
   int             n  = MatrixSize(A);
	          
   double         *up = VectorData(u);
   double         *fp = VectorData(f);
	          
   int             i, j, jj;
   int             relax_error;


   /*-----------------------------------------------------------------------
    * Start Relaxation sweep.
    *-----------------------------------------------------------------------*/


   for (i = 0; i < n; i++)
   { 
       if (vtype==0 || (vtype==1 && igc[i]<0) || (vtype==3 && icg[i]>0)
       {
          if (a[ia[i]-1] != 0.0)
          {
             for (jj = ia[i]; jj < ia[i+1]-1; jj++)
	     {
                 res = fp[i];
	         j = ja[jj]-1;
	         res -= a[jj] * xp[j];
	     }
          }
	  up[i] = res/a[ia[i]-1];
        }
   }
}


