/*--------------------------------------------------------------------------
 * Purpose:       Solves a linear system given a right-hand side and the
 *                linear solver structure that was created by 
 *                ic_initialize and ic_setup.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/

#include "general.h"

/* Include headers for problem and solver data structure */

#include "ic_data.h"

/* Include C interface to cg_driver */
#include "ict_facsol_f.h"

int ic_solve ( void *input_data, double *x, double *b )
{
   ICData    *ic_data = input_data;
   int            n, nnz, ierr_solver, ierr_input;
   double        *b_copy;
   Matrix        *A, *preconditioner;
   int            i, ierr_ict;
   int            lenpmx;
   int            scale, reorder;




  preconditioner = ICDataPreconditioner(ic_data);

  A =        ICDataA(ic_data);

  n = MatrixSize(A);
  nnz = MatrixNNZ(A);
  lenpmx = ICDataLenpmx(ic_data);

  if ( ICDataMode(ic_data) == 2 ) 
  {
    /* Will have to do the
       scaling and reordering of the vectors ourselves outside
       of CG and temporarily set flags telling CG that
       scaling and reordering were not performed */

    reorder = ICDataIpar(ic_data)[0];
    scale = ICDataIpar(ic_data)[1];
    ICDataIpar(ic_data)[0] = 0;
    ICDataIpar(ic_data)[1] = 0;

    b_copy = talloc(double,n);

    for ( i=0; i< n; i++ )
    {
       b_copy[i] = b[i];
    }

#ifdef ILUFact
    if ((scale == 1) || (scale == 3))
#elif defined ICFact
    if ( scale != 0 )
#endif
    {
      for ( i=0; i< n; i++ )
      {
         b[i] *= ICDataRscale(ic_data)[i];
      }
    }

    if ( reorder )
    {
      CALL_DVPERM( n, b, ICDataPerm(ic_data) );
    }
  }


#ifdef ILUFact
  CALL_CG_DRIVER((n), (nnz),
                  MatrixIA(A), MatrixJA(A), MatrixData(A),
                  b, x,
                  ICDataIpar(ic_data),
                  ICDataRpar(ic_data),
                  lenpmx,
                  MatrixData(preconditioner),
                  MatrixJA(preconditioner),
                  MatrixIA(preconditioner),
                  ICDataPerm(ic_data), ICDataInversePerm(ic_data), 
                  ICDataRscale(ic_data), ICDataCscale(ic_data),
                  ICDataRwork(ic_data), (ICDataLRwork(ic_data)),
                  ierr_solver, ierr_input);
#elif defined ICFact
  CALL_CG_DRIVER((n), (nnz),
                  MatrixIA(A), MatrixJA(A), MatrixData(A),
                  b, x,
                  ICDataIpar(ic_data),
                  ICDataRpar(ic_data),
                  lenpmx,
                  MatrixData(preconditioner),
                  MatrixJA(preconditioner),
                  ICDataPerm(ic_data), ICDataInversePerm(ic_data), 
                  ICDataScale(ic_data),
                  ICDataRwork(ic_data), (ICDataLRwork(ic_data)),
                  ierr_solver, ierr_input);
#endif

  if ( ICDataMode(ic_data) == 2 ) 
  {
    if ( reorder )
    {
      CALL_DVPERM( n, x, ICDataInversePerm(ic_data) );
    }

#ifdef ILUFact
    if ((scale == 2) || (scale == 3))
#elif defined ICFact
    if ( scale != 0 )
#endif
    {
      for ( i=0; i< n; i++ )
         x[i] *= ICDataCscale(ic_data)[i];
    }

    ICDataIpar(ic_data)[0] = reorder;
    ICDataIpar(ic_data)[1] = scale;

    for ( i=0; i< n; i++ )
    {
       b[i] = b_copy[i];
    }
    tfree(b_copy);
    
  }

  if( ierr_input != 0 ) 
  {
    printf("Input error to CG, error %d\n", ierr_input);
    return(ierr_input);
  }

  if( ierr_solver != 0 ) 
  {
    if( ierr_solver != -1 ) 
    { 
      printf("Warning: Nonzero error code in CG, error %d\n", ierr_solver);
      return(0);
    } else
    {
      ierr_solver = 0;
    }
  }


  return(ierr_solver); 
}
