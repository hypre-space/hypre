/*--------------------------------------------------------------------------
 * Purpose:       Solves a linear system given a right-hand side and the
 *                linear solver structure that was created by 
 *                incfact_initialize and incfact_setup.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/

#include "general.h"

/* Include headers for problem and solver data structure */

#include "incfact_data.h"

/* Include C interface to ksp_driver */
#include "incfactt_facsol_f.h"

int incfact_solve ( void *input_data, double *x, double *b )
{
   INCFACTData    *incfact_data = input_data;
   int            n, nnz, ierr_solver, ierr_input;
   double        *b_copy;
   Matrix        *A, *preconditioner;
   int            i, ierr_incfactt;
   int            lenpmx;
   int            scale, reorder;




  preconditioner = INCFACTDataPreconditioner(incfact_data);

  A =        INCFACTDataA(incfact_data);

  n = MatrixSize(A);
  nnz = MatrixNNZ(A);
  lenpmx = INCFACTDataLenpmx(incfact_data);

  if ( INCFACTDataMode(incfact_data) == 2 ) 
  {
    /* Will have to do the
       scaling and reordering of the vectors ourselves outside
       of KSP and temporarily set flags telling KSP that
       scaling and reordering were not performed */

    reorder = INCFACTDataIpar(incfact_data)[0];
    scale = INCFACTDataIpar(incfact_data)[1];
    INCFACTDataIpar(incfact_data)[0] = 0;
    INCFACTDataIpar(incfact_data)[1] = 0;

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
         b[i] *= INCFACTDataRscale(incfact_data)[i];
      }
    }

    if ( reorder )
    {
      CALL_DVPERM( n, b, INCFACTDataPerm(incfact_data) );
    }
  }


#ifdef ILUFact
  CALL_KSP_DRIVER((n), (nnz),
                  MatrixIA(A), MatrixJA(A), MatrixData(A),
                  b, x,
                  INCFACTDataIpar(incfact_data),
                  INCFACTDataRpar(incfact_data),
                  lenpmx,
                  MatrixData(preconditioner),
                  MatrixJA(preconditioner),
                  MatrixIA(preconditioner),
                  INCFACTDataPerm(incfact_data), INCFACTDataInversePerm(incfact_data), 
                  INCFACTDataRscale(incfact_data), INCFACTDataCscale(incfact_data),
                  INCFACTDataRwork(incfact_data), (INCFACTDataLRwork(incfact_data)),
                  ierr_solver, ierr_input);
#elif defined ICFact
  CALL_KSP_DRIVER((n), (nnz),
                  MatrixIA(A), MatrixJA(A), MatrixData(A),
                  b, x,
                  INCFACTDataIpar(incfact_data),
                  INCFACTDataRpar(incfact_data),
                  lenpmx,
                  MatrixData(preconditioner),
                  MatrixJA(preconditioner),
                  INCFACTDataPerm(incfact_data), INCFACTDataInversePerm(incfact_data), 
                  INCFACTDataScale(incfact_data),
                  INCFACTDataRwork(incfact_data), (INCFACTDataLRwork(incfact_data)),
                  ierr_solver, ierr_input);
#endif

  if ( INCFACTDataMode(incfact_data) == 2 ) 
  {
    if ( reorder )
    {
      CALL_DVPERM( n, x, INCFACTDataInversePerm(incfact_data) );
    }

#ifdef ILUFact
    if ((scale == 2) || (scale == 3))
#elif defined ICFact
    if ( scale != 0 )
#endif
    {
      for ( i=0; i< n; i++ )
         x[i] *= INCFACTDataCscale(incfact_data)[i];
    }

    INCFACTDataIpar(incfact_data)[0] = reorder;
    INCFACTDataIpar(incfact_data)[1] = scale;

    for ( i=0; i< n; i++ )
    {
       b[i] = b_copy[i];
    }
    tfree(b_copy);
    
  }

  if( ierr_input != 0 ) 
  {
    printf("Input error to KSP, error %d\n", ierr_input);
    return(ierr_input);
  }

  if( ierr_solver != 0 ) 
  {
    if( ierr_solver != -1 ) 
    { 
      printf("Warning: Nonzero error code in KSP, error %d\n", ierr_solver);
      return(0);
    } else
    {
      ierr_solver = 0;
    }
  }


  return(ierr_solver); 
}
