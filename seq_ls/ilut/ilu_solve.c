/*--------------------------------------------------------------------------
 * Purpose:       Solves a linear system given a right-hand side and the
 *                linear solver structure that was created by 
 *                ilu_initialize and ilu_setup.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/

#include "general.h"

/* Include headers for problem and solver data structure */

#include "ilu_data.h"

/* Include C interface to gmres_driver */
#include "ilut_facsol_f.h"

int ilu_solve ( void *input_data, double *x, double *b )
{
   ILUData    *ilu_data = input_data;
   int            n, nnz, ierr_solver, ierr_input;
   double        *b_copy;
   Matrix        *A, *preconditioner;
   int            i, ierr_ilut;
   int            lenpmx;
   int            scale, reorder;




  preconditioner = ILUDataPreconditioner(ilu_data);

  A =        ILUDataA(ilu_data);

  n = MatrixSize(A);
  nnz = MatrixNNZ(A);
  lenpmx = ILUDataLenpmx(ilu_data);

  if ( ILUDataMode(ilu_data) == 2 ) 
  {
    /* Will have to do the
       scaling and reordering of the vectors ourselves outside
       of GMRES and temporarily set flags telling GMRES that
       scaling and reordering were not performed */

    reorder = ILUDataIpar(ilu_data)[0];
    scale = ILUDataIpar(ilu_data)[1];
    ILUDataIpar(ilu_data)[0] = 0;
    ILUDataIpar(ilu_data)[1] = 0;

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
         b[i] *= ILUDataRscale(ilu_data)[i];
      }
    }

    if ( reorder )
    {
      CALL_DVPERM( n, b, ILUDataPerm(ilu_data) );
    }
  }


#ifdef ILUFact
  CALL_GMRES_DRIVER((n), (nnz),
                  MatrixIA(A), MatrixJA(A), MatrixData(A),
                  b, x,
                  ILUDataIpar(ilu_data),
                  ILUDataRpar(ilu_data),
                  lenpmx,
                  MatrixData(preconditioner),
                  MatrixJA(preconditioner),
                  MatrixIA(preconditioner),
                  ILUDataPerm(ilu_data), ILUDataInversePerm(ilu_data), 
                  ILUDataRscale(ilu_data), ILUDataCscale(ilu_data),
                  ILUDataRwork(ilu_data), (ILUDataLRwork(ilu_data)),
                  ierr_solver, ierr_input);
#elif defined ICFact
  CALL_GMRES_DRIVER((n), (nnz),
                  MatrixIA(A), MatrixJA(A), MatrixData(A),
                  b, x,
                  ILUDataIpar(ilu_data),
                  ILUDataRpar(ilu_data),
                  lenpmx,
                  MatrixData(preconditioner),
                  MatrixJA(preconditioner),
                  ILUDataPerm(ilu_data), ILUDataInversePerm(ilu_data), 
                  ILUDataScale(ilu_data),
                  ILUDataRwork(ilu_data), (ILUDataLRwork(ilu_data)),
                  ierr_solver, ierr_input);
#endif

  if ( ILUDataMode(ilu_data) == 2 ) 
  {
    if ( reorder )
    {
      CALL_DVPERM( n, x, ILUDataInversePerm(ilu_data) );
    }

#ifdef ILUFact
    if ((scale == 2) || (scale == 3))
#elif defined ICFact
    if ( scale != 0 )
#endif
    {
      for ( i=0; i< n; i++ )
         x[i] *= ILUDataCscale(ilu_data)[i];
    }

    ILUDataIpar(ilu_data)[0] = reorder;
    ILUDataIpar(ilu_data)[1] = scale;

    for ( i=0; i< n; i++ )
    {
       b[i] = b_copy[i];
    }
    tfree(b_copy);
    
  }

  if( ierr_input != 0 ) 
  {
    printf("Input error to GMRES, error %d\n", ierr_input);
    return(ierr_input);
  }

  if( ierr_solver != 0 ) 
  {
    if( ierr_solver != -1 ) 
    { 
      printf("Warning: Nonzero error code in GMRES, error %d\n", ierr_solver);
      return(0);
    } else
    {
      ierr_solver = 0;
    }
  }


  return(ierr_solver); 
}
