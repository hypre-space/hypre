/* Include BJ Headers */
#include "BlockJacobiPcKsp.h"

/* Include solver calling sequence */

/***********************************************************************/
int ILU_Apply ( void *solver_data, Vec b, Vec x)
     /* Wrapper around inc_fact_solve in form needed by Petsc */
{
  double    *ILU_b, *ILU_x;
  void      *ilu_data;
  Scalar     zero = 0.0;
  BJData    *BJ_data = solver_data;
  int        ierr;
  int        size;
  int        ierr_ilut, ierr_solver, ierr_input;
  int        n, nnz, lenpmx;
  int        scale, reorder;


  /***********                                    ***********/
  /* Convert Petsc formatted vectors to that expected for ILU */
  ierr = VecGetArray( b, &ILU_b ); CHKERRA(ierr);
  ierr = VecGetLocalSize( b, &size ); CHKERRA(ierr);

  ierr = VecSet( &zero, x );
  ierr = VecGetArray( x, &ILU_x ); CHKERRA(ierr);
  ierr = VecGetLocalSize( x, &size ); CHKERRA(ierr);
  /***********                                    ***********/

  ilu_data = BJDataLsData( BJ_data );

  ierr = ilu_solve ( ilu_data, ILU_x, ILU_b );
  /*
  ierr = VecCopy( b, x );
  */

  ierr = VecRestoreArray( b, &ILU_b); CHKERRA(ierr);
  ierr = VecRestoreArray( x, &ILU_x); CHKERRA(ierr);

  return(ierr); 

}
 
