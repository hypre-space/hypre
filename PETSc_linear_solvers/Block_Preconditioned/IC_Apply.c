/* Include BJ Headers */
#include "BlockJacobiPcKsp.h"

/* Include solver calling sequence */

/***********************************************************************/
int IC_Apply ( void *solver_data, Vec b, Vec x)
     /* Wrapper around inc_fact_solve in form needed by Petsc */
{
  double    *IC_b, *IC_x;
  void      *ic_data;
  Scalar     zero = 0.0;
  BJData    *BJ_data = solver_data;
  int        ierr;
  int        size;
  int        ierr_ict, ierr_solver, ierr_input;
  int        n, nnz, lenpmx;
  int        scale, reorder;


  /***********                                    ***********/
  /* Convert Petsc formatted vectors to that expected for IC */
  ierr = VecGetArray( b, &IC_b ); CHKERRA(ierr);
  ierr = VecGetLocalSize( b, &size ); CHKERRA(ierr);

  ierr = VecSet( &zero, x );
  ierr = VecGetArray( x, &IC_x ); CHKERRA(ierr);
  ierr = VecGetLocalSize( x, &size ); CHKERRA(ierr);
  /***********                                    ***********/

  ic_data = BJDataLsData( BJ_data );

  ierr = ic_solve ( ic_data, IC_x, IC_b );
  /*
  ierr = VecCopy( b, x );
  */

  ierr = VecRestoreArray( b, &IC_b); CHKERRA(ierr);
  ierr = VecRestoreArray( x, &IC_x); CHKERRA(ierr);

  return(ierr); 

}
 
