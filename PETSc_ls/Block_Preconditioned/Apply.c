/* Include BJ Headers */
#include "BlockJacobiPcKsp.h"

/* Include solver calling sequence */

/***********************************************************************/
int INCFACT_Apply ( void *solver_data, Vec b, Vec x)
     /* Wrapper around inc_fact_solve in form needed by Petsc */
{
  double    *INCFACT_b, *INCFACT_x;
  void      *incfact_data;
  Scalar     zero = 0.0;
  BJData    *BJ_data = solver_data;
  int        ierr;
  int        size;
  int        ierr_incfactt, ierr_solver, ierr_input;
  int        n, nnz, lenpmx;
  int        scale, reorder;


  /***********                                    ***********/
  /* Convert Petsc formatted vectors to that expected for INCFACT */
  ierr = VecGetArray( b, &INCFACT_b ); CHKERRA(ierr);
  ierr = VecGetLocalSize( b, &size ); CHKERRA(ierr);

  ierr = VecSet( &zero, x );
  ierr = VecGetArray( x, &INCFACT_x ); CHKERRA(ierr);
  ierr = VecGetLocalSize( x, &size ); CHKERRA(ierr);
  /***********                                    ***********/

  incfact_data = BJDataLsData( BJ_data );

  ierr = incfact_solve ( incfact_data, INCFACT_x, INCFACT_b );
  /*
  ierr = VecCopy( b, x );
  */

  ierr = VecRestoreArray( b, &INCFACT_b); CHKERRA(ierr);
  ierr = VecRestoreArray( x, &INCFACT_x); CHKERRA(ierr);

  return(ierr); 

}
 
