/* Include headers for problem and solver data structure */
#include "BlockJacobiINCFACTPcKsp.h"


int BlockJacobiINCFACTPcKsp(Mat A, Vec x, Vec b, void *data )
     /* Uses INCFACT as an approximate linear system solver on each
        processor as the block solver in BlockJacobi Preconditioner */
{


  BJData   *bj_data;
  SLES     *sles_ptr;  
  int       flg, its;


  bj_data = (BJData *) data;

  sles_ptr = BJDataSles_ptr( bj_data );

  /* Call Petsc solver */
#if 0
  printf("about to call slessolve\n");
#endif

  flg = SLESSolve(*sles_ptr,b,x,&its); CHKERRA(flg);

#if 0
  PetscPrintf(MPI_COMM_WORLD, "iterations = %d\n",its);
#endif


  return flg;
}

