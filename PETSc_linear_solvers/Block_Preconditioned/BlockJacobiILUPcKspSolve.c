/* Include headers for problem and solver data structure */
#include "BlockJacobiPcKsp.h"


int BlockJacobiILUPcKspSolve( void *data, Mat A, Vec x, Vec b )
     /* Uses ILU as an approximate linear system solver on each
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

