/* Include headers */
#include "BlockJacobiPcKsp.h"
#include <string.h>

/* debugging header */
#include <cegdb.h>


int main(int argc,char **args)
{
  void       *bj_data;
  Mat         A;
  Vec         x, b, error, rhs, exact_soln;
  int         soln_provided;
  char       *problem;
  char        dlamch_arg[1]="e";
  char        file_name[255];
  int         ierr, n;
  int         myid;
  double      minus_1 = -1.0, zero = 0.0;
  double      norm_A, scaled_resid, norm_resid, norm_soln, eps;
  double      dlamch_( char *arg1, int arg2);



  /* Initialize Petsc */
  PetscInitialize(&argc,&args,(char *)0,NULL);

  MPI_Comm_rank(MPI_COMM_WORLD, &myid );

  cegdb(&argc, &args, myid);

  if (argc < 2)
    {
      fprintf(stderr, "Usage: driver.ILU <problem> [Petsc options]\n");
      exit(1);
    }

  /* Initialize solver structure */
  bj_data = BlockJacobiILUPcKspInitialize( (void *) NULL );

  problem = args[1];
  strcpy( file_name, problem);
  
  /* Read in matrix and vectors and put in Petsc format */
  /* Create and assemble matrix */
  ierr = ReadMPIAIJ_matread( &A, strcat( file_name, ".matx"), &n ); CHKERRA( ierr );
  strcpy( file_name, problem);
  PetscPrintf(MPI_COMM_WORLD, "Finished reading in matrix\n");

  /* Create vectors */
  ierr = ReadMPIVec( &b, strcat( file_name, ".rhs") ); CHKERRA( ierr );
  strcpy( file_name, problem);
  ierr = VecDuplicate (b, &rhs ); CHKERRA( ierr );
  ierr = VecCopy ( b, rhs); CHKERRA( ierr );

  ierr = ReadMPIVec( &x, strcat( file_name, ".initu") ); 
  strcpy( file_name, problem); 

  if ( ierr == 1) 
  { /* No initial guess was provided; set initial guess to zero */
    ierr = VecDuplicate (b, &x ); CHKERRA( ierr );
    ierr = VecSet( &zero, x );
  }
  else
    CHKERRA( ierr );

  ierr = ReadMPIVec( &exact_soln, strcat( file_name, ".soln") );
  soln_provided = ( ierr != 1 ); ierr = 0;

  PetscPrintf(MPI_COMM_WORLD, "Finished reading in vectors\n");


  /* call setup of routine */
  PetscPrintf(MPI_COMM_WORLD, "calling setup routine\n");
  ierr = BlockJacobiILUPcKspSetup( bj_data, A, x, b ); CHKERRA( ierr );

  /* call solve routine */
  PetscPrintf(MPI_COMM_WORLD, "calling solver routine\n");
  ierr = BlockJacobiILUPcKspSolve( bj_data, A, x, b ); CHKERRA( ierr );

  /* Output */
  /*
  ierr = VecView ( x, 0 );
  */

  ierr = MatNorm( A, NORM_1, &norm_A ); CHKERRA( ierr );

  ierr = VecNorm ( x, NORM_2, &norm_soln); CHKERRA(ierr);
#ifdef debug
  PetscPrintf(MPI_COMM_WORLD, "Here is the RHS\n");
  ierr = VecView (rhs, 0 );
#endif

  ierr = VecDuplicate( x, &error ); CHKERRA( ierr );
  ierr = MatMult ( A, x, error ); CHKERRA( ierr );
#ifdef debug
  PetscPrintf(MPI_COMM_WORLD, "Here is A*soln\n");
  ierr = VecView (error, 0 );
#endif

  ierr = VecAXPY ( &minus_1, rhs, error ); CHKERRA(ierr);
#ifdef debug
  PetscPrintf(MPI_COMM_WORLD, "Here is the residual vector\n");
  ierr = VecView (error, 0 );
#endif
  ierr = VecNorm ( error, NORM_2, &norm_resid ); CHKERRA(ierr);
  PetscPrintf(MPI_COMM_WORLD, "Norm of residual of answer is %e \n",norm_resid);

  eps = dlamch_( dlamch_arg, strlen("e") );

#ifdef debug
  PetscPrintf(MPI_COMM_WORLD, 
    "Scaled residual is calculated as %e/(%d*%e*%e*%e) \n",
             norm_resid, n, eps, norm_A, norm_soln );
#endif
  scaled_resid = norm_resid/( n*eps*norm_A*norm_soln);
  PetscPrintf(MPI_COMM_WORLD, "Scaled residual of answer is %e \n",scaled_resid);

  if ( soln_provided ) {
     ierr = VecAXPY ( &minus_1, x, exact_soln ); CHKERRA( ierr );
     ierr = VecNorm ( exact_soln, NORM_2, &norm_resid ); CHKERRA ( ierr ) ;
     PetscPrintf(MPI_COMM_WORLD, "Scaled norm of error from actual soln is %e \n",
         norm_resid/norm_soln);
  }

  /* Cleanup after routine */
  BlockJacobiILUPcKspFinalize ( bj_data ); CHKERRA(ierr);

  ierr = VecDestroy(error); CHKERRA(ierr);
  ierr = VecDestroy(rhs); CHKERRA(ierr);  
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(exact_soln); CHKERRA(exact_soln);
  ierr = VecDestroy(b); CHKERRA(ierr);  
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
}
