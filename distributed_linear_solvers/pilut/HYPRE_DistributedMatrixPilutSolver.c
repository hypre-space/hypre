/* Include headers for problem and solver data structure */
#include "./DistributedMatrixPilutSolver.h"


/*--------------------------------------------------------------------------
 * HYPRE_NewDistributedMatrixPilutSolver
 *--------------------------------------------------------------------------*/

HYPRE_DistributedMatrixPilutSolver  HYPRE_NewDistributedMatrixPilutSolver( 
                                  MPI_Comm comm,
                                  HYPRE_DistributedMatrix matrix )
     /* Allocates and Initializes solver structure */
{

   hypre_DistributedMatrixPilutSolver     *solver;
   hypre_PilutSolverGlobals *globals;
   int            ierr, nprocs, myid;

   /* Allocate structure for holding solver data */
   solver = (hypre_DistributedMatrixPilutSolver *) 
            hypre_CTAlloc( hypre_DistributedMatrixPilutSolver, 1);

   /* Initialize components of solver */
   hypre_DistributedMatrixPilutSolverComm(solver) = comm;
   hypre_DistributedMatrixPilutSolverDataDist(solver) = 
         (DataDistType *) hypre_CTAlloc( DataDistType, 1 );

   /* Structure for holding "global variables"; makes code thread safe(r) */
   globals = hypre_DistributedMatrixPilutSolverGlobals(solver) = 
       (hypre_PilutSolverGlobals *) hypre_CTAlloc( hypre_PilutSolverGlobals, 1 );

   /* Set some variables in the "global variables" section */
   pilut_comm = comm;

   MPI_Comm_size( comm, &nprocs );
   npes = nprocs;

   MPI_Comm_rank( comm, &myid );
   mype = myid;

   /* Data distribution structure */
   DataDistTypeRowdist(hypre_DistributedMatrixPilutSolverDataDist(solver))
       = (int *) hypre_CTAlloc( int, nprocs+1 );

   hypre_DistributedMatrixPilutSolverFactorMat(solver) = 
          (FactorMatType *) hypre_CTAlloc( FactorMatType, 1 );

   /* Note that because we allow matrix to be NULL at this point so that it can
      be set later with a SetMatrix call, we do nothing with matrix except insert
      it into the structure */
   hypre_DistributedMatrixPilutSolverMatrix(solver) = matrix;

   /* Defaults for Parameters controlling the incomplete factorization */
   hypre_DistributedMatrixPilutSolverGmaxnz(solver)   = 20;     /* Maximum nonzeroes per row of factor */
   hypre_DistributedMatrixPilutSolverTol(solver)   = 0.000001;  /* Drop tolerance for factor */

   /* Return created structure to calling routine */
   return( (HYPRE_DistributedMatrixPilutSolver) solver );

}

/*--------------------------------------------------------------------------
 * HYPRE_FreeDistributedMatrixPilutSolver
 *--------------------------------------------------------------------------*/

int HYPRE_FreeDistributedMatrixPilutSolver ( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr )
{
   hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_TFree( DataDistTypeRowdist(hypre_DistributedMatrixPilutSolverDataDist(solver)));
  hypre_TFree( hypre_DistributedMatrixPilutSolverDataDist(solver) );
  
  hypre_TFree( hypre_DistributedMatrixPilutSolverFactorMat(solver) );

  hypre_TFree( hypre_DistributedMatrixPilutSolverGlobals(solver) );

  hypre_TFree(solver);

  return(0);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverInitialize
 *--------------------------------------------------------------------------*/

int HYPRE_DistributedMatrixPilutSolverInitialize ( 
                  HYPRE_DistributedMatrixPilutSolver solver )
{

   return(0);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSetMatrix
 *--------------------------------------------------------------------------*/

int HYPRE_DistributedMatrixPilutSolverSetMatrix( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr,
                  HYPRE_DistributedMatrix matrix )
{
  int ierr=0;
  hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverMatrix( solver ) = matrix;
  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverGetMatrix
 *--------------------------------------------------------------------------*/

HYPRE_DistributedMatrix
   HYPRE_DistributedMatrixPilutSolverGetMatrix( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr )
{
  hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  return( hypre_DistributedMatrixPilutSolverMatrix( solver ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSetFirstLocalRow
 *--------------------------------------------------------------------------*/

int HYPRE_DistributedMatrixPilutSolverSetNumLocalRow( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr,
                  int FirstLocalRow )
{
  int ierr=0;
  hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;
   hypre_PilutSolverGlobals *globals = hypre_DistributedMatrixPilutSolverGlobals(solver);

  DataDistTypeRowdist(hypre_DistributedMatrixPilutSolverDataDist( solver ))[mype] = 
     FirstLocalRow;

  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSetFactorRowSize
 *   Sets the maximum number of entries to be kept in the incomplete factors
 *   This number applies both to the row of L, and also separately to the
 *   row of U.
 *--------------------------------------------------------------------------*/

int HYPRE_DistributedMatrixPilutSolverSetFactorRowSize( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr,
                  int size )
{
  int ierr=0;
  hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverGmaxnz( solver ) = size;

  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSetDropTolerance
 *--------------------------------------------------------------------------*/

int HYPRE_DistributedMatrixPilutSolverSetDropTolerance( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr,
                  double tolerance )
{
  int ierr=0;
  hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverTol( solver ) = tolerance;

  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSetMaxIts
 *--------------------------------------------------------------------------*/

int HYPRE_DistributedMatrixPilutSolverSetMaxIts( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr,
                  int its )
{
  int ierr=0;
  hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverMaxIts( solver ) = its;

  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSetup
 *--------------------------------------------------------------------------*/

int HYPRE_DistributedMatrixPilutSolverSetup( HYPRE_DistributedMatrixPilutSolver in_ptr )
{
   int ierr=0;
   int m, n, nprocs, start, end, *rowdist;
   hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;
   hypre_PilutSolverGlobals *globals = hypre_DistributedMatrixPilutSolverGlobals(solver);


   if(hypre_DistributedMatrixPilutSolverMatrix(solver) == NULL )
   {
      ierr = -1;
      /* printf("Cannot call setup to solver until matrix has been set\n");*/
      return(ierr);
   }

   /* Set up the DataDist structure */

   HYPRE_GetDistributedMatrixDims(
      hypre_DistributedMatrixPilutSolverMatrix(solver), &m, &n);

   DataDistTypeNrows( hypre_DistributedMatrixPilutSolverDataDist( solver ) ) = m;

   HYPRE_GetDistributedMatrixLocalRange(
      hypre_DistributedMatrixPilutSolverMatrix(solver), &start, &end);

   DataDistTypeLnrows(hypre_DistributedMatrixPilutSolverDataDist( solver )) = 
      end - start;

   /* Set up DataDist entry in distributed_solver */
   /* This requires that each processor know which rows are owned by each proc */
   nprocs = npes;

   rowdist = DataDistTypeRowdist( hypre_DistributedMatrixPilutSolverDataDist( solver ) );

   MPI_Allgather( &start, 1, MPI_INT, rowdist, 1, MPI_INT, 
      hypre_DistributedMatrixPilutSolverComm(solver) );

   rowdist[ nprocs ] = n;


   /* Perform approximate factorization */
   ILUT( hypre_DistributedMatrixPilutSolverDataDist (solver),
         hypre_DistributedMatrixPilutSolverMatrix (solver),
         hypre_DistributedMatrixPilutSolverFactorMat (solver),
         hypre_DistributedMatrixPilutSolverGmaxnz (solver),
         hypre_DistributedMatrixPilutSolverTol (solver),
         hypre_DistributedMatrixPilutSolverGlobals (solver)
       );

   SetUpLUFactor( hypre_DistributedMatrixPilutSolverDataDist (solver), 
               hypre_DistributedMatrixPilutSolverFactorMat (solver),
               hypre_DistributedMatrixPilutSolverGmaxnz (solver),
               hypre_DistributedMatrixPilutSolverGlobals (solver) );

#ifdef HYPRE_DEBUG
   fflush(stdout);
   printf("Nlevels: %d\n",
          hypre_DistributedMatrixPilutSolverFactorMat (solver)->nlevels);
#endif

   return(0);
}


/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSolve
 *--------------------------------------------------------------------------*/

int HYPRE_DistributedMatrixPilutSolverSolve( HYPRE_DistributedMatrixPilutSolver in_ptr,
                                           double *x, double *b )
{

   hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

   /******** NOTE: Since I am using this currently as a preconditioner, I am only
     doing a single front and back solve. To be a general-purpose solver, this
     call should really be in a loop checking convergence and counting iterations.
     AC - 2/12/98 
   */
   /* It should be obvious, but the current treatment of vectors is pretty
      insufficient. -AC 2/12/98 
   */
   LDUSolve( hypre_DistributedMatrixPilutSolverDataDist (solver),
         hypre_DistributedMatrixPilutSolverFactorMat (solver),
         x,
         b,
         hypre_DistributedMatrixPilutSolverGlobals (solver)
       );

  return(0);
}

