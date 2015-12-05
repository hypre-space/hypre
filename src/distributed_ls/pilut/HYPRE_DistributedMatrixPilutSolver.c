/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/* Include headers for problem and solver data structure */
#include "./DistributedMatrixPilutSolver.h"


/*--------------------------------------------------------------------------
 * HYPRE_NewDistributedMatrixPilutSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int  HYPRE_NewDistributedMatrixPilutSolver( 
                                  MPI_Comm comm,
                                  HYPRE_DistributedMatrix matrix,
                                  HYPRE_DistributedMatrixPilutSolver *new_solver )
     /* Allocates and Initializes solver structure */
{

   hypre_DistributedMatrixPilutSolver     *solver;
   hypre_PilutSolverGlobals *globals;
   HYPRE_Int            nprocs, myid;
   FactorMatType *ldu;

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

   jr = NULL;
   lr = NULL;
   jw = NULL;
   w  = NULL;

   /* Set some variables in the "global variables" section */
   pilut_comm = comm;

   hypre_MPI_Comm_size( comm, &nprocs );
   npes = nprocs;

   hypre_MPI_Comm_rank( comm, &myid );
   mype = myid;

#ifdef HYPRE_TIMING
   globals->CCI_timer = hypre_InitializeTiming( "hypre_ComputeCommInfo" );
   globals->SS_timer = hypre_InitializeTiming( "hypre_SelectSet" );
   globals->SFR_timer = hypre_InitializeTiming( "hypre_SendFactoredRows" );
   globals->CR_timer = hypre_InitializeTiming( "hypre_ComputeRmat" );
   globals->FL_timer = hypre_InitializeTiming( "hypre_FactorLocal" );
   globals->SLUD_timer = hypre_InitializeTiming( "SeparateLU_byDIAG" );
   globals->SLUM_timer = hypre_InitializeTiming( "SeparateLU_byMIS" );
   globals->UL_timer = hypre_InitializeTiming( "hypre_UpdateL" );
   globals->FNR_timer = hypre_InitializeTiming( "hypre_FormNRmat" );

   globals->Ll_timer = hypre_InitializeTiming( "Local part of front solve" );
   globals->Lp_timer = hypre_InitializeTiming( "Parallel part of front solve" );
   globals->Up_timer = hypre_InitializeTiming( "Parallel part of back solve" );
   globals->Ul_timer = hypre_InitializeTiming( "Local part of back solve" );
#endif

   /* Data distribution structure */
   DataDistTypeRowdist(hypre_DistributedMatrixPilutSolverDataDist(solver))
       = (HYPRE_Int *) hypre_CTAlloc( HYPRE_Int, nprocs+1 );

   hypre_DistributedMatrixPilutSolverFactorMat(solver) = 
          (FactorMatType *) hypre_CTAlloc( FactorMatType, 1 );

   ldu = hypre_DistributedMatrixPilutSolverFactorMat(solver);

   ldu->lsrowptr = NULL;
   ldu->lerowptr = NULL;
   ldu->lcolind  = NULL;
   ldu->lvalues  = NULL;
   ldu->usrowptr = NULL;
   ldu->uerowptr = NULL;
   ldu->ucolind  = NULL;
   ldu->uvalues  = NULL;
   ldu->dvalues  = NULL;
   ldu->nrm2s    = NULL;
   ldu->perm     = NULL;
   ldu->iperm    = NULL;

   /* Note that because we allow matrix to be NULL at this point so that it can
      be set later with a SetMatrix call, we do nothing with matrix except insert
      it into the structure */
   hypre_DistributedMatrixPilutSolverMatrix(solver) = matrix;

   /* Defaults for Parameters controlling the incomplete factorization */
   hypre_DistributedMatrixPilutSolverGmaxnz(solver)   = 20;     /* Maximum nonzeroes per row of factor */
   hypre_DistributedMatrixPilutSolverTol(solver)   = 0.000001;  /* Drop tolerance for factor */

   /* Return created structure to calling routine */
   *new_solver = ( (HYPRE_DistributedMatrixPilutSolver) solver );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeDistributedMatrixPilutSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_FreeDistributedMatrixPilutSolver ( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr )
{
  FactorMatType *ldu;

   hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

#ifdef HYPRE_TIMING
   hypre_PilutSolverGlobals *globals;
  globals = hypre_DistributedMatrixPilutSolverGlobals(solver);
#endif

  hypre_TFree( DataDistTypeRowdist(hypre_DistributedMatrixPilutSolverDataDist(solver)));
  hypre_TFree( hypre_DistributedMatrixPilutSolverDataDist(solver) );
  
  /* Free malloced members of the FactorMat member */
  ldu = hypre_DistributedMatrixPilutSolverFactorMat(solver);

  hypre_TFree( ldu->lcolind );
  hypre_TFree( ldu->ucolind );

  hypre_TFree( ldu->lvalues );
  hypre_TFree( ldu->uvalues );

  hypre_TFree( ldu->lrowptr );
  hypre_TFree( ldu->urowptr );

  hypre_TFree( ldu->dvalues );
  hypre_TFree( ldu->nrm2s );
  hypre_TFree( ldu->perm );
  hypre_TFree( ldu->iperm );

  hypre_TFree( ldu->gatherbuf );

  hypre_TFree( ldu->lx );
  hypre_TFree( ldu->ux );

    /* Beginning of TriSolveCommType freeing */
    hypre_TFree( ldu->lcomm.raddr );
    hypre_TFree( ldu->ucomm.raddr );

    hypre_TFree( ldu->lcomm.spes );
    hypre_TFree( ldu->ucomm.spes );

    hypre_TFree( ldu->lcomm.sptr );
    hypre_TFree( ldu->ucomm.sptr );

    hypre_TFree( ldu->lcomm.sindex );
    hypre_TFree( ldu->ucomm.sindex );

    hypre_TFree( ldu->lcomm.auxsptr );
    hypre_TFree( ldu->ucomm.auxsptr );

    hypre_TFree( ldu->lcomm.rpes );
    hypre_TFree( ldu->ucomm.rpes );

    hypre_TFree( ldu->lcomm.rdone );
    hypre_TFree( ldu->ucomm.rdone );

    hypre_TFree( ldu->lcomm.rnum );
    hypre_TFree( ldu->ucomm.rnum );

    /* End of TriSolveCommType freeing */

  hypre_TFree( hypre_DistributedMatrixPilutSolverFactorMat(solver) );
  /* End of FactorMat member */

#ifdef HYPRE_TIMING
  hypre_FinalizeTiming( globals->CCI_timer );
  hypre_FinalizeTiming( globals->SS_timer  );
  hypre_FinalizeTiming( globals->SFR_timer );
  hypre_FinalizeTiming( globals->CR_timer );
  hypre_FinalizeTiming( globals->FL_timer  );
  hypre_FinalizeTiming( globals->SLUD_timer  );
  hypre_FinalizeTiming( globals->SLUM_timer );
  hypre_FinalizeTiming( globals->UL_timer  );
  hypre_FinalizeTiming( globals->FNR_timer  );

  hypre_FinalizeTiming( globals->Ll_timer  );
  hypre_FinalizeTiming( globals->Lp_timer );
  hypre_FinalizeTiming( globals->Up_timer );
  hypre_FinalizeTiming( globals->Ul_timer );
#endif

  hypre_TFree( hypre_DistributedMatrixPilutSolverGlobals(solver) );

  hypre_TFree(solver);

  return(0);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_DistributedMatrixPilutSolverInitialize ( 
                  HYPRE_DistributedMatrixPilutSolver solver )
{

   return(0);
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSetMatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetMatrix( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr,
                  HYPRE_DistributedMatrix matrix )
{
  hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverMatrix( solver ) = matrix;
  return hypre_error_flag;
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

HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetNumLocalRow( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr,
                  HYPRE_Int FirstLocalRow )
{
  hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;
   hypre_PilutSolverGlobals *globals = hypre_DistributedMatrixPilutSolverGlobals(solver);

  DataDistTypeRowdist(hypre_DistributedMatrixPilutSolverDataDist( solver ))[mype] = 
     FirstLocalRow;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSetFactorRowSize
 *   Sets the maximum number of entries to be kept in the incomplete factors
 *   This number applies both to the row of L, and also separately to the
 *   row of U.
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetFactorRowSize( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr,
                  HYPRE_Int size )
{
  hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverGmaxnz( solver ) = size;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSetDropTolerance
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetDropTolerance( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr,
                  double tolerance )
{
  hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverTol( solver ) = tolerance;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSetMaxIts
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetMaxIts( 
                  HYPRE_DistributedMatrixPilutSolver in_ptr,
                  HYPRE_Int its )
{
  hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverMaxIts( solver ) = its;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetup( HYPRE_DistributedMatrixPilutSolver in_ptr )
{
   HYPRE_Int m, n, nprocs, start, end, *rowdist, col0, coln, ierr;
   hypre_DistributedMatrixPilutSolver *solver = 
      (hypre_DistributedMatrixPilutSolver *) in_ptr;
   hypre_PilutSolverGlobals *globals = hypre_DistributedMatrixPilutSolverGlobals(solver);


   if(hypre_DistributedMatrixPilutSolverMatrix(solver) == NULL )
   {
       hypre_error_in_arg(1);
      /* hypre_printf("Cannot call setup to solver until matrix has been set\n");*/
      /* return hypre_error_flag; */
   }

   /* Set up the DataDist structure */

   HYPRE_DistributedMatrixGetDims(
      hypre_DistributedMatrixPilutSolverMatrix(solver), &m, &n);

   DataDistTypeNrows( hypre_DistributedMatrixPilutSolverDataDist( solver ) ) = m;

   HYPRE_DistributedMatrixGetLocalRange(
      hypre_DistributedMatrixPilutSolverMatrix(solver), &start, &end, &col0, &coln);

   DataDistTypeLnrows(hypre_DistributedMatrixPilutSolverDataDist( solver )) = 
      end - start + 1;

   /* Set up DataDist entry in distributed_solver */
   /* This requires that each processor know which rows are owned by each proc */
   nprocs = npes;

   rowdist = DataDistTypeRowdist( hypre_DistributedMatrixPilutSolverDataDist( solver ) );

   hypre_MPI_Allgather( &start, 1, HYPRE_MPI_INT, rowdist, 1, HYPRE_MPI_INT, 
      hypre_DistributedMatrixPilutSolverComm(solver) );

   rowdist[ nprocs ] = n;

#ifdef HYPRE_TIMING
{
   HYPRE_Int ilut_timer;

   ilut_timer = hypre_InitializeTiming( "hypre_ILUT factorization" );

   hypre_BeginTiming( ilut_timer );
#endif

   /* Perform approximate factorization */
   ierr = hypre_ILUT( hypre_DistributedMatrixPilutSolverDataDist (solver),
         hypre_DistributedMatrixPilutSolverMatrix (solver),
         hypre_DistributedMatrixPilutSolverFactorMat (solver),
         hypre_DistributedMatrixPilutSolverGmaxnz (solver),
         hypre_DistributedMatrixPilutSolverTol (solver),
         hypre_DistributedMatrixPilutSolverGlobals (solver)
       );

#ifdef HYPRE_TIMING
   hypre_EndTiming( ilut_timer );
   /* hypre_FinalizeTiming( ilut_timer ); */
}
#endif

   if (ierr) 
   {
       hypre_error(HYPRE_ERROR_GENERIC);
       /* return hypre_error_flag; */
   }

#ifdef HYPRE_TIMING
{
   HYPRE_Int Setup_timer;

   Setup_timer = hypre_InitializeTiming( "hypre_SetUpLUFactor: setup for triangular solvers");

   hypre_BeginTiming( Setup_timer );
#endif

   ierr = hypre_SetUpLUFactor( hypre_DistributedMatrixPilutSolverDataDist (solver), 
               hypre_DistributedMatrixPilutSolverFactorMat (solver),
               hypre_DistributedMatrixPilutSolverGmaxnz (solver),
               hypre_DistributedMatrixPilutSolverGlobals (solver) );

#ifdef HYPRE_TIMING
   hypre_EndTiming( Setup_timer );
   /* hypre_FinalizeTiming( Setup_timer ); */
}
#endif

   if (ierr) 
   {
       hypre_error(HYPRE_ERROR_GENERIC);
       /* return hypre_error_flag; */
   }

#ifdef HYPRE_DEBUG
   fflush(stdout);
   hypre_printf("Nlevels: %d\n",
          hypre_DistributedMatrixPilutSolverFactorMat (solver)->nlevels);
#endif

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * HYPRE_DistributedMatrixPilutSolverSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_DistributedMatrixPilutSolverSolve( HYPRE_DistributedMatrixPilutSolver in_ptr,
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
#ifdef HYPRE_TIMING
{
   HYPRE_Int LDUSolve_timer;

   LDUSolve_timer = hypre_InitializeTiming( "hypre_ILUT application" );

   hypre_BeginTiming( LDUSolve_timer );
#endif

   hypre_LDUSolve( hypre_DistributedMatrixPilutSolverDataDist (solver),
         hypre_DistributedMatrixPilutSolverFactorMat (solver),
         x,
         b,
         hypre_DistributedMatrixPilutSolverGlobals (solver)
       );
#ifdef HYPRE_TIMING
   hypre_EndTiming( LDUSolve_timer );
   /* hypre_FinalizeTiming ( LDUSolve_timer ); */
}
#endif


  return hypre_error_flag;
}

