/* Include headers for problem and solver data structure */
#include "./PETScSolverParILUT.h"

/* 
   This routine replaces SLESSetup for a PETSc user. 

   The user has a choice: one of the following must have been done, so that
   the solver can access the matrix that defines the preconditioner:

   1) The following PETSc
   routines (and any of their precursors such as PETScInitialize)
   must have been called prior to this routine:

   SLESCreate
   SLESSetOperators

   and the resultant SLES context must have been set through a call to 

   BlockJacobiINCFACTKspSetSLES

       OR

   2) The routine

   BlockJacobiINCFACTKspSetPreconditionerMatrix

   must have been called.


   The following non-PETSc routine must have been called prior to this routine:

   BlockJacobiINCFACTPcKspInitialize

   The PETSc routines

   SLESSetFromOptions
   SLESSetup

   should not be called by the user; this routine calls them.

   Author: Andrew J. Cleary

   History:
   12/10/97: Initial version. AJC
*/
   

/*--------------------------------------------------------------------------
 * HYPRE_PETScSolverParILUTSetup
 *--------------------------------------------------------------------------*/

int HYPRE_PETScSolverParILUTSetup( HYPRE_PETScSolverParILUT in_ptr,
                            Vec x, Vec b )
{
   hypre_PETScSolverParILUT *solver = 
      (hypre_PETScSolverParILUT *) in_ptr;

   int         nlocal, its, first, lens;
   Scalar      zero = 0.0, one = 1.0, norm;
   int        *rowdist, count, nprocs, myproc, start, end, n;

   MPI_Comm comm = hypre_PETScSolverParILUTComm(solver);
  
   SLES        sles;
   PC          pc;
   PCType      pc_type;
   KSP         ksp;
   Mat         PreconditionerMatrix, SystemMatrix, TempMatrix;
   HYPRE_PETScMatPilutSolver PETScMatPilutSolver;
   MatStructure MatStructureFlag; /* Retrieved from user, determines whether
                             we can re-use info in the preconditioner */
   MatType     MatTypeName; /* used to check that matrix is correct type */

   int         i, ierr=0;


   sles = hypre_PETScSolverParILUTSles( solver );
   SystemMatrix = hypre_PETScSolverParILUTSystemMatrix( solver );

   PreconditionerMatrix = hypre_PETScSolverParILUTPreconditionerMatrix( solver );

   if( !sles )
   {
     if( !SystemMatrix )
     {
       printf(
        "HYPRE_PETScSolverParILUTSetup: you must call either SetSLES or ");
       printf("SetSystemMatrix before Setup\n");
       ierr = -1; CHKERRA( ierr );
     }
     else
     {
       /* User has given us matrix; we must set up SLES */

       hypre_PETScSolverParILUTSlesOwner(solver) = ParILUTLibrary;

       ierr = SLESCreate(comm,&sles); CHKERRA(ierr);

       if( PreconditionerMatrix )
       {
         ierr = SLESSetOperators(sles, SystemMatrix,
                 PreconditionerMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRA(ierr);
       }
       else
       {
         ierr = SLESSetOperators(sles, SystemMatrix,
                 SystemMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRA(ierr);
       }
       hypre_PETScSolverParILUTSles(solver) = sles;

     }
   }
   else
   {
     /* If user gives us both an SLES *and* one or both matrices, we assume
        that he wants to replace his existing matrices with the new ones. If
        he only gives us one new matrix, we will preserve the other one by
        extracting it out and then putting it back in. AC */

     hypre_PETScSolverParILUTSlesOwner(solver) = ParILUTUser;

     ierr = SLESGetKSP(sles,&ksp); CHKERRA(ierr);

     ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
     
     if( SystemMatrix  )
     {
       if( PreconditionerMatrix  )
       {
         ierr = SLESSetOperators(sles, SystemMatrix,
                 PreconditionerMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRA(ierr);
       }
       else
       {
         ierr = PCGetOperators( pc, &TempMatrix,
                 &PreconditionerMatrix, &MatStructureFlag ); 
              CHKERRA(ierr);
         ierr = SLESSetOperators(sles, SystemMatrix,
                 PreconditionerMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRA(ierr);
       }
     }
     else
     {
       if( PreconditionerMatrix != PETSC_NULL )
       {
         ierr = PCGetOperators( pc, &SystemMatrix,
                 &TempMatrix, &MatStructureFlag ); 
              CHKERRA(ierr);
         ierr = SLESSetOperators(sles, SystemMatrix,
                 PreconditionerMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRA(ierr);
       }
     }
   }

   /* Get PETSc constructs for further manipulation */

   ierr = SLESGetKSP(sles,&ksp); CHKERRA(ierr);

   ierr = KSPSetMonitor(ksp,KSPDefaultSMonitor,(void *)0);CHKERRQ(ierr);

   ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);

   ierr = PCGetOperators( pc, &SystemMatrix, &PreconditionerMatrix, &MatStructureFlag ); 
     CHKERRA(ierr);

   if( SystemMatrix == PETSC_NULL )
   {
     ierr = -1; CHKERRA( ierr );
   }

   /* Set defaults that can be overridden by command line options */

   /* Set preconditioner to a shell routine */
   ierr = PCSetType(pc,PCSHELL); CHKERRA(ierr);

   /* We assume that a user that has a nonzero x wants to use that as
      an initial guess. AC */
   ierr = VecNorm( x, NORM_1, &norm ); CHKERRA( ierr );
   if( norm != 0.0 ) 
     { ierr = KSPSetInitialGuessNonzero( ksp ); CHKERRA( ierr ); }


   /* User can change the above on the command line; import those changes */
   ierr = SLESSetFromOptions( sles );


   /* From this point on, we override user choices */

   /* Get the linear solver structures for the preconditioner solves... */
   ierr = PCGetType( pc, &pc_type ); CHKERRA(ierr);

   if ( !strcmp( pc_type, PCSHELL ) )
   {

     PETScMatPilutSolver = 
        hypre_PETScSolverParILUTPETScMatPilutSolver( solver );

     /* Setup Petsc to use PARILUT as preconditioner solver */
     ierr = PCShellSetApply(pc, HYPRE_PETScMatPilutSolverApply, PETScMatPilutSolver ); 
     CHKERRA(ierr);
  
     /* Call the PETSc Setup function */
     ierr = SLESSetUp( sles,b,x); CHKERRA(ierr);

     ierr = HYPRE_PETScMatPilutSolverSetMatrix( PETScMatPilutSolver,
                PreconditionerMatrix ); if(ierr) return(ierr);

     /* Since we are using the parallel incomplete factorization routine as
        a preconditioner, we only want to do one iteration per "solve" */
     ierr = HYPRE_PETScMatPilutSolverSetMaxIts( PETScMatPilutSolver,
                1 ); if(ierr) return(ierr);

     /* Complete setup of preconditioner routine, etc. */
     ierr = HYPRE_PETScMatPilutSolverSetup ( PETScMatPilutSolver, x, b ); 
     
     if ( ierr != 0 )
     {
       printf("Error returned by HYPRE_PETScMatPilutSolverSetup = %d\n",ierr);

       return( ierr );
     }
   }
   else
   {
     /* Call the PETSc Setup function */
     ierr = SLESSetUp(sles,b,x); CHKERRA(ierr);
   }

   return( 0 );

}

