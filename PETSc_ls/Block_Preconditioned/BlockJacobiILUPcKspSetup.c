/* Include headers for problem and solver data structure */
#include "BlockJacobiPcKsp.h"

/* Include PETSc's externally viewable SLES header */
#include "sles.h"

/* Include Petsc implementation details so that the matrix
   structure can be dereferenced and fed into the ILU structure */
#include "mpiaij.h"

/* include solver prototypes */
#include "ilu_protos.h"

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

   BlockJacobiILUKspSetSLES

       OR

   2) The routine

   BlockJacobiILUKspSetPreconditionerMatrix

   must have been called.


   The following non-PETSc routine must have been called prior to this routine:

   BlockJacobiILUPcKspInitialize

   The PETSc routines

   SLESSetFromOptions
   SLESSetup

   should not be called by the user; this routine calls them.

   Author: Andrew J. Cleary

   History:
   12/10/97: Initial version. AJC
*/
   

int BlockJacobiILUPcKspSetup(void *in_ptr, Vec x, Vec b )
     /* Sets up data for ILU from Petsc matrix */
{
   int         nlocal, its, first, lens;
   Scalar      zero = 0.0, one = 1.0, norm;
  
   SLES        *sles, *subsles;
   PC          pc, subpc;
   PCType      pc_type;
   KSP         ksp, subksp;
   Mat         *PreconditionerMatrix, *SystemMatrix, *TempMatrix;
   MatStructure MatStructureFlag; /* Retrieved from user, determines whether
                             we can re-use info in the preconditioner */

   void       *ilu_data;
   BJData     *BJ_data = (BJData *) in_ptr;
   Matrix     *ILU_A;
   int         i, ierr, flg, size, first_row, last_row;

   Mat         local_pmat;
   /* variables for dereferencing Petsc matrix */
   Mat_SeqAIJ *local_diag_block_impl;


   sles = BJDataSles_ptr( BJ_data );
   SystemMatrix = BJDataSystemMatrixPtr( BJ_data );

   PreconditionerMatrix = BJDataPreconditionerMatrixPtr( BJ_data );

   if( !sles )
   {
     if( !SystemMatrix )
     {
       printf(
        "BlockJacobiILUPcKspSetup: you must call either SetSLES or ");
       printf("SetSystemMatrix before Setup\n");
       ierr = -1; CHKERRA( ierr );
     }
     else
     {
       /* User has given us matrix; we must set up SLES */

       sles = (SLES *) talloc( SLES, 1 );

       BJDataSlesOwner(BJ_data) = BJLibrary;

       ierr = SLESCreate(MPI_COMM_WORLD,sles); CHKERRA(ierr);

       if( PreconditionerMatrix )
       {
         ierr = SLESSetOperators(*sles, *SystemMatrix,
                 *PreconditionerMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRA(ierr);
       }
       else
       {
         ierr = SLESSetOperators(*sles, *SystemMatrix,
                 *SystemMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRA(ierr);
       }
       BJDataSles_ptr(BJ_data) = sles;

     }
   }
   else
   {
     /* If user gives us both an SLES *and* one or both matrices, we assume
        that he wants to replace his existing matrices with the new ones. If
        he only gives us one new matrix, we will preserve the other one by
        extracting it out and then putting it back in. AC */

     BJDataSlesOwner(BJ_data) = BJUser;

     ierr = SLESGetKSP(*sles,&ksp); CHKERRA(ierr);

     ierr = SLESGetPC(*sles,&pc); CHKERRA(ierr);
     
     if( SystemMatrix  )
     {
       if( PreconditionerMatrix  )
       {
         ierr = SLESSetOperators(*sles, *SystemMatrix,
                 *PreconditionerMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRA(ierr);
       }
       else
       {
         ierr = PCGetOperators( pc, TempMatrix,
                 PreconditionerMatrix, &MatStructureFlag ); 
              CHKERRA(ierr);
         ierr = SLESSetOperators(*sles, *SystemMatrix,
                 *PreconditionerMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRA(ierr);
       }
     }
     else
     {
       if( PreconditionerMatrix != PETSC_NULL )
       {
         ierr = PCGetOperators( pc, SystemMatrix,
                 TempMatrix, &MatStructureFlag ); 
              CHKERRA(ierr);
         ierr = SLESSetOperators(*sles, *SystemMatrix,
                 *PreconditionerMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRA(ierr);
       }
     }
   }

   /* Get PETSc constructs for further manipulation */

   ierr = SLESGetKSP(*sles,&ksp); CHKERRA(ierr);

   ierr = SLESGetPC(*sles,&pc); CHKERRA(ierr);

   ierr = PCGetOperators( pc, SystemMatrix, PreconditionerMatrix, &MatStructureFlag ); 
     CHKERRA(ierr);

   if( SystemMatrix == PETSC_NULL )
   {
     ierr = -1; CHKERRA( ierr );
   }

   /* Set defaults that can be overridden by command line options */

   /* Set preconditioner to Additive Schwarz, one block per processor. */
   ierr = PCSetType(pc,PCASM); CHKERRA(ierr);

#ifdef LocalSolverILU
   /* Default to preconditioning on the right side */
   ierr = KSPSetPreconditionerSide( ksp, PC_RIGHT );
#endif
#ifdef LocalSolverIC
   /* Default to preconditioning on the left side */
   ierr = KSPSetPreconditionerSide( ksp, PC_LEFT );
#endif

   /* We assume that a user that has a nonzero x wants to use that as
      an initial guess. AC */
   ierr = VecNorm( x, NORM_1, &norm ); CHKERRA( ierr );
   if( norm != 0.0 ) 
     { ierr = KSPSetInitialGuessNonzero( ksp ); CHKERRA( ierr ); }


   /* User can change the above on the command line; import those changes */
   ierr = SLESSetFromOptions( *sles );


   /* From this point on, we override user choices */

   /* Get the linear solver structures for the block solves... */
   ierr = PCGetType( pc, &pc_type, PETSC_NULL ); CHKERRA(ierr);

   /* Note that SLESSetUp() MUST be called before PC(solver)GetSubSLES(). */
   ierr = SLESSetUp(*sles,b,x); CHKERRA(ierr);

   if ( pc_type == PCASM )
   {
     ierr = PCASMGetSubSLES(pc,&nlocal,&first,&subsles); CHKERRA(ierr);
   } else if (pc_type == PCBJACOBI )
   {
     ierr = PCBJacobiGetSubSLES(pc,&nlocal,&first,&subsles); CHKERRA(ierr);
   } else
   {
     ierr = 1;
     PetscPrintf( MPI_COMM_WORLD, 
       "Preconditioner must be Block Jacobi or Additive Schwarz\n");
     CHKERRA(ierr);
     return(PETSC_NULL);
   }

   /* Tell Petsc to use our sequential solvers as the block solver. */
   ierr = SLESGetKSP(subsles[0],&subksp); CHKERRA(ierr);
   ierr = KSPSetType(subksp,KSPPREONLY); CHKERRA(ierr);

   ierr = SLESGetPC(subsles[0],&subpc); CHKERRA(ierr);
   ierr = PCSetType(subpc,PCSHELL); CHKERRA(ierr);

   /* Choose the local block solver...*/

   /* Setup Petsc to use ILU as local block solvers */
   ierr = PCShellSetApply(subpc, ILU_Apply, BJ_data); CHKERRA(ierr);
  
   /* Get the matrix to be used on each processor as preconditioner */

   ierr = PCGetOperators( subpc, PETSC_NULL, &local_pmat, PETSC_NULL );
   local_diag_block_impl = (Mat_SeqAIJ *) local_pmat->data;


   /* Set up ILU_A to have information from Petsc structure */
   ILU_A = NewMatrix( 
		     (local_diag_block_impl)->  a,
		     (local_diag_block_impl)->  i,
		     (local_diag_block_impl)->  j,
		     (local_diag_block_impl)->  m
                    );


   /* Get structure holding sequential solver information */
   ilu_data = BJDataLsData(BJ_data);

   /* If using Schwarz... */
   if ( pc_type == PCASM )
   {
      SetILUMode( ilu_data, 2 );
   }
   else
   {
      /* Else (using Jacobi) */
      SetILUMode( ilu_data, 1 );
   }


   /* Complete setup of ilu_data structure with computation of
         preconditioner, etc. */
   ierr = ilu_setup ( ilu_data, ILU_A );

   if ( ierr != 0 )
   {
     printf("Error returned by ilu_setup = %d\n",ierr);
     CHKERRA(ierr);
   }

   /* Insert ILU information into BJ structure */
   BJDataA(BJ_data) = ILU_A;

   /* Return created BJ structure to calling routine */
   return( 0 );

}

