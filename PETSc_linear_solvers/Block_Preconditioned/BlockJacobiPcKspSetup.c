/* Include headers for problem and solver data structure */
#include "BlockJacobiPcKsp.h"

/* Include Petsc implementation details so that the matrix
   structure can be dereferenced and fed into the INCFACT structure */
#include "mpiaij.h"

/* include solver prototypes */
#include "incfact_protos.h"


int BlockJacobiINCFACTPcKspSetup(void *in_ptr, Mat A, Vec x, Vec b )
     /* Sets up data for INCFACT from Petsc matrix */
{
   int         nlocal, its, first, lens;
   Scalar      zero = 0.0, one = 1.0;
  
   SLES       *sles, *subsles;
   PC          pc, subpc;
   PCType      pc_type;
   KSP         ksp, subksp;

   void       *incfact_data;
   BJData     *BJ_data;
   Matrix     *INCFACT_A;
   int         i, ierr, flg, size, first_row, last_row;

   Mat         local_pmat;
   /* variables for dereferencing Petsc matrix */
   Mat_SeqAIJ *local_diag_block_impl;



   BJ_data = (BJData *) in_ptr;

   sles = BJDataSles_ptr(BJ_data);

   ierr = SLESSetOperators(*sles,A,A,DIFFERENT_NONZERO_PATTERN);
         CHKERRA(ierr);   


   /* Note that SLESSetUp() MUST be called before PC(solver)GetSubSLES(). */
   ierr = SLESSetUp(*sles,b,x); CHKERRA(ierr);

   /* Get the linear solver structures for the block solves... */
   ierr = SLESGetPC(*sles,&pc); CHKERRA(ierr);
   ierr = PCGetType( pc, &pc_type, PETSC_NULL ); CHKERRA(ierr);

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

   /* If we want to use INCFACT as the local block solver...*/

     /* Setup Petsc to use INCFACT as local block solvers */
     ierr = PCShellSetApply(subpc, INCFACT_Apply, BJ_data); CHKERRA(ierr);
  
   /* Endif */


   /* Get the matrix to be used on each processor as preconditioner */

   ierr = PCGetOperators( subpc, PETSC_NULL, &local_pmat, PETSC_NULL );
   local_diag_block_impl = (Mat_SeqAIJ *) local_pmat->data;


   /* Set up INCFACT_A to have information from Petsc structure */
   INCFACT_A = NewMatrix( 
		     (local_diag_block_impl)->  a,
		     (local_diag_block_impl)->  i,
		     (local_diag_block_impl)->  j,
		     (local_diag_block_impl)->  m
                    );


   /* Initial setup of incfact_data structure */
   incfact_data = incfact_initialize( (void *) NULL );

   /* Include any other INCFACT settings here */
   /* This is where scaling and reordering should be decided */
   GetINCFACTDataIpar(incfact_data)[0] = 1; /* Reordering? 1=yes*/
   GetINCFACTDataIpar(incfact_data)[1] = 3; /* Scaling? 3 is both */
   GetINCFACTDataIpar(incfact_data)[2] = 0; /* Output message device; default is iout=0 */
   GetINCFACTDataIpar(incfact_data)[5] = 90;/* lfil_incfactt: # of fills per row. No Default */
#ifdef LocalSolverILU
   /*   GetINCFACTDataIpar(incfact_data)[6] = 30;*//* Dimension of GMRES subspace. No Default */
   GetINCFACTDataIpar(incfact_data)[6] = 4;/* Dimension of GMRES subspace. No Default */
   GetINCFACTDataIpar(incfact_data)[7] = 3; /* Maxits for GMRES. Default (100)*/
#endif
#ifdef LocalSolverIC
   GetINCFACTDataIpar(incfact_data)[6] = 0;/* Unused */
   GetINCFACTDataIpar(incfact_data)[7] = 4; /* Maxits for CG. Default (100)*/
#endif

   /*  GetINCFACTDataRpar(incfact_data)[0] = 0.0001;*/ /* Drop tolerance; default 0.0001 */
   GetINCFACTDataRpar(incfact_data)[0] = 0.000001; /* Drop tolerance; default 0.0001 */
   GetINCFACTDataRpar(incfact_data)[1] = 0.000000000001; /* Convergence criterion; default 0.00001 */


   /* If using Schwarz... */
   if ( pc_type == PCASM )
   {
      SetINCFACTMode( incfact_data, 2 );
   }
   else
   {
      /* Else (using Jacobi) */
      SetINCFACTMode( incfact_data, 1 );
   }


   /* Complete setup of incfact_data structure with computation of
         preconditioner, etc. */
   ierr = incfact_setup ( incfact_data, INCFACT_A );

   if ( ierr != 0 )
   {
     printf("Error returned by incfact_setup = %d\n",ierr);
     exit(ierr);
   }


   /* Insert INCFACT information into BJ structure */
   BJDataA(BJ_data) = INCFACT_A;
   BJDataLsData(BJ_data) = incfact_data;


   /* Insert other information into BJ structure (if any) */


   /* Return created BJ structure to calling routine */
   return( 0 );

}

