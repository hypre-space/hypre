/* Include headers for problem and solver data structure */
#include "BlockJacobiPcKsp.h"

/* Include Petsc implementation details so that the matrix
   structure can be dereferenced and fed into the ILU structure */
#include "mpiaij.h"

/* include solver prototypes */
#include "ilu_protos.h"


int BlockJacobiILUPcKspSetup(void *in_ptr, Mat A, Vec x, Vec b )
     /* Sets up data for ILU from Petsc matrix */
{
   int         nlocal, its, first, lens;
   Scalar      zero = 0.0, one = 1.0;
  
   SLES       *sles, *subsles;
   PC          pc, subpc;
   PCType      pc_type;
   KSP         ksp, subksp;

   void       *ilu_data;
   BJData     *BJ_data;
   Matrix     *ILU_A;
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

   /* If we want to use ILU as the local block solver...*/

     /* Setup Petsc to use ILU as local block solvers */
     ierr = PCShellSetApply(subpc, ILU_Apply, BJ_data); CHKERRA(ierr);
  
   /* Endif */


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


   /* Initial setup of ilu_data structure */
   ilu_data = ilu_initialize( (void *) NULL );

   /* Include any other ILU settings here */
   /* This is where scaling and reordering should be decided */
   GetILUDataIpar(ilu_data)[0] = 1; /* Reordering? 1=yes*/
   GetILUDataIpar(ilu_data)[1] = 3; /* Scaling? 3 is both */
   GetILUDataIpar(ilu_data)[2] = 0; /* Output message device; default is iout=0 */
   GetILUDataIpar(ilu_data)[5] = 90;/* lfil_ilut: # of fills per row. No Default */
#ifdef LocalSolverILU
   /*   GetILUDataIpar(ilu_data)[6] = 30;*//* Dimension of GMRES subspace. No Default */
   GetILUDataIpar(ilu_data)[6] = 4;/* Dimension of GMRES subspace. No Default */
   GetILUDataIpar(ilu_data)[7] = 3; /* Maxits for GMRES. Default (100)*/
#endif
#ifdef LocalSolverIC
   GetILUDataIpar(ilu_data)[6] = 0;/* Unused */
   GetILUDataIpar(ilu_data)[7] = 4; /* Maxits for CG. Default (100)*/
#endif

   /*  GetILUDataRpar(ilu_data)[0] = 0.0001;*/ /* Drop tolerance; default 0.0001 */
   GetILUDataRpar(ilu_data)[0] = 0.000001; /* Drop tolerance; default 0.0001 */
   GetILUDataRpar(ilu_data)[1] = 0.000000000001; /* Convergence criterion; default 0.00001 */


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
     exit(ierr);
   }


   /* Insert ILU information into BJ structure */
   BJDataA(BJ_data) = ILU_A;
   BJDataLsData(BJ_data) = ilu_data;


   /* Insert other information into BJ structure (if any) */


   /* Return created BJ structure to calling routine */
   return( 0 );

}

