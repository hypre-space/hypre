
/******************************************************
 *
 *  File:  Hypre_PCG.c
 *
 *********************************************************/

#include "Hypre_PCG_Skel.h" 
#include "Hypre_PCG_Data.h" 

#include "Hypre_StructMatrix_Skel.h"
#include "Hypre_StructVector_Skel.h"
#include "Hypre_StructVector_Data.h"
#include "Hypre_MPI_Com_Skel.h"
#include "Hypre_MPI_Com_Data.h"
#include "Hypre_StructJacobi.h"
#include "Hypre_StructSMG.h"
#include "math.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_PCG_constructor(Hypre_PCG this) {

   this->d_table = (struct Hypre_PCG_private_type *)
      malloc( sizeof( struct Hypre_PCG_private_type ) );

   this->d_table->hssolver = (HYPRE_StructSolver *)
     malloc( sizeof( HYPRE_StructSolver ) );

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_PCG_destructor(Hypre_PCG this) {

   struct Hypre_PCG_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   HYPRE_StructPCGDestroy( *S );
   free(this->d_table);

} /* end destructor */

/* ********************************************************
 * impl_Hypre_PCGApply
 **********************************************************/
void  impl_Hypre_PCG_Apply
(Hypre_PCG this, Hypre_StructVector b, Hypre_StructVector* x) {

   struct Hypre_PCG_private_type *HPCGp = this->d_table;
   HYPRE_StructSolver *S = HPCGp->hssolver;

   Hypre_StructMatrix A = this->d_table->hsmatrix;
   struct Hypre_StructMatrix_private_type *SMp = A->d_table;
   HYPRE_StructMatrix *MA = SMp->hsmat;

   struct Hypre_StructVector_private_type *SVbp = b->d_table;
   HYPRE_StructVector *Vb = SVbp->hsvec;

   struct Hypre_StructVector_private_type *SVxp = (*x)->d_table;
   HYPRE_StructVector *Vx = SVxp->hsvec;

   HYPRE_StructPCGSolve( *S, *MA, *Vb, *Vx );

} /* end impl_Hypre_PCGApply */

/* ********************************************************
 * impl_Hypre_PCGGetSystemOperator
 **********************************************************/
Hypre_StructMatrix  impl_Hypre_PCG_GetSystemOperator(Hypre_PCG this) {

   return this->d_table->hsmatrix;

} /* end impl_Hypre_PCGGetSystemOperator */

/* ********************************************************
 * impl_Hypre_PCGGetResidual
 **********************************************************/
Hypre_StructVector  impl_Hypre_PCG_GetResidual(Hypre_PCG this) {
  /* ******* MORE TO DO *****
    There is no residual function at the HYPRE level; I haven't looked lower yet.
    There is a HYPRE_StructPCGGetFinalRelativeResidualNorm which provides a
    double*.

    For now, all we do is make a dummy object and return it.  It can't even be
    of the right size because the grid information is quite buried and it's not
    worthwhile to store a StructuredGrid object just to support a function that
    doesn't work.
  */

   Hypre_StructVector vec = Hypre_StructVector_new();

   printf( "called Hypre_PCG_GetResidual, which doesn't work!\n");

   return vec;

} /* end impl_Hypre_PCGGetResidual */

/* ********************************************************
 * impl_Hypre_PCGGetConvergenceInfo
 **********************************************************/
void  impl_Hypre_PCG_GetConvergenceInfo(Hypre_PCG this, char* name, double* value) {
   int ivalue;

   struct Hypre_PCG_private_type *HSp = this->d_table;
   HYPRE_StructSolver *S = HSp->hssolver;

   if ( !strcmp(name,"number of iterations") ) {
      ivalue = -1;
      HYPRE_StructPCGGetNumIterations( *S, &ivalue );
      *value = ivalue;
      return;
   }
   else if ( !strcmp(name,"residual norm") ) {
      HYPRE_StructPCGGetFinalRelativeResidualNorm( *S, value );
      return;
   }
   else {
      printf(
         "Don't understand keyword %s to Hypre_PCG_GetConvergenceInfo\n",
         name );
      *value = 0;
      return;
   }


} /* end impl_Hypre_PCGGetConvergenceInfo */

/* ********************************************************
 * impl_Hypre_PCGGetPreconditioner
 **********************************************************/
Hypre_Solver  impl_Hypre_PCG_GetPreconditioner(Hypre_PCG this) {
/* ******* MORE TO DO ******
 this doesn't exist at HYPRE level, I haven't looked lower yet;
 obviously the information exists.
*/
} /* end impl_Hypre_PCGGetPreconditioner */

/* ********************************************************
 * impl_Hypre_PCGSetSystemOperator
 **********************************************************/
void  impl_Hypre_PCG_SetSystemOperator(Hypre_PCG this, Hypre_StructMatrix op) {

/* Sets the matrix.  Setup should (probably) be called before anything is
   done with it. */

   this->d_table->hsmatrix = op ;

} /* end impl_Hypre_PCGSetSystemOperator */

/* ********************************************************
 * impl_Hypre_PCGGetDoubleParameter
 **********************************************************/
double  impl_Hypre_PCG_GetDoubleParameter(Hypre_PCG this, char* name) {
/* The parameters exist, but there are no HYPRE-level Get functions.
   I'll implement pieces of this when needed. */

   printf( "Hypre_PCG_GetDoubleParameter does not recognize name ~s\n", name );
   return 0;

} /* end impl_Hypre_PCGGetDoubleParameter */

/* ********************************************************
 * impl_Hypre_PCGGetIntParameter
 **********************************************************/
int  impl_Hypre_PCG_GetIntParameter(Hypre_PCG this, char* name) {
/* The parameters exist, but there are no HYPRE-level Get functions.
   I'll implement pieces of this when needed. */

   printf( "Hypre_PCG_GetIntParameter does not recognize name ~s\n", name );
   return 0;

} /* end impl_Hypre_PCGGetIntParameter */

/* ********************************************************
 * impl_Hypre_PCGSetDoubleParameter
 **********************************************************/
void  impl_Hypre_PCG_SetDoubleParameter(Hypre_PCG this, char* name, double value) {
/* JFP: This function just dispatches to the parameter's set function. */

   struct Hypre_PCG_private_type *HSp = this->d_table;
   HYPRE_StructSolver *S = HSp->hssolver;

   if ( !strcmp(name,"tol") ) {
      HYPRE_StructPCGSetTol( *S, value );
      return;
   }
   else {
      printf( "Hypre_PCG_SetDoubleParameter does not recognize name ~s\n", name );
   } ;
} /* end impl_Hypre_PCGSetDoubleParameter */

/* ********************************************************
 * impl_Hypre_PCGSetIntParameter
 **********************************************************/
void  impl_Hypre_PCG_SetIntParameter(Hypre_PCG this, char* name, int value) {
/* JFP: This function just dispatches to the parameter's set function. */

   struct Hypre_PCG_private_type *HSp = this->d_table;
   HYPRE_StructSolver *S = HSp->hssolver;

   if ( !strcmp(name,"max_iter" )) {
      HYPRE_StructPCGSetMaxIter( *S, value );
      return;
   }
   else if ( !strcmp(name, "2-norm" ) ) {
      HYPRE_StructPCGSetTwoNorm( *S, value );
      return;
   }
   else if ( !strcmp(name,"relative change test") ) {
      HYPRE_StructPCGSetRelChange( *S, value );
      return;
   }
   else if (  !strcmp(name,"log") ) {
      HYPRE_StructPCGSetLogging( *S, value );
      return;
   }
   else {
      printf( "Hypre_PCG_SetIntParameter does not recognize name ~s\n", name );
   } ;
} /* end impl_Hypre_PCGSetIntParameter */

/* ********************************************************
 * impl_Hypre_PCGNew
 **********************************************************/
void  impl_Hypre_PCG_New(Hypre_PCG this, Hypre_MPI_Com comm) {

   struct Hypre_PCG_private_type *HSPCGp = this->d_table;
   HYPRE_StructSolver *S = HSPCGp->hssolver;

   struct Hypre_MPI_Com_private_type * HMCp = comm->d_table;
   MPI_Comm *C = HMCp->hcom;

/* the StructSolver this inherits from keeps its own pointer to the
   underlying HYPRE object.  Make sure they are the same.
*/
   Hypre_StructSolver HSS = Hypre_PCG_castTo( this, "Hypre_StructSolver" );
   struct Hypre_StructSolver_private_type *HSSp = HSS->d_table;
   HSSp->hssolver = S;

   HYPRE_StructPCGCreate( *C, S );
} /* end impl_Hypre_PCGNew */

/* ********************************************************
 * impl_Hypre_PCGConstructor
 **********************************************************/
Hypre_PCG  impl_Hypre_PCG_Constructor(Hypre_MPI_Com comm) {
   /* declared static; just combines the new and New functions */
   Hypre_PCG SPCG = Hypre_PCG_new();
   Hypre_PCG_New( SPCG, comm );
   return SPCG;
} /* end impl_Hypre_PCGConstructor */

/* ********************************************************
 * impl_Hypre_PCGSetup
 **********************************************************/
void  impl_Hypre_PCG_Setup
(Hypre_PCG this, Hypre_StructMatrix A, Hypre_StructVector b, Hypre_StructVector x) {
   struct Hypre_PCG_private_type *HSp = this->d_table;
   HYPRE_StructSolver *S = HSp->hssolver;

   struct Hypre_StructMatrix_private_type *SMp = A->d_table;
   HYPRE_StructMatrix *MA = SMp->hsmat;

   struct Hypre_StructVector_private_type *SVbp = b->d_table;
   HYPRE_StructVector *Vb = SVbp->hsvec;

   struct Hypre_StructVector_private_type *SVxp = x->d_table;
   HYPRE_StructVector *Vx = SVxp->hsvec;

   this->d_table->hsmatrix = A;

   HYPRE_StructPCGSetup( *S, *MA, *Vb, *Vx );

} /* end impl_Hypre_PCGSetup */

/* ********************************************************
 * impl_Hypre_PCGGetConstructedObject
 *       insert the library code below
 **********************************************************/
Hypre_Solver impl_Hypre_PCG_GetConstructedObject(Hypre_PCG this) {

   return (Hypre_Solver) this;

} /* end impl_Hypre_PCGGetConstructedObject */

/* ********************************************************
 * impl_Hypre_PCGSetPreconditioner
 *       insert the library code below
 **********************************************************/
void  impl_Hypre_PCG_SetPreconditioner
(Hypre_PCG this, Hypre_StructSolver precond) {

   struct Hypre_PCG_private_type *HSPCGp = this->d_table;
   HYPRE_StructSolver *S = HSPCGp->hssolver;

   struct Hypre_StructSolver_private_type *HSSPp = precond->d_table;
   HYPRE_StructSolver * precond_solver = HSSPp->hssolver;

/* attempt to cast to the supported preconditioners ... */
   Hypre_StructJacobi precond_SJ = (Hypre_StructJacobi)
      Hypre_StructSolver_castTo( precond, "Hypre_StructJacobi" );
   Hypre_StructSMG precond_SMG = (Hypre_StructSMG)
      Hypre_StructSolver_castTo( precond, "Hypre_StructSMG" );
/* call the SetPrecond function for whichever preconditioner we could cast to ... */
   if ( precond_SJ != 0 ) {
      HYPRE_StructPCGSetPrecond( *S,
                                 HYPRE_StructJacobiSolve,
                                 HYPRE_StructJacobiSetup,
                                 *precond_solver );
   }
   else if ( precond_SMG != 0 ) {
      HYPRE_StructPCGSetPrecond( *S,
                                 HYPRE_StructSMGSolve,
                                 HYPRE_StructSMGSetup,
                                 *precond_solver );
   }
   else {
      printf( "Hypre_PCG_SetPreconditioner does not recognize preconditioner!\n" );
   }
   ;

} /* end impl_Hypre_PCGSetPreconditioner */


/* Print Logging; not in the SIDL file */

void
HYPRE_StructPCG_PrintLogging( HYPRE_StructSolver  solver )
{
   hypre_PCGPrintLogging( (void *) solver, 0 ) ;
}

void  Hypre_PCG_PrintLogging( Hypre_PCG this ) {

   struct Hypre_PCG_private_type *HSPCGp = this->d_table;
   HYPRE_StructSolver *S = HSPCGp->hssolver;
   HYPRE_StructPCG_PrintLogging( *S );
}
;
