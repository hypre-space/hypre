
/******************************************************
 *
 *  File:  Hypre_StructSMG.c
 *
 *********************************************************/

#include "Hypre_StructSMG_Skel.h" 
#include "Hypre_StructSMG_Data.h" 

#include "Hypre_StructMatrix_Skel.h"
#include "Hypre_StructMatrix_Data.h"
#include "Hypre_StructVector_Skel.h"
#include "Hypre_StructVector_Data.h"
#include "Hypre_MPI_Com_Skel.h"
#include "Hypre_MPI_Com_Data.h"
#include "math.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructSMG_constructor(Hypre_StructSMG this) {
   this->d_table = (struct Hypre_StructSMG_private_type *)
      malloc( sizeof( struct Hypre_StructSMG_private_type ) );

   this->d_table->hssolver = (HYPRE_StructSolver *)
     malloc( sizeof( HYPRE_StructSolver ) );
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructSMG_destructor(Hypre_StructSMG this) {
   struct Hypre_StructSMG_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   HYPRE_StructSMGDestroy( *S );
   free(this->d_table);

} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructSMGApply
 **********************************************************/
void  impl_Hypre_StructSMG_Apply
(Hypre_StructSMG this, Hypre_StructVector b, Hypre_StructVector* x) {
   struct Hypre_StructSMG_private_type *HSMGp = this->d_table;
   HYPRE_StructSolver *S = HSMGp->hssolver;

   Hypre_StructMatrix A = this->d_table->hsmatrix;
   struct Hypre_StructMatrix_private_type *SMp = A->d_table;
   HYPRE_StructMatrix *MA = SMp->hsmat;

   struct Hypre_StructVector_private_type *SVbp = b->d_table;
   HYPRE_StructVector *Vb = SVbp->hsvec;

   struct Hypre_StructVector_private_type *SVxp = (*x)->d_table;
   HYPRE_StructVector *Vx = SVxp->hsvec;

   HYPRE_StructSMGSolve( *S, *MA, *Vb, *Vx );
} /* end impl_Hypre_StructSMGApply */

/* ********************************************************
 * impl_Hypre_StructSMGGetSystemOperator
 **********************************************************/
Hypre_StructMatrix  impl_Hypre_StructSMG_GetSystemOperator
(Hypre_StructSMG this) {

   return this->d_table->hsmatrix;

} /* end impl_Hypre_StructSMGGetSystemOperator */

/* ********************************************************
 * impl_Hypre_StructSMGGetResidual
 **********************************************************/
Hypre_StructVector  impl_Hypre_StructSMG_GetResidual(Hypre_StructSMG this) {
  
  /*
    The present HYPRE_struct_smg.c code in Hypre doesn't provide a residual.
    I haven't bothered to look at other Hypre SMG files. (jfp)
    
    For now, all we do is make a dummy object and return it.  It can't even be
    of the right size because the grid information is quite buried and it's not
    worthwhile to store a StructuredGrid object just to support a function that
    doesn't work.
  */

   Hypre_StructVector vec = Hypre_StructVector_new();

   printf( "called Hypre_StructSMG_GetResidual, which doesn't work!\n");

   return vec;
} /* end impl_Hypre_StructSMGGetResidual */

/* ********************************************************
 * impl_Hypre_StructSMGGetConvergenceInfo
 **********************************************************/
void  impl_Hypre_StructSMG_GetConvergenceInfo
(Hypre_StructSMG this, char* name, double* value) {
   int ivalue;

   struct Hypre_StructSMG_private_type *HSMGp = this->d_table;
   HYPRE_StructSolver *S = HSMGp->hssolver;

   if ( !strcmp(name,"num iterations") ) {
      HYPRE_StructSMGGetNumIterations( *S, &ivalue );
      *value = ivalue;
      return;
   }
   if ( !strcmp(name,"final relative residual norm") ) {
      HYPRE_StructSMGGetFinalRelativeResidualNorm( *S, value );
      return;
   }

   printf( "Hypre_StructJacobi_GetParameter does not recognize name ~s\n", name );

   return;
} /* end impl_Hypre_StructSMGGetConvergenceInfo */

/* ********************************************************
 * impl_Hypre_StructSMGSetSystemOperator
 **********************************************************/
void  impl_Hypre_StructSMG_SetSystemOperator
(Hypre_StructSMG this, Hypre_StructMatrix op) {

/* Sets the matrix.  Setup should (probably) be called before anything is
   done with it. */

   this->d_table->hsmatrix = op ;

} /* end impl_Hypre_StructSMGSetSystemOperator */

/* ********************************************************
 * impl_Hypre_StructSMGGetParameter
 **********************************************************/
double  impl_Hypre_StructSMG_GetParameter(Hypre_StructSMG this, char* name) {
   double value;
   int ivalue;
   printf( "Hypre_StructJacobi_GetParameter does not recognize name ~s\n", name );
   return 0;
} /* end impl_Hypre_StructSMGGetParameter */

/* ********************************************************
 * impl_Hypre_StructSMGSetParameter
 **********************************************************/
void  impl_Hypre_StructSMG_SetParameter
(Hypre_StructSMG this, char* name, double value) {

/* JFP: This function just dispatches to the parameter's set function. */

   struct Hypre_StructSMG_private_type *HSMGp = this->d_table;
   HYPRE_StructSolver *S = HSMGp->hssolver;

   if ( !strcmp(name,"tol") ) {
      HYPRE_StructSMGSetTol( *S, value );
      return;
   };
   if ( !strcmp(name,"max_iter" )) {
      HYPRE_StructSMGSetMaxIter( *S, floor(value*1.001) );
      /* ... floor(value*1.001) is a simple adequate way to undo an
         int->double conversion */
      return;
   };
   if ( !strcmp(name,"max iter" )) {
      HYPRE_StructSMGSetMaxIter( *S, floor(value*1.001) );
      /* ... floor(value*1.001) is a simple adequate way to undo an
         int->double conversion */
      return;
   };
   if ( !strcmp(name,"zero guess") ) {
      HYPRE_StructSMGSetZeroGuess( *S );
      return;
   };
   if (  !strcmp(name,"nonzero guess") ) {
      HYPRE_StructSMGSetNonZeroGuess( *S );
      return;
   };
   if ( !strcmp(name,"memory use") ) {
      HYPRE_StructSMGSetMemoryUse( *S, floor(value*1.001) );
   };
   if ( !strcmp(name,"rel change") ) {
      HYPRE_StructSMGSetRelChange( *S, floor(value*1.001) );
   };
   if ( !strcmp(name,"num prerelax") ) {
      HYPRE_StructSMGSetNumPreRelax( *S, floor(value*1.001) );
   };
   if ( !strcmp(name,"num postrelax") ) {
      HYPRE_StructSMGSetNumPostRelax( *S, floor(value*1.001) );
   };
   if ( !strcmp(name,"logging") ) {
      HYPRE_StructSMGSetLogging( *S, floor(value*1.001) );
   };

} /* end impl_Hypre_StructSMGSetParameter */

/* ********************************************************
 * impl_Hypre_StructSMGNew
 **********************************************************/
void  impl_Hypre_StructSMG_New(Hypre_StructSMG this, Hypre_MPI_Com comm) {

   struct Hypre_StructSMG_private_type *HSMGp = this->d_table;
   HYPRE_StructSolver *S = HSMGp->hssolver;

   struct Hypre_MPI_Com_private_type * HMCp = comm->d_table;
   MPI_Comm *C = HMCp->hcom;

/* the StructSolver this inherits from keeps its own pointer to the
   underlying HYPRE object.  Make sure they are the same.
*/
   Hypre_StructSolver HSS = Hypre_StructSMG_castTo
      ( this, "Hypre_StructSolver" );
   struct Hypre_StructSolver_private_type *HSSp = HSS->d_table;
   HSSp->hssolver = S;

   HYPRE_StructSMGCreate( *C, S );
} /* end impl_Hypre_StructSMGNew */

/* ********************************************************
 * impl_Hypre_StructSMGSetup
 **********************************************************/
void  impl_Hypre_StructSMG_Setup
(Hypre_StructSMG this, Hypre_StructMatrix A, Hypre_StructVector b,
 Hypre_StructVector x) {

   struct Hypre_StructSMG_private_type *HSMGp = this->d_table;
   HYPRE_StructSolver *S = HSMGp->hssolver;

   struct Hypre_StructMatrix_private_type *SMp = A->d_table;
   HYPRE_StructMatrix *MA = SMp->hsmat;

   struct Hypre_StructVector_private_type *SVbp = b->d_table;
   HYPRE_StructVector *Vb = SVbp->hsvec;

   struct Hypre_StructVector_private_type *SVxp = x->d_table;
   HYPRE_StructVector *Vx = SVxp->hsvec;

   this->d_table->hsmatrix = A;

   HYPRE_StructSMGSetup( *S, *MA, *Vb, *Vx );
} /* end impl_Hypre_StructSMGSetup */

/* ********************************************************
 * impl_Hypre_StructSMGConstructor
 **********************************************************/
Hypre_StructSMG  impl_Hypre_StructSMG_Constructor(Hypre_MPI_Com comm) {
   /* declared static; just combines the new and New functions */
   Hypre_StructSMG SMG = Hypre_StructSMG_new();
   Hypre_StructSMG_New( SMG, comm );
   return SMG;
} /* end impl_Hypre_StructSMGConstructor */

/* ********************************************************
 * impl_Hypre_StructSMGGetConstructedObject
 **********************************************************/
Hypre_Solver  impl_Hypre_StructSMG_GetConstructedObject(Hypre_StructSMG this) {

   return (Hypre_Solver) this;

} /* end impl_Hypre_StructSMGGetConstructedObject */

