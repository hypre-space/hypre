
/******************************************************
 *
 *  File:  Hypre_StructJacobi.c
 *
 *********************************************************/

#include "Hypre_StructJacobi_Skel.h" 
#include "Hypre_StructJacobi_Data.h" 

 /* gkk: added ... */
#include "Hypre_StructMatrix_Skel.h"
#include "Hypre_StructMatrix_Data.h"
#include "Hypre_StructVector_Skel.h"
#include "Hypre_StructVector_Data.h"
#include "Hypre_MPI_Com_Skel.h"
#include "Hypre_MPI_Com_Data.h"
#include "math.h"

/* JFP: In Hypre, there is no struct with a name like HYPRE_StructJacobi,
   etc.  but hand-name-mangling of functions indicates that such a "class"
   exists in somebody's mind.  If the data be in a HYPRE_StructSolver
   (or what it points to), then HYPRE_StructJacobi could be seen as
   an interpretation of the object that has the data.
   Actually, there is a struct hypre_PointRelaxData, and the HYPRE_StructSolver
   data points to that (through void * casts, dereferences, etc.) when its
   correct interpretation is as a Jacobi solver.
   */

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructJacobi_constructor(Hypre_StructJacobi this) {
  
   this->d_table = (struct Hypre_StructJacobi_private_type *)
      malloc( sizeof( struct Hypre_StructJacobi_private_type ) );

   this->d_table->hssolver = (HYPRE_StructSolver *)
     malloc( sizeof( HYPRE_StructSolver ) );

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructJacobi_destructor(Hypre_StructJacobi this) {

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   HYPRE_StructJacobiDestroy( *S );
   free(this->d_table);

} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructJacobiApply
 **********************************************************/
int  impl_Hypre_StructJacobi_Apply
(Hypre_StructJacobi this, Hypre_Vector b, Hypre_Vector* x) {
/* old args:
 (Hypre_StructJacobi this, Hypre_StructVector b, Hypre_StructVector* x)
*/

   Hypre_StructVector SVb, SVx;
   struct Hypre_StructVector_private_type * SVbp;
   HYPRE_StructVector * Vb;
   struct Hypre_StructVector_private_type * SVxp;
   HYPRE_StructVector * Vx;

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   Hypre_StructMatrix A = this->d_table->hsmatrix;
   struct Hypre_StructMatrix_private_type *SMp = A->d_table;
   HYPRE_StructMatrix *MA = SMp->hsmat;

   SVb = (Hypre_StructVector) Hypre_Vector_castTo( b, "Hypre_StructVector" );
   if ( SVb==NULL ) return 1;
   SVx = (Hypre_StructVector) Hypre_Vector_castTo( *x, "Hypre_StructVector" );
   if ( SVb==NULL ) return 1;

   SVbp = SVb->d_table;
   Vb = SVbp->hsvec;

   SVxp = SVx->d_table;
   Vx = SVxp->hsvec;

   return HYPRE_StructJacobiSolve( *S, *MA, *Vb, *Vx );

} /* end impl_Hypre_StructJacobiApply */

/* ********************************************************
 * impl_Hypre_StructJacobiGetDims
 **********************************************************/
int  impl_Hypre_StructJacobi_GetDims(Hypre_StructJacobi this, int* m, int* n) {
   Hypre_StructMatrix hsmatrix = this->d_table->hsmatrix;
   return Hypre_StructMatrix_GetDims( hsmatrix, m, n );
} /* end impl_Hypre_StructJacobiGetDims */

/* ********************************************************
 * impl_Hypre_StructJacobiGetSystemOperator
 **********************************************************/
int impl_Hypre_StructJacobi_GetSystemOperator
( Hypre_StructJacobi this, Hypre_LinearOperator *op ) 
{
   Hypre_StructMatrix SM = this->d_table->hsmatrix;
   Hypre_LinearOperator LO = (Hypre_LinearOperator)
      Hypre_StructMatrix_castTo( SM, "Hypre_LinearOperator" );

   *op = LO;
   return 0;
} /* end impl_Hypre_StructJacobiGetSystemOperator */

/* ********************************************************
 * impl_Hypre_StructJacobiGetResidual
 **********************************************************/
int impl_Hypre_StructJacobi_GetResidual
( Hypre_StructJacobi this, Hypre_Vector *resid ) {
  
  /*
    The present StructJacobi code in Hypre doesn't provide a residual.
    c.f. files, point_relax.c (the end of the iteration is around line 605)
    jacobi.c, and HYPRE_struct_jacobi.c.  In the last file is an
    unimplemented function HYPRE_StructJacobiGetFinalRelativeResidualNorm.
    
    For now, all we do is make a dummy object and return it.  It can't even be
    of the right size because the grid information is quite buried and it's not
    worthwhile to store a StructuredGrid object just to support a function that
    doesn't work.
  */

   *resid = Hypre_Vector_new();

   printf( "called Hypre_StructJacobi_GetResidual, which doesn't work!\n");

   return 1;

} /* end impl_Hypre_StructJacobiGetResidual */

/* ********************************************************
 * impl_Hypre_StructJacobiGetConvergenceInfo
 **********************************************************/
int  impl_Hypre_StructJacobi_GetConvergenceInfo
(Hypre_StructJacobi this, char* name, double* value) {
   /* As the only HYPRE function called here is an unimplemented no-op,
      this function does nothing useful except to demonstrate how I would
      write such a function. */

   int ivalue, ierr;

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   if ( !strcmp(name,"number of iterations") ) {
      ivalue = -1;
      ierr = HYPRE_StructJacobiGetNumIterations( *S, &ivalue );
      *value = ivalue;
      return ierr;
   }
   else {
      printf(
         "Don't understand keyword %s to Hypre_StructJacobi_GetConvergenceInfo\n",
         name );
      *value = 0;
      return 1;
   }

} /* end impl_Hypre_StructJacobiGetConvergenceInfo */

/* ********************************************************
 * impl_Hypre_StructJacobiGetParameterDouble
 **********************************************************/
int impl_Hypre_StructJacobi_GetParameterDouble
( Hypre_StructJacobi this, char* name, double *value) {
   printf( "Hypre_StructJacobi_GetParameterDouble does not recognize name %s\n", name );
   *value = -123.456;
   return 1;
} /* end impl_Hypre_StructJacobiGetParameterDouble */

/* ********************************************************
 * impl_Hypre_StructJacobiGetParameterInt
 **********************************************************/
int  impl_Hypre_StructJacobi_GetParameterInt
( Hypre_StructJacobi this, char* name, int *value ) {
   printf( "Hypre_StructJacobi_GetParameterInt does not recognize name %s\n", name );
   *value = -123456;
   return 1;
} /* end impl_Hypre_StructJacobiGetParameterInt */

/* ********************************************************
 * impl_Hypre_StructJacobiSetParameterDouble
 **********************************************************/
int impl_Hypre_StructJacobi_SetParameterDouble
(Hypre_StructJacobi this, char* name, double value) {

/* JFP: This function just dispatches to the parameter's set function. */

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   if ( !strcmp(name,"tol") ) {
      return HYPRE_StructJacobiSetTol( *S, value );
   };
   if ( !strcmp(name,"zero guess") ) {
      return HYPRE_StructJacobiSetZeroGuess( *S );
   };
   if (  !strcmp(name,"nonzero guess") ) {
      return HYPRE_StructJacobiSetNonZeroGuess( *S );
   };
   printf( "Hypre_StructJacobi_SetParameterDouble does not recognize name %s\n", name );
   return 1;

} /* end impl_Hypre_StructJacobiSetParameterDouble */

/* ********************************************************
 * impl_Hypre_StructJacobiSetParameterInt
 **********************************************************/
int  impl_Hypre_StructJacobi_SetParameterInt
(Hypre_StructJacobi this, char* name, int value) {

/* JFP: This function just dispatches to the parameter's set function. */

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   if ( !strcmp(name,"max_iter" )) {
      HYPRE_StructJacobiSetMaxIter( *S, value );
      return;
   };
   if ( !strcmp(name,"zero guess") ) {
      HYPRE_StructJacobiSetZeroGuess( *S );
      return;
   };
   if (  !strcmp(name,"nonzero guess") ) {
      HYPRE_StructJacobiSetNonZeroGuess( *S );
      return;
   };
   printf( "Hypre_StructJacobi_SetParameterInt does not recognize name %s\n", name );
   return 1;

} /* end impl_Hypre_StructJacobiSetParameterInt */

/* ********************************************************
 * impl_Hypre_StructJacobiSetParameterString
 **********************************************************/
int impl_Hypre_StructJacobi_SetParameterString
( Hypre_StructJacobi this, char* name, char* value ) {
   printf( "Hypre_StructJacobi_SetParameterString does not recognize name %s\n", name );
   return 1;
} /* end impl_Hypre_StructJacobiSetParameterString */

/* ********************************************************
 * impl_Hypre_StructJacobiNew
 **********************************************************/
int  impl_Hypre_StructJacobi_New(Hypre_StructJacobi this, Hypre_MPI_Com comm) {

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   struct Hypre_MPI_Com_private_type * HMCp = comm->d_table;
   MPI_Comm *C = HMCp->hcom; /*gkk: ??? HMCp was CP */

/* the StructSolver this inherits from keeps its own pointer to the
   underlying HYPRE object.  Make sure they are the same.
   Hypre_StructSolver HSS = Hypre_StructJacobi_castTo
      ( this, "Hypre_StructSolver" );
   struct Hypre_StructSolver_private_type *HSSp = HSS->d_table;
   HSSp->hssolver = S;
*/

   return HYPRE_StructJacobiCreate( *C, S );

} /* end impl_Hypre_StructJacobiNew */

/* ********************************************************
 * impl_Hypre_StructJacobiSetup
 **********************************************************/
int  impl_Hypre_StructJacobi_Setup
(Hypre_StructJacobi this, Hypre_LinearOperator A, Hypre_Vector b,
 Hypre_Vector x) {

/* We try cast the arguments to the data types which can really be used by the
   HYPRE struct Jacobi solver.  If they can't be cast, return an error flag.
   It the cast succeeds, pull out the pointers and call the HYPRE struct Jacobi
   setup function.  The argument list we would really like for this function is:
 (Hypre_StructSMG this, Hypre_StructMatrix A, Hypre_StructVector b,
  Hypre_StructVector x)
 */

   Hypre_StructMatrix SM;
   Hypre_StructVector SVb, SVx;
   struct Hypre_StructMatrix_private_type * SMp;
   HYPRE_StructMatrix * MA;
   struct Hypre_StructVector_private_type * SVbp;
   HYPRE_StructVector * Vb;
   struct Hypre_StructVector_private_type * SVxp;
   HYPRE_StructVector * Vx;

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   SM = (Hypre_StructMatrix) Hypre_LinearOperator_castTo( A, "Hypre_StructMatrix" );
   if ( SM==NULL ) return 1;
   SVb = (Hypre_StructVector) Hypre_Vector_castTo( b, "Hypre_StructVector" );
   if ( SVb==NULL ) return 1;
   SVx = (Hypre_StructVector) Hypre_Vector_castTo( x, "Hypre_StructVector" );
   if ( SVb==NULL ) return 1;

   SMp = SM->d_table;
   MA = SMp->hsmat;

   SVbp = SVb->d_table;
   Vb = SVbp->hsvec;

   SVxp = SVx->d_table;
   Vx = SVxp->hsvec;

   this->d_table->hsmatrix = SM;

   return HYPRE_StructJacobiSetup( *S, *MA, *Vb, *Vx );

} /* end impl_Hypre_StructJacobiSetup */

/* ********************************************************
 * impl_Hypre_StructJacobiConstructor
 **********************************************************/
Hypre_StructJacobi  impl_Hypre_StructJacobi_Constructor(Hypre_MPI_Com comm) {
   /* declared static; just combines the new and New functions */
   Hypre_StructJacobi SJ = Hypre_StructJacobi_new();
   Hypre_StructJacobi_New( SJ, comm );
   return SJ;
} /* end impl_Hypre_StructJacobiConstructor */

/* ********************************************************
 * impl_Hypre_StructJacobiGetConstructedObject
 **********************************************************/
int impl_Hypre_StructJacobi_GetConstructedObject
( Hypre_StructJacobi this, Hypre_Solver *obj ) {

   *obj = (Hypre_Solver) this;
   return 0;
} /* end impl_Hypre_StructJacobiGetConstructedObject */

