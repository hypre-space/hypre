
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
   this->Hypre_StructSMG_data = (struct Hypre_StructSMG_private_type *)
      malloc( sizeof( struct Hypre_StructSMG_private_type ) );

   this->Hypre_StructSMG_data->hssolver = (HYPRE_StructSolver *)
     malloc( sizeof( HYPRE_StructSolver ) );
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructSMG_destructor(Hypre_StructSMG this) {
   struct Hypre_StructSMG_private_type *HSJp = this->Hypre_StructSMG_data;
   HYPRE_StructSolver *S = HSJp->hssolver;

   HYPRE_StructSMGDestroy( *S );
   free(this->Hypre_StructSMG_data);

} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructSMGApply
 * This function only works if the provided Hypre_Vector's
 * are really Hypre_StructVector's.  That requirement is checked.
 **********************************************************/
int  impl_Hypre_StructSMG_Apply
(Hypre_StructSMG this, Hypre_Vector b, Hypre_Vector* x) {
   struct Hypre_StructSMG_private_type *HSMGp = this->Hypre_StructSMG_data;
   HYPRE_StructSolver *S = HSMGp->hssolver;

   Hypre_StructMatrix A = this->Hypre_StructSMG_data->hsmatrix;
   struct Hypre_StructMatrix_private_type *SMp = A->Hypre_StructMatrix_data;
   HYPRE_StructMatrix *MA = SMp->hsmat;

   Hypre_StructVector Sb, Sx;
   struct Hypre_StructVector_private_type *SVbp, *SVxp;
   HYPRE_StructVector *Vb, *Vx;

   Sb = (Hypre_StructVector) Hypre_Vector_castTo( b, "Hypre_StructVector" );
   if ( Sb == NULL ) return 1;

   Sx = (Hypre_StructVector) Hypre_Vector_castTo( *x, "Hypre_StructVector" );
   if ( Sx == NULL ) return 1;

   SVbp = Sb->Hypre_StructVector_data;
   Vb = SVbp->hsvec;

   SVxp = Sx->Hypre_StructVector_data;
   Vx = SVxp->hsvec;

   return HYPRE_StructSMGSolve( *S, *MA, *Vb, *Vx );
} /* end impl_Hypre_StructSMGApply */

/* ********************************************************
 * impl_Hypre_StructSMGGetDims
 **********************************************************/
int  impl_Hypre_StructSMG_GetDims(Hypre_StructSMG this, int* m, int* n) {
   Hypre_StructMatrix hsmatrix = this->Hypre_StructSMG_data->hsmatrix;
   return Hypre_StructMatrix_GetDims( hsmatrix, m, n );
} /* end impl_Hypre_StructSMGGetDims */

/* ********************************************************
 * impl_Hypre_StructSMGGetSystemOperator
 **********************************************************/
int impl_Hypre_StructSMG_GetSystemOperator
( Hypre_StructSMG this, Hypre_LinearOperator *op ) {

   Hypre_StructMatrix mat =  this->Hypre_StructSMG_data->hsmatrix;
   
   *op = (Hypre_LinearOperator)
      Hypre_StructMatrix_castTo( mat, "Hypre_LinearOperator" );
   return 0;
} /* end impl_Hypre_StructSMGGetSystemOperator */

/* ********************************************************
 * impl_Hypre_StructSMGGetResidual
 **********************************************************/
int impl_Hypre_StructSMG_GetResidual
( Hypre_StructSMG this, Hypre_Vector *resid ) {
  
  /*
    The present HYPRE_struct_smg.c code in Hypre doesn't provide a residual.
    I haven't bothered to look at other Hypre SMG files. (jfp)
    If we implement this function, the way to do it would be to compute it,
    using saved matrix and vectors.
    
    For now, all we do is make a dummy object and return it.  It can't even be
    of the right size because the grid information is quite buried and it's not
    worthwhile to store a StructuredGrid object just to support a function that
    doesn't work.
  */

   *resid = (Hypre_Vector) NULL;

   printf( "called Hypre_StructSMG_GetResidual, which doesn't work!\n");

   return 1;
} /* end impl_Hypre_StructSMGGetResidual */

/* ********************************************************
 * impl_Hypre_StructSMGGetConvergenceInfo
 **********************************************************/
int  impl_Hypre_StructSMG_GetConvergenceInfo
(Hypre_StructSMG this, char* name, double* value) {
   int ivalue, ierr;

   struct Hypre_StructSMG_private_type *HSMGp = this->Hypre_StructSMG_data;
   HYPRE_StructSolver *S = HSMGp->hssolver;

   if ( !strcmp(name,"num iterations") ) {
      ierr = HYPRE_StructSMGGetNumIterations( *S, &ivalue );
      *value = ivalue;
      return ierr;
   }
   if ( !strcmp(name,"final relative residual norm") ) {
      ierr = HYPRE_StructSMGGetFinalRelativeResidualNorm( *S, value );
      return ierr;
   }

   printf( "Hypre_StructJacobi_GetConvergenceInfo does not recognize name %s\n", name );

   return 1;
} /* end impl_Hypre_StructSMGGetConvergenceInfo */

/* ********************************************************
 * impl_Hypre_StructSMGGetParameterDouble
 **********************************************************/
int impl_Hypre_StructSMG_GetParameterDouble
( Hypre_StructSMG this, char* name, double *value ) {
   printf( "Hypre_StructJacobi_GetParameterDouble does not recognize name %s\n", name );
   return 1;
} /* end impl_Hypre_StructSMGGetParameterDouble */

/* ********************************************************
 * impl_Hypre_StructSMGGetParameterInt
 **********************************************************/
int impl_Hypre_StructSMG_GetParameterInt
( Hypre_StructSMG this, char* name, int *value ) {
   printf( "Hypre_StructJacobi_GetParameterInt does not recognize name %s\n", name );
   return 1;
} /* end impl_Hypre_StructSMGGetParameterInt */

/* ********************************************************
 * impl_Hypre_StructSMGSetParameterDouble
 **********************************************************/
int  impl_Hypre_StructSMG_SetParameterDouble
(Hypre_StructSMG this, char* name, double value) {

/* This function just dispatches to the parameter's set function. */

   struct Hypre_StructSMG_private_type *HSMGp = this->Hypre_StructSMG_data;
   HYPRE_StructSolver *S = HSMGp->hssolver;

   if ( !strcmp(name,"tol") ) {
      return HYPRE_StructSMGSetTol( *S, value );
   };
   if ( !strcmp(name,"zero guess") ) {
      return HYPRE_StructSMGSetZeroGuess( *S );
   };
   if (  !strcmp(name,"nonzero guess") ) {
      return HYPRE_StructSMGSetNonZeroGuess( *S );
   };
   return 1;

} /* end impl_Hypre_StructSMGSetParameterDouble */

/* ********************************************************
 * impl_Hypre_StructSMGSetParameterInt
 **********************************************************/
int impl_Hypre_StructSMG_SetParameterInt
(Hypre_StructSMG this, char* name, int value) {

/* This function just dispatches to the parameter's set function. */

   struct Hypre_StructSMG_private_type *HSMGp = this->Hypre_StructSMG_data;
   HYPRE_StructSolver *S = HSMGp->hssolver;

   if ( !strcmp(name,"max_iter" )) {
      return HYPRE_StructSMGSetMaxIter( *S, value );
   };
   if ( !strcmp(name,"max iter" )) {
      return HYPRE_StructSMGSetMaxIter( *S, value );
   };
   if ( !strcmp(name,"zero guess") ) {
      return HYPRE_StructSMGSetZeroGuess( *S );
   };
   if (  !strcmp(name,"nonzero guess") ) {
      return HYPRE_StructSMGSetNonZeroGuess( *S );
   };
   if ( !strcmp(name,"memory use") ) {
      return HYPRE_StructSMGSetMemoryUse( *S, value );
   };
   if ( !strcmp(name,"rel change") ) {
      return HYPRE_StructSMGSetRelChange( *S, value );
   };
   if ( !strcmp(name,"num prerelax") ) {
      return HYPRE_StructSMGSetNumPreRelax( *S, value );
   };
   if ( !strcmp(name,"num postrelax") ) {
      return HYPRE_StructSMGSetNumPostRelax( *S, value );
   };
   if ( !strcmp(name,"logging") ) {
      return HYPRE_StructSMGSetLogging( *S, value );
   };
   printf( "Hypre_StructJacobi_SetParameterInt does not recognize name %s\n", name );
   return 1;

} /* end impl_Hypre_StructSMGSetParameterInt */

/* ********************************************************
 * impl_Hypre_StructSMGSetParameterString
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_StructSMG_SetParameterString
( Hypre_StructSMG this, char* name, char* value ) {
   printf( "Hypre_StructJacobi_SetParameterString does not recognize name %s\n", name );
   return 1;
} /* end impl_Hypre_StructSMGSetParameterString */

/* ********************************************************
 * impl_Hypre_StructSMG_Start
 **********************************************************/
int impl_Hypre_StructSMG_Start(Hypre_StructSMG this, Hypre_MPI_Com comm) {
   struct Hypre_StructSMG_private_type *HSMGp = this->Hypre_StructSMG_data;
   HYPRE_StructSolver *S = HSMGp->hssolver;

   struct Hypre_MPI_Com_private_type * HMCp = comm->Hypre_MPI_Com_data;
   MPI_Comm *C = HMCp->hcom;

   return HYPRE_StructSMGCreate( *C, S );
} /* end impl_Hypre_StructSMG_Start */

/* ********************************************************
 * impl_Hypre_StructSMGSetup
 * A really has to be Hypre_StructMatrix, and b,x really have to be
 * Hypre_StructVector We check for that.
 **********************************************************/
int  impl_Hypre_StructSMG_Setup
( Hypre_StructSMG this, Hypre_LinearOperator A, Hypre_Vector b, Hypre_Vector x)
{
   
/* We try cast the arguments to the data types which can really be used by the
   HYPRE SMG.  If they can't be cast, return an error flag.  It the cast
   succeeds, pull out the pointers and call the HYPRE SMG setup function.
   The argument list we would really like for this function is:
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

   struct Hypre_StructSMG_private_type *HSMGp = this->Hypre_StructSMG_data;
   HYPRE_StructSolver *S = HSMGp->hssolver;

   SM = (Hypre_StructMatrix) Hypre_LinearOperator_castTo( A, "Hypre_StructMatrix" );
   if ( SM==NULL ) return 1;
   SVb = (Hypre_StructVector) Hypre_Vector_castTo( b, "Hypre_StructVector" );
   if ( SVb==NULL ) return 1;
   SVx = (Hypre_StructVector) Hypre_Vector_castTo( x, "Hypre_StructVector" );
   if ( SVb==NULL ) return 1;

   SMp = SM->Hypre_StructMatrix_data;
   MA = SMp->hsmat;

   SVbp = SVb->Hypre_StructVector_data;
   Vb = SVbp->hsvec;

   SVxp = SVx->Hypre_StructVector_data;
   Vx = SVxp->hsvec;

   this->Hypre_StructSMG_data->hsmatrix = SM;

   return HYPRE_StructSMGSetup( *S, *MA, *Vb, *Vx );
} /* end impl_Hypre_StructSMGSetup */

/* ********************************************************
 * impl_Hypre_StructSMGConstructor
 **********************************************************/
Hypre_StructSMG  impl_Hypre_StructSMG_Constructor(Hypre_MPI_Com comm) {
   /* declared static; just combines the New and Start functions */
   Hypre_StructSMG SMG = Hypre_StructSMG_New();
   Hypre_StructSMG_Start( SMG, comm );
   return SMG;
} /* end impl_Hypre_StructSMGConstructor */

/* ********************************************************
 * impl_Hypre_StructSMGGetConstructedObject
 **********************************************************/
int impl_Hypre_StructSMG_GetConstructedObject
( Hypre_StructSMG this, Hypre_Solver *obj ) {

   *obj = (Hypre_Solver) this;
   return 0;

} /* end impl_Hypre_StructSMGGetConstructedObject */

