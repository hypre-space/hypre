/* *****************************************************
 *
 *	File:  Hypre_StructJacobi.c
 *
 ********************************************************/

#include "Hypre_StructJacobi_Skel.h" 
#include <math.h>
#include "Hypre_StructJacobi_Data.h" /* gkk: added (is automatic as of version 0.3.0) */
#include "Hypre_StructMatrix_Skel.h" /* gkk: added */
#include "Hypre_StructMatrix_Data.h" /* gkk: added */
#include "Hypre_StructVector_Skel.h" /* gkk: added */
#include "Hypre_StructVector_Data.h" /* gkk: added */
#include "Hypre_MPI_Com_Skel.h"      /* gkk: added */
#include "Hypre_MPI_Com_Data.h"      /* gkk: added */

/* JFP: In Hypre, there is no struct with a name like HYPRE_StructJacobi,
   etc.  but hand-name-mangling of functions indicates that such a "class"
   exists in somebody's mind.  If the data be in a HYPRE_StructSolver
   (or what it points to), then HYPRE_StructJacobi could be seen as
   an interpretation of the object that has the data.
   Actually, there is a struct hypre_PointRelaxData, and the HYPRE_StructSolver
   data points to that (through void * casts, dereferences, etc.) when its
   correct interpretation is as a Jacobi solver.
   */


/*#************************************************
#	Constructor
#**************************************************/

void Hypre_StructJacobi_constructor(Hypre_StructJacobi this) {

/* JFP: Allocates Memory */
  
   this->d_table = (struct Hypre_StructJacobi_private_type *)
      malloc( sizeof( struct Hypre_StructJacobi_private_type ) );

   this->d_table->hssolver = (HYPRE_StructSolver *)
     malloc( sizeof( HYPRE_StructSolver ) );

}


/*#************************************************
#	Destructor
#**************************************************/

void Hypre_StructJacobi_destructor(Hypre_StructJacobi this) {

   /* JFP: Deallocates memory. */

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   HYPRE_StructJacobiDestroy( *S );
   free(this->d_table);
}

void  impl_Hypre_StructJacobi_Apply(
   Hypre_StructJacobi this, Hypre_StructVector b, Hypre_StructVector* x) {

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   Hypre_StructMatrix A = this->d_table->hsmatrix;
   struct Hypre_StructMatrix_private_type *SMp = A->d_table;
   HYPRE_StructMatrix *MA = SMp->hsmat;

   struct Hypre_StructVector_private_type *SVbp = b->d_table;
   HYPRE_StructVector *Vb = SVbp->hsvec;

   struct Hypre_StructVector_private_type *SVxp = (*x)->d_table;
   HYPRE_StructVector *Vx = SVxp->hsvec;

   HYPRE_StructJacobiSolve( *S, *MA, *Vb, *Vx );
}

Hypre_StructMatrix  impl_Hypre_StructJacobi_GetSystemOperator(Hypre_StructJacobi this) {

   return this->d_table->hsmatrix;

}

Hypre_StructVector  impl_Hypre_StructJacobi_GetResidual(Hypre_StructJacobi this) {
  
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

   Hypre_StructVector vec = Hypre_StructVector_new();

   printf( "called Hypre_StructJacobi_GetResidual, which doesn't work!\n");

   return vec;

}

/*void  impl_Hypre_StructJacobi_GetConvergenceInfo(Hypre_StructJacobi this) {

  * *******************************************************
   * 
   * 	Put Library code here!!!!!!
   * 
   *********************************************************
   }*/

/* After the next Babel run, replace the above with something like the
  following (replace _JFP_ with __ ) */
void  impl_Hypre_StructJacobi_GetConvergenceInfo
( Hypre_StructJacobi this, char* name, double *value )
{
   /* As the only HYPRE function called here is an unimplemented no-op,
      this function does nothing useful except to demonstrate how I would
      write such a function. */

   int ivalue;

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   if ( !strcmp(name,"number of iterations") ) {
      ivalue = -1;
      HYPRE_StructJacobiGetNumIterations( *S, &ivalue );
      *value = ivalue;
      return;
   }
   else {
      printf(
         "Don't understand keyword %s to Hypre_StructJacobi_GetConvergenceInfo\n",
         name );
      *value = 0;
      return;
   }

}


void  impl_Hypre_StructJacobi_SetSystemOperator( Hypre_StructJacobi this,
                                                  Hypre_StructMatrix op) {

/* Sets the matrix.  Setup should (probably) be called before anything is
   done with it. */

   this->d_table->hsmatrix = op ;

}

void  impl_Hypre_StructJacobi_SetParameter(
   Hypre_StructJacobi this, char* name, double value) {

/* JFP: This function just dispatches to the parameter's set function. */

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   if ( !strcmp(name,"tol") ) {
      HYPRE_StructJacobiSetTol( *S, value );
      return;
   };
   if ( !strcmp(name,"max_iter" )) {
      HYPRE_StructJacobiSetMaxIter( *S, floor(value*1.001) );
      /* ... floor(value*1.001) is a simple adequate way to undo an
         int->double conversion */
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

}

double  impl_Hypre_StructJacobi_GetParameter(Hypre_StructJacobi this, char* name) {
  /* gkk: Added: does nothing right now */
  return 0;
}

void  impl_Hypre_StructJacobi_Setup(
   Hypre_StructJacobi this, Hypre_StructMatrix A, Hypre_StructVector b,
   Hypre_StructVector x ){ /* gkk: removed Hypre_MPI_Com comm) {*/

/* after next Babel run, won't need comm argument */

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   struct Hypre_StructMatrix_private_type *SMp = A->d_table;
   HYPRE_StructMatrix *MA = SMp->hsmat;

   struct Hypre_StructVector_private_type *SVbp = b->d_table;
   HYPRE_StructVector *Vb = SVbp->hsvec;

   struct Hypre_StructVector_private_type *SVxp = x->d_table;
   HYPRE_StructVector *Vx = SVxp->hsvec;

   this->d_table->hsmatrix = A;

   HYPRE_StructJacobiSetup( *S, *MA, *Vb, *Vx );

}

void impl_Hypre_StructJacobi_NewSolver /*gkk was Hypre_StructJacobi_NewSolver*/
( Hypre_StructJacobi this, Hypre_MPI_Com comm) {

/* after next Babel run, will have a name like
   impl_Hypre_StructJacobi_NewSolver */

   struct Hypre_StructJacobi_private_type *HSJp = this->d_table;
   HYPRE_StructSolver *S = HSJp->hssolver;

   struct Hypre_MPI_Com_private_type * HMCp = comm->d_table;
   MPI_Comm *C = HMCp->hcom; /*gkk: ??? HMCp was CP */

   HYPRE_StructJacobiCreate( *C, S );

}

Hypre_SolverBuilder  impl_Hypre_StructJacobi_GetConstructedObject(Hypre_StructJacobi this) {

   return (Hypre_SolverBuilder) this;

}


