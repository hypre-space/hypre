/*#*****************************************************
#
#	File:  Hypre_StructJacobi.c
#
#********************************************************/

#include "Hypre_StructJacobi_Skel.h" 
#include <math.h>

/* JFP: In Hypre, there is no struct with a name like HYPRE_StructJacobi,
   etc.  but hand-name-mangling of functions indicates that such a "class"
   exists in somebody's mind.  If the data be in a HYPRE_StructSolver
   (or what it points to), then HYPRE_StructJacobi could be seen as
   an interpretation of the object that has the data.
   Actually, there is a struct hypre_PointRelaxData, and the HYPRE_StructSolver
   data points to that (through void * casts, dereferences, etc.) when its
   correct interpretation is as a Jacobi solver.
   */

/* JFP >>>>>>> If I keep the following structs, they need to be moved
 to a .h file <<<<<< */

/*--------------------------------------------------------------------------
 * hypre_JacobiData data structure copied from jacobi.c
 *--------------------------------------------------------------------------*/

typedef struct
{
   void  *relax_data;

} hypre_JacobiData;

/*--------------------------------------------------------------------------
 * hypre_PointRelaxData data structure copied from point_relax.c
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;
                       
   double                  tol;                /* not yet used */
   int                     max_iter;
   int                     rel_change;         /* not yet used */
   int                     zero_guess;
   double                  weight;
                         
   int                     num_pointsets;
   int                    *pointset_sizes;
   int                    *pointset_ranks;
   hypre_Index            *pointset_strides;
   hypre_Index           **pointset_indices;
                       
   hypre_StructMatrix     *A;
   hypre_StructVector     *b;
   hypre_StructVector     *x;

   hypre_StructVector     *t;

   int                     diag_rank;

   hypre_ComputePkg      **compute_pkgs;

   /* log info (always logged) */
   int                     num_iterations;
   int                     time_index;
   int                     flops;

} hypre_PointRelaxData;


/*#************************************************
#	Constructor
#**************************************************/

void Hypre_StructJacobi_constructor(Hypre_StructJacobi this) {

/* JFP: Allocates Memory */

   struct Hypre_StructJacobi_private * HSJp;
   HSJp = (struct Hypre_StructJacobi_private *)
      malloc( sizeof( struct Hypre_StructJacobi_private ) );
   this->d_table = (Hypre_StructJacobi_Private) HSJp;

   this->d_table->hssolver = (HYPRE_StructSolver *)
      malloc( sizeof( HYPRE_StructSolver ) );

}


/*#************************************************
#	Destructor
#**************************************************/

void Hypre_StructJacobi_destructor(Hypre_StructJacobi this) {

   /* JFP: Deallocates memory. */

   Hypre_StructJacobi_Private HSJP = this->d_table;
   struct Hypre_StructJacobi_private *HSJp = HSJP;
   HYPRE_StructSolver *S = HSJp->hssolver;

   HYPRE_StructJacobiDestroy( *S );

   free(this->d_table);

}

void  impl__Hypre_StructJacobi_Apply(
   Hypre_StructJacobi this, Hypre_StructVector b, Hypre_StructVector* x) {

   Hypre_StructJacobi_Private HSJP = this->d_table;
   struct Hypre_StructJacobi_private *HSJp = HSJP;
   HYPRE_StructSolver *S = HSJp->hssolver;

/* It's a bit tricky to get the matrix back out of where it's hidden:
   when the HYPRE_StructSolver representing a StructJacobi is created, the matrix
   is passed via some void* function arguments and saved in a struct defined only
   in a .c file.  Maybe instead I should save my own pointer to the Hypre_StructMatrix
   object in Hypre_StructJacobi_Data.h ... */
   hypre_JacobiData *jacobi_data =  (void *) *S; /* as in jacobi.c */
   hypre_PointRelaxData *relax_data = jacobi_data -> relax_data; /* as in point_relax.c */
   hypre_StructMatrix * hA = hypre_StructMatrixRef( relax_data -> A );
   HYPRE_StructMatrix HA = (HYPRE_StructMatrix) hA;

   Hypre_StructVector_Private SVbP = b->d_table;
   struct Hypre_StructVector_private *SVbp = SVbP;
   HYPRE_StructVector *Vb = SVbp->hsvec;

   Hypre_StructVector_Private SVxP = (*x)->d_table;
   struct Hypre_StructVector_private *SVxp = SVxP;
   HYPRE_StructVector *Vx = SVxp->hsvec;

   HYPRE_StructJacobiSolve( *S, HA, *Vb, *Vx );

}

Hypre_StructMatrix  impl__Hypre_StructJacobi_GetSystemOperator(Hypre_StructJacobi this) {

	/*#*******************************************************
	#
	#	Put Library code here!!!!!!
	#
	#*********************************************************/

}

Hypre_StructVector  impl__Hypre_StructJacobi_GetResidual(Hypre_StructJacobi this) {

	/*#*******************************************************
	#
	#	Put Library code here!!!!!!
	#
	#*********************************************************/

}

void  impl__Hypre_StructJacobi_GetConvergenceInfo(Hypre_StructJacobi this) {

	/*#*******************************************************
	#
	#	Put Library code here!!!!!!
	#
	#*********************************************************/

}

void  impl__Hypre_StructJacobi_SetSystemOperator(Hypre_StructJacobi this, Hypre_StructMatrix op) {

	/*#*******************************************************
	#
	#	Put Library code here!!!!!!
	#
	#*********************************************************/

}

void  impl__Hypre_StructJacobi_SetParameter(
   Hypre_StructJacobi this, char* name, double value) {

/* JFP: This function just dispatches to the parameter's set function. */

   Hypre_StructJacobi_Private HSJP = this->d_table;
   struct Hypre_StructJacobi_private *HSJp = HSJP;
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

void  impl__Hypre_StructJacobi_Setup(
   Hypre_StructJacobi this, Hypre_StructMatrix A, Hypre_StructVector b,
   Hypre_StructVector x, Hypre_MPI_Com comm) {

/* after next Babel run, won't need comm argument */

   Hypre_StructJacobi_Private HSJP = this->d_table;
   struct Hypre_StructJacobi_private *HSJp = HSJP;
   HYPRE_StructSolver *S = HSJp->hssolver;

   Hypre_StructMatrix_Private SMP = A->d_table;
   struct Hypre_StructMatrix_private *SMp = SMP;
   HYPRE_StructMatrix *MA = SMp->hsmat;

   Hypre_StructVector_Private SVbP = b->d_table;
   struct Hypre_StructVector_private *SVbp = SVbP;
   HYPRE_StructVector *Vb = SVbp->hsvec;

   Hypre_StructVector_Private SVxP = x->d_table;
   struct Hypre_StructVector_private *SVxp = SVxP;
   HYPRE_StructVector *Vx = SVxp->hsvec;

   HYPRE_StructJacobiSetup( *S, *MA, *Vb, *Vx );

}

void Hypre_StructJacobi_NewSolver
( Hypre_StructJacobi this, Hypre_MPI_Com comm) {

/* after next Babel run, will have a name like
   impl__Hypre_StructJacobi_NewSolver */

   Hypre_StructJacobi_Private HSJP = this->d_table;
   struct Hypre_StructJacobi_private *HSJp = HSJP;
   HYPRE_StructSolver *S = HSJp->hssolver;

   Hypre_MPI_Com_Private CP = comm->d_table;
   struct Hypre_MPI_Com_private * HMCp = CP;
   MPI_Comm *C = CP->hcom;

   HYPRE_StructJacobiCreate( *C, S );

}

Hypre_SolverBuilder  impl__Hypre_StructJacobi_GetConstructedObject(Hypre_StructJacobi this) {

	/*#*******************************************************
	#
	#	Put Library code here!!!!!!
	#
	#*********************************************************/

}


