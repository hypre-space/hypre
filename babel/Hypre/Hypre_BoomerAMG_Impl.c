/*
 * File:          Hypre_BoomerAMG_Impl.c
 * Symbol:        Hypre.BoomerAMG-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.BoomerAMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1232
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.BoomerAMG" (version 0.1.7)
 * 
 * Algebraic multigrid solver, based on classical Ruge-Stueben.
 * 
 * The following optional parameters are available and may be set
 * using the appropriate {\tt Parameter} function (as indicated in
 * parentheses):
 * 
 * \begin{description}
 * 
 * \item[Max Levels] ({\tt Int}) - maximum number of multigrid
 * levels.
 * 
 * \item[Strong Threshold] ({\tt Double}) - AMG strength threshold.
 * 
 * \item[Max Row Sum] ({\tt Double}) -
 * 
 * \item[Coarsen Type] ({\tt Int}) - type of parallel coarsening
 * algorithm used.
 * 
 * \item[Measure Type] ({\tt Int}) - type of measure used; local or
 * global.
 * 
 * \item[Cycle Type] ({\tt Int}) - type of cycle used; a V-cycle
 * (default) or a W-cycle.
 * 
 * \item[Num Grid Sweeps] ({\tt IntArray}) - number of sweeps for
 * fine and coarse grid, up and down cycle.
 * 
 * \item[Grid Relax Type] ({\tt IntArray}) - type of smoother used
 * on fine and coarse grid, up and down cycle.
 * 
 * \item[Grid Relax Points] ({\tt IntArray}) - point ordering used
 * in relaxation.
 * 
 * \item[Relax Weight] ({\tt DoubleArray}) - relaxation weight for
 * smoothed Jacobi and hybrid SOR.
 * 
 * \item[Truncation Factor] ({\tt Double}) - truncation factor for
 * interpolation.
 * 
 * \item[Smooth Type] ({\tt Int}) - more complex smoothers.
 * 
 * \item[Smooth Num Levels] ({\tt Int}) - number of levels for more
 * complex smoothers.
 * 
 * \item[Smooth Num Sweeps] ({\tt Int}) - number of sweeps for more
 * complex smoothers.
 * 
 * \item[Print File Name] ({\tt String}) - name of file printed to
 * in association with {\tt SetPrintLevel}.  (not yet implemented).
 * 
 * \item[Num Functions] ({\tt Int}) - size of the system of PDEs
 * (when using the systems version).
 * 
 * \item[DOF Func] ({\tt IntArray}) - mapping that assigns the
 * function to each variable (when using the systems version).
 * 
 * \item[Variant] ({\tt Int}) - variant of Schwarz used.
 * 
 * \item[Overlap] ({\tt Int}) - overlap for Schwarz.
 * 
 * \item[Domain Type] ({\tt Int}) - type of domain used for
 * Schwarz.
 * 
 * \item[Schwarz Relaxation Weight] ({\tt Double}) - the smoothing
 * parameter for additive Schwarz.
 * 
 * \item[Debug Flag] ({\tt Int}) -
 * 
 * \end{description}
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Changed name from 'ParAMG' (x)
 * 
 */

#include "Hypre_BoomerAMG_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "Hypre_IJParCSRMatrix_Impl.h"
#include "Hypre_IJParCSRVector_Impl.h"
/* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG__ctor"

void
impl_Hypre_BoomerAMG__ctor(
  Hypre_BoomerAMG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG._ctor) */
  /* Insert the implementation of the constructor method here... */

   int ierr=0;
   HYPRE_Solver dummy;
   /* will really be initialized by Create call */
   HYPRE_Solver * solver = &dummy;
   struct Hypre_BoomerAMG__data * data;
   data = hypre_CTAlloc( struct Hypre_BoomerAMG__data, 1 );
   data -> comm = NULL;
   ierr += HYPRE_BoomerAMGCreate( solver );
   data -> solver = *solver;
   /* set any other data components here */
   Hypre_BoomerAMG__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG__dtor"

void
impl_Hypre_BoomerAMG__dtor(
  Hypre_BoomerAMG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct Hypre_BoomerAMG__data * data;
   data = Hypre_BoomerAMG__get_data( self );
   ierr += HYPRE_BoomerAMGDestroy( data->solver );
   Hypre_IJParCSRMatrix_deleteRef( data->matrix );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_SetCommunicator"

int32_t
impl_Hypre_BoomerAMG_SetCommunicator(
  Hypre_BoomerAMG self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   struct Hypre_BoomerAMG__data * data = Hypre_BoomerAMG__get_data( self );
   data -> comm = (MPI_Comm *) mpi_comm;
   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_SetIntParameter"

int32_t
impl_Hypre_BoomerAMG_SetIntParameter(
  Hypre_BoomerAMG self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Coarsen Type")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCoarsenType( solver, value );      
   }
   else if ( strcmp(name,"Measure Type")==0 ) 
   {
      ierr += HYPRE_BoomerAMGSetMeasureType( solver, value );
   }
   else if ( strcmp(name,"Print Level")==0 )
   {
      ierr += HYPRE_BoomerAMGSetPrintLevel( solver, value );
   }
   else if ( strcmp(name,"Cycle Type")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCycleType( solver, value );
   }
   else if ( strcmp(name,"Smooth Type")==0 )
   {
      ierr += HYPRE_BoomerAMGSetSmoothType( solver, value );
   }
   else if ( strcmp(name,"Smooth Num Levels")==0 )
   {
      ierr += HYPRE_BoomerAMGSetSmoothNumLevels( solver, value );
   }
   else if ( strcmp(name,"Smooth Num Sweeps")==0 )
   {
      ierr += HYPRE_BoomerAMGSetSmoothNumSweeps( solver, value );
   }
   else if ( strcmp(name,"Max Levels")==0 )
   {
      ierr += HYPRE_BoomerAMGSetMaxLevels( solver, value );
   }
   else if ( strcmp(name,"Debug Flag")==0 )
   {
      ierr += HYPRE_BoomerAMGSetDebugFlag( solver, value );
   }
   else if ( strcmp(name,"Variant")==0 )
   {
      ierr += HYPRE_BoomerAMGSetVariant( solver, value );
   }
   else if ( strcmp(name,"Overlap")==0 )
   {
      ierr += HYPRE_BoomerAMGSetOverlap( solver, value );
   }
   else if ( strcmp(name,"Domain Type")==0 )
   {
      ierr += HYPRE_BoomerAMGSetDomainType( solver, value );
   }
   else if ( strcmp(name,"Num Functions")==0 )
   {
      ierr += HYPRE_BoomerAMGSetNumFunctions( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_SetDoubleParameter"

int32_t
impl_Hypre_BoomerAMG_SetDoubleParameter(
  Hypre_BoomerAMG self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Strong Threshold")==0 )
   {
      ierr += HYPRE_BoomerAMGSetStrongThreshold( solver, value );
   }
   else if ( strcmp(name,"Truncation Factor")==0 )
   {
      ierr += HYPRE_BoomerAMGSetTruncFactor( solver, value );
   }
   else if ( strcmp(name,"Schwarz Relaxation Weight")==0 )
   {
      ierr += HYPRE_BoomerAMGSetSchwarzRlxWeight( solver, value );
   }
   else if ( strcmp(name,"Max Row Sum")==0 )
   {
      ierr += HYPRE_BoomerAMGSetMaxRowSum( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_SetStringParameter"

int32_t
impl_Hypre_BoomerAMG_SetStringParameter(
  Hypre_BoomerAMG self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Print File Name")==0 )
   {
      ierr += hypre_BoomerAMGSetPrintFileName( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.SetStringParameter) */
}

/*
 * Set the int array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_SetIntArrayParameter"

int32_t
impl_Hypre_BoomerAMG_SetIntArrayParameter(
  Hypre_BoomerAMG self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */

   int ierr = 0;
   int dim, lb0, ub0, lb1, ub1, i, j;
   int * data1_c;  /* the data in a C 1d array */
   int ** data2_c; /* the data in a C 2d array */
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;
   data1_c = value->d_firstElement;

   dim = SIDL_int__array_dimen( value );

   if ( strcmp(name,"Num Grid Sweeps")==0 )
   {
      assert( dim==1 );
      ierr += HYPRE_BoomerAMGSetNumGridSweeps( solver, data1_c );
   }
   else if ( strcmp(name,"Grid Relax Type")==0 )
   {
      assert( dim==1 );
      ierr += HYPRE_BoomerAMGSetGridRelaxType( solver, data1_c );
   }
   else if ( strcmp(name,"Grid Relax Points")==0 )
   {
      assert( dim==2 );
      lb0 = SIDL_int__array_lower( value, 0 );
      ub0 = SIDL_int__array_upper( value, 0 );
      lb1 = SIDL_int__array_lower( value, 1 );
      ub1 = SIDL_int__array_upper( value, 1 );
      assert( lb0==0 );
      assert( lb1==0 );
      data2_c = hypre_CTAlloc(int *,ub0);
      for ( i=0; i<ub0; ++i )
      {
         data2_c[i] = hypre_CTAlloc(int,ub1);
         for ( j=0; j<ub1; ++j )
         {
            data2_c[i][j] = SIDL_int__array_get2( value, i, j );
         }
      }
      ierr += HYPRE_BoomerAMGSetGridRelaxPoints( solver, data2_c );
   }
   else if ( strcmp(name,"DOF Func")==0 )
   {
      assert( dim==1 );
      ierr += HYPRE_BoomerAMGSetDofFunc( solver, data1_c );
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.SetIntArrayParameter) */
}

/*
 * Set the double array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_SetDoubleArrayParameter"

int32_t
impl_Hypre_BoomerAMG_SetDoubleArrayParameter(
  Hypre_BoomerAMG self, const char* name, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.SetDoubleArrayParameter) */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */

   int ierr = 0, dim;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   dim = SIDL_double__array_dimen( value );

   if ( strcmp(name,"Relax Weight")==0 )
   {
      assert( dim==1 );
      ierr += HYPRE_BoomerAMGSetRelaxWeight( solver, value->d_firstElement );
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.SetDoubleArrayParameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_GetIntValue"

int32_t
impl_Hypre_BoomerAMG_GetIntValue(
  Hypre_BoomerAMG self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr = 1;

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_GetDoubleValue"

int32_t
impl_Hypre_BoomerAMG_GetDoubleValue(
  Hypre_BoomerAMG self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr = 1;

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_Setup"

int32_t
impl_Hypre_BoomerAMG_Setup(
  Hypre_BoomerAMG self, Hypre_Vector b, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;
   struct Hypre_IJParCSRMatrix__data * dataA;
   struct Hypre_IJParCSRVector__data * datab, * datax;
   Hypre_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix HypreP_A;
   Hypre_IJParCSRVector HypreP_b, HypreP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = Hypre_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   HypreP_A = (HYPRE_ParCSRMatrix) objectA;

   if ( Hypre_Vector_queryInt(b, "Hypre.IJParCSRVector" ) )
   {
      HypreP_b = Hypre_IJParCSRVector__cast( b );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }

   datab = Hypre_IJParCSRVector__get_data( HypreP_b );
   Hypre_IJParCSRVector_deleteRef( HypreP_b );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   if ( Hypre_Vector_queryInt( x, "Hypre.IJParCSRVector" ) )
   {
      HypreP_x = Hypre_IJParCSRVector__cast( x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)(x) );
   }

   datax = Hypre_IJParCSRVector__get_data( HypreP_x );
   Hypre_IJParCSRVector_deleteRef( HypreP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   ierr += HYPRE_BoomerAMGSetup( solver, HypreP_A, bb, xx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_Apply"

int32_t
impl_Hypre_BoomerAMG_Apply(
  Hypre_BoomerAMG self, Hypre_Vector b, Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.Apply) */
  /* Insert the implementation of the Apply method here... */

   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;
   struct Hypre_IJParCSRMatrix__data * dataA;
   struct Hypre_IJParCSRVector__data * datab, * datax;
   Hypre_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix HypreP_A;
   Hypre_IJParCSRVector HypreP_b, HypreP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = Hypre_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   HypreP_A = (HYPRE_ParCSRMatrix) objectA;

   if ( Hypre_Vector_queryInt(b, "Hypre.IJParCSRVector" ) )
   {
      HypreP_b = Hypre_IJParCSRVector__cast( b );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }

   datab = Hypre_IJParCSRVector__get_data( HypreP_b );
   Hypre_IJParCSRVector_deleteRef( HypreP_b );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   if ( *x==NULL )
   {
      /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or assert(x
       * has the right size) */
      Hypre_Vector_Clone( b, x );
      Hypre_Vector_Clear( *x );
   }
   if ( Hypre_Vector_queryInt( *x, "Hypre.IJParCSRVector" ) )
   {
      HypreP_x = Hypre_IJParCSRVector__cast( *x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)(*x) );
   }

   datax = Hypre_IJParCSRVector__get_data( HypreP_x );
   Hypre_IJParCSRVector_deleteRef( HypreP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   ierr += HYPRE_BoomerAMGSolve( solver, HypreP_A, bb, xx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.Apply) */
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_SetOperator"

int32_t
impl_Hypre_BoomerAMG_SetOperator(
  Hypre_BoomerAMG self, Hypre_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   int ierr = 0;
   struct Hypre_BoomerAMG__data * data;
   Hypre_IJParCSRMatrix Amat;

   if ( Hypre_Operator_queryInt( A, "Hypre.IJParCSRMatrix" ) )
   {
      Amat = Hypre_IJParCSRMatrix__cast( A );
   }
   else
   {
      assert( "Unrecognized operator type."==(char *)A );
   }

   data = Hypre_BoomerAMG__get_data( self );
   data->matrix = Amat;

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_SetTolerance"

int32_t
impl_Hypre_BoomerAMG_SetTolerance(
  Hypre_BoomerAMG self, double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_BoomerAMGSetTol( solver, tolerance );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_SetMaxIterations"

int32_t
impl_Hypre_BoomerAMG_SetMaxIterations(
  Hypre_BoomerAMG self, int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_BoomerAMGSetMaxIter( solver, max_iterations );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.SetMaxIterations) */
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_SetLogging"

int32_t
impl_Hypre_BoomerAMG_SetLogging(
  Hypre_BoomerAMG self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */

   /* This function should be called before Setup.  Log level changes
    * may require allocation or freeing of arrays, which is presently
    * only done there.  It may be possible to support log_level
    * changes at other times, but there is little need.  */
   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr += HYPRE_BoomerAMGSetLogging( solver, level );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.SetLogging) */
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_SetPrintLevel"

int32_t
impl_Hypre_BoomerAMG_SetPrintLevel(
  Hypre_BoomerAMG self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr += HYPRE_BoomerAMGSetPrintLevel( solver, level );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_GetNumIterations"

int32_t
impl_Hypre_BoomerAMG_GetNumIterations(
  Hypre_BoomerAMG self, int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr += HYPRE_BoomerAMGGetNumIterations( solver, num_iterations );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_BoomerAMG_GetRelResidualNorm"

int32_t
impl_Hypre_BoomerAMG_GetRelResidualNorm(
  Hypre_BoomerAMG self, double* norm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.BoomerAMG.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_BoomerAMG__data * data;

   data = Hypre_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr += HYPRE_BoomerAMGGetFinalRelativeResidualNorm( solver, norm );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.BoomerAMG.GetRelResidualNorm) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
