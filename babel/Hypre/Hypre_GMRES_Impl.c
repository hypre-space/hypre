/*
 * File:          Hypre_GMRES_Impl.c
 * Symbol:        Hypre.GMRES-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.GMRES
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1262
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.GMRES" (version 0.1.7)
 * 
 * Objects of this type can be cast to PreconditionedSolver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here. (x)
 * 
 */

#include "Hypre_GMRES_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.GMRES._includes) */
/* Put additional includes or other arbitrary code here... */
#include "Hypre_IJParCSRMatrix.h"
#include "Hypre_IJParCSRMatrix_Impl.h"
#include "Hypre_IJParCSRVector.h"
#include "Hypre_IJParCSRVector_Impl.h"
#include "Hypre_BoomerAMG.h"
#include "Hypre_BoomerAMG_Impl.h"
#include "Hypre_ParCSRDiagScale.h"
#include "Hypre_ParCSRDiagScale_Impl.h"
#include <assert.h>
#include "mpi.h"

/* >>> To do: impl_Hypre_GMRES_Copy_Parameters_from_HYPRE_struct (see
 * comments in Hypre_PCG_Impl.c). */

int impl_Hypre_GMRES_Copy_Parameters_to_HYPRE_struct( Hypre_GMRES self )
/* Copy parameter cache from the Hypre_GMRES__data object into the
 * HYPRE_Solver object */
/* >>> Possible BUG if this routine is not called earlier (at
 * initialization).  See comment in Hypre_PCG_Impl.c */
{
   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_GMRES__data * data;

   data = Hypre_GMRES__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   /* double parameters: */
   ierr += HYPRE_GMRESSetTol( solver, data->tol );

   /* int parameters: */
   ierr += HYPRE_GMRESSetKDim( solver, data->k_dim );
   ierr += HYPRE_GMRESSetMaxIter( solver, data->max_iter );
   ierr += HYPRE_GMRESSetMinIter( solver, data->min_iter );
   ierr += HYPRE_GMRESSetRelChange( solver, data->rel_change );
   ierr += HYPRE_GMRESSetStopCrit( solver, data->stop_crit );

   ierr += HYPRE_GMRESSetPrintLevel( solver, data->printlevel );
   ierr += HYPRE_GMRESSetLogging( solver, data->log_level );

   return ierr;
}

/* DO-NOT-DELETE splicer.end(Hypre.GMRES._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES__ctor"

void
impl_Hypre_GMRES__ctor(
  Hypre_GMRES self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct Hypre_GMRES__data * data;
   data = hypre_CTAlloc( struct Hypre_GMRES__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> solver = NULL;
   data -> matrix = NULL;
   data -> vector_type = NULL;
   /* We would like to call HYPRE_<vector type>GMRESCreate at this
    * point, but it's impossible until we know the vector type.
    * That's needed because the C-language Krylov solvers need to be
    * told exactly what functions to call.  If we were to switch to a
    * Babel-based GMRES solver, we would be able to use generic
    * function names; hence we could really initialize GMRES here. */

   /* default values (copied from gmres.c; better to get them by
    * function calls)...*/
   data -> tol        = 1.0e-06;
   data -> k_dim      = 5;
   data -> min_iter   = 0;
   data -> max_iter   = 1000;
   data -> rel_change = 0;
   data -> stop_crit  = 0; /* rel. residual norm */

   /* set any other data components here */

   Hypre_PCG__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES__dtor"

void
impl_Hypre_GMRES__dtor(
  Hypre_GMRES self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );

   if ( data->vector_type == "ParVector" )
   {
      ierr += HYPRE_ParGMRESDestroy( data->solver );
   }
   /* To Do: support more vector types */
   else
   {
      /* Unsupported vector type.  We're unlikely to reach this point. */
      ierr++;
   }
   Hypre_Operator_deleteRef( data->matrix );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetCommunicator"

int32_t
impl_Hypre_GMRES_SetCommunicator(
  Hypre_GMRES self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );
   data -> comm = (MPI_Comm) mpi_comm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetIntParameter"

int32_t
impl_Hypre_GMRES_SetIntParameter(
  Hypre_GMRES self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   /* The normal way to implement this function would be to call the
    * corresponding HYPRE function to set the parameter.  That can't
    * always be done because the HYPRE struct may not exist.  The
    * HYPRE struct may not exist because it can't be created until we
    * know the vector type - and that is not known until Apply is
    * first called.  So what we do is save the parameter in a cache
    * belonging to this Babel interface, and copy it into the HYPRE
    * struct once Apply is called.  */
   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );

   if ( strcmp(name,"KDim")==0 || strcmp(name,"K Dim")==0 )
   {
      data -> k_dim = value;
   }
   else if ( strcmp(name,"Min Iter")==0 )
   {
      data -> min_iter = value;
   }
   else if ( strcmp(name,"Rel Change")==0 )
   {
      data -> rel_change = value;
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetDoubleParameter"

int32_t
impl_Hypre_GMRES_SetDoubleParameter(
  Hypre_GMRES self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   /* The normal way to implement this function would be to call the
    * corresponding HYPRE function to set the parameter.  That can't
    * always be done because the HYPRE struct may not exist.  The
    * HYPRE struct may not exist because it can't be created until we
    * know the vector type - and that is not known until Apply is
    * first called.  So what we do is save the parameter in a cache
    * belonging to this Babel interface, and copy it into the HYPRE
    * struct once Apply is called.  */
   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );

   ierr = 1;

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetStringParameter"

int32_t
impl_Hypre_GMRES_SetStringParameter(
  Hypre_GMRES self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetStringParameter) */
}

/*
 * Set the int array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetIntArrayParameter"

int32_t
impl_Hypre_GMRES_SetIntArrayParameter(
  Hypre_GMRES self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetIntArrayParameter) */
}

/*
 * Set the double array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetDoubleArrayParameter"

int32_t
impl_Hypre_GMRES_SetDoubleArrayParameter(
  Hypre_GMRES self, const char* name, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetDoubleArrayParameter) */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetDoubleArrayParameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetIntValue"

int32_t
impl_Hypre_GMRES_GetIntValue(
  Hypre_GMRES self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   /* >>> We should add a Get for everything in SetParameter.  There
    * are two values for each parameter: the Hypre cache, and the
    * HYPRE value.  The cache gets copied to HYPRE when Apply is
    * called.  What we want to return is the cache value if the
    * corresponding Set had been called, otherwise the real (HYPRE)
    * value.  Assuming the HYPRE interface is not used simultaneously
    * with the Babel interface, it is sufficient to initialize the
    * cache with whatever HYPRE is using. */
   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_GMRES__data * data;

   data = Hypre_GMRES__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   ierr = 1;

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetDoubleValue"

int32_t
impl_Hypre_GMRES_GetDoubleValue(
  Hypre_GMRES self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   /* >>> We should add a Get for everything in SetParameter.  There
    * are two values for each parameter: the Hypre cache, and the
    * HYPRE value.  The cache gets copied to HYPRE when Apply is
    * called.  What we want to return is the cache value if the
    * corresponding Set had been called, otherwise the real (HYPRE)
    * value.  Assuming the HYPRE interface is not used simultaneously
    * with the Babel interface, it is sufficient to initialize the
    * cache with whatever HYPRE is using. */
   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_GMRES__data * data;

   data = Hypre_GMRES__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   ierr = 1;

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_Setup"

int32_t
impl_Hypre_GMRES_Setup(
  Hypre_GMRES self, Hypre_Vector b, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr=0;
   MPI_Comm comm;
   HYPRE_Solver solver;
   HYPRE_Solver * psolver = &solver; /* will get a real value later */
   struct Hypre_GMRES__data * data;
   Hypre_Operator mat;
   HYPRE_Matrix HYPRE_A;
   Hypre_IJParCSRMatrix HypreP_A;
   HYPRE_ParCSRMatrix AA;
   HYPRE_IJMatrix ij_A;
   HYPRE_Vector HYPRE_x, HYPRE_b;
   Hypre_IJParCSRVector HypreP_b, HypreP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;
   struct Hypre_IJParCSRMatrix__data * dataA;
   struct Hypre_IJParCSRVector__data * datab, * datax;
   void * objectA, * objectb, * objectx;

   data = Hypre_GMRES__get_data( self );
   comm = data->comm;
   /* SetCommunicator should have been called earlier */
   assert( comm != MPI_COMM_NULL );
   mat = data->matrix;
   /* SetOperator should have been called earlier */
   assert( mat != NULL );

   if ( data -> vector_type == NULL )
   {
      /* This is the first time this Babel GMRES object has seen a
       * vector.  So we are ready to create the Hypre GMRES object. */
      if ( Hypre_Vector_queryInt( b, "Hypre.IJParCSRVector") )
      {
         data -> vector_type = "ParVector";
         HYPRE_ParCSRGMRESCreate( comm, psolver );
         assert( solver != NULL );
         data -> solver = *psolver;
      }
      /* Add more vector types here */
      else
      {
         assert( "only IJParCSRVector supported by GMRES"==0 );
      }
      Hypre_GMRES__set_data( self, data );
   }
   else
   {
      solver = data->solver;
      assert( solver != NULL );
   }
   /* The SetParameter functions set parameters in the local
    * Babel-interface struct, "data".  That is because the HYPRE
    * struct (where they are actually used) may not exist yet when the
    * functions are called.  At this point we finally know the HYPRE
    * struct exists, so we copy the parameters to it. */
   ierr += impl_Hypre_GMRES_Copy_Parameters_to_HYPRE_struct( self );
   if ( data->vector_type == "ParVector" )
   {
      HypreP_b = Hypre_IJParCSRVector__cast
         ( Hypre_Vector_queryInt( b, "Hypre.IJParCSRVector") );
      datab = Hypre_IJParCSRVector__get_data( HypreP_b );
      ij_b = datab -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
      bb = (HYPRE_ParVector) objectb;
      HYPRE_b = (HYPRE_Vector) bb;

      HypreP_x = Hypre_IJParCSRVector__cast
         ( Hypre_Vector_queryInt( x, "Hypre.IJParCSRVector") );
      datax = Hypre_IJParCSRVector__get_data( HypreP_x );
      ij_x = datax -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
      xx = (HYPRE_ParVector) objectx;
      HYPRE_x = (HYPRE_Vector) xx;

      HypreP_A = Hypre_IJParCSRMatrix__cast
         ( Hypre_Operator_queryInt( mat, "Hypre.IJParCSRMatrix") );
      assert( HypreP_A != NULL );
      dataA = Hypre_IJParCSRMatrix__get_data( HypreP_A );
      ij_A = dataA -> ij_A;
      ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
      AA = (HYPRE_ParCSRMatrix) objectA;
      HYPRE_A = (HYPRE_Matrix) AA;
   }
   else
   {
      assert( "only IJParCSRVector supported by GMRES"==0 );
   }
      
   ierr += HYPRE_GMRESSetPrecond( solver, data->precond, data->precond_setup,
                                  *(data->solverprecond) );
   HYPRE_GMRESSetup( solver, HYPRE_A, HYPRE_b, HYPRE_x );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_Apply"

int32_t
impl_Hypre_GMRES_Apply(
  Hypre_GMRES self, Hypre_Vector b, Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.Apply) */
  /* Insert the implementation of the Apply method here... */

   /* In the long run, the solver should be implemented right here,
    * calling the appropriate Hypre functions.  But for now we are
    * calling the existing HYPRE solver.  Advantages: don't want to
    * have two versions of the same GMRES solver lying around.
    * Disadvantage: we have to cache user-supplied parameters until
    * the Apply call, where we make the GMRES object and really set
    * the parameters - messy and unnatural. */
   int ierr=0;
   MPI_Comm comm;
   HYPRE_Solver solver;
   HYPRE_Solver * psolver = &solver; /* will get a real value later */
   struct Hypre_GMRES__data * data;
   Hypre_Operator mat;
   HYPRE_Matrix HYPRE_A;
   Hypre_IJParCSRMatrix HypreP_A;
   HYPRE_ParCSRMatrix AA;
   HYPRE_IJMatrix ij_A;
   HYPRE_Vector HYPRE_x, HYPRE_b;
   Hypre_IJParCSRVector HypreP_b, HypreP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;
   struct Hypre_IJParCSRMatrix__data * dataA;
   struct Hypre_IJParCSRVector__data * datab, * datax;
   void * objectA, * objectb, * objectx;

   data = Hypre_GMRES__get_data( self );
   comm = data->comm;
   /* SetCommunicator should have been called earlier */
   assert( comm != MPI_COMM_NULL );
   mat = data->matrix;
   /* SetOperator should have been called earlier */
   assert( mat != NULL );

   if ( data -> vector_type == NULL )
   {
      /* This is the first time this Babel GMRES object has seen a
       * vector.  So we are ready to create the Hypre GMRES object. */
      if ( Hypre_Vector_queryInt( b, "Hypre.IJParCSRVector") )
      {
         data -> vector_type = "ParVector";
         HYPRE_ParCSRGMRESCreate( comm, psolver );
         assert( solver != NULL );
         data -> solver = *psolver;
      }
      /* Add more vector types here */
      else
      {
         assert( "only IJParCSRVector supported by GMRES"==0 );
      }
      Hypre_GMRES__set_data( self, data );
   }
   else
   {
      solver = data->solver;
      assert( solver != NULL );
   }
   /* The SetParameter functions set parameters in the local
    * Babel-interface struct, "data".  That is because the HYPRE
    * struct (where they are actually used) may not exist yet when the
    * functions are called.  At this point we finally know the HYPRE
    * struct exists, so we copy the parameters to it. */
   ierr += impl_Hypre_GMRES_Copy_Parameters_to_HYPRE_struct( self );
   if ( data->vector_type == "ParVector" )
   {
      HypreP_b = Hypre_IJParCSRVector__cast
         ( Hypre_Vector_queryInt( b, "Hypre.IJParCSRVector") );
      datab = Hypre_IJParCSRVector__get_data( HypreP_b );
      ij_b = datab -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
      bb = (HYPRE_ParVector) objectb;
      HYPRE_b = (HYPRE_Vector) bb;

      HypreP_x = Hypre_IJParCSRVector__cast
         ( Hypre_Vector_queryInt( *x, "Hypre.IJParCSRVector") );
      datax = Hypre_IJParCSRVector__get_data( HypreP_x );
      ij_x = datax -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
      xx = (HYPRE_ParVector) objectx;
      HYPRE_x = (HYPRE_Vector) xx;

      HypreP_A = Hypre_IJParCSRMatrix__cast
         ( Hypre_Operator_queryInt( mat, "Hypre.IJParCSRMatrix") );
      assert( HypreP_A != NULL );
      dataA = Hypre_IJParCSRMatrix__get_data( HypreP_A );
      ij_A = dataA -> ij_A;
      ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
      AA = (HYPRE_ParCSRMatrix) objectA;
      HYPRE_A = (HYPRE_Matrix) AA;
   }
   else
   {
      assert( "only IJParCSRVector supported by GMRES"==0 );
   }
      
   ierr += HYPRE_GMRESSetPrecond( solver, data->precond, data->precond_setup,
                                  *(data->solverprecond) );

   HYPRE_GMRESSolve( solver, HYPRE_A, HYPRE_b, HYPRE_x );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.Apply) */
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetOperator"

int32_t
impl_Hypre_GMRES_SetOperator(
  Hypre_GMRES self, Hypre_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
   int ierr = 0;
   struct Hypre_GMRES__data * data;

   data = Hypre_GMRES__get_data( self );
   data->matrix = A;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetTolerance"

int32_t
impl_Hypre_GMRES_SetTolerance(
  Hypre_GMRES self, double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );

   data -> tol = tolerance;

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetMaxIterations"

int32_t
impl_Hypre_GMRES_SetMaxIterations(
  Hypre_GMRES self, int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );

   data -> max_iter = max_iterations;

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetMaxIterations) */
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
#define __FUNC__ "impl_Hypre_GMRES_SetLogging"

int32_t
impl_Hypre_GMRES_SetLogging(
  Hypre_GMRES self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */
   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );

   data -> log_level = level;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetLogging) */
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
#define __FUNC__ "impl_Hypre_GMRES_SetPrintLevel"

int32_t
impl_Hypre_GMRES_SetPrintLevel(
  Hypre_GMRES self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */

   /* The normal way to implement this function would be to call the
    * corresponding HYPRE function to set the print level.  That can't
    * always be done because the HYPRE struct may not exist.  The
    * HYPRE struct may not exist because it can't be created until we
    * know the vector type - and that is not known until Apply is
    * first called.  So what we do is save the print level in a cache
    * belonging to this Babel interface, and copy it into the HYPRE
    * struct once Apply is called.  */
   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );

   data -> printlevel = level;

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetNumIterations"

int32_t
impl_Hypre_GMRES_GetNumIterations(
  Hypre_GMRES self, int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_GMRES__data * data;

   data = Hypre_GMRES__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   ierr += HYPRE_GMRESGetNumIterations( solver, num_iterations );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetRelResidualNorm"

int32_t
impl_Hypre_GMRES_GetRelResidualNorm(
  Hypre_GMRES self, double* norm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_GMRES__data * data;

   data = Hypre_GMRES__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   ierr += HYPRE_GMRESGetFinalRelativeResidualNorm( solver, norm );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetRelResidualNorm) */
}

/*
 * Set the preconditioner.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetPreconditioner"

int32_t
impl_Hypre_GMRES_SetPreconditioner(
  Hypre_GMRES self, Hypre_Solver s)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetPreconditioner) */
  /* Insert the implementation of the SetPreconditioner method here... */

   int ierr = 0;
   HYPRE_Solver * solverprecond;
   struct Hypre_GMRES__data * dataself;
   struct Hypre_BoomerAMG__data * AMG_dataprecond;
   Hypre_BoomerAMG AMG_s;
   HYPRE_PtrToSolverFcn precond, precond_setup; /* functions */

   dataself = Hypre_GMRES__get_data( self );

   if ( Hypre_Solver_queryInt( s, "Hypre.BoomerAMG" ) )
   {
      /* s is a Hypre_BoomerAMG */
      AMG_s = Hypre_BoomerAMG__cast
         ( Hypre_Solver_queryInt( s, "Hypre.BoomerAMG") );
      AMG_dataprecond = Hypre_BoomerAMG__get_data( AMG_s );
      solverprecond = &AMG_dataprecond->solver;
      assert( solverprecond != NULL );
      precond = (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve;
      precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup;
   }
   else if ( Hypre_Solver_queryInt( s, "Hypre.ParCSRDiagScale" ) )
   {
      solverprecond = (HYPRE_Solver *) hypre_CTAlloc( double, 1 );
      /* ... HYPRE diagonal scaling needs no solver object, but we
       * must provide a HYPRE_Solver object.  It will be totally
       * ignored. */
      precond = (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale;
      precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup;
   }
   /* put other preconditioner types here */
   else
   {
      assert( "GMRES_SetPreconditioner cannot recognize preconditioner"==0 );
   }

   /* We can't actually set the HYPRE preconditioner, because that
    * requires knowing what the solver object is - but that requires
    * knowing its data type but _that_ requires knowing the kind of
    * matrix and vectors we'll need; not known until Apply is called.
    * So save the information in the Hypre data structure, and stick
    * it in HYPRE later... */
   dataself->precond = precond;
   dataself->precond_setup = precond_setup;
   dataself->solverprecond = solverprecond;
   /* For an example call, see test/IJ_linear_solvers.c, line 1686.
    * The four arguments are: self's (solver) data; and, for the
    * preconditioner: solver function, setup function, data */

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetPreconditioner) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
