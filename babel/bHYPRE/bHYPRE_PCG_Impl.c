/*
 * File:          bHYPRE_PCG_Impl.c
 * Symbol:        bHYPRE.PCG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:29 PST
 * Description:   Server-side implementation for bHYPRE.PCG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1237
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.PCG" (version 1.0.0)
 * 
 * Objects of this type can be cast to PreconditionedSolver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 */

#include "bHYPRE_PCG_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._includes) */
/* Put additional includes or other arbitrary code here... */
#include "bHYPRE_IJParCSRMatrix.h"
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "bHYPRE_BoomerAMG.h"
#include "bHYPRE_BoomerAMG_Impl.h"
#include "bHYPRE_ParCSRDiagScale.h"
#include "bHYPRE_ParCSRDiagScale_Impl.h"
#include <assert.h>
#include "mpi.h"

/* This can't be implemented until the HYPRE_PCG Get functions are implemented.
 * But this function should be used to initialize the parameter cache
 * in the bHYPRE_PCG__data object, so that we can have bHYPRE_PCG Get
 * functions for all settable parameters...
int impl_bHYPRE_PCG_Copy_Parameters_from_HYPRE_struct( bHYPRE_PCG self )
{
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_PCG__data * data;

   data = bHYPRE_PCG__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   / * double parameters: * /
   ierr += HYPRE_PCGGetTol( solver, &(data->tol) );
   ierr += HYPRE_PCGGetAbsoluteTolFactor( solver, &(data->atolf) );
   ierr += HYPRE_PCGGetConvergenceFactorTol( solver, &(data->cf_tol) );

   / * int parameters: * /
   ierr += HYPRE_PCGGetMaxIter( solver, &(data->maxiter) );
   ierr += HYPRE_PCGGetRelChange( solver, &(data->relchange) );
   ierr += HYPRE_PCGGetTwoNorm( solver, &(data->twonorm) );
   ierr += HYPRE_PCGGetStopCrit( solver, &(data->stop_crit) );

   ierr += HYPRE_PCGGetPrintLevel( solver, &(data->printlevel) );
   ierr += HYPRE_PCGGetLogging( solver, *(data->log_level) );

   return ierr;
}
*/

int impl_bHYPRE_PCG_Copy_Parameters_to_HYPRE_struct( bHYPRE_PCG self )
/* Copy parameter cache from the bHYPRE_PCG__data object into the
 * HYPRE_Solver object */
/* >>> Possible BUG: If the default (initial) values in the HYPRE code
 * are different from those used in the Babel interface, and the user
 * didn't set everything, calling this function will give the wrong
 * defaults (defining "correct" defaults to be those used in the HYPRE
 * interface).  The solution is to intialize the Babel PCG parameters
 * by calling HYPRE-level Get functions (which haven't been written
 * yet). */
{
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_PCG__data * data;

   data = bHYPRE_PCG__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   /* double parameters: */
   ierr += HYPRE_PCGSetTol( solver, data->tol );
   ierr += HYPRE_PCGSetAbsoluteTolFactor( solver,
                                          data->atolf );
   ierr += HYPRE_PCGSetConvergenceFactorTol( solver,
                                             data->cf_tol );

   /* int parameters: */
   ierr += HYPRE_PCGSetMaxIter( solver, data->maxiter );
   ierr += HYPRE_PCGSetRelChange( solver, data->relchange );
   ierr += HYPRE_PCGSetTwoNorm( solver, data->twonorm );
   ierr += HYPRE_PCGSetStopCrit( solver, data->stop_crit );

   ierr += HYPRE_PCGSetPrintLevel( solver, data->printlevel );
   ierr += HYPRE_PCGSetLogging( solver, data->log_level );

   return ierr;
}
/* DO-NOT-DELETE splicer.end(bHYPRE.PCG._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG__ctor"

void
impl_bHYPRE_PCG__ctor(
  bHYPRE_PCG self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct bHYPRE_PCG__data * data;
   data = hypre_CTAlloc( struct bHYPRE_PCG__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> solver = NULL;
   data -> matrix = NULL;
   data -> vector_type = NULL;
   /* We would like to call HYPRE_<vector type>PCGCreate at this
    * point, but it's impossible until we know the vector type.
    * That's needed because the C-language Krylov solvers need to be
    * told exactly what functions to call.  If we were to switch to a
    * Babel-based PCG solver, we would be able to use generic function
    * names; hence we could really initialize PCG here. */

   /* default values (copied from pcg.c; better to get them by
    * function calls)...*/
   data -> tol = 1.0e-6;
   data -> atolf = 0.0;
   data -> cf_tol = 0.0;
   data -> maxiter = 1000;
   data -> relchange = 0;
   data -> twonorm = 0;
   data ->log_level = 0;
   data -> printlevel = 0;
   data -> stop_crit = 0;

   /* set any other data components here */
   bHYPRE_PCG__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG__dtor"

void
impl_bHYPRE_PCG__dtor(
  bHYPRE_PCG self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data;
   data = bHYPRE_PCG__get_data( self );

   if ( data->vector_type == "ParVector" )
   {
      ierr += HYPRE_ParCSRPCGDestroy( data->solver );
   }
   /* To Do: support more vector types */
   else
   {
      /* Unsupported vector type.  We're unlikely to reach this point. */
      ierr++;
   }
   bHYPRE_Operator_deleteRef( data->matrix );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetCommunicator"

int32_t
impl_bHYPRE_PCG_SetCommunicator(
  bHYPRE_PCG self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data;
   data = bHYPRE_PCG__get_data( self );
   data -> comm = (MPI_Comm) mpi_comm;

   return ierr;
  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetIntParameter"

int32_t
impl_bHYPRE_PCG_SetIntParameter(
  bHYPRE_PCG self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetIntParameter) */
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
   struct bHYPRE_PCG__data * data;
   data = bHYPRE_PCG__get_data( self );

   if ( strcmp(name,"TwoNorm")==0 )
   {
      data -> twonorm = value;
   }
   else if ( strcmp(name,"RelChange")==0 )
   {
      data -> relchange = value;
   }
   else
   {
      ierr=1;
   }

   return ierr;
  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetDoubleParameter"

int32_t
impl_bHYPRE_PCG_SetDoubleParameter(
  bHYPRE_PCG self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetDoubleParameter) */
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
   struct bHYPRE_PCG__data * data;
   data = bHYPRE_PCG__get_data( self );

   if ( strcmp(name,"AbsoluteTolFactor")==0 )
   {
      data -> atolf = value;
   }
   else if ( strcmp(name,"ConvergenceFactorTol")==0 )
   {
      /* tolerance for special test for slow convergence */
      data -> cf_tol = value;
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetStringParameter"

int32_t
impl_bHYPRE_PCG_SetStringParameter(
  bHYPRE_PCG self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetIntArray1Parameter"

int32_t
impl_bHYPRE_PCG_SetIntArray1Parameter(
  bHYPRE_PCG self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetIntArray2Parameter"

int32_t
impl_bHYPRE_PCG_SetIntArray2Parameter(
  bHYPRE_PCG self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetDoubleArray1Parameter"

int32_t
impl_bHYPRE_PCG_SetDoubleArray1Parameter(
  bHYPRE_PCG self, const char* name, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetDoubleArray2Parameter"

int32_t
impl_bHYPRE_PCG_SetDoubleArray2Parameter(
  bHYPRE_PCG self, const char* name, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_GetIntValue"

int32_t
impl_bHYPRE_PCG_GetIntValue(
  bHYPRE_PCG self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   /* >>> We should add a Get for everything in SetParameter.  There
    * are two values for each parameter: the bHYPRE cache, and the
    * HYPRE value.  The cache gets copied to HYPRE when Apply is
    * called.  What we want to return is the cache value if the
    * corresponding Set had been called, otherwise the real (HYPRE)
    * value.  Assuming the HYPRE interface is not used simultaneously
    * with the Babel interface, it is sufficient to initialize the
    * cache with whatever HYPRE is using. */
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_PCG__data * data;

   data = bHYPRE_PCG__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   ierr=1;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_GetDoubleValue"

int32_t
impl_bHYPRE_PCG_GetDoubleValue(
  bHYPRE_PCG self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   /* >>> We should add a Get for everything in SetParameter.  There
    * are two values for each parameter: the bHYPRE cache, and the
    * HYPRE value.  The cache gets copied to HYPRE when Apply is
    * called.  What we want to return is the cache value if the
    * corresponding Set had been called, otherwise the real (HYPRE)
    * value.  Assuming the HYPRE interface is not used simultaneously
    * with the Babel interface, it is sufficient to initialize the
    * cache with whatever HYPRE is using. */
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_PCG__data * data;

   data = bHYPRE_PCG__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   ierr = 1;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_Setup"

int32_t
impl_bHYPRE_PCG_Setup(
  bHYPRE_PCG self, bHYPRE_Vector b, bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr=0;
   MPI_Comm comm;
   HYPRE_Solver solver;
   HYPRE_Solver * psolver = &solver; /* will get a real value later */
   struct bHYPRE_PCG__data * data;
   bHYPRE_Operator mat;
   HYPRE_Matrix HYPRE_A;
   bHYPRE_IJParCSRMatrix bHYPREP_A;
   HYPRE_ParCSRMatrix AA;
   HYPRE_IJMatrix ij_A;
   HYPRE_Vector HYPRE_x, HYPRE_b;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   void * objectA, * objectb, * objectx;

   data = bHYPRE_PCG__get_data( self );
   comm = data->comm;
   /* SetCommunicator should have been called earlier */
   assert( comm != MPI_COMM_NULL );
   mat = data->matrix;
   /* SetOperator should have been called earlier */
   assert( mat != NULL );

   if ( data -> vector_type == NULL )
   {
      /* This is the first time this Babel PCG object has seen a
       * vector.  So we are ready to create the bHYPRE PCG object. */
      if ( bHYPRE_Vector_queryInt( b, "bHYPRE.IJParCSRVector") )
      {
         bHYPRE_Vector_deleteRef( b );  /* extra ref created by queryInt */
         data -> vector_type = "ParVector";
         HYPRE_ParCSRPCGCreate( comm, psolver );
         assert( solver != NULL );
         data -> solver = *psolver;
      }
      /* Add more vector types here */
      else
      {
         assert( "only IJParCSRVector supported by PCG"==0 );
      }
      bHYPRE_PCG__set_data( self, data );
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
   ierr += impl_bHYPRE_PCG_Copy_Parameters_to_HYPRE_struct( self );
   if ( data->vector_type == "ParVector" )
   {
      bHYPREP_b = bHYPRE_IJParCSRVector__cast
         ( bHYPRE_Vector_queryInt( b, "bHYPRE.IJParCSRVector") );
      datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
      bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b ); /* extra reference from queryInt */
      ij_b = datab -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
      bb = (HYPRE_ParVector) objectb;
      HYPRE_b = (HYPRE_Vector) bb;

      bHYPREP_x = bHYPRE_IJParCSRVector__cast
         ( bHYPRE_Vector_queryInt( x, "bHYPRE.IJParCSRVector") );
      datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
      bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x ); /* extra reference from queryInt */
      ij_x = datax -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
      xx = (HYPRE_ParVector) objectx;
      HYPRE_x = (HYPRE_Vector) xx;

      bHYPREP_A = bHYPRE_IJParCSRMatrix__cast
         ( bHYPRE_Operator_queryInt( mat, "bHYPRE.IJParCSRMatrix") );
      assert( bHYPREP_A != NULL );
      dataA = bHYPRE_IJParCSRMatrix__get_data( bHYPREP_A );
      bHYPRE_IJParCSRMatrix_deleteRef( bHYPREP_A ); /* extra reference from queryInt */
      ij_A = dataA -> ij_A;
      ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
      AA = (HYPRE_ParCSRMatrix) objectA;
      HYPRE_A = (HYPRE_Matrix) AA;
   }
   else
   {
      assert( "only IJParCSRVector supported by PCG"==0 );
   }
      
   ierr += HYPRE_PCGSetPrecond( solver, data->precond, data->precond_setup,
                                *(data->solverprecond) );
   HYPRE_PCGSetup( solver, HYPRE_A, HYPRE_b, HYPRE_x );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_Apply"

int32_t
impl_bHYPRE_PCG_Apply(
  bHYPRE_PCG self, bHYPRE_Vector b, bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.Apply) */
  /* Insert the implementation of the Apply method here... */

   /* In the long run, the solver should be implemented right here,
    * calling the appropriate bHYPRE functions.  But for now we are
    * calling the existing HYPRE solver.  Advantages: don't want to
    * have two versions of the same PCG solver lying around.
    * Disadvantage: we have to cache user-supplied parameters until
    * the Apply call, where we make the PCG object and really set the
    * parameters - messy and unnatural. */
   int ierr=0;
   MPI_Comm comm;
   HYPRE_Solver solver;
   HYPRE_Solver * psolver = &solver; /* will get a real value later */
   struct bHYPRE_PCG__data * data;
   bHYPRE_Operator mat;
   HYPRE_Matrix HYPRE_A;
   bHYPRE_IJParCSRMatrix bHYPREP_A;
   HYPRE_ParCSRMatrix AA;
   HYPRE_IJMatrix ij_A;
   HYPRE_Vector HYPRE_x, HYPRE_b;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   void * objectA, * objectb, * objectx;

   data = bHYPRE_PCG__get_data( self );
   comm = data->comm;
   /* SetCommunicator should have been called earlier */
   assert( comm != MPI_COMM_NULL );
   mat = data->matrix;
   /* SetOperator should have been called earlier */
   assert( mat != NULL );

   if ( data -> vector_type == NULL )
   {
      /* This is the first time this Babel PCG object has seen a
       * vector.  So we are ready to create the bHYPRE PCG object. */
      if ( bHYPRE_Vector_queryInt( b, "bHYPRE.IJParCSRVector") )
      {
         bHYPRE_Vector_deleteRef( b ); /* extra ref created by queryInt */
         data -> vector_type = "ParVector";
         HYPRE_ParCSRPCGCreate( comm, psolver );
         assert( solver != NULL );
         data -> solver = *psolver;
      }
      /* Add more vector types here */
      else
      {
         assert( "only IJParCSRVector supported by PCG"==0 );
      }
      bHYPRE_PCG__set_data( self, data );
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
   ierr += impl_bHYPRE_PCG_Copy_Parameters_to_HYPRE_struct( self );
   if ( data->vector_type == "ParVector" )
   {
      bHYPREP_b = bHYPRE_IJParCSRVector__cast
         ( bHYPRE_Vector_queryInt( b, "bHYPRE.IJParCSRVector") );
      datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
      bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b ); /* extra ref created by queryInt */
      ij_b = datab -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
      bb = (HYPRE_ParVector) objectb;
      HYPRE_b = (HYPRE_Vector) bb;

      bHYPREP_x = bHYPRE_IJParCSRVector__cast
         ( bHYPRE_Vector_queryInt( *x, "bHYPRE.IJParCSRVector") );
      datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
      bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x ); /* extra ref created by queryInt */
      ij_x = datax -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
      xx = (HYPRE_ParVector) objectx;
      HYPRE_x = (HYPRE_Vector) xx;

      bHYPREP_A = bHYPRE_IJParCSRMatrix__cast
         ( bHYPRE_Operator_queryInt( mat, "bHYPRE.IJParCSRMatrix") );
      assert( bHYPREP_A != NULL );
      dataA = bHYPRE_IJParCSRMatrix__get_data( bHYPREP_A );
      bHYPRE_IJParCSRMatrix_deleteRef( bHYPREP_A ); /* extra ref created by queryInt */
      ij_A = dataA -> ij_A;
      ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
      AA = (HYPRE_ParCSRMatrix) objectA;
      HYPRE_A = (HYPRE_Matrix) AA;
   }
   else
   {
      assert( "only IJParCSRVector supported by PCG"==0 );
   }
      
   ierr += HYPRE_PCGSetPrecond( solver, data->precond, data->precond_setup,
                                *(data->solverprecond) );

   HYPRE_PCGSolve( solver, HYPRE_A, HYPRE_b, HYPRE_x );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.Apply) */
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetOperator"

int32_t
impl_bHYPRE_PCG_SetOperator(
  bHYPRE_PCG self, bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data;

   data = bHYPRE_PCG__get_data( self );
   data->matrix = A;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetTolerance"

int32_t
impl_bHYPRE_PCG_SetTolerance(
  bHYPRE_PCG self, double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data;
   data = bHYPRE_PCG__get_data( self );

   data -> tol = tolerance;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetMaxIterations"

int32_t
impl_bHYPRE_PCG_SetMaxIterations(
  bHYPRE_PCG self, int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data;
   data = bHYPRE_PCG__get_data( self );

   data -> maxiter = max_iterations;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_PCG_SetLogging"

int32_t
impl_bHYPRE_PCG_SetLogging(
  bHYPRE_PCG self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */

   /* The normal way to implement this function would be to call the
    * corresponding HYPRE function to set the print level.  That can't
    * always be done because the HYPRE struct may not exist.  The
    * HYPRE struct may not exist because it can't be created until we
    * know the vector type - and that is not known until Apply is
    * first called.  So what we do is save the print level in a cache
    * belonging to this Babel interface, and copy it into the HYPRE
    * struct once Apply is called.  */
   int ierr = 0;
   struct bHYPRE_PCG__data * data;
   data = bHYPRE_PCG__get_data( self );

   data -> log_level = level;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_PCG_SetPrintLevel"

int32_t
impl_bHYPRE_PCG_SetPrintLevel(
  bHYPRE_PCG self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetPrintLevel) */
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
   struct bHYPRE_PCG__data * data;
   data = bHYPRE_PCG__get_data( self );

   data -> printlevel = level;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_GetNumIterations"

int32_t
impl_bHYPRE_PCG_GetNumIterations(
  bHYPRE_PCG self, int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_PCG__data * data;

   data = bHYPRE_PCG__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   ierr += HYPRE_PCGGetNumIterations( solver, num_iterations );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_GetRelResidualNorm"

int32_t
impl_bHYPRE_PCG_GetRelResidualNorm(
  bHYPRE_PCG self, double* norm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_PCG__data * data;

   data = bHYPRE_PCG__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   ierr += HYPRE_PCGGetFinalRelativeResidualNorm( solver, norm );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.GetRelResidualNorm) */
}

/*
 * Set the preconditioner.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetPreconditioner"

int32_t
impl_bHYPRE_PCG_SetPreconditioner(
  bHYPRE_PCG self, bHYPRE_Solver s)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetPreconditioner) */
  /* Insert the implementation of the SetPreconditioner method here... */

   int ierr = 0;
   HYPRE_Solver * solverprecond;
   struct bHYPRE_PCG__data * dataself;
   struct bHYPRE_BoomerAMG__data * AMG_dataprecond;
   bHYPRE_BoomerAMG AMG_s;
   HYPRE_PtrToSolverFcn precond, precond_setup; /* functions */

   dataself = bHYPRE_PCG__get_data( self );

   if ( bHYPRE_Solver_queryInt( s, "bHYPRE.BoomerAMG" ) )
   {
      /* s is a bHYPRE_BoomerAMG */
      AMG_s = bHYPRE_BoomerAMG__cast
         ( bHYPRE_Solver_queryInt( s, "bHYPRE.BoomerAMG") );
      AMG_dataprecond = bHYPRE_BoomerAMG__get_data( AMG_s );
      solverprecond = &AMG_dataprecond->solver;
      assert( solverprecond != NULL );
      precond = (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve;
      precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup;
      bHYPRE_BoomerAMG_deleteRef( AMG_s ); /* extra reference from queryInt */
   }
   else if ( bHYPRE_Solver_queryInt( s, "bHYPRE.ParCSRDiagScale" ) )
   {
      bHYPRE_Solver_deleteRef( s ); /* extra reference from queryInt */
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
      assert( "PCG_SetPreconditioner cannot recognize preconditioner"==0 );
   }

   /* We can't actually set the HYPRE preconditioner, because that
    * requires knowing what the solver object is - but that requires
    * knowing its data type but _that_ requires knowing the kind of
    * matrix and vectors we'll need; not known until Apply is called.
    * So save the information in the bHYPRE data structure, and stick
    * it in HYPRE later... */
   dataself->precond = precond;
   dataself->precond_setup = precond_setup;
   dataself->solverprecond = solverprecond;
   /* For an example call, see test/IJ_linear_solvers.c, line 1686.
    * The four arguments are: self's (solver) data; and, for the
    * preconditioner: solver function, setup function, data */

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetPreconditioner) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
