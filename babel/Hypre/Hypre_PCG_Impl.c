/*
 * File:          Hypre_PCG_Impl.c
 * Symbol:        Hypre.PCG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:34 PDT
 * Description:   Server-side implementation for Hypre.PCG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.PCG" (version 0.1.5)
 */

#include "Hypre_PCG_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.PCG._includes) */
/* Put additional includes or other arbitrary code here... */
#include "Hypre_ParCSRMatrix.h"
#include "Hypre_ParCSRMatrix_Impl.h"
#include "Hypre_ParCSRVector.h"
#include "Hypre_ParCSRVector_Impl.h"
#include "Hypre_ParAMG.h"
#include "Hypre_ParAMG_Impl.h"
#include "Hypre_ParDiagScale.h"
#include "Hypre_ParDiagScale_Impl.h"
#include <assert.h>

/* This can't be implemented until the HYPRE_PCG Get functions are implemented.
   But this function should be used to initialize the parameter cache
   in the Hypre_PCG__data object, so that we can have Hypre_PCG Get
   functions for all settable parameters...
int impl_Hypre_PCG_Copy_Parameters_from_HYPRE_struct( Hypre_PCG self )
{
   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_PCG__data * data;

   data = Hypre_PCG__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   / * double parameters: * /
   ierr += HYPRE_PCGGetTol( solver, &(data->tol) );
   ierr += HYPRE_PCGGetAbsoluteTolFactor( solver,
                                          &(data->atolf) );
   ierr += HYPRE_PCGGetConvergenceFactorTol( solver,
                                             &(data->cf_tol) );

   / * int parameters: * /
   ierr += HYPRE_PCGGetMaxIter( solver, &(data->maxiter) );
   ierr += HYPRE_PCGGetRelChange( solver, &(data->relchange) );
   ierr += HYPRE_PCGGetTwoNorm( solver, &(data->twonorm) );
   ierr += HYPRE_PCGGetStopCrit( solver, &(data->stop_crit) );

   ierr += HYPRE_PCGGetPrintLevel( solver, &(data->printlevel) );
   ierr += HYPRE_PCGGetLogLevel( solver, *(data->log_level) );

   return ierr;
}
*/

int impl_Hypre_PCG_Copy_Parameters_to_HYPRE_struct( Hypre_PCG self )
/* Copy parameter cache from the Hypre_PCG__data object into the
   HYPRE_Solver object */
/* >>> Possible BUG: If the default (initial) values in the HYPRE code
   are different from those used in the Babel interface, and the user didn't
   set everything, calling this function will give the wrong defaults
   (defining "correct" defaults to be those used in the HYPRE interface).
   The solution is to intialize the Babel PCG parameters by calling HYPRE-level
   Get functions (which haven't been written yet). */
{
   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_PCG__data * data;

   data = Hypre_PCG__get_data( self );
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
   ierr += HYPRE_PCGSetLogLevel( solver, data->log_level );

   return ierr;
}
/* DO-NOT-DELETE splicer.end(Hypre.PCG._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG__ctor"

void
impl_Hypre_PCG__ctor(
  Hypre_PCG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG._ctor) */
  /* Insert the implementation of the constructor method here... */
   struct Hypre_PCG__data * data;
   data = hypre_CTAlloc( struct Hypre_PCG__data, 1 );
   data -> comm = NULL;
   data -> solver = NULL;
   data -> matrix = NULL;
   data -> vector_type = NULL;
   /* We would like to call HYPRE_<vector type>PCGCreate at this point, but
      it's impossible until we know the vector type.  That's needed because
      the C-language Krylov solvers need to be told exactly what functions
      to call.  If we were to switch to a Babel-based PCG solver, we would be
      able to use generic function names; hence we could really initialize PCG
      here. */

   /* default values (copied from pcg.c; better to get them by function calls)...*/
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
   Hypre_PCG__set_data( self, data );
  /* DO-NOT-DELETE splicer.end(Hypre.PCG._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG__dtor"

void
impl_Hypre_PCG__dtor(
  Hypre_PCG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG._dtor) */
  /* Insert the implementation of the destructor method here... */
   int ierr = 0;
   struct Hypre_PCG__data * data;
   data = Hypre_PCG__get_data( self );

   ierr += HYPRE_ParCSRPCGDestroy( data->solver );
   Hypre_Operator_deleteReference( data->matrix );
   /* delete any nontrivial data components here */
   hypre_TFree( data );
  /* DO-NOT-DELETE splicer.end(Hypre.PCG._dtor) */
}

/*
 * Method:  Apply
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_Apply"

int32_t
impl_Hypre_PCG_Apply(
  Hypre_PCG self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.Apply) */
  /* Insert the implementation of the Apply method here... */
   /* In the long run, the solver should be implemented right here, calling
      the appropriate Hypre functions.  But for now we are calling the existing
      HYPRE solver.  Advantages: don't want to have two versions of the same
      PCG solver lying around.  Disadvantage: we have to cache user-supplied
      parameters until the Apply call, where we make the PCG object and really
      set the parameters - messy and unnatural. */
   int ierr=0;
   MPI_Comm comm;
   HYPRE_Solver solver;
   HYPRE_Solver * psolver = &solver; /* will get a real value later */
   struct Hypre_PCG__data * data;
   Hypre_Operator mat;
   HYPRE_Matrix HYPRE_A;
   Hypre_ParCSRMatrix HypreP_A;
   HYPRE_ParCSRMatrix AA;
   HYPRE_IJMatrix ij_A;
   HYPRE_Vector HYPRE_y, HYPRE_x;
   Hypre_ParCSRVector HypreP_x, HypreP_y;
   HYPRE_ParVector xx, yy;
   HYPRE_IJVector ij_x, ij_y;
   struct Hypre_ParCSRMatrix__data * dataA;
   struct Hypre_ParCSRVector__data * datax, * datay;
   void * objectA, * objectx, * objecty;

   data = Hypre_PCG__get_data( self );
   comm = data->comm;
   assert( comm != NULL ); /* SetCommunicator should have been called earlier */
   mat = data->matrix;
   assert( mat != NULL ); /* SetOperator should have been called earlier */

   if ( data -> vector_type == NULL ) {
      /* This is the first time this Babel PCG object has seen a vector.
         So we are ready to create the Hypre PCG object. */
      if ( Hypre_Vector_queryInterface( x, "Hypre.ParCSRVector") ) {
         data -> vector_type = "ParVector";
         HYPRE_ParCSRPCGCreate( comm, psolver );
         assert( solver != NULL );
         data -> solver = *psolver;
      }
      /* Add more vector types here */
      else {
         assert( "only ParCSRVector supported by PCG"==0 );
      }
      Hypre_PCG__set_data( self, data );
   }
   else {
      solver = data->solver;
      assert( solver != NULL );
   };
   /* The SetParameter functions set parameters in the local Babel-interface struct,
      "data".  That is because the HYPRE struct (where they are actually used) may
      not exist yet when the functions are called.  At this point we finally know
      the HYPRE struct exists, so we copy the parameters to it. */
   ierr += impl_Hypre_PCG_Copy_Parameters_to_HYPRE_struct( self );
   if ( data->vector_type == "ParVector" ) {
         HypreP_x = Hypre_Vector__cast2
            ( Hypre_Vector_queryInterface( x, "Hypre.ParCSRVector"),
              "Hypre.ParCSRVector" );
         datax = Hypre_ParCSRVector__get_data( HypreP_x );
         ij_x = datax -> ij_b;
         ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
         xx = (HYPRE_ParVector) objectx;
         HYPRE_x = (HYPRE_Vector) xx;

         HypreP_y = Hypre_Vector__cast2
            ( Hypre_Vector_queryInterface( *y, "Hypre.ParCSRVector"),
              "Hypre.ParCSRVector" );
         datay = Hypre_ParCSRVector__get_data( HypreP_y );
         ij_y = datay -> ij_b;
         ierr += HYPRE_IJVectorGetObject( ij_y, &objecty );
         yy = (HYPRE_ParVector) objecty;
         HYPRE_y = (HYPRE_Vector) yy;

         HypreP_A = Hypre_Operator__cast2
            ( Hypre_Operator_queryInterface( mat, "Hypre.ParCSRMatrix"),
              "Hypre.ParCSRMatrix" );
         assert( HypreP_A != NULL );
         dataA = Hypre_ParCSRMatrix__get_data( HypreP_A );
         ij_A = dataA -> ij_A;
         ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
         AA = (HYPRE_ParCSRMatrix) objectA;
         HYPRE_A = (HYPRE_Matrix) AA;

   }
   else {
         assert( "only ParCSRVector supported by PCG"==0 );
   }
      
   ierr += HYPRE_PCGSetPrecond( solver, data->precond, data->precond_setup,
                                *(data->solverprecond) );
   HYPRE_PCGSetup( solver, HYPRE_A, HYPRE_x, HYPRE_y );
   HYPRE_PCGSolve( solver, HYPRE_A, HYPRE_x, HYPRE_y );

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.Apply) */
}

/*
 * Method:  GetDoubleValue
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_GetDoubleValue"

int32_t
impl_Hypre_PCG_GetDoubleValue(
  Hypre_PCG self,
  const char* name,
  double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   /* >>> We should add a Get for everything in SetParameter.
      There are two values for each parameter: the Hypre cache, and the HYPRE value.
      The cache gets copied to HYPRE when Apply is called.  What we want to return
      is the cache value if the corresponding Set had been called, otherwise the
      real (HYPRE) value.  Assuming the HYPRE interface is not used simultaneously
      with the Babel interface, it is sufficient to initialize the cache with
      whatever HYPRE is using. */
   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_PCG__data * data;

   data = Hypre_PCG__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   if ( strcmp(name,"FinalRelativeResidualNorm")==0 ||
        strcmp(name,"Final Relative Residual Norm")==0 ) {
      ierr += HYPRE_PCGGetFinalRelativeResidualNorm( solver, value );
   }
   /* Get other values here. */
   else ierr=1;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.GetDoubleValue) */
}

/*
 * Method:  GetIntValue
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_GetIntValue"

int32_t
impl_Hypre_PCG_GetIntValue(
  Hypre_PCG self,
  const char* name,
  int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   /* >>> We should add a Get for everything in SetParameter.
      There are two values for each parameter: the Hypre cache, and the HYPRE value.
      The cache gets copied to HYPRE when Apply is called.  What we want to return
      is the cache value if the corresponding Set had been called, otherwise the
      real (HYPRE) value.  Assuming the HYPRE interface is not used simultaneously
      with the Babel interface, it is sufficient to initialize the cache with
      whatever HYPRE is using. */
   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_PCG__data * data;

   data = Hypre_PCG__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   printf("data->maxiter=%i\n",data->maxiter);
   if ( strcmp(name,"NumIterations")==0 || strcmp(name,"Num Iterations")==0
      || strcmp(name,"Number of Iterations")==0 ) {
      ierr += HYPRE_PCGGetNumIterations( solver, value );
      printf("num iterations=%i",*value);
   }
   /* Get other values here. */
   else ierr=1;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.GetIntValue) */
}

/*
 * Method:  GetPreconditionedResidual
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_GetPreconditionedResidual"

int32_t
impl_Hypre_PCG_GetPreconditionedResidual(
  Hypre_PCG self,
  Hypre_Vector* r)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.GetPreconditionedResidual) */
  /* Insert the implementation of the GetPreconditionedResidual method here... 
    */
   /* This is all wrong, and the whole function is not needed.
      >>> Delete this entire implementation soon. <<< */
   /* The preconditioned residual is s = C*r in the file krylov/pcg.c, pcg.h.
      (r is the residual b-A*x, C the preconditioner, an approx. inverse of A) */
   int ierr = 0;
   void * objectr;
   char *vector_type;
   HYPRE_Solver solver;
   struct Hypre_PCG__data * data;

   /* declarations for ParCSR matrix/vector type: */
   struct Hypre_ParCSRVector__data * datar;
   Hypre_ParCSRVector HypreP_r;
   HYPRE_ParVector rr;
   HYPRE_ParVector rr2;
   HYPRE_ParVector * prr = &rr2;
   HYPRE_IJVector ij_r;

   data = Hypre_PCG__get_data( self );
   solver = data->solver;
   vector_type = data -> vector_type;

   if ( vector_type=="ParVector" ) {
      HypreP_r = Hypre_Vector__cast2
         ( Hypre_Vector_queryInterface( *r, "Hypre.ParCSRVector"),
           "Hypre.ParCSRVector" );
      assert( HypreP_r!=NULL );
      datar = Hypre_ParCSRVector__get_data( HypreP_r );
      ij_r = datar -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_r, &objectr );
      rr = (HYPRE_ParVector) objectr;

      ierr += HYPRE_PCGGetPreconditionedResidual( solver, (void **) prr );
      HYPRE_ParVectorCopy( *prr, rr );
   }
   else {
      /* Unsupported vector type */
      ++ierr;
      return ierr;
   }

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.GetPreconditionedResidual) */
}

/*
 * Method:  GetResidual
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_GetResidual"

int32_t
impl_Hypre_PCG_GetResidual(
  Hypre_PCG self,
  Hypre_Vector* r)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.GetResidual) */
  /* Insert the implementation of the GetResidual method here... */
   int ierr = 0;
   void * objectr;
   char *vector_type;
   HYPRE_Solver solver;
   struct Hypre_PCG__data * data;

   /* declarations for ParCSR matrix/vector type: */
   struct Hypre_ParCSRVector__data * datar;
   Hypre_ParCSRVector HypreP_r;
   HYPRE_ParVector rr;
   HYPRE_ParVector rr2;
   HYPRE_ParVector * prr = &rr2;
   HYPRE_IJVector ij_r;

   data = Hypre_PCG__get_data( self );
   solver = data->solver;
   vector_type = data -> vector_type;

   if ( vector_type=="ParVector" ) {
      HypreP_r = Hypre_Vector__cast2
         ( Hypre_Vector_queryInterface( *r, "Hypre.ParCSRVector"),
           "Hypre.ParCSRVector" );
      assert( HypreP_r!=NULL );
      datar = Hypre_ParCSRVector__get_data( HypreP_r );
      ij_r = datar -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_r, &objectr );
      rr = (HYPRE_ParVector) objectr;

      ierr += HYPRE_PCGGetResidual( solver, (void**) prr );
      HYPRE_ParVectorCopy( *prr, rr );
   }
   else {
      /* Unsupported vector type */
      ++ierr;
      return ierr;
   }

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.GetResidual) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetCommunicator"

int32_t
impl_Hypre_PCG_SetCommunicator(
  Hypre_PCG self,
  void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   int ierr = 0;
   struct Hypre_PCG__data * data;
   data = Hypre_PCG__get_data( self );
   data -> comm = (MPI_Comm) comm;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetCommunicator) */
}

/*
 * Method:  SetDoubleArrayParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetDoubleArrayParameter"

int32_t
impl_Hypre_PCG_SetDoubleArrayParameter(
  Hypre_PCG self,
  const char* name,
  struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetDoubleArrayParameter) */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */
   /* no such parameters, return error if called */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetDoubleArrayParameter) */
}

/*
 * Method:  SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetDoubleParameter"

int32_t
impl_Hypre_PCG_SetDoubleParameter(
  Hypre_PCG self,
  const char* name,
  double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   /* The normal way to implement this function would be to call the corresponding
      HYPRE function to set the parameter.  That can't always be done because the
      HYPRE struct may not exist.  The HYPRE struct may not exist because it can't
      be created until we know the vector type - and that is not known until Apply
      is first called.  So what we do is save the parameter in a cache belonging to
      this Babel interface, and copy it into the HYPRE struct once Apply is called.
   */
   int ierr = 0;
   struct Hypre_PCG__data * data;
   data = Hypre_PCG__get_data( self );

   if ( strcmp(name,"Tol")==0 || strcmp(name,"Tolerance")==0 ) {
      data -> tol = value;
   }
   else if ( strcmp(name,"AbsoluteTolFactor")==0 ||
             strcmp(name,"Absolute Tol Factor")==0 ||
             strcmp(name,"Absolute Tolerance Factor")==0 ) {
      data -> atolf = value;
   }
   else if (strcmp(name,"ConvergenceFactorTol")==0 ||
            strcmp(name,"Convergence Factor Tol")==0 ||
            strcmp(name,"Convergence Factor Tolerance")==0 ) {
      /* tolerance for special test for slow convergence */
      data -> cf_tol = value;
   }
   /* Set other parameters here. */
   else ierr=1;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetDoubleParameter) */
}

/*
 * Method:  SetIntArrayParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetIntArrayParameter"

int32_t
impl_Hypre_PCG_SetIntArrayParameter(
  Hypre_PCG self,
  const char* name,
  struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */
   /* no such parameters, return error if called */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetIntArrayParameter) */
}

/*
 * Method:  SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetIntParameter"

int32_t
impl_Hypre_PCG_SetIntParameter(
  Hypre_PCG self,
  const char* name,
  int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   /* The normal way to implement this function would be to call the corresponding
      HYPRE function to set the parameter.  That can't always be done because the
      HYPRE struct may not exist.  The HYPRE struct may not exist because it can't
      be created until we know the vector type - and that is not known until Apply
      is first called.  So what we do is save the parameter in a cache belonging to
      this Babel interface, and copy it into the HYPRE struct once Apply is called.
   */
   int ierr = 0;
   struct Hypre_PCG__data * data;
   data = Hypre_PCG__get_data( self );

   if ( strcmp(name,"MaxIter")==0 || strcmp(name,"Max Iter")==0 ||
      strcmp(name,"Maximum Number of Iterations")==0 ) {
      data -> maxiter = value;
   }
   else if ( strcmp(name,"TwoNorm")==0 || strcmp(name,"Two Norm")==0 ||
            strcmp(name,"2-Norm")==0 ) {
      data -> twonorm = value;
   }
   else if ( strcmp(name,"RelChange")==0 || strcmp(name,"Rel Change")==0 ||
            strcmp(name,"Relative Change Test")==0 ) {
      data -> relchange = value;
   }
   else if ( strcmp(name,"StopCrit")==0 || strcmp(name,"Stop Crit")==0 ||
             strcmp(name,"Pure Absolute Error Stopping Criterion")==0 ) {
      /* this parameter is obsolete but still supported */
      data -> stop_crit = value;
   }
   else if ( strcmp(name,"PrintLevel")==0 || strcmp(name,"Print Level")==0 ) {
      /* also settable through SetPrintLevel */
      data -> printlevel = value;
   }
   else if ( strcmp(name,"LogLevel")==0 || strcmp(name,"Log Level")==0 ) {
      /* also settable through SetLogging */
      data -> log_level = value;
   }
   /* Set other parameters here. */
   else ierr=1;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetIntParameter) */
}

/*
 * Method:  SetLogging
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetLogging"

int32_t
impl_Hypre_PCG_SetLogging(
  Hypre_PCG self,
  int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */
   /* The normal way to implement this function would be to call the corresponding
      HYPRE function to set the print level.  That can't always be done because the
      HYPRE struct may not exist.  The HYPRE struct may not exist because it can't
      be created until we know the vector type - and that is not known until Apply
      is first called.  So what we do is save the print level in a cache belonging to
      this Babel interface, and copy it into the HYPRE struct once Apply is called.
   */
   int ierr = 0;
   struct Hypre_PCG__data * data;
   data = Hypre_PCG__get_data( self );

   data -> log_level = level;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetLogging) */
}

/*
 * Method:  SetOperator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetOperator"

int32_t
impl_Hypre_PCG_SetOperator(
  Hypre_PCG self,
  Hypre_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
   int ierr = 0;
   struct Hypre_PCG__data * data;

   data = Hypre_PCG__get_data( self );
   data->matrix = A;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetOperator) */
}

/*
 * Method:  SetPreconditioner
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetPreconditioner"

int32_t
impl_Hypre_PCG_SetPreconditioner(
  Hypre_PCG self,
  Hypre_Solver s)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetPreconditioner) */
  /* Insert the implementation of the SetPreconditioner method here... */
   int ierr = 0;
   HYPRE_Solver * solverprecond;
   struct Hypre_PCG__data * dataself;
   struct Hypre_ParAMG__data * AMG_dataprecond;
   Hypre_ParAMG AMG_s;
   struct Hypre_ParDiagScale__data * DiagScale_dataprecond;
   Hypre_ParDiagScale DiagScale_s;
   HYPRE_PtrToSolverFcn precond, precond_setup; /* functions */

   dataself = Hypre_PCG__get_data( self );
/*   solver = dataself->solver;
     assert( solver != NULL );*/

   if ( Hypre_Solver_queryInterface( s, "Hypre.ParAMG" ) ) {
      /* s is a Hypre_ParAMG */
      AMG_s = Hypre_Operator__cast2
         ( Hypre_Solver_queryInterface( s, "Hypre.ParAMG"),
           "Hypre.ParAMG" );
      AMG_dataprecond = Hypre_ParAMG__get_data( AMG_s );
      solverprecond = &AMG_dataprecond->solver;
      assert( solverprecond != NULL );
      precond = (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve;
      precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup;
   }
   else if ( Hypre_Solver_queryInterface( s, "Hypre.ParDiagScale" ) ) {
      /* s is a Hypre_ParDiagScale */
      DiagScale_s = Hypre_Operator__cast2
         ( Hypre_Solver_queryInterface( s, "Hypre.ParDiagScale"),
           "Hypre.ParDiagScale" );
      DiagScale_dataprecond = Hypre_ParDiagScale__get_data( DiagScale_s );
      solverprecond = (HYPRE_Solver *) hypre_CTAlloc( double, 1 );
      /* ... HYPRE diagonal scaling needs no solver object, but we must provide a
         HYPRE_Solver object.  It will be totally ignored. */
      precond = (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale;
      precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup;
   }
   /* put other preconditioner types here */
   else {
      assert( "PCG_SetPreconditioner cannot recognize preconditioner"==0 );
   }

   /* We can't actually set the HYPRE preconditioner, because that requires
      knowing what the solver object is - but that requires knowing its data type
      but _that_ requires knowing the kind of matrix and vectors we'll need;
      not known until Apply is called.  So save the information in the Hypre
      data structure, and stick it in HYPRE later... */
   dataself->precond = precond;
   dataself->precond_setup = precond_setup;
   dataself->solverprecond = solverprecond;
   /*   for example call, see test/IJ_linear_solvers.c, line 1686.
        The four arguments  are:  self's (solver) data; and, for the preconditioner:
        solver function, setup function, data */

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetPreconditioner) */
}

/*
 * Method:  SetPrintLevel
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetPrintLevel"

int32_t
impl_Hypre_PCG_SetPrintLevel(
  Hypre_PCG self,
  int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */
   /* The normal way to implement this function would be to call the corresponding
      HYPRE function to set the print level.  That can't always be done because the
      HYPRE struct may not exist.  The HYPRE struct may not exist because it can't
      be created until we know the vector type - and that is not known until Apply
      is first called.  So what we do is save the print level in a cache belonging to
      this Babel interface, and copy it into the HYPRE struct once Apply is called.
   */
   int ierr = 0;
   struct Hypre_PCG__data * data;
   data = Hypre_PCG__get_data( self );

   data -> printlevel = level;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetPrintLevel) */
}

/*
 * Method:  SetStringParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetStringParameter"

int32_t
impl_Hypre_PCG_SetStringParameter(
  Hypre_PCG self,
  const char* name,
  const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   /* There are no string parameters, so return an error. */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetStringParameter) */
}

/*
 * Method:  Setup
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_Setup"

int32_t
impl_Hypre_PCG_Setup(
  Hypre_PCG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.Setup) */
  /* Insert the implementation of the Setup method here... */
   /* Setup is not done here because it requires the entire problem.
      It is done in Apply instead. */
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.Setup) */
}
