/*
 * File:          Hypre_GMRES_Impl.c
 * Symbol:        Hypre.GMRES-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:52 PDT
 * Description:   Server-side implementation for Hypre.GMRES
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.GMRES" (version 0.1.5)
 */

#include "Hypre_GMRES_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.GMRES._includes) */
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

/* >>> To do: impl_Hypre_GMRES_Copy_Parameters_from_HYPRE_struct
   (see comments in Hypre_PCG_Impl.c). */

int impl_Hypre_GMRES_Copy_Parameters_to_HYPRE_struct( Hypre_GMRES self )
/* Copy parameter cache from the Hypre_GMRES__data object into the
   HYPRE_Solver object */
/* >>> Possible BUG if impl_Hypre_GMRES_Copy_Parameters_from_HYPRE_struct
   be not called earlier (at initialization).
   See comment in Hypre_PCG_Impl.c */
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
   ierr += HYPRE_GMRESSetLogLevel( solver, data->log_level );

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
   data -> comm = (MPI_Comm)NULL;
   data -> solver = NULL;
   data -> matrix = NULL;
   data -> vector_type = NULL;
   /* We would like to call HYPRE_<vector type>GMRESCreate at this point, but
      it's impossible until we know the vector type.  That's needed because
      the C-language Krylov solvers need to be told exactly what functions
      to call.  If we were to switch to a Babel-based GMRES solver, we would be
      able to use generic function names; hence we could really initialize GMRES
      here. */

   /* default values (copied from gmres.c; better to get them by function
      calls)...*/
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

/* >>> TO DO switch according to vector_type; it's not always ParCSR */
   ierr += HYPRE_ParCSRGMRESDestroy( data->solver );
   Hypre_Operator_deleteReference( data->matrix );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES._dtor) */
}

/*
 * Method:  Apply
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_Apply"

int32_t
impl_Hypre_GMRES_Apply(
  Hypre_GMRES self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.Apply) */
  /* Insert the implementation of the Apply method here... */
   /* In the long run, the solver should be implemented right here, calling
      the appropriate Hypre functions.  But for now we are calling the existing
      HYPRE solver.  Advantages: don't want to have two versions of the same
      GMRES solver lying around.  Disadvantage: we have to cache user-supplied
      parameters until the Apply call, where we make the GMRES object and really
      set the parameters - messy and unnatural. */
   int ierr=0;
   MPI_Comm comm;
   HYPRE_Solver solver;
   HYPRE_Solver * psolver = &solver; /* will get a real value later */
   struct Hypre_GMRES__data * data;
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

   data = Hypre_GMRES__get_data( self );
   comm = data->comm;
   assert( comm != (MPI_Comm)NULL ); /* SetCommunicator should have been called earlier */
   mat = data->matrix;
   assert( mat != NULL ); /* SetOperator should have been called earlier */

   if ( data -> vector_type == NULL ) {
      /* This is the first time this Babel GMRES object has seen a vector.
         So we are ready to create the Hypre GMRES object. */
      if ( Hypre_Vector_queryInterface( x, "Hypre.ParCSRVector") ) {
         data -> vector_type = "ParVector";
         HYPRE_ParCSRGMRESCreate( comm, psolver );
         assert( solver != NULL );
         data -> solver = *psolver;
      }
      /* Add more vector types here */
      else {
         assert( "only ParCSRVector supported by GMRES"==0 );
      }
      Hypre_GMRES__set_data( self, data );
   }
   else {
      solver = data->solver;
      assert( solver != NULL );
   };
   /* The SetParameter functions set parameters in the local Babel-interface struct,
      "data".  That is because the HYPRE struct (where they are actually used) may
      not exist yet when the functions are called.  At this point we finally know
      the HYPRE struct exists, so we copy the parameters to it. */
   ierr += impl_Hypre_GMRES_Copy_Parameters_to_HYPRE_struct( self );
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
         assert( "only ParCSRVector supported by GMRES"==0 );
   }
      
   ierr += HYPRE_GMRESSetPrecond( solver, data->precond, data->precond_setup,
                                *(data->solverprecond) );
   HYPRE_GMRESSetup( solver, HYPRE_A, HYPRE_x, HYPRE_y );
   HYPRE_GMRESSolve( solver, HYPRE_A, HYPRE_x, HYPRE_y );

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.Apply) */
}

/*
 * Method:  GetDoubleValue
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetDoubleValue"

int32_t
impl_Hypre_GMRES_GetDoubleValue(
  Hypre_GMRES self,
  const char* name,
  double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetDoubleValue) */
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
   struct Hypre_GMRES__data * data;

   data = Hypre_GMRES__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   if ( strcmp(name,"FinalRelativeResidualNorm")==0 ||
        strcmp(name,"Final Relative Residual Norm")==0 ) {
      ierr += HYPRE_GMRESGetFinalRelativeResidualNorm( solver, value );
   }
   /* Get other values here. */
   else ierr=1;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetDoubleValue) */
}

/*
 * Method:  GetIntValue
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetIntValue"

int32_t
impl_Hypre_GMRES_GetIntValue(
  Hypre_GMRES self,
  const char* name,
  int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetIntValue) */
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
   struct Hypre_GMRES__data * data;

   data = Hypre_GMRES__get_data( self );
   assert( data->solver != NULL );
   solver = data->solver;

   if ( strcmp(name,"NumIterations")==0 || strcmp(name,"Num Iterations")==0
      || strcmp(name,"Number of Iterations")==0 ) {
      ierr += HYPRE_GMRESGetNumIterations( solver, value );
      printf("num iterations=%i",*value);
   }
   /* Get other values here. */
   else ierr=1;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetIntValue) */
}

/*
 * Method:  GetPreconditionedResidual
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetPreconditionedResidual"

int32_t
impl_Hypre_GMRES_GetPreconditionedResidual(
  Hypre_GMRES self,
  Hypre_Vector* r)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetPreconditionedResidual) */
  /* Insert the implementation of the GetPreconditionedResidual method here... 
    */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetPreconditionedResidual) */
}

/*
 * Method:  GetResidual
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetResidual"

int32_t
impl_Hypre_GMRES_GetResidual(
  Hypre_GMRES self,
  Hypre_Vector* r)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetResidual) */
  /* Insert the implementation of the GetResidual method here... */
  /* Insert the implementation of the GetResidual method here... */
   /* >>> this doesn't work yet because the necessary capability
      hasn't been implemented in krylov/gmres.c and krylov/gmres.h */
   int ierr = 0;
   void * objectr;
   char *vector_type;
   HYPRE_Solver solver;
   struct Hypre_GMRES__data * data;

   /* declarations for ParCSR matrix/vector type: */
   struct Hypre_ParCSRVector__data * datar;
   Hypre_ParCSRVector HypreP_r;
   HYPRE_ParVector rr;
   HYPRE_ParVector rr2;
   HYPRE_ParVector * prr = &rr2;
   HYPRE_IJVector ij_r;

   assert( strcmp("not ready","to be called")==0 );  /* >>> finish gmres.[c,h] first */

   data = Hypre_GMRES__get_data( self );
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

      ierr += HYPRE_GMRESGetResidual( solver, (void**) prr );
      HYPRE_ParVectorCopy( *prr, rr );
   }
   else {
      /* Unsupported vector type */
      ++ierr;
   }
   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetResidual) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetCommunicator"

int32_t
impl_Hypre_GMRES_SetCommunicator(
  Hypre_GMRES self,
  void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );
   data -> comm = (MPI_Comm) comm;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetCommunicator) */
}

/*
 * Method:  SetDoubleArrayParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetDoubleArrayParameter"

int32_t
impl_Hypre_GMRES_SetDoubleArrayParameter(
  Hypre_GMRES self,
  const char* name,
  struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetDoubleArrayParameter) */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */
   /* no such parameters, return error if called */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetDoubleArrayParameter) */
}

/*
 * Method:  SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetDoubleParameter"

int32_t
impl_Hypre_GMRES_SetDoubleParameter(
  Hypre_GMRES self,
  const char* name,
  double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   /* The normal way to implement this function would be to call the corresponding
      HYPRE function to set the parameter.  That can't always be done because the
      HYPRE struct may not exist.  The HYPRE struct may not exist because it can't
      be created until we know the vector type - and that is not known until Apply
      is first called.  So what we do is save the parameter in a cache belonging to
      this Babel interface, and copy it into the HYPRE struct once Apply is called.
   */
   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );

   if ( strcmp(name,"Tol")==0 || strcmp(name,"Tolerance")==0 ) {
      data -> tol = value;
   }
   /* Set other parameters here. */
   else ierr=1;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetDoubleParameter) */
}

/*
 * Method:  SetIntArrayParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetIntArrayParameter"

int32_t
impl_Hypre_GMRES_SetIntArrayParameter(
  Hypre_GMRES self,
  const char* name,
  struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */
   /* no such parameters, return error if called */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetIntArrayParameter) */
}

/*
 * Method:  SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetIntParameter"

int32_t
impl_Hypre_GMRES_SetIntParameter(
  Hypre_GMRES self,
  const char* name,
  int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   /* The normal way to implement this function would be to call the corresponding
      HYPRE function to set the parameter.  That can't always be done because the
      HYPRE struct may not exist.  The HYPRE struct may not exist because it can't
      be created until we know the vector type - and that is not known until Apply
      is first called.  So what we do is save the parameter in a cache belonging to
      this Babel interface, and copy it into the HYPRE struct once Apply is called.
   */
   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );

   if ( strcmp(name,"KDim")==0 || strcmp(name,"K Dim")==0 ) {
      data -> k_dim = value;
   }
   if ( strcmp(name,"MaxIter")==0 || strcmp(name,"Max Iter")==0 ||
      strcmp(name,"Maximum Number of Iterations")==0 ) {
      data -> max_iter = value;
   }
   if ( strcmp(name,"MinIter")==0 || strcmp(name,"Min Iter")==0 ||
      strcmp(name,"Minimum Number of Iterations")==0 ) {
      data -> min_iter = value;
   }
   else if ( strcmp(name,"RelChange")==0 || strcmp(name,"Rel Change")==0 ||
            strcmp(name,"Relative Change Test")==0 ) {
      data -> rel_change = value;
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
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetIntParameter) */
}

/*
 * Method:  SetLogging
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetLogging"

int32_t
impl_Hypre_GMRES_SetLogging(
  Hypre_GMRES self,
  int32_t level)
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
 * Method:  SetOperator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetOperator"

int32_t
impl_Hypre_GMRES_SetOperator(
  Hypre_GMRES self,
  Hypre_Operator A)
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
 * Method:  SetPreconditioner
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetPreconditioner"

int32_t
impl_Hypre_GMRES_SetPreconditioner(
  Hypre_GMRES self,
  Hypre_Solver s)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetPreconditioner) */
  /* Insert the implementation of the SetPreconditioner method here... */
   int ierr = 0;
   HYPRE_Solver * solverprecond;
   struct Hypre_GMRES__data * dataself;
   struct Hypre_ParAMG__data * AMG_dataprecond;
   Hypre_ParAMG AMG_s;
/* not used   struct Hypre_ParDiagScale__data * DiagScale_dataprecond;*/
/* not used   Hypre_ParDiagScale DiagScale_s;*/
   HYPRE_PtrToSolverFcn precond, precond_setup; /* functions */

   dataself = Hypre_GMRES__get_data( self );
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
/* not used      DiagScale_s = Hypre_Operator__cast2
         ( Hypre_Solver_queryInterface( s, "Hypre.ParDiagScale"),
         "Hypre.ParDiagScale" );*/
/* not used      DiagScale_dataprecond = Hypre_ParDiagScale__get_data( DiagScale_s );*/
      solverprecond = (HYPRE_Solver *) hypre_CTAlloc( double, 1 );
      /* ... HYPRE diagonal scaling needs no solver object, but we must provide a
         HYPRE_Solver object.  It will be totally ignored. */
      precond = (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale;
      precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup;
   }
   /* put other preconditioner types here */
   else {
      assert( "GMRES_SetPreconditioner cannot recognize preconditioner"==0 );
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
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetPreconditioner) */
}

/*
 * Method:  SetPrintLevel
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetPrintLevel"

int32_t
impl_Hypre_GMRES_SetPrintLevel(
  Hypre_GMRES self,
  int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */
   /* The normal way to implement this function would be to call the corresponding
      HYPRE function to set the print level.  That can't always be done because the
      HYPRE struct may not exist.  The HYPRE struct may not exist because it can't
      be created until we know the vector type - and that is not known until Apply
      is first called.  So what we do is save the print level in a cache belonging to
      this Babel interface, and copy it into the HYPRE struct once Apply is called.
   */
   int ierr = 0;
   struct Hypre_GMRES__data * data;
   data = Hypre_GMRES__get_data( self );

   data -> printlevel = level;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetPrintLevel) */
}

/*
 * Method:  SetStringParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetStringParameter"

int32_t
impl_Hypre_GMRES_SetStringParameter(
  Hypre_GMRES self,
  const char* name,
  const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   /* There are no string parameters, so return an error. */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetStringParameter) */
}

/*
 * Method:  Setup
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_Setup"

int32_t
impl_Hypre_GMRES_Setup(
  Hypre_GMRES self,
  Hypre_Vector x,
  Hypre_Vector y)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.Setup) */
  /* Insert the implementation of the Setup method here... */
   /* Setup is not done here because it requires the entire problem.
      It is done in Apply instead. */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.Setup) */
   return 0;
}
