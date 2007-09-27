/*
 * File:          bHYPRE_PCG_Impl.c
 * Symbol:        bHYPRE.PCG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.PCG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.PCG" (version 1.0.0)
 * 
 * PCG solver.
 * This calls Babel-interface matrix and vector functions, so it will work
 * with any consistent matrix, vector, and preconditioner classes.
 */

#include "bHYPRE_PCG_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._includes) */
/* Put additional includes or other arbitrary code here... */


#include "bHYPRE_MPICommunicator_Impl.h"
#include "bHYPRE_IdentitySolver_Impl.h"
#include "bHYPRE_MatrixVectorView.h"
#include <math.h>

#include "hypre_babel_exception_handler.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.PCG._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_PCG__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._load) */
  /* Insert-Code-Here {bHYPRE.PCG._load} (static class initializer method) */
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_PCG__ctor(
  /* in */ bHYPRE_PCG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct bHYPRE_PCG__data * data;
   data = hypre_CTAlloc( struct bHYPRE_PCG__data, 1 );

   data -> mpicomm      = bHYPRE_MPICommunicator_CreateC( (void *)MPI_COMM_NULL, _ex ); SIDL_CHECK(*_ex);
   data -> matrix       = (bHYPRE_Operator)NULL;
   data -> tol          = 1.0e-06;
   data -> atolf        = 0.0;
   data -> cf_tol       = 0.0;
   data -> max_iter     = 1000;
   data -> two_norm     = 0;
   data -> rel_change   = 0;
   data -> stop_crit    = 0;
   data -> converged    = 0;
   data -> num_iterations = 0;
   data -> rel_residual_norm = 0.0;
   data -> precond      = (bHYPRE_Solver)NULL;
   data -> print_level  = 0;
   data -> logging      = 0;
   data -> norms        = NULL;
   data -> rel_norms    = NULL;
   data -> p            = (bHYPRE_Vector)NULL;
   data -> s            = (bHYPRE_Vector)NULL;
   data -> r            = (bHYPRE_Vector)NULL;

   bHYPRE_PCG__set_data( self, data );

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_PCG__ctor2(
  /* in */ bHYPRE_PCG self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._ctor2) */
    /* Insert-Code-Here {bHYPRE.PCG._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_PCG__dtor(
  /* in */ bHYPRE_PCG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._dtor) */
  /* Insert the implementation of the destructor method here... */

   struct bHYPRE_PCG__data * data;
   data = bHYPRE_PCG__get_data( self );

   if (data)
   {
      if ( (data -> norms) != NULL )
      {
         hypre_TFree( data -> norms );
         data -> norms = NULL;
      } 
      if ( (data -> rel_norms) != NULL )
      {
         hypre_TFree( data -> rel_norms );
         data -> rel_norms = NULL;
      }

      if ( data -> p != (bHYPRE_Vector)NULL )
      {
         bHYPRE_Vector_deleteRef( data->p, _ex ); SIDL_CHECK(*_ex);
      }
      if ( data -> s != (bHYPRE_Vector)NULL )
      {
         bHYPRE_Vector_deleteRef( data->s, _ex ); SIDL_CHECK(*_ex);
      }
      if ( data -> r != (bHYPRE_Vector)NULL )
      {
         bHYPRE_Vector_deleteRef( data->r, _ex ); SIDL_CHECK(*_ex);
      }
      if ( data -> matrix != (bHYPRE_Operator)NULL )
      {
         bHYPRE_Operator_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
      }
      if ( data -> precond != (bHYPRE_Solver)NULL )
      {
         bHYPRE_Solver_deleteRef( data->precond, _ex ); SIDL_CHECK(*_ex);
      }

      hypre_TFree( data );
   }

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG._dtor) */
  }
}

/*
 *  This function is the preferred way to create a PCG solver. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_PCG
impl_bHYPRE_PCG_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.Create) */
  /* Insert-Code-Here {bHYPRE.PCG.Create} (Create method) */

   bHYPRE_PCG solver = bHYPRE_PCG__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( solver );
   bHYPRE_IdentitySolver Id  = bHYPRE_IdentitySolver_Create( mpi_comm, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_Solver IdS = bHYPRE_Solver__cast( Id, _ex ); SIDL_CHECK(*_ex);

   if (data->mpicomm) {
      bHYPRE_MPICommunicator_deleteRef( data->mpicomm, _ex ); SIDL_CHECK(*_ex);
   }
   data->mpicomm = mpi_comm;
   if( data->matrix != (bHYPRE_Operator)NULL )
   {
      bHYPRE_Operator_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   }

   data->matrix = A;
   bHYPRE_Operator_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   data->precond = IdS;

   bHYPRE_IdentitySolver_deleteRef( Id, _ex ); SIDL_CHECK(*_ex);
   /* ...Create and cast created 2 references, we're keeping only one (data->precond) */

   return solver;

   hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.Create) */
  }
}

/*
 * Set the preconditioner.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetPreconditioner"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetPreconditioner(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Solver s,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetPreconditioner) */
  /* Insert the implementation of the SetPreconditioner method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );
   if( data->precond != (bHYPRE_Solver)NULL )
      bHYPRE_Solver_deleteRef( data->precond, _ex ); SIDL_CHECK(*_ex);

   data->precond = s;
   bHYPRE_Solver_addRef( data->precond, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetPreconditioner) */
  }
}

/*
 * Method:  GetPreconditioner[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_GetPreconditioner"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_GetPreconditioner(
  /* in */ bHYPRE_PCG self,
  /* out */ bHYPRE_Solver* s,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.GetPreconditioner) */
  /* Insert-Code-Here {bHYPRE.PCG.GetPreconditioner} (GetPreconditioner method) */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );

   *s = data->precond;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.GetPreconditioner) */
  }
}

/*
 * Method:  Clone[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_Clone"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_Clone(
  /* in */ bHYPRE_PCG self,
  /* out */ bHYPRE_PreconditionedSolver* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.Clone) */
  /* Insert-Code-Here {bHYPRE.PCG.Clone} (Clone method) */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );
   struct bHYPRE_PCG__data * datax;
   bHYPRE_PCG PCG_x;

   PCG_x = bHYPRE_PCG_Create( data->mpicomm, data->matrix, _ex ); SIDL_CHECK(*_ex);

   /* Copy most data members.
      The preconditioner copy will be a shallow copy (just the pointer);
      it is likely to be replaced later.
      But don't copy anything created in Setup (p,s,r,norms,rel_norms).
      The user will call Setup on x, later
      Also don't copy the end-of-solve diagnostics (converged,num_iterations,
      rel_residual_norm) */

   datax = bHYPRE_PCG__get_data( PCG_x );
   datax->tol               = data->tol;
   datax->atolf             = data->atolf;
   datax->cf_tol            = data->cf_tol;
   datax->max_iter          = data->max_iter;
   datax->two_norm          = data->two_norm;
   datax->rel_change        = data->rel_change;
   datax->stop_crit         = data->stop_crit;
   datax->print_level       = data->print_level;
   datax->logging           = data->logging;

   bHYPRE_PCG_SetPreconditioner( PCG_x, data->precond, _ex ); SIDL_CHECK(*_ex);

   *x = bHYPRE_PreconditionedSolver__cast( PCG_x, _ex ); SIDL_CHECK(*_ex);

   bHYPRE_PCG_deleteRef( PCG_x,_ex ); SIDL_CHECK(*_ex);
   return ierr;

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.Clone) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetOperator(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   /* DEPRECATED  the second argument in Create does the same thing */

   int ierr = 0;
   struct bHYPRE_PCG__data * data;

   data = bHYPRE_PCG__get_data( self );
   if( data->matrix != (bHYPRE_Operator)NULL )
      bHYPRE_Operator_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   data->matrix = A;
   bHYPRE_Operator_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetTolerance(
  /* in */ bHYPRE_PCG self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );

   data -> tol = tolerance;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetMaxIterations(
  /* in */ bHYPRE_PCG self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );

   data -> max_iter = max_iterations;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetMaxIterations) */
  }
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetLogging(
  /* in */ bHYPRE_PCG self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );

   data -> logging = level;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetLogging) */
  }
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetPrintLevel(
  /* in */ bHYPRE_PCG self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );

   data -> print_level = level;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_GetNumIterations(
  /* in */ bHYPRE_PCG self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );

   *num_iterations = data->num_iterations;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_GetRelResidualNorm(
  /* in */ bHYPRE_PCG self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );

   *norm = data->rel_residual_norm;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetCommunicator(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1;  /* DEPRECATED and will never be implemented */
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_PCG_Destroy(
  /* in */ bHYPRE_PCG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.Destroy) */
    /* Insert-Code-Here {bHYPRE.PCG.Destroy} (Destroy method) */
     bHYPRE_PCG_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetIntParameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );

   if ( strcmp(name,"TwoNorm")==0 || strcmp(name,"2-norm")==0 )
   {
      data -> two_norm = value;
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      data -> max_iter = value;
   }
   else if ( strcmp(name,"RelChange")==0 || strcmp(name,"relative change test")==0 )
   {
      data -> rel_change = value;
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      data -> logging = value;
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      data -> print_level = value;
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetDoubleParameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );

   if ( strcmp(name,"AbsoluteTolFactor")==0 )
   {
      data -> atolf = value;
   }
   else if ( strcmp(name,"ConvergenceFactorTol")==0 )
   {
      /* tolerance for special test for slow convergence */
      data -> cf_tol = value;
   }
   else if ( strcmp(name,"Tolerance")==0  || strcmp(name,"Tol")==0 )
   {
      data -> tol = value;
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetStringParameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetIntArray1Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetIntArray2Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_GetIntValue(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );

   if ( strcmp(name,"NumIterations")==0 )
   {
      *value = data -> num_iterations;
   }
   else if ( strcmp(name,"Converged")==0 )
   {
      *value = data -> converged;
   }
   else if ( strcmp(name,"TwoNorm")==0 || strcmp(name,"2-norm")==0 )
   {
      *value = data -> two_norm;
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      *value = data -> max_iter;
   }
   else if ( strcmp(name,"RelChange")==0 || strcmp(name,"relative change test")==0 )
   {
      *value = data -> rel_change;
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      *value = data -> logging;
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      *value = data -> print_level;
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_GetDoubleValue(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );

   if ( strcmp(name,"Final Relative Residual Norm")==0 ||
        strcmp(name,"FinalRelativeResidualNorm")==0 ||
        strcmp(name,"RelResidualNorm")==0 ||
        strcmp(name,"RelativeResidualNorm")==0 )
   {
      *value = data -> rel_residual_norm;
   }
   else if ( strcmp(name,"AbsoluteTolFactor")==0 )
   {
      *value = data -> atolf;
   }
   else if ( strcmp(name,"ConvergenceFactorTol")==0 )
   {
      /* tolerance for special test for slow convergence */
      *value = data -> cf_tol;
   }
   else if ( strcmp(name,"Tolerance")==0  || strcmp(name,"Tol")==0 )
   {
      *value = data -> tol;
   }
   else
   {
      ierr = 1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_Setup(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );
   int   max_iter = data->max_iter;
   bHYPRE_MatrixVectorView Vp, Vs, Vr;

   /* Setup should not be called more than once. */
   hypre_assert( data->p == (bHYPRE_Vector)NULL );
   hypre_assert( data->s == (bHYPRE_Vector)NULL );
   hypre_assert( data->r == (bHYPRE_Vector)NULL );

   ierr += bHYPRE_Vector_Clone( x, &(data->p), _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Clone( x, &(data->s), _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Clone( b, &(data->r), _ex ); SIDL_CHECK(*_ex);
   Vp = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->p, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex);
   if ( Vp!=NULL )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vp, _ex ); SIDL_CHECK(*_ex);
   }
   Vs = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->s, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex);
   if ( Vs!=NULL )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vs, _ex ); SIDL_CHECK(*_ex);
   }
   Vr = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->r, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex);
   if ( Vr!=NULL )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vr, _ex ); SIDL_CHECK(*_ex);
   }

   ierr += bHYPRE_Solver_Setup( data->precond, b, x, _ex ); SIDL_CHECK(*_ex);

   if ( data->logging>0  || data->print_level>0 ) 
   {  /* arrays needed for logging */
      if ( data->norms != NULL )
         hypre_TFree( data->norms );
      data->norms = hypre_CTAlloc( double, max_iter + 1 );

      if ( data->rel_norms != NULL )
         hypre_TFree( data -> rel_norms );
      data->rel_norms = hypre_CTAlloc( double, max_iter + 1 );
   }

   bHYPRE_MatrixVectorView_deleteRef( Vp, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_MatrixVectorView_deleteRef( Vs, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_MatrixVectorView_deleteRef( Vr, _ex ); SIDL_CHECK(*_ex);

   return ierr;
   
     hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_Apply(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.Apply) */
  /* Insert the implementation of the Apply method here... */

/*--------------------------------------------------------------------------
 *
 * We use the following convergence test as the default (see Ashby, Holst,
 * Manteuffel, and Saylor):
 *
 *       ||e||_A                           ||r||_C
 *       -------  <=  [kappa_A(C*A)]^(1/2) -------  < tol
 *       ||x||_A                           ||b||_C
 *
 * where we let (for the time being) kappa_A(CA) = 1.
 * We implement the test as:
 *
 *       gamma = <C*r,r>/<C*b,b>  <  (tol^2) = eps
 *
 *--------------------------------------------------------------------------*/

   int ierr = 0;
   struct bHYPRE_PCG__data * data = bHYPRE_PCG__get_data( self );
   bHYPRE_Operator A = data->matrix;
   double          tol          = data -> tol;
   double          atolf        = data -> atolf;
   double          cf_tol       = data -> cf_tol;
   int             max_iter     = data -> max_iter;
   int             two_norm     = data -> two_norm;
   int             rel_change   = data -> rel_change;
   int             stop_crit    = data -> stop_crit;
   bHYPRE_Vector   p            = data -> p;
   bHYPRE_Vector   s            = data -> s;
   bHYPRE_Vector   r            = data -> r;
   bHYPRE_Solver   precond      = data -> precond;
   int             print_level  = data -> print_level;
   int             logging      = data -> logging;
   double         *norms        = data -> norms;
   double         *rel_norms    = data  -> rel_norms;
   bHYPRE_MPICommunicator  bmpicomm = data -> mpicomm;
                
   double          alpha, beta;
   double          gamma, gamma_old;
   double          bi_prod, i_prod, eps;
   double          pi_prod, xi_prod;
   double          ieee_check = 0.;
                
   double          i_prod_0;
   double          cf_ave_0 = 0.0;
   double          cf_ave_1 = 0.0;
   double          weight;
   double          ratio;

   double          guard_zero_residual, sdotp;

   int             i = 0;
   MPI_Comm        comm;
   int             my_id, num_procs;

   comm = bHYPRE_MPICommunicator__get_data(bmpicomm)->mpi_comm;
   MPI_Comm_size( comm, &num_procs );
   MPI_Comm_rank( comm, &my_id );

   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/

   guard_zero_residual = 0.0;

   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/

   /* compute eps */
   if (two_norm)
   {
      /* bi_prod = <b,b> */
      ierr += bHYPRE_Vector_Dot( b, b, &bi_prod, _ex ); SIDL_CHECK(*_ex);
      if (print_level > 1 && my_id == 0) 
          printf("<b,b>: %e\n",bi_prod);
   }
   else
   {
      /* bi_prod = <C*b,b> */
      ierr += bHYPRE_Vector_Clear( p, _ex ); SIDL_CHECK(*_ex);
      ierr += bHYPRE_Solver_Apply( precond, b, &p, _ex ); SIDL_CHECK(*_ex);
      ierr += bHYPRE_Vector_Dot( p, b, &bi_prod, _ex ); SIDL_CHECK(*_ex);
      if (print_level > 1 && my_id == 0)
          printf("<C*b,b>: %e\n",bi_prod);
   };

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (bi_prod != 0.) ieee_check = bi_prod/bi_prod; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0 || logging > 0)
      {
        printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        printf("ERROR -- hypre_PCGSolve: INFs and/or NaNs detected in input.\n");
        printf("User probably placed non-numerics in supplied b.\n");
        printf("Returning error flag += 101.  Program not terminated.\n");
        printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

   eps = tol*tol;
   if ( bi_prod > 0.0 ) {
      if ( stop_crit && !rel_change && atolf<=0 ) {  /* pure absolute tolerance */
         eps = eps / bi_prod;
         /* Note: this section is obsolete.  Aside from backwards comatability
            concerns, we could delete the stop_crit parameter and related code,
            using tol & atolf instead. */
      }
      else if ( atolf>0 )  /* mixed relative and absolute tolerance */
         bi_prod += atolf;
   }
   else    /* bi_prod==0.0: the rhs vector b is zero */
   {
      /* Set x equal to zero and return */
      ierr += bHYPRE_Vector_Copy( *x, b, _ex ); SIDL_CHECK(*_ex);
      if (logging>0 || print_level>0)
      {
         norms[0]     = 0.0;
         rel_norms[i] = 0.0;
      }
      data -> converged = 1;
      ierr = 0;
      return ierr;
      /* In this case, for the original parcsr pcg, the code would take special
         action to force iterations even though the exact value was known. */
   };

   /* r = b - Ax */
   /* It would be better define a matvec operation which directly calls the
      HYPRE matvec, so we can do this in fewer primitive operations.
      However, the cost savings would be negligible. */
   ierr += bHYPRE_Operator_Apply( A, *x, &r, _ex );  /* r = Ax */
   SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Axpy( r, -1.0, b, _ex );   /* r = r - b = Ax - b */
   SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Scale( r, -1.0, _ex );     /* r = -r = b - Ax */
   SIDL_CHECK(*_ex);
 
   /* p = C*r */
   ierr += bHYPRE_Vector_Clear( p, _ex );  /* is this needed ? */
   SIDL_CHECK(*_ex);
   ierr += bHYPRE_Solver_Apply( precond, r, &p, _ex );  /* A`p = r */
   SIDL_CHECK(*_ex);

   /* gamma = <r,p> */
   ierr += bHYPRE_Vector_Dot( r, p, &gamma, _ex ); SIDL_CHECK(*_ex);

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (gamma != 0.) ieee_check = gamma/gamma; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0 || logging > 0)
      {
        printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        printf("ERROR -- hypre_PCGSolve: INFs and/or NaNs detected in input.\n");
        printf("User probably placed non-numerics in supplied A or x_0.\n");
        printf("Returning error flag += 101.  Program not terminated.\n");
        printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

   /* Set initial residual norm */
   if ( logging>0 || print_level > 0 || cf_tol > 0.0 )
   {
      if (two_norm)
      {
         ierr += bHYPRE_Vector_Dot( r, r, &i_prod_0, _ex ); SIDL_CHECK(*_ex);
      }
      else
         i_prod_0 = gamma;

      if ( logging>0 || print_level>0 ) norms[0] = sqrt(i_prod_0);
   }
   if ( print_level > 1 && my_id==0 )  /* formerly for par_csr only */
   {
      printf("\n\n");
      if (two_norm)
      {
         if ( stop_crit && !rel_change && atolf==0 ) {  /* pure absolute tolerance */
            printf("Iters       ||r||_2     conv.rate\n");
            printf("-----    ------------   ---------\n");
         }
         else {
            printf("Iters       ||r||_2     conv.rate  ||r||_2/||b||_2\n");
            printf("-----    ------------   ---------  ------------ \n");
         }
      }
      else  /* !two_norm */
      {
         printf("Iters       ||r||_C     conv.rate  ||r||_C/||b||_C\n");
         printf("-----    ------------    ---------  ------------ \n");
      }
   }

   hypre_assert( ierr == 0 );

   /* ********************************************************
    *
    * Main solve loop
    *
    ********************************************************** */

   while ((i+1) <= max_iter)
   {
      i++;

      /* s = A*p */
      ierr += bHYPRE_Operator_Apply( A, p, &s, _ex ); SIDL_CHECK(*_ex);

      /* alpha = gamma / <s,p> */
      ierr += bHYPRE_Vector_Dot( s, p, &sdotp, _ex ); SIDL_CHECK(*_ex);
      if ( sdotp==0.0 )
      {
         ++ierr;
         if (i==1) i_prod=i_prod_0;
         break;
      }
      alpha = gamma / sdotp;

      gamma_old = gamma;

      /* x = x + alpha*p */
      ierr += bHYPRE_Vector_Axpy( *x, alpha, p, _ex ); SIDL_CHECK(*_ex);

      /* r = r - alpha*s */
      ierr += bHYPRE_Vector_Axpy( r, -alpha, s, _ex ); SIDL_CHECK(*_ex);
         
      /* s = C*r */
      ierr += bHYPRE_Vector_Clear( s, _ex ); /* is this needed ? */
      SIDL_CHECK(*_ex);
      ierr += bHYPRE_Solver_Apply( precond, r, &s, _ex ); /* A`s = r */
      SIDL_CHECK(*_ex);

      /* gamma = <r,s> */
      ierr += bHYPRE_Vector_Dot( r, s, &gamma, _ex ); SIDL_CHECK(*_ex);

      /* set i_prod for convergence test */
      if (two_norm)
      {
         ierr += bHYPRE_Vector_Dot( r, r, &i_prod, _ex ); SIDL_CHECK(*_ex);
      }
      else
         i_prod = gamma;

#if 0
      if (two_norm)
         printf("Iter (%d): ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
                i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
      else
         printf("Iter (%d): ||r||_C = %e, ||r||_C/||b||_C = %e\n",
                i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
#endif
 
      /* print norm info */
      if ( logging>0 || print_level>0 )
      {
         norms[i]     = sqrt(i_prod);
         rel_norms[i] = bi_prod ? sqrt(i_prod/bi_prod) : 0;
      }
      if ( print_level > 1 && my_id==0 )
      {
         if (two_norm)
         {
            if ( stop_crit && !rel_change && atolf==0 ) {  /* pure absolute tolerance */
               printf("% 5d    %e    %f\n", i, norms[i],
                      norms[i]/norms[i-1] );
            }
            else 
            {
               printf("% 5d    %e    %f    %e\n", i, norms[i],
                      norms[i]/norms[i-1], rel_norms[i] );
            }
         }
         else 
         {
               printf("% 5d    %e    %f    %e\n", i, norms[i],
                      norms[i]/norms[i-1], rel_norms[i] );
         }
      }


      /* check for convergence */
      if (i_prod / bi_prod < eps)
      {
         if (rel_change && i_prod > guard_zero_residual)
         {
            ierr += bHYPRE_Vector_Dot( p, p, &pi_prod, _ex ); SIDL_CHECK(*_ex);
            ierr += bHYPRE_Vector_Dot( *x, *x, &xi_prod, _ex ); SIDL_CHECK(*_ex);
            ratio = alpha*alpha*pi_prod/xi_prod;
            if (ratio < eps)
 	    {
               (data -> converged) = 1;
               break;
 	    }
         }
         else
         {
            (data -> converged) = 1;
            break;
         }
      }

      if ( (gamma<1.0e-292) && ((-gamma)<1.0e-292) ) {
         ierr = 1;
         break;
      }
      /* ... gamma should be >=0.  IEEE subnormal numbers are < 2**(-1022)=2.2e-308
         (and >= 2**(-1074)=4.9e-324).  So a gamma this small means we're getting
         dangerously close to subnormal or zero numbers (usually if gamma is small,
         so will be other variables).  Thus further calculations risk a crash.
         Such small gamma generally means no hope of progress anyway. */

      /*--------------------------------------------------------------------
       * Optional test to see if adequate progress is being made.
       * The average convergence factor is recorded and compared
       * against the tolerance 'cf_tol'. The weighting factor is  
       * intended to pay more attention to the test when an accurate
       * estimate for average convergence factor is available.  
       *--------------------------------------------------------------------*/

      if (cf_tol > 0.0)
      {
         cf_ave_0 = cf_ave_1;
         if ( i_prod_0<1.0e-292 ) {
            /* i_prod_0 is zero, or (almost) subnormal, yet i_prod wasn't small
               enough to pass the convergence test.  Therefore initial guess was good,
               and we're just calculating garbage - time to bail out before the
               next step, which will be a divide by zero (or close to it). */
            ierr = 1;
            break;
         }
	 cf_ave_1 = pow( i_prod / i_prod_0, 1.0/(2.0*i)); 

         weight   = fabs(cf_ave_1 - cf_ave_0);
         weight   = weight / hypre_max(cf_ave_1, cf_ave_0);
         weight   = 1.0 - weight;
#if 0
         printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
                i, cf_ave_1, cf_ave_0, weight );
#endif
         if (weight * cf_ave_1 > cf_tol) break;
      }

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = s + beta p */
      ierr += bHYPRE_Vector_Scale( p, beta, _ex ); SIDL_CHECK(*_ex);
      ierr += bHYPRE_Vector_Axpy( p, 1.0, s, _ex ); SIDL_CHECK(*_ex);
   }

   if ( print_level > 1 && my_id==0 )  /* formerly for par_csr only */
      printf("\n\n");

   data -> num_iterations = i;
   if (bi_prod > 0.0)
      data -> rel_residual_norm = sqrt(i_prod/bi_prod);
   else /* actually, we'll never get here... */
      data -> rel_residual_norm = 0.0;

   return ierr;

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_PCG_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_PCG_ApplyAdjoint(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.PCG.ApplyAdjoint} (ApplyAdjoint method) */

   return 1; /* not implemented */

    /* DO-NOT-DELETE splicer.end(bHYPRE.PCG.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_MPICommunicator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_PCG_fcast_bHYPRE_MPICommunicator(void* bi, sidl_BaseInterface* 
  _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_PCG_fconnect_bHYPRE_Operator(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_PCG_fcast_bHYPRE_Operator(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_PCG__object* impl_bHYPRE_PCG_fconnect_bHYPRE_PCG(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_PCG__connectI(url, ar, _ex);
}
struct bHYPRE_PCG__object* impl_bHYPRE_PCG_fcast_bHYPRE_PCG(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_PCG__cast(bi, _ex);
}
struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_PreconditionedSolver(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_PreconditionedSolver__connectI(url, ar, _ex);
}
struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_PCG_fcast_bHYPRE_PreconditionedSolver(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_PreconditionedSolver__cast(bi, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_PCG_fconnect_bHYPRE_Solver(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_PCG_fcast_bHYPRE_Solver(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_PCG_fconnect_bHYPRE_Vector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_PCG_fcast_bHYPRE_Vector(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_PCG_fconnect_sidl_BaseClass(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_PCG_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* impl_bHYPRE_PCG_fconnect_sidl_BaseInterface(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* impl_bHYPRE_PCG_fcast_sidl_BaseInterface(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_PCG_fconnect_sidl_ClassInfo(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_PCG_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_PCG_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_PCG_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface* 
  _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}

