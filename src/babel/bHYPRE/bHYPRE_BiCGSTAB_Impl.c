/*
 * File:          bHYPRE_BiCGSTAB_Impl.c
 * Symbol:        bHYPRE.BiCGSTAB-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.BiCGSTAB
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.BiCGSTAB" (version 1.0.0)
 * 
 * BiCGSTAB solver.
 * This calls Babel-interface matrix and vector functions, so it will work
 * with any consistent matrix, vector, and preconditioner classes.
 */

#include "bHYPRE_BiCGSTAB_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB._includes) */
/* Insert-Code-Here {bHYPRE.BiCGSTAB._includes} (includes and arbitrary code) */


#include "bHYPRE_MPICommunicator_Impl.h"
#include "bHYPRE_IdentitySolver_Impl.h"
#include "bHYPRE_MatrixVectorView.h"
#include <math.h>

#include "hypre_babel_exception_handler.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_BiCGSTAB__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB._load) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB._load} (static class initializer method) */
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_BiCGSTAB__ctor(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB._ctor) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB._ctor} (constructor method) */

   struct bHYPRE_BiCGSTAB__data * data;
   data = hypre_CTAlloc( struct bHYPRE_BiCGSTAB__data, 1 );


   /* additional log info (logged when `logging' > 0) */

   data -> mpicomm      = bHYPRE_MPICommunicator_CreateC( (void *)MPI_COMM_NULL, _ex ); SIDL_CHECK(*_ex);
   data -> matrix       = (bHYPRE_Operator)NULL;
   data -> precond      = (bHYPRE_Solver)NULL;

   data -> tol           = 1.0e-06;
   data -> cf_tol        = 0.0;
   data -> rel_residual_norm = 0.0;
   data -> min_iter      = 0;
   data -> max_iter      = 1000;
   data -> stop_crit     = 0;
   data -> converged     = 0;
   data -> num_iterations = 0;

   data -> p            = (bHYPRE_Vector)NULL;
   data -> q            = (bHYPRE_Vector)NULL;
   data -> r            = (bHYPRE_Vector)NULL;
   data -> r0           = (bHYPRE_Vector)NULL;
   data -> s            = (bHYPRE_Vector)NULL;
   data -> v            = (bHYPRE_Vector)NULL;

   data -> print_level   = 0;
   data -> logging       = 0;
   data -> norms         = NULL;
   data -> log_file_name = NULL;

   bHYPRE_BiCGSTAB__set_data( self, data );

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_BiCGSTAB__ctor2(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB._ctor2) */
    /* Insert-Code-Here {bHYPRE.BiCGSTAB._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_BiCGSTAB__dtor(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB._dtor) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB._dtor} (destructor method) */

   struct bHYPRE_BiCGSTAB__data * data;
   data = bHYPRE_BiCGSTAB__get_data( self );

   if (data)
   {
      if ( (data -> norms) != NULL )
      {
         hypre_TFree( data -> norms );
         data -> norms = NULL;
      } 

      if ( data -> p != (bHYPRE_Vector)NULL )
         bHYPRE_Vector_deleteRef( data->p, _ex ); SIDL_CHECK(*_ex);
      if ( data -> q != (bHYPRE_Vector)NULL )
         bHYPRE_Vector_deleteRef( data->q, _ex ); SIDL_CHECK(*_ex);
      if ( data -> r != (bHYPRE_Vector)NULL )
         bHYPRE_Vector_deleteRef( data->r, _ex ); SIDL_CHECK(*_ex);
      if ( data -> r0 != (bHYPRE_Vector)NULL )
         bHYPRE_Vector_deleteRef( data->r0, _ex ); SIDL_CHECK(*_ex);
      if ( data -> s != (bHYPRE_Vector)NULL )
         bHYPRE_Vector_deleteRef( data->s, _ex ); SIDL_CHECK(*_ex);
      if ( data -> v != (bHYPRE_Vector)NULL )
         bHYPRE_Vector_deleteRef( data->v, _ex ); SIDL_CHECK(*_ex);
      if ( data -> matrix != (bHYPRE_Operator)NULL )
         bHYPRE_Operator_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
      if ( data -> precond != (bHYPRE_Solver)NULL )
         bHYPRE_Solver_deleteRef( data->precond, _ex ); SIDL_CHECK(*_ex);

      hypre_TFree( data );
   }

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB._dtor) */
  }
}

/*
 *  This function is the preferred way to create a BiCGSTAB solver. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_BiCGSTAB
impl_bHYPRE_BiCGSTAB_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.Create) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.Create} (Create method) */

   bHYPRE_BiCGSTAB solver = bHYPRE_BiCGSTAB__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( solver );
   bHYPRE_IdentitySolver Id  = bHYPRE_IdentitySolver_Create( mpi_comm, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_Solver IdS = bHYPRE_Solver__cast( Id, _ex ); SIDL_CHECK(*_ex);

   if (data->mpicomm) {
      bHYPRE_MPICommunicator_deleteRef( data->mpicomm, _ex ); SIDL_CHECK(*_ex);
   }
   data->mpicomm = mpi_comm;
   if( data->matrix != (bHYPRE_Operator)NULL )
      bHYPRE_Operator_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   data->matrix = A;
   bHYPRE_Operator_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   data->precond = IdS;

   bHYPRE_IdentitySolver_deleteRef( Id, _ex ); SIDL_CHECK(*_ex);
   /* ...Create and cast created 2 references, we're keeping only one (data->precond) */

   return solver;

   hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.Create) */
  }
}

/*
 * Set the preconditioner.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetPreconditioner"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetPreconditioner(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Solver s,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetPreconditioner) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetPreconditioner} (SetPreconditioner method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );
   if( data->precond != (bHYPRE_Solver)NULL )
      bHYPRE_Solver_deleteRef( data->precond, _ex ); SIDL_CHECK(*_ex);

   data->precond = s;
   bHYPRE_Solver_addRef( data->precond, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetPreconditioner) */
  }
}

/*
 * Method:  GetPreconditioner[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_GetPreconditioner"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_GetPreconditioner(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ bHYPRE_Solver* s,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.GetPreconditioner) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.GetPreconditioner} (GetPreconditioner method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   *s = data->precond;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.GetPreconditioner) */
  }
}

/*
 * Method:  Clone[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_Clone"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_Clone(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ bHYPRE_PreconditionedSolver* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.Clone) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.Clone} (Clone method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );
   struct bHYPRE_BiCGSTAB__data * datax;
   bHYPRE_BiCGSTAB BiCGSTAB_x;

   BiCGSTAB_x = bHYPRE_BiCGSTAB_Create( data->mpicomm, data->matrix, _ex ); SIDL_CHECK(*_ex);

   /* Copy most data members.
      The preconditioner copy will be a shallow copy (just the pointer);
      it is likely to be replaced later.
      But don't copy anything created in Setup (p,q,r,r0,s,v,norms,log_file_name).
      The user will call Setup on x, later
      Also don't copy the end-of-solve diagnostics (converged,num_iterations,
      rel_residual_norm) */

   datax = bHYPRE_BiCGSTAB__get_data( BiCGSTAB_x );
   datax->tol               = data->tol;
   datax->tol               = data->cf_tol;
   datax->min_iter          = data->min_iter;
   datax->max_iter          = data->max_iter;
   datax->stop_crit         = data->stop_crit;
   datax->print_level       = data->print_level;
   datax->logging           = data->logging;

   bHYPRE_BiCGSTAB_SetPreconditioner( BiCGSTAB_x, data->precond, _ex ); SIDL_CHECK(*_ex);

   *x = bHYPRE_PreconditionedSolver__cast( BiCGSTAB_x, _ex ); SIDL_CHECK(*_ex);

   bHYPRE_BiCGSTAB_deleteRef( BiCGSTAB_x,_ex ); SIDL_CHECK(*_ex);
   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.Clone) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetOperator(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetOperator) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetOperator} (SetOperator method) */

   /* DEPRECATED  the second argument in Create does the same thing */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data;

   data = bHYPRE_BiCGSTAB__get_data( self );
   if( data->matrix != (bHYPRE_Operator)NULL )
      bHYPRE_Operator_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   data->matrix = A;
   bHYPRE_Operator_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetTolerance(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetTolerance) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetTolerance} (SetTolerance method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   data -> tol = tolerance;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetMaxIterations(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetMaxIterations) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetMaxIterations} (SetMaxIterations method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   data -> max_iter = max_iterations;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetLogging(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetLogging) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetLogging} (SetLogging method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   data -> logging = level;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetPrintLevel(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetPrintLevel} (SetPrintLevel method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   data -> print_level = level;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_GetNumIterations(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.GetNumIterations) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.GetNumIterations} (GetNumIterations method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   *num_iterations = data->num_iterations;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_GetRelResidualNorm(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.GetRelResidualNorm) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.GetRelResidualNorm} (GetRelResidualNorm method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   *norm = data->rel_residual_norm;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetCommunicator(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetCommunicator} (SetCommunicator method) */
   return 1;  /* DEPRECATED and will never be implemented */
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_BiCGSTAB_Destroy(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.Destroy) */
    /* Insert-Code-Here {bHYPRE.BiCGSTAB.Destroy} (Destroy method) */
     bHYPRE_BiCGSTAB_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetIntParameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetIntParameter) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetIntParameter} (SetIntParameter method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   if ( strcmp(name,"MinIter")==0 || strcmp(name,"MinIterations")==0 )
   {
      data -> min_iter = value;
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      data -> max_iter = value;
   }
   else if ( strcmp(name,"StopCrit")==0 || strcmp(name,"stopping criterion")==0 )
   {
      data -> stop_crit = value;
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

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetDoubleParameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetDoubleParameter) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetDoubleParameter} (SetDoubleParameter method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   if ( strcmp(name,"Tolerance")==0  || strcmp(name,"Tol")==0 )
   {
      data -> tol = value;
   }
   else if ( strcmp(name,"ConvergenceFactorTol")==0 )
   {
      data -> cf_tol = value;
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetStringParameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetStringParameter) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetStringParameter} (SetStringParameter method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   if ( strcmp(name,"LogFileName")==0  || strcmp(name,"log file name")==0 )
   {
      data -> log_file_name = value;
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetIntArray1Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetIntArray2Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetDoubleArray1Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_SetDoubleArray2Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_GetIntValue(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.GetIntValue} (GetIntValue method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   if ( strcmp(name,"NumIterations")==0 )
   {
      *value = data -> num_iterations;
   }
   else if ( strcmp(name,"Converged")==0 )
   {
      *value = data -> converged;
   }
   else if ( strcmp(name,"MinIter")==0 || strcmp(name,"MinIterations")==0 )
   {
      *value = data -> max_iter;
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      *value = data -> max_iter;
   }
   else if ( strcmp(name,"StopCrit")==0 || strcmp(name,"stopping criterion")==0 )
   {
      *value = data -> stop_crit;
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

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_GetDoubleValue(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.GetDoubleValue} (GetDoubleValue method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );

   if ( strcmp(name,"Final Relative Residual Norm")==0 ||
        strcmp(name,"FinalRelativeResidualNorm")==0 ||
        strcmp(name,"RelResidualNorm")==0 ||
        strcmp(name,"RelativeResidualNorm")==0 )
   {
      *value = data -> rel_residual_norm;
   }
   else if ( strcmp(name,"Tolerance")==0  || strcmp(name,"Tol")==0 )
   {
      *value = data -> tol;
   }
   else if ( strcmp(name,"ConvergenceFactorTol")==0 )
   {
      *value = data -> cf_tol;
   }
   else
   {
      ierr = 1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_Setup(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.Setup) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.Setup} (Setup method) */

   int ierr = 0;
   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );
   int   max_iter = data->max_iter;
   bHYPRE_MatrixVectorView Vp, Vq, Vr, Vr0, Vs, Vv;

   /* Setup should not be called more than once. */
   hypre_assert( data->p == (bHYPRE_Vector)NULL );
   hypre_assert( data->q == (bHYPRE_Vector)NULL );
   hypre_assert( data->r == (bHYPRE_Vector)NULL );
   hypre_assert( data->r0 == (bHYPRE_Vector)NULL );
   hypre_assert( data->s == (bHYPRE_Vector)NULL );
   hypre_assert( data->v == (bHYPRE_Vector)NULL );

   ierr += bHYPRE_Vector_Clone( b, &(data->p), _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Clone( b, &(data->q), _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Clone( b, &(data->r), _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Clone( b, &(data->r0), _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Clone( b, &(data->s), _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Clone( b, &(data->v), _ex ); SIDL_CHECK(*_ex);
   Vp = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->p, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex);
   Vq = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->q, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex);
   Vr = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->r, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex);
   Vr0 = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->r0, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex);
   Vs = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->s, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex);
   Vv = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->v, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex);
   if ( Vp )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vp, _ex ); SIDL_CHECK(*_ex);
   }
   if ( Vq )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vq, _ex ); SIDL_CHECK(*_ex);
   }
   if ( Vr )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vr, _ex ); SIDL_CHECK(*_ex);
   }
   if ( Vr0 )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vr0, _ex ); SIDL_CHECK(*_ex);
   }
   if ( Vs )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vs, _ex ); SIDL_CHECK(*_ex);
   }
   if ( Vv )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vv, _ex ); SIDL_CHECK(*_ex);
   }

   ierr += bHYPRE_Solver_Setup( data->precond, b, x, _ex ); SIDL_CHECK(*_ex);

   if ( data->logging>0  || data->print_level>0 ) 
   {  /* arrays needed for logging */
      if ( data->norms == NULL )
         data->norms = hypre_CTAlloc( double, max_iter + 1 );
      if ( data->log_file_name == NULL )
         data->log_file_name = "bicgstab.out.log";
   }

   bHYPRE_MatrixVectorView_deleteRef( Vp, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_MatrixVectorView_deleteRef( Vq, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_MatrixVectorView_deleteRef( Vr, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_MatrixVectorView_deleteRef( Vr0, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_MatrixVectorView_deleteRef( Vs, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_MatrixVectorView_deleteRef( Vv, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_Apply(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.Apply) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.Apply} (Apply method) */

   struct bHYPRE_BiCGSTAB__data * data = bHYPRE_BiCGSTAB__get_data( self );
   bHYPRE_Operator   A            = data -> matrix;
   int               min_iter     = data -> min_iter;
   int 		     max_iter     = data -> max_iter;
   int 		     stop_crit    = data -> stop_crit;
   double 	     accuracy     = data -> tol;
   double 	     cf_tol       = data -> cf_tol;
   bHYPRE_Vector     p            = data -> p;
   bHYPRE_Vector     q            = data -> q;
   bHYPRE_Vector     r            = data -> r;
   bHYPRE_Vector     r0           = data -> r0;
   bHYPRE_Vector     s            = data -> s;
   bHYPRE_Vector     v            = data -> v;
   bHYPRE_Solver     precond      = data -> precond;
   int             logging        = data -> logging;
   int             print_level    = data -> print_level;
   double        * norms          = data -> norms;
/*   char          * log_file_name  = data -> log_file_name;*/
   
   int        ierr = 0;
   int        iter; 
   double     alpha, beta, gamma, epsilon, temp, res, r_norm, b_norm;
   double     epsmac = 1.e-128; 
   double     ieee_check = 0.;
   double     cf_ave_0 = 0.0;
   double     cf_ave_1 = 0.0;
   double     weight;
   double     r_norm_0;
   bHYPRE_MPICommunicator  bmpicomm = data -> mpicomm;
   MPI_Comm        comm;
   int        my_id, num_procs;

   comm = bHYPRE_MPICommunicator__get_data(bmpicomm)->mpi_comm;
   MPI_Comm_size( comm, &num_procs );
   MPI_Comm_rank( comm, &my_id );

   /* initialize work arrays */
   /* b=r0 */  /* >>> Should the user be able to initialize r0? <<< */
   ierr += bHYPRE_Vector_Copy( r0, b, _ex ); SIDL_CHECK(*_ex);

   /* compute initial residual */
   /* note: This would be a bit simpler with a matvec.  BTW, the standard hypre
    * matvec( a, A, x, b, y ) does y = aAx + by, a & b scalars, x & y vectors */
   /* p=r=r0 = r0 - Ax */
   ierr += bHYPRE_Operator_Apply( A, *x, &r, _ex ); /* r = Ax */
   SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Axpy( r0, -1.0, r, _ex );  /* r0 = r0 - r = r0 - Ax */
   SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Copy( r, r0, _ex );        /* r = r0 */
   SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Copy( p, r0, _ex );        /* p = r0 */
   SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Dot( b, b, &b_norm, _ex ); /* b_norm = <b,b> */
   SIDL_CHECK(*_ex);
   b_norm = sqrt( b_norm );           /* b_norm = sqrt(b_norm) = L2-norm of b */

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (b_norm != 0.) ieee_check = b_norm/b_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0)
      {
        printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        printf("ERROR -- hypre_BiCGSTABSolve: INFs and/or NaNs detected in input.\n");
        printf("User probably placed non-numerics in supplied b.\n");
        printf("Returning error flag += 101.  Program not terminated.\n");
        printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

   ierr += bHYPRE_Vector_Dot( r0, r0, &res, _ex );   SIDL_CHECK(*_ex);
   r_norm = sqrt(res);
   r_norm_0 = r_norm;
 
   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (r_norm != 0.) ieee_check = r_norm/r_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0)
      {
        printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        printf("ERROR -- hypre_BiCGSTABSolve: INFs and/or NaNs detected in input.\n");
        printf("User probably placed non-numerics in supplied A or x_0.\n");
        printf("Returning error flag += 101.  Program not terminated.\n");
        printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

   if (logging > 0)
   {
      norms[0] = r_norm;
      if (print_level > 0 && my_id == 0)
      {
  	 printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("Initial L2 norm of residual: %e\n", r_norm);
      }
      
   }
   iter = 0;

   if (b_norm > 0.0)
   {
/* convergence criterion |r_i| <= accuracy*|b| if |b| > 0 */
     epsilon = accuracy * b_norm;
   }
   else
   {
/* convergence criterion |r_i| <= accuracy*|r0| if |b| = 0 */
     epsilon = accuracy * r_norm;
   };

/* convergence criterion |r_i| <= accuracy , absolute residual norm*/
   if (stop_crit)
      epsilon = accuracy;

   if (print_level > 0 && my_id == 0)
   {
      if (b_norm > 0.0)
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
          printf("-----    ------------    ---------- ------------\n");
      }
      else
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate\n");
          printf("-----    ------------    ----------\n");
      
      }
   }

   data -> num_iterations = iter;
   if (b_norm > 0.0)
      data -> rel_residual_norm = r_norm/b_norm;

   /* **********************************************
    *             main iteration loop:
    * ********************************************** */

   while (iter < max_iter)
   {

        if (r_norm == 0.0)
        {
           data -> converged = 1;
	   ierr = 0;
	   return ierr;
	}

        /* check for convergence, evaluate actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           ierr += bHYPRE_Vector_Copy( r, b, _ex );   SIDL_CHECK(*_ex);
           ierr += bHYPRE_Operator_Apply( A, *x, &v, _ex ); /* v = Ax */
           SIDL_CHECK(*_ex);
           ierr += bHYPRE_Vector_Axpy( r, -1.0, v, _ex );   /* r = r - v = b - Ax */
           SIDL_CHECK(*_ex);
           ierr += bHYPRE_Vector_Dot( r, r, &r_norm, _ex ); /* r_norm = <r,r> */
           SIDL_CHECK(*_ex);
           r_norm = sqrt( r_norm );  /* r_norm = L2 norm of r */
	   if (r_norm <= epsilon)
           {
              if (print_level > 0 && my_id == 0)
              {
                 printf("\n\n");
                 printf("Final L2 norm of residual: %e\n\n", r_norm);
              }
              data -> converged = 1;
              break;
           }
	   else
	   {
              ierr += bHYPRE_Vector_Copy( p, r, _ex );  /* p = r */
              SIDL_CHECK(*_ex);
	   }
	}

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
           cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));

           weight   = fabs(cf_ave_1 - cf_ave_0);
           weight   = weight / hypre_max(cf_ave_1, cf_ave_0);
           weight   = 1.0 - weight;
#if 0
           printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
                i, cf_ave_1, cf_ave_0, weight );
#endif
           if (weight * cf_ave_1 > cf_tol) break;
        }

        iter++;

        ierr += bHYPRE_Solver_Apply( precond, p, &v, _ex ); /* v=Cp */
        SIDL_CHECK(*_ex);
        ierr += bHYPRE_Operator_Apply( A, v, &q, _ex );  /* q=Av=Acp */
        SIDL_CHECK(*_ex);
        ierr += bHYPRE_Vector_Dot( r0, q, &temp, _ex );  /* temp=<r0,q> */
        SIDL_CHECK(*_ex);
      	if (fabs(temp) >= epsmac)
	   alpha = res/temp;
	else
	{
	   printf("BiCGSTAB broke down!! divide by near zero\n");
	   return(1);
	}
        ierr += bHYPRE_Vector_Axpy( *x, alpha, v, _ex );  /* x = x + alpha*v */
        SIDL_CHECK(*_ex);
        ierr += bHYPRE_Vector_Axpy( r, -alpha, q, _ex );  /* r = r - alpha*q */
        SIDL_CHECK(*_ex);
        ierr += bHYPRE_Solver_Apply( precond, r, &v, _ex ); /* v=Cr */
        SIDL_CHECK(*_ex);
        ierr += bHYPRE_Operator_Apply( A, v, &s, _ex );   /* s = Av */
        SIDL_CHECK(*_ex);
        ierr += bHYPRE_Vector_Dot( r, s, &temp, _ex );    /* temp = <r,s> */
        SIDL_CHECK(*_ex);
        ierr += bHYPRE_Vector_Dot( s, s, &gamma, _ex );   /* gamma = <s,s> */
        SIDL_CHECK(*_ex);
        gamma = temp / gamma;               /* gamma = <r,s> / <s,s> */
        ierr += bHYPRE_Vector_Axpy( *x, gamma, v, _ex );  /* x = x + gamma*v */
        SIDL_CHECK(*_ex);
        ierr += bHYPRE_Vector_Axpy( r, -gamma, s, _ex );  /* r = r - gamma*s */
        SIDL_CHECK(*_ex);
      	if (fabs(res) >= epsmac)
           beta = 1.0/res;
	else
	{
	   printf("BiCGSTAB broke down!! res=0 \n");
	   return(2);
	}
        ierr += bHYPRE_Vector_Dot( r0, r, &res, _ex ); /* res = <r0,r> */
        SIDL_CHECK(*_ex);
        beta *= res;
        ierr += bHYPRE_Vector_Axpy( p, -gamma, q, _ex ); /* p = p - gamma*q */
        SIDL_CHECK(*_ex);
      	if (fabs(gamma) >= epsmac)
        {
           ierr += bHYPRE_Vector_Scale( p, beta*alpha/gamma, _ex );
           SIDL_CHECK(*_ex);
           /* ... p *= beta * alpha / gamma */
        }
	else
	{
	   printf("BiCGSTAB broke down!! gamma=0 \n");
	   return(3);
	}
        ierr += bHYPRE_Vector_Axpy( p, 1.0, r, _ex ); /* p = p + r */
        SIDL_CHECK(*_ex);

        ierr += bHYPRE_Vector_Dot( r, r, &r_norm, _ex ); /* r_norm = <r,r> */
        SIDL_CHECK(*_ex);
        r_norm = sqrt( r_norm );  /* r_norm = L2 norm of r */
	if (logging > 0)
	{
	   norms[iter] = r_norm;
	}

        if (print_level > 0 && my_id == 0)
	{
           if (b_norm > 0.0)
              printf("% 5d    %e    %f   %e\n", iter, norms[iter],
			norms[iter]/norms[iter-1], norms[iter]/b_norm);
           else
              printf("% 5d    %e    %f\n", iter, norms[iter],
		norms[iter]/norms[iter-1]);
	}
   }

   data -> num_iterations = iter;
   if (b_norm > 0.0)
      data -> rel_residual_norm = r_norm/b_norm;
   if (b_norm == 0.0)
      data -> rel_residual_norm = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BiCGSTAB_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BiCGSTAB_ApplyAdjoint(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB.ApplyAdjoint} (ApplyAdjoint method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_BiCGSTAB__object* impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_BiCGSTAB(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_BiCGSTAB__connectI(url, ar, _ex);
}
struct bHYPRE_BiCGSTAB__object* impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_BiCGSTAB(
  void* bi, sidl_BaseInterface* _ex) {
  return bHYPRE_BiCGSTAB__cast(bi, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Operator(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_Operator(
  void* bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_PreconditionedSolver(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_PreconditionedSolver__connectI(url, ar, _ex);
}
struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_PreconditionedSolver(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_PreconditionedSolver__cast(bi, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Solver(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Vector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_BiCGSTAB_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_BiCGSTAB_fconnect_sidl_BaseClass(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_BiCGSTAB_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_BaseInterface(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_BiCGSTAB_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* 
  _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_BiCGSTAB_fconnect_sidl_ClassInfo(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_BiCGSTAB_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_BiCGSTAB_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
