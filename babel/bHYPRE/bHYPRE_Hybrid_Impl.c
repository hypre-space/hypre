/*
 * File:          bHYPRE_Hybrid_Impl.c
 * Symbol:        bHYPRE.Hybrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.Hybrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.Hybrid" (version 1.0.0)
 * 
 * Hybrid solver
 * first tries to solve with the specified Krylov solver, preconditioned by
 * If that fails to converge, it will try again with the user-specified
 * 
 * Specify the preconditioner  by calling SecondSolver's SetPreconditioner
 * method.  If no preconditioner is specified (equivalently, if the
 * preconditioner for SecondSolver is IdentitySolver), the preconditioner for
 * the second try will be one of the following defaults.
 * StructMatrix: SMG.  other matrix types: not implemented
 * 
 * The Hybrid solver's Setup method will call Setup on KrylovSolver, so the
 * user should not call Setup on KrylovSolver.
 */

#include "bHYPRE_Hybrid_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid._includes) */
/* Insert-Code-Here {bHYPRE.Hybrid._includes} (includes and arbitrary code) */



/* There are many possible generalizations, most of them obvious, but not worth
   doing unless there's a need. */

#include "bHYPRE_MPICommunicator_Impl.h"
#include <math.h>

#include "hypre_babel_exception_handler.h"
#include "bHYPRE_StructSMG.h"
#include "bHYPRE_StructDiagScale.h"
#include "bHYPRE_StructDiagScale_Impl.h"
#include "bHYPRE_ParCSRDiagScale.h"
#include "bHYPRE_SStructDiagScale.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Hybrid__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid._load) */
  /* Insert-Code-Here {bHYPRE.Hybrid._load} (static class initializer method) */
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Hybrid__ctor(
  /* in */ bHYPRE_Hybrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid._ctor) */
  /* Insert-Code-Here {bHYPRE.Hybrid._ctor} (constructor method) */

   struct bHYPRE_Hybrid__data * data;
   data = hypre_CTAlloc( struct bHYPRE_Hybrid__data, 1 );

   data -> mpicomm      = bHYPRE_MPICommunicator_CreateC( (void *)MPI_COMM_NULL, _ex ); SIDL_CHECK(*_ex);
   data -> krylov_solver_1 = (bHYPRE_PreconditionedSolver)NULL;
   data -> krylov_solver_2 = (bHYPRE_PreconditionedSolver)NULL;
   data -> solver_used = 0;

   bHYPRE_Hybrid__set_data( self, data );

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Hybrid__ctor2(
  /* in */ bHYPRE_Hybrid self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid._ctor2) */
    /* Insert-Code-Here {bHYPRE.Hybrid._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Hybrid__dtor(
  /* in */ bHYPRE_Hybrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid._dtor) */
  /* Insert-Code-Here {bHYPRE.Hybrid._dtor} (destructor method) */

   struct bHYPRE_Hybrid__data * data;
   data = bHYPRE_Hybrid__get_data( self );

   if (data)
   {
      if ( data -> krylov_solver_1 != (bHYPRE_PreconditionedSolver)NULL )
      {
         bHYPRE_PreconditionedSolver_deleteRef( data->krylov_solver_1, _ex ); SIDL_CHECK(*_ex);
      }
      if ( data -> krylov_solver_2 != (bHYPRE_PreconditionedSolver)NULL )
      {
         bHYPRE_PreconditionedSolver_deleteRef( data->krylov_solver_2, _ex ); SIDL_CHECK(*_ex);
      }
      if ( data->operator != (bHYPRE_Operator)NULL )
      {
         bHYPRE_Operator_deleteRef( data->operator, _ex ); SIDL_CHECK(*_ex);
      }

      hypre_TFree( data );
   }

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid._dtor) */
  }
}

/*
 *  This function is the preferred way to create a Hybrid solver. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_Hybrid
impl_bHYPRE_Hybrid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_PreconditionedSolver SecondSolver,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.Create) */
  /* Insert-Code-Here {bHYPRE.Hybrid.Create} (Create method) */

   bHYPRE_Hybrid solver = bHYPRE_Hybrid__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_Hybrid__data * data = bHYPRE_Hybrid__get_data( solver );

   if (data->mpicomm) {
      bHYPRE_MPICommunicator_deleteRef( data->mpicomm, _ex ); SIDL_CHECK(*_ex);
   }
   data->mpicomm = mpi_comm;
   data->krylov_solver_2 = SecondSolver;
   bHYPRE_PreconditionedSolver_addRef( data->krylov_solver_2, _ex ); SIDL_CHECK(*_ex);
   /* ... krylov_solver_2 may get modified during Setup */

   data->operator = A;
   bHYPRE_Operator_addRef( data->operator, _ex );

   return solver;

   hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.Create) */
  }
}

/*
 * Method:  GetFirstSolver[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_GetFirstSolver"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_GetFirstSolver(
  /* in */ bHYPRE_Hybrid self,
  /* out */ bHYPRE_PreconditionedSolver* FirstSolver,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.GetFirstSolver) */
  /* Insert-Code-Here {bHYPRE.Hybrid.GetFirstSolver} (GetFirstSolver method) */

   struct bHYPRE_Hybrid__data * data = bHYPRE_Hybrid__get_data( self );
   *FirstSolver = data -> krylov_solver_1;

   return 0;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.GetFirstSolver) */
  }
}

/*
 * Method:  GetSecondSolver[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_GetSecondSolver"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_GetSecondSolver(
  /* in */ bHYPRE_Hybrid self,
  /* out */ bHYPRE_PreconditionedSolver* SecondSolver,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.GetSecondSolver) */
  /* Insert-Code-Here {bHYPRE.Hybrid.GetSecondSolver} (GetSecondSolver method) */

   struct bHYPRE_Hybrid__data * data = bHYPRE_Hybrid__get_data( self );
   *SecondSolver = data -> krylov_solver_2;

   return 0;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.GetSecondSolver) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetOperator(
  /* in */ bHYPRE_Hybrid self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetOperator) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetOperator} (SetOperator method) */

   /* This function is not normally needed, as an operator is provided
      in Create */

   int ierr = 0;
   struct bHYPRE_Hybrid__data * data = bHYPRE_Hybrid__get_data( self );

   data->operator = A;
   bHYPRE_Operator_addRef( data->operator, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetTolerance(
  /* in */ bHYPRE_Hybrid self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetTolerance) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetTolerance} (SetTolerance method) */

   /* The tolerance should be set in the Krylov solver instead, prior to
      creating the Hybrid solver. */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetMaxIterations(
  /* in */ bHYPRE_Hybrid self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetMaxIterations) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetMaxIterations} (SetMaxIterations method) */

   /* MaxIterations should be set in the Krylov solver instead, prior to
      creating the Hybrid solver. */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_Hybrid_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetLogging(
  /* in */ bHYPRE_Hybrid self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetLogging) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetLogging} (SetLogging method) */

   /* Logging should be set in the Krylov solver instead, prior to
      creating the Hybrid solver. */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_Hybrid_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetPrintLevel(
  /* in */ bHYPRE_Hybrid self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetPrintLevel} (SetPrintLevel method) */

   /* PrintLevel should be set in the Krylov solver instead, prior to
      creating the Hybrid solver. */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_GetNumIterations(
  /* in */ bHYPRE_Hybrid self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.GetNumIterations) */

   int ierr = 0;
   int num_iterations_1, num_iterations_2;
   struct bHYPRE_Hybrid__data * data = bHYPRE_Hybrid__get_data( self );

   ierr += bHYPRE_PreconditionedSolver_GetNumIterations(
      data->krylov_solver_1, &num_iterations_1, _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_PreconditionedSolver_GetNumIterations(
      data->krylov_solver_2, &num_iterations_2, _ex ); SIDL_CHECK(*_ex);
   *num_iterations = num_iterations_1 + num_iterations_2;

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* Insert-Code-Here {bHYPRE.Hybrid.GetNumIterations} (GetNumIterations method) */

   
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_GetRelResidualNorm(
  /* in */ bHYPRE_Hybrid self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.GetRelResidualNorm) */
  /* Insert-Code-Here {bHYPRE.Hybrid.GetRelResidualNorm} (GetRelResidualNorm method) */

   int ierr = 0;
   struct bHYPRE_Hybrid__data * data = bHYPRE_Hybrid__get_data( self );

   switch ( data->solver_used )
   {
   case 0:
      ++ierr; break;
   case 1:
      ierr += bHYPRE_PreconditionedSolver_GetRelResidualNorm(
         data->krylov_solver_1, norm, _ex ); SIDL_CHECK(*_ex);
      break;
   case 2:
      ierr += bHYPRE_PreconditionedSolver_GetRelResidualNorm(
         data->krylov_solver_2, norm, _ex ); SIDL_CHECK(*_ex);
      break;
   }

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetCommunicator(
  /* in */ bHYPRE_Hybrid self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetCommunicator} (SetCommunicator method) */
   return 1;  /* deprecated and will never be implemented */
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Hybrid_Destroy(
  /* in */ bHYPRE_Hybrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.Destroy) */
    /* Insert-Code-Here {bHYPRE.Hybrid.Destroy} (Destroy method) */
     bHYPRE_Hybrid_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetIntParameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetIntParameter) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetIntParameter} (SetIntParameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetDoubleParameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetDoubleParameter) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetDoubleParameter} (SetDoubleParameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetStringParameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetStringParameter) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetStringParameter} (SetStringParameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetIntArray1Parameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetIntArray2Parameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.Hybrid.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_GetIntValue(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.Hybrid.GetIntValue} (GetIntValue method) */

   int ierr = 0;
   struct bHYPRE_Hybrid__data * data = bHYPRE_Hybrid__get_data( self );

   if ( strcmp(name,"NumIterations")==0 )
   {
      ierr += bHYPRE_Hybrid_GetNumIterations( self, value, _ex ); SIDL_CHECK(*_ex);
   }
   else if ( strcmp(name,"SolverUsed")==0 )
   {
      *value = data -> solver_used;
   }
   else
   {
      ierr = 1;
   }

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_GetDoubleValue(
  /* in */ bHYPRE_Hybrid self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.Hybrid.GetDoubleValue} (GetDoubleValue method) */

   int ierr = 0;

   if ( strcmp(name,"Final Relative Residual Norm")==0 ||
        strcmp(name,"FinalRelativeResidualNorm")==0 ||
        strcmp(name,"RelResidualNorm")==0 ||
        strcmp(name,"RelativeResidualNorm")==0 )
   {
      ierr += bHYPRE_Hybrid_GetRelResidualNorm( self, value, _ex ); SIDL_CHECK(*_ex);
   }
   else
   {
      ierr = 1;
   }

   return ierr;


   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_Setup(
  /* in */ bHYPRE_Hybrid self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.Setup) */
  /* Insert-Code-Here {bHYPRE.Hybrid.Setup} (Setup method) */

   int ierr = 0;
   struct bHYPRE_Hybrid__data * data = bHYPRE_Hybrid__get_data( self );
   bHYPRE_Solver precond_1, precond_2;
   bHYPRE_StructSMG SMG;
   bHYPRE_StructDiagScale StructDiagScale;
   bHYPRE_ParCSRDiagScale ParCSRDiagScale;
   bHYPRE_SStructDiagScale SStructDiagScale;
   bHYPRE_IJParCSRMatrix bHIJ_A;
   bHYPRE_StructMatrix bHS_A;

   ierr += bHYPRE_PreconditionedSolver_GetPreconditioner(
      data->krylov_solver_2, &precond_2, _ex ); SIDL_CHECK(*_ex);

   if ( bHYPRE_Solver__cast2( precond_2, "bHYPRE.IdentitySolver", _ex ) )
   {  /* Reset KrylovSolver's preconditioner to the Hybrid default.*/
      SIDL_CHECK(*_ex);
      bHYPRE_Solver_deleteRef( precond_2, _ex ); SIDL_CHECK(*_ex);
      /* The default preconditioner depends on the matrix/vector type. */
      if ( bHYPRE_Vector__cast2( b, "bHYPRE_StructVector", _ex ) )
     {
         SIDL_CHECK(*_ex);
         bHYPRE_Vector_deleteRef( b, _ex ); SIDL_CHECK(*_ex);
         bHS_A = (bHYPRE_StructMatrix) bHYPRE_Operator__cast2( data->operator, "bHYPRE.StructMatrix", _ex );
         SIDL_CHECK(*_ex);
         hypre_assert( bHS_A!=NULL );
         SMG = bHYPRE_StructSMG_Create( data->mpicomm, bHS_A, _ex ); SIDL_CHECK(*_ex);
         bHYPRE_StructMatrix_deleteRef( bHS_A, _ex ); SIDL_CHECK(*_ex );
         if ( precond_2!=(bHYPRE_Solver)NULL )
         {
            bHYPRE_Solver_deleteRef( precond_2, _ex ); SIDL_CHECK(*_ex);
         }
         precond_2 = bHYPRE_Solver__cast( SMG, _ex ); SIDL_CHECK(*_ex);
         ierr += bHYPRE_PreconditionedSolver_SetPreconditioner(
            data->krylov_solver_2, precond_2, _ex ); SIDL_CHECK(*_ex);
         bHYPRE_Solver_deleteRef( precond_2, _ex ); SIDL_CHECK(*_ex );
      }
      else
      {  /* default preconditioner not defined yet */
         SIDL_CHECK(*_ex);
         ++ierr;
      }
   }
   else SIDL_CHECK(*_ex);

   /* Make krylov_solver_1, same as krylov_solver_2 but with diagonal scaling as
    * preconditioner. */
   
   if ( data->krylov_solver_1 != (bHYPRE_PreconditionedSolver)NULL )
   {
      bHYPRE_PreconditionedSolver_deleteRef( data->krylov_solver_1, _ex ); SIDL_CHECK(*_ex);
   }
   ierr += bHYPRE_PreconditionedSolver_Clone( data->krylov_solver_2,
                                              &data->krylov_solver_1, _ex ); SIDL_CHECK(*_ex);
   if ( bHYPRE_Vector__cast2( b, "bHYPRE.StructVector", _ex ) )
   {
      SIDL_CHECK(*_ex);
      bHYPRE_Vector_deleteRef( b, _ex ); SIDL_CHECK(*_ex);

      bHS_A = (bHYPRE_StructMatrix) bHYPRE_Operator__cast2( data->operator, "bHYPRE.StructMatrix", _ex );
      SIDL_CHECK(*_ex);
      hypre_assert( bHS_A!=NULL );

      StructDiagScale = bHYPRE_StructDiagScale_Create(
         data->mpicomm, bHS_A, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_StructMatrix_deleteRef( bHS_A, _ex ); SIDL_CHECK(*_ex );
      if ( precond_1!=(bHYPRE_Solver)NULL )
      {
         bHYPRE_Solver_deleteRef( precond_1, _ex ); SIDL_CHECK(*_ex);
      }
      precond_1 = bHYPRE_Solver__cast( StructDiagScale, _ex ); SIDL_CHECK(*_ex);
      ierr += bHYPRE_PreconditionedSolver_SetPreconditioner(
         data->krylov_solver_1, precond_1, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_Solver_deleteRef( precond_1, _ex ); SIDL_CHECK(*_ex );
      bHYPRE_StructDiagScale_deleteRef(StructDiagScale, _ex); SIDL_CHECK(*_ex);
   }
   else SIDL_CHECK(*_ex);
   if ( bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex ) )
   {
      SIDL_CHECK(*_ex);
      bHYPRE_Vector_deleteRef( b, _ex ); SIDL_CHECK(*_ex);

      bHIJ_A = (bHYPRE_IJParCSRMatrix) bHYPRE_Operator__cast2( data->operator, "bHYPRE.IJParCSRMatrix", _ex );
      SIDL_CHECK(*_ex);

      ParCSRDiagScale = bHYPRE_ParCSRDiagScale_Create(
         data->mpicomm, bHIJ_A, _ex ); SIDL_CHECK(*_ex);
      if ( precond_1!=(bHYPRE_Solver)NULL )
      {
         bHYPRE_Solver_deleteRef( precond_1, _ex ); SIDL_CHECK(*_ex);
      }
      precond_1 = bHYPRE_Solver__cast( ParCSRDiagScale, _ex ); SIDL_CHECK(*_ex);
      ierr += bHYPRE_PreconditionedSolver_SetPreconditioner(
         data->krylov_solver_1, precond_1, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_Solver_deleteRef( precond_1, _ex ); SIDL_CHECK(*_ex );
      bHYPRE_ParCSRDiagScale_deleteRef(ParCSRDiagScale, _ex); SIDL_CHECK(*_ex);
   }
   else SIDL_CHECK(*_ex);
   if ( bHYPRE_Vector__cast2( b, "bHYPRE.SStructVector", _ex ) )
   {
      SIDL_CHECK(*_ex);
      bHYPRE_Vector_deleteRef( b, _ex ); SIDL_CHECK(*_ex);
      SStructDiagScale = bHYPRE_SStructDiagScale_Create(
         data->mpicomm, data->operator, _ex ); SIDL_CHECK(*_ex);
      if ( precond_1!=(bHYPRE_Solver)NULL )
      {
         bHYPRE_Solver_deleteRef( precond_1, _ex ); SIDL_CHECK(*_ex);
      }
      precond_1 = bHYPRE_Solver__cast( SStructDiagScale, _ex ); SIDL_CHECK(*_ex);
      ierr += bHYPRE_PreconditionedSolver_SetPreconditioner(
         data->krylov_solver_1, precond_1, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_Solver_deleteRef( precond_1, _ex ); SIDL_CHECK(*_ex );
      bHYPRE_SStructDiagScale_deleteRef(SStructDiagScale, _ex); SIDL_CHECK(*_ex);
   }
   else SIDL_CHECK(*_ex);

   /*  The user should not have called Setup on Krylov solver. */
   ierr += bHYPRE_PreconditionedSolver_Setup( data->krylov_solver_2, b, x, _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_PreconditionedSolver_Setup( data->krylov_solver_1, b, x, _ex ); SIDL_CHECK(*_ex);
   /* >>> It would decrease the memory footprint, and make it more correct for
      the user to set parameters, if we didn't  to Setup on krylov_solver_? until
      Apply.  We would destroy krylov_solver_1 before doing Setup on krylov_solver_2.
   */


   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_Apply(
  /* in */ bHYPRE_Hybrid self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.Apply) */
  /* Insert-Code-Here {bHYPRE.Hybrid.Apply} (Apply method) */

   int ierr = 0;
   struct bHYPRE_Hybrid__data * data = bHYPRE_Hybrid__get_data( self );
   int converged;

   ierr += bHYPRE_PreconditionedSolver_Apply( data->krylov_solver_1, b, x, _ex ); SIDL_CHECK(*_ex);

   data -> solver_used = 1;
   ierr += bHYPRE_PreconditionedSolver_GetIntValue(
      data->krylov_solver_1, "Converged", &converged, _ex ); SIDL_CHECK(*_ex);

   if ( converged == 0 )
   {
      ierr += bHYPRE_PreconditionedSolver_Apply( data->krylov_solver_2, b, x, _ex ); SIDL_CHECK(*_ex);
      data -> solver_used = 2;
   }

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Hybrid_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Hybrid_ApplyAdjoint(
  /* in */ bHYPRE_Hybrid self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Hybrid.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.Hybrid.ApplyAdjoint} (ApplyAdjoint method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.Hybrid.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_Hybrid__object* impl_bHYPRE_Hybrid_fconnect_bHYPRE_Hybrid(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Hybrid__connectI(url, ar, _ex);
}
struct bHYPRE_Hybrid__object* impl_bHYPRE_Hybrid_fcast_bHYPRE_Hybrid(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_Hybrid__cast(bi, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Hybrid_fconnect_bHYPRE_MPICommunicator(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Hybrid_fcast_bHYPRE_MPICommunicator(void* bi, sidl_BaseInterface* 
  _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_Hybrid_fconnect_bHYPRE_Operator(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_Hybrid_fcast_bHYPRE_Operator(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_Hybrid_fconnect_bHYPRE_PreconditionedSolver(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_PreconditionedSolver__connectI(url, ar, _ex);
}
struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_Hybrid_fcast_bHYPRE_PreconditionedSolver(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_PreconditionedSolver__cast(bi, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_Hybrid_fconnect_bHYPRE_Solver(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_Hybrid_fcast_bHYPRE_Solver(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_Hybrid_fconnect_bHYPRE_Vector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_Hybrid_fcast_bHYPRE_Vector(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_Hybrid_fconnect_sidl_BaseClass(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_Hybrid_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_Hybrid_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* impl_bHYPRE_Hybrid_fcast_sidl_BaseInterface(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_Hybrid_fconnect_sidl_ClassInfo(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_Hybrid_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_Hybrid_fconnect_sidl_RuntimeException(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_Hybrid_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface* 
  _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}

