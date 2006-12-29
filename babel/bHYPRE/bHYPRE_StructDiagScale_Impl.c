/*
 * File:          bHYPRE_StructDiagScale_Impl.c
 * Symbol:        bHYPRE.StructDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.StructDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructDiagScale" (version 1.0.0)
 * 
 * Diagonal scaling preconditioner for STruct matrix class.
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 */

#include "bHYPRE_StructDiagScale_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale._includes) */
/* Insert-Code-Here {bHYPRE.StructDiagScale._includes} (includes and arbitrary code) */


#include "HYPRE_struct_ls.h"
#include "bHYPRE_MPICommunicator_Impl.h"
#include "bHYPRE_StructVector_Impl.h"
#include "bHYPRE_StructMatrix_Impl.h"
#include <assert.h>
#include "hypre_babel_exception_handler.h"

/* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructDiagScale__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale._load) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructDiagScale__ctor(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale._ctor) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale._ctor} (constructor method) */

   struct bHYPRE_StructDiagScale__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructDiagScale__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> matrix = NULL;
   bHYPRE_StructDiagScale__set_data( self, data );
   /* hypre diagonal scaling requires no constructor or setup. */

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructDiagScale__ctor2(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale._ctor2) */
    /* Insert-Code-Here {bHYPRE.StructDiagScale._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructDiagScale__dtor(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale._dtor) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale._dtor} (destructor method) */

   struct bHYPRE_StructDiagScale__data * data;
   data = bHYPRE_StructDiagScale__get_data( self );

   bHYPRE_StructMatrix_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   /* delete any nontrivial data components here */
   hypre_TFree( data );

   return; hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale._dtor) */
  }
}

/*
 *  This function is the preferred way to create a Struct DiagScale solver. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_StructDiagScale
impl_bHYPRE_StructDiagScale_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructMatrix A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.Create) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.Create} (Create method) */

   bHYPRE_StructDiagScale solver = bHYPRE_StructDiagScale__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_StructDiagScale__data * data =
      bHYPRE_StructDiagScale__get_data( solver );

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   data->matrix = A;
   bHYPRE_StructMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return solver;

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.Create) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetOperator(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetOperator) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetOperator} (SetOperator method) */

   int ierr = 0;
   struct bHYPRE_StructDiagScale__data * data;
   bHYPRE_StructMatrix bH_A;

   bH_A = (bHYPRE_StructMatrix) bHYPRE_Operator__cast2( A, "bHYPRE.IJParCSRMatrix", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bH_A!=NULL );

   data = bHYPRE_StructDiagScale__get_data( self );
   data->matrix = bH_A;
   bHYPRE_StructMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetTolerance(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetTolerance) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetTolerance} (SetTolerance method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetMaxIterations(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetMaxIterations) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetMaxIterations} (SetMaxIterations method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetLogging(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetLogging) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetLogging} (SetLogging method) */

   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetPrintLevel(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetPrintLevel} (SetPrintLevel method) */

   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_GetNumIterations(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.GetNumIterations) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.GetNumIterations} (GetNumIterations method) */

   /* diagonal scaling is like 1 step of Jacobi */
   *num_iterations = 1;
   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_GetRelResidualNorm(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.GetRelResidualNorm) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.GetRelResidualNorm} (GetRelResidualNorm method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetCommunicator(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetCommunicator} (SetCommunicator method) */

   /* DEPRECATED   Use Create */

   int ierr = 0;
   struct bHYPRE_StructDiagScale__data * data;
   data = bHYPRE_StructDiagScale__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   bHYPRE_StructDiagScale__set_data( self, data );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructDiagScale_Destroy(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.Destroy) */
    /* Insert-Code-Here {bHYPRE.StructDiagScale.Destroy} (Destroy method) */
     bHYPRE_StructDiagScale_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetIntParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetIntParameter) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetIntParameter} (SetIntParameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetDoubleParameter) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetDoubleParameter} (SetDoubleParameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetStringParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetStringParameter) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetStringParameter} (SetStringParameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetIntArray1Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetIntArray2Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_GetIntValue(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.GetIntValue} (GetIntValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_GetDoubleValue(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.GetDoubleValue} (GetDoubleValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_Setup(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.Setup) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.Setup} (Setup method) */

   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_Apply(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.Apply) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.Apply} (Apply method) */

   int ierr = 0;
   MPI_Comm comm;
   HYPRE_StructSolver HSSdummy; /* required arg, not used */
   struct bHYPRE_StructDiagScale__data * data;
   bHYPRE_StructMatrix bA;
   HYPRE_StructMatrix HA;
   bHYPRE_StructVector b_b, b_x;
   HYPRE_StructVector Hb, Hx;
   struct bHYPRE_StructMatrix__data * dataA;
   struct bHYPRE_StructVector__data * datab, * datax;

   data = bHYPRE_StructDiagScale__get_data( self );
   comm = data->comm;
   hypre_assert( comm != MPI_COMM_NULL );
   bA = data->matrix;
   hypre_assert( bA != NULL );

   if ( *x==NULL )
   {  /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or
       * hypre_assert(x-has-the-right-size) */
      bHYPRE_Vector_Clone( b, x, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_Vector_Clear( *x, _ex ); SIDL_CHECK(*_ex);
   }

   b_b = (bHYPRE_StructVector) bHYPRE_Vector__cast2( b, "bHYPRE.StructVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( b_b!=NULL );

   datab = bHYPRE_StructVector__get_data( b_b );
   Hb = datab -> vec;

   b_x = (bHYPRE_StructVector) bHYPRE_Vector__cast2( *x, "bHYPRE.StructVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( b_x!=NULL );

   datax = bHYPRE_StructVector__get_data( b_x );
   Hx = datax -> vec;

   dataA = bHYPRE_StructMatrix__get_data( bA );
   HA = dataA -> matrix;

   /* does x = y/diagA as approximation to solving Ax=y for x ... */
   ierr += HYPRE_StructDiagScale( HSSdummy, HA, Hb, Hx );

   bHYPRE_StructVector_deleteRef( b_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_StructVector_deleteRef( b_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructDiagScale_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructDiagScale_ApplyAdjoint(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale.ApplyAdjoint} (ApplyAdjoint method) */

   /* The adjoint of a (real) diagonal matrix is itself, so just call Apply: */
     int32_t ierr = impl_bHYPRE_StructDiagScale_Apply( self, b, x, _ex ); SIDL_CHECK(*_ex);
     return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Solver(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_StructDiagScale__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructDiagScale(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_StructDiagScale__connectI(url, ar, _ex);
}
struct bHYPRE_StructDiagScale__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_StructDiagScale(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_StructDiagScale__cast(bi, _ex);
}
struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_StructMatrix__connectI(url, ar, _ex);
}
struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_StructMatrix(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_StructMatrix__cast(bi, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructDiagScale_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructDiagScale_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
