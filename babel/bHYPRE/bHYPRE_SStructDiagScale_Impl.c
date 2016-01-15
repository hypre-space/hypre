/*
 * File:          bHYPRE_SStructDiagScale_Impl.c
 * Symbol:        bHYPRE.SStructDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.SStructDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructDiagScale" (version 1.0.0)
 */

#include "bHYPRE_SStructDiagScale_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale._includes) */
/* Insert-Code-Here {bHYPRE.SStructDiagScale._includes} (includes and arbitrary code) */



#include "HYPRE_sstruct_ls.h"
#include "bHYPRE_MPICommunicator_Impl.h"
#include "bHYPRE_SStructVector_Impl.h"
#include "bHYPRE_SStructMatrix_Impl.h"

#include "hypre_babel_exception_handler.h"

/* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructDiagScale__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale._load) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale._load} (static class initializer method) */
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructDiagScale__ctor(
  /* in */ bHYPRE_SStructDiagScale self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale._ctor) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale._ctor} (constructor method) */

   struct bHYPRE_SStructDiagScale__data * data;
   data = hypre_CTAlloc( struct bHYPRE_SStructDiagScale__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> matrix = NULL;
   bHYPRE_SStructDiagScale__set_data( self, data );
   /* hypre diagonal scaling requires no constructor or setup. */

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructDiagScale__ctor2(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale._ctor2) */
    /* Insert-Code-Here {bHYPRE.SStructDiagScale._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructDiagScale__dtor(
  /* in */ bHYPRE_SStructDiagScale self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale._dtor) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale._dtor} (destructor method) */

   struct bHYPRE_SStructDiagScale__data * data;
   data = bHYPRE_SStructDiagScale__get_data( self );

   bHYPRE_Operator_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   /* delete any nontrivial data components here */
   hypre_TFree( data );

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale._dtor) */
  }
}

/*
 *  This function is the preferred way to create a SStruct DiagScale solver. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_SStructDiagScale
impl_bHYPRE_SStructDiagScale_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.Create) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.Create} (Create method) */

   bHYPRE_SStructDiagScale solver = bHYPRE_SStructDiagScale__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_SStructDiagScale__data * data =
      bHYPRE_SStructDiagScale__get_data( solver );

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   data->matrix = A;
   bHYPRE_Operator_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return solver;

   hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.Create) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetOperator(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetOperator) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetOperator} (SetOperator method) */

   int ierr = 0;
   struct bHYPRE_SStructDiagScale__data * data;

   data = bHYPRE_SStructDiagScale__get_data( self );
   data->matrix = A;
   bHYPRE_Operator_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetTolerance(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetTolerance) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetTolerance} (SetTolerance method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetMaxIterations(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetMaxIterations) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetMaxIterations} (SetMaxIterations method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetLogging(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetLogging) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetLogging} (SetLogging method) */

   return 0;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetPrintLevel(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetPrintLevel} (SetPrintLevel method) */

   return 0;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_GetNumIterations(
  /* in */ bHYPRE_SStructDiagScale self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.GetNumIterations) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.GetNumIterations} (GetNumIterations method) */

   return 0;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_GetRelResidualNorm(
  /* in */ bHYPRE_SStructDiagScale self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.GetRelResidualNorm) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.GetRelResidualNorm} (GetRelResidualNorm method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetCommunicator(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetCommunicator} (SetCommunicator method) */

   /* DEPRECATED   Use Create */

   int ierr = 0;
   struct bHYPRE_SStructDiagScale__data * data;
   data = bHYPRE_SStructDiagScale__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   bHYPRE_SStructDiagScale__set_data( self, data );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructDiagScale_Destroy(
  /* in */ bHYPRE_SStructDiagScale self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.Destroy) */
    /* Insert-Code-Here {bHYPRE.SStructDiagScale.Destroy} (Destroy method) */
     bHYPRE_SStructDiagScale_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetIntParameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetIntParameter) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetIntParameter} (SetIntParameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetDoubleParameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetDoubleParameter) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetDoubleParameter} (SetDoubleParameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetStringParameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetStringParameter) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetStringParameter} (SetStringParameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_GetIntValue(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.GetIntValue} (GetIntValue method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_GetDoubleValue(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.GetDoubleValue} (GetDoubleValue method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_Setup(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.Setup) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.Setup} (Setup method) */

   return 0;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_Apply(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.Apply) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.Apply} (Apply method) */

   int ierr = 0;
   MPI_Comm comm;
   HYPRE_SStructSolver HSSdummy; /* required arg, not used */
   struct bHYPRE_SStructDiagScale__data * data;
   bHYPRE_Operator mat;
   bHYPRE_SStructMatrix bA;
   HYPRE_SStructMatrix HA;
   bHYPRE_SStructVector b_b, b_x;
   HYPRE_SStructVector Hb, Hx;
   struct bHYPRE_SStructMatrix__data * dataA;
   struct bHYPRE_SStructVector__data * datab, * datax;

   data = bHYPRE_SStructDiagScale__get_data( self );
   comm = data->comm;
   hypre_assert( comm != MPI_COMM_NULL );
   mat = data->matrix;
   /* SetOperator should have been called earlier */
   hypre_assert( mat != NULL );

   if ( *x==NULL )
   {  /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or
       * hypre_assert(x-has-the-right-size) */
      bHYPRE_Vector_Clone( b, x, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_Vector_Clear( *x, _ex ); SIDL_CHECK(*_ex);
   }

   b_b = (bHYPRE_SStructVector) bHYPRE_Vector__cast2( b, "bHYPRE.SStructVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( b_b!=NULL );

   datab = bHYPRE_SStructVector__get_data( b_b );
   Hb = datab -> vec;

   b_x = (bHYPRE_SStructVector) bHYPRE_Vector__cast2( *x, "bHYPRE.SStructVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( b_x!=NULL );

   datax = bHYPRE_SStructVector__get_data( b_x );
   Hx = datax -> vec;

   bA = (bHYPRE_SStructMatrix) bHYPRE_Operator__cast2( mat, "bHYPRE.SStructMatrix", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bA!=NULL );

   dataA = bHYPRE_SStructMatrix__get_data( bA );
   HA = dataA -> matrix;

   /* does x = y/diagA as approximation to solving Ax=y for x ... */
   ierr += HYPRE_SStructDiagScale( HSSdummy, HA, Hb, Hx );

   bHYPRE_SStructVector_deleteRef( b_b, _ex );
   bHYPRE_SStructVector_deleteRef( b_x, _ex );
   bHYPRE_SStructMatrix_deleteRef( bA, _ex );

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructDiagScale_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructDiagScale_ApplyAdjoint(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale.ApplyAdjoint} (ApplyAdjoint method) */

   /* The adjoint of a (real) diagonal matrix is itself, so just call Apply: */
     int32_t ierr = impl_bHYPRE_SStructDiagScale_Apply( self, b, x, _ex ); SIDL_CHECK(*_ex);
     return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructDiagScale_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_Operator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructDiagScale_fcast_bHYPRE_Operator(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_SStructDiagScale__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_SStructDiagScale(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructDiagScale__connectI(url, ar, _ex);
}
struct bHYPRE_SStructDiagScale__object* 
  impl_bHYPRE_SStructDiagScale_fcast_bHYPRE_SStructDiagScale(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructDiagScale__cast(bi, _ex);
}
struct bHYPRE_Solver__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_Solver(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_SStructDiagScale_fcast_bHYPRE_Solver(
  void* bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_Vector(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_SStructDiagScale_fcast_bHYPRE_Vector(
  void* bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructDiagScale_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructDiagScale_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructDiagScale_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructDiagScale_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}

