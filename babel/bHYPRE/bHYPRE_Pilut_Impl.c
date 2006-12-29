/*
 * File:          bHYPRE_Pilut_Impl.c
 * Symbol:        bHYPRE.Pilut-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.Pilut
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.Pilut" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * Pilut has not been implemented yet.
 */

#include "bHYPRE_Pilut_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._includes) */
/* Put additional includes or other arbitrary code here... */


#include "hypre_babel_exception_handler.h"

/* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Pilut__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._load) */
  /* Insert-Code-Here {bHYPRE.Pilut._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Pilut__ctor(
  /* in */ bHYPRE_Pilut self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Pilut__ctor2(
  /* in */ bHYPRE_Pilut self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._ctor2) */
    /* Insert-Code-Here {bHYPRE.Pilut._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Pilut__dtor(
  /* in */ bHYPRE_Pilut self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._dtor) */
  }
}

/*
 *  This function is the preferred way to create a Pilut solver. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_Pilut
impl_bHYPRE_Pilut_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.Create) */
  /* Insert-Code-Here {bHYPRE.Pilut.Create} (Create method) */

   /* int ierr = 0;*/
   bHYPRE_Pilut solver = bHYPRE_Pilut__create(_ex); /* assumed to call HYPRE_PilutCreate()*/
   SIDL_CHECK(*_ex);
   /* >>> requires data to be set up...
      struct bHYPRE_Pilut__data * data = bHYPRE_Pilut__get_data( self );

      data -> comm = (MPI_Comm *) mpi_comm;
   */

   return solver;

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.Create) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetOperator(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetTolerance(
  /* in */ bHYPRE_Pilut self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetMaxIterations(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_Pilut_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetLogging(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_Pilut_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetPrintLevel(
  /* in */ bHYPRE_Pilut self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_GetNumIterations(
  /* in */ bHYPRE_Pilut self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_GetRelResidualNorm(
  /* in */ bHYPRE_Pilut self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetCommunicator(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Pilut_Destroy(
  /* in */ bHYPRE_Pilut self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.Destroy) */
    /* Insert-Code-Here {bHYPRE.Pilut.Destroy} (Destroy method) */
     bHYPRE_Pilut_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetIntParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetDoubleParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetStringParameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetIntArray1Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetIntArray2Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_GetIntValue(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_GetDoubleValue(
  /* in */ bHYPRE_Pilut self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_Setup(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.Setup) */
  /* Insert the implementation of the Setup method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_Apply(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.Apply) */
  /* Insert the implementation of the Apply method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Pilut_ApplyAdjoint(
  /* in */ bHYPRE_Pilut self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.Pilut.ApplyAdjoint} (ApplyAdjoint method) */

   return 1; /* not implemented */

  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Pilut_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Pilut_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_Pilut_fconnect_bHYPRE_Operator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_Pilut_fcast_bHYPRE_Operator(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_Pilut__object* impl_bHYPRE_Pilut_fconnect_bHYPRE_Pilut(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Pilut__connectI(url, ar, _ex);
}
struct bHYPRE_Pilut__object* impl_bHYPRE_Pilut_fcast_bHYPRE_Pilut(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Pilut__cast(bi, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_Pilut_fconnect_bHYPRE_Solver(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_Pilut_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_Pilut_fconnect_bHYPRE_Vector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_Pilut_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_Pilut_fconnect_sidl_BaseClass(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_Pilut_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_Pilut_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_Pilut_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_Pilut_fconnect_sidl_ClassInfo(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_Pilut_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_Pilut_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_Pilut_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
