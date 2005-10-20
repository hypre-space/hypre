/*
 * File:          bHYPRE_IdentitySolver_Impl.c
 * Symbol:        bHYPRE.IdentitySolver-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for bHYPRE.IdentitySolver
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.IdentitySolver" (version 1.0.0)
 * 
 * Identity solver, just solves an identity matrix, for when you don't really
 * want a preconditioner
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 */

#include "bHYPRE_IdentitySolver_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_IdentitySolver__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver._load) */
  /* Insert-Code-Here {bHYPRE.IdentitySolver._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_IdentitySolver__ctor(
  /* in */ bHYPRE_IdentitySolver self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_IdentitySolver__dtor(
  /* in */ bHYPRE_IdentitySolver self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver._dtor) */
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_IdentitySolver
impl_bHYPRE_IdentitySolver_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.Create) */
  /* Insert-Code-Here {bHYPRE.IdentitySolver.Create} (Create method) */

   return bHYPRE_IdentitySolver__create();

  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.Create) */
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetCommunicator(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_MPICommunicator mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetIntParameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetDoubleParameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetStringParameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetIntArray1Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetIntArray2Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetDoubleArray1Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetDoubleArray2Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_GetIntValue(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* out */ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_GetDoubleValue(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* out */ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_Setup(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.Setup) */
  /* Insert the implementation of the Setup method here... */
   return 0;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_Apply(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.Apply) */
  /* Insert the implementation of the Apply method here... */
   return bHYPRE_Vector_Copy( *x, b );  /* copies data of b to x */
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.Apply) */
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetOperator(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetTolerance(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetMaxIterations(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetMaxIterations) */
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetLogging(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetLogging) */
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_SetPrintLevel(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_GetNumIterations(
  /* in */ bHYPRE_IdentitySolver self,
  /* out */ int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */
   num_iterations = 0;
   return 0;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IdentitySolver_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IdentitySolver_GetRelResidualNorm(
  /* in */ bHYPRE_IdentitySolver self,
  /* out */ double* norm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */
   *norm = 0.0;
   return 0;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver.GetRelResidualNorm) */
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_Solver__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connect(url, _ex);
}
char * impl_bHYPRE_IdentitySolver_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj) {
  return bHYPRE_Solver__getURL(obj);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connect(url, _ex);
}
char * impl_bHYPRE_IdentitySolver_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) {
  return bHYPRE_MPICommunicator__getURL(obj);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connect(url, _ex);
}
char * impl_bHYPRE_IdentitySolver_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) {
  return bHYPRE_Operator__getURL(obj);
}
struct bHYPRE_IdentitySolver__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_IdentitySolver(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_IdentitySolver__connect(url, _ex);
}
char * impl_bHYPRE_IdentitySolver_fgetURL_bHYPRE_IdentitySolver(struct 
  bHYPRE_IdentitySolver__object* obj) {
  return bHYPRE_IdentitySolver__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_IdentitySolver_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_IdentitySolver_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connect(url, _ex);
}
char * impl_bHYPRE_IdentitySolver_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) {
  return bHYPRE_Vector__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_IdentitySolver_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_IdentitySolver_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_IdentitySolver_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_IdentitySolver_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
