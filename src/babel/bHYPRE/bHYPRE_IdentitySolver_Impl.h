/*
 * File:          bHYPRE_IdentitySolver_Impl.h
 * Symbol:        bHYPRE.IdentitySolver-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.IdentitySolver
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_IdentitySolver_Impl_h
#define included_bHYPRE_IdentitySolver_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_IdentitySolver_h
#include "bHYPRE_IdentitySolver.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_Solver_h
#include "bHYPRE_Solver.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver._includes) */
/* Insert-Code-Here {bHYPRE.IdentitySolver._includes} (include files) */
/* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver._includes) */

/*
 * Private data for class bHYPRE.IdentitySolver
 */

struct bHYPRE_IdentitySolver__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IdentitySolver._data) */
  /* Put private data members here... */
   char * vector_type;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IdentitySolver._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_IdentitySolver__data*
bHYPRE_IdentitySolver__get_data(
  bHYPRE_IdentitySolver);

extern void
bHYPRE_IdentitySolver__set_data(
  bHYPRE_IdentitySolver,
  struct bHYPRE_IdentitySolver__data*);

extern
void
impl_bHYPRE_IdentitySolver__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IdentitySolver__ctor(
  /* in */ bHYPRE_IdentitySolver self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IdentitySolver__ctor2(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IdentitySolver__dtor(
  /* in */ bHYPRE_IdentitySolver self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_IdentitySolver
impl_bHYPRE_IdentitySolver_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_IdentitySolver__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_IdentitySolver(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IdentitySolver__object* 
  impl_bHYPRE_IdentitySolver_fcast_bHYPRE_IdentitySolver(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IdentitySolver_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_Operator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IdentitySolver_fcast_bHYPRE_Operator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_Solver(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_IdentitySolver_fcast_bHYPRE_Solver(void* bi, sidl_BaseInterface* 
  _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_Vector(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IdentitySolver_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IdentitySolver_fconnect_sidl_BaseClass(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IdentitySolver_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IdentitySolver_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IdentitySolver_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IdentitySolver_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IdentitySolver_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IdentitySolver_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IdentitySolver_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_IdentitySolver_SetOperator(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetTolerance(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetMaxIterations(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetLogging(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetPrintLevel(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_GetNumIterations(
  /* in */ bHYPRE_IdentitySolver self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_GetRelResidualNorm(
  /* in */ bHYPRE_IdentitySolver self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetCommunicator(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IdentitySolver_Destroy(
  /* in */ bHYPRE_IdentitySolver self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetIntParameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetDoubleParameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetStringParameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetIntArray1Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetIntArray2Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetDoubleArray1Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_SetDoubleArray2Parameter(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_GetIntValue(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_GetDoubleValue(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_Setup(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_Apply(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IdentitySolver_ApplyAdjoint(
  /* in */ bHYPRE_IdentitySolver self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_IdentitySolver__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_IdentitySolver(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IdentitySolver__object* 
  impl_bHYPRE_IdentitySolver_fcast_bHYPRE_IdentitySolver(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IdentitySolver_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_Operator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IdentitySolver_fcast_bHYPRE_Operator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_Solver(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_IdentitySolver_fcast_bHYPRE_Solver(void* bi, sidl_BaseInterface* 
  _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IdentitySolver_fconnect_bHYPRE_Vector(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IdentitySolver_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IdentitySolver_fconnect_sidl_BaseClass(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IdentitySolver_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IdentitySolver_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IdentitySolver_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IdentitySolver_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IdentitySolver_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IdentitySolver_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IdentitySolver_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);

#ifdef __cplusplus
}
#endif
#endif
