/*
 * File:          bHYPRE_ParCSRDiagScale_Impl.h
 * Symbol:        bHYPRE.ParCSRDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_ParCSRDiagScale_Impl_h
#define included_bHYPRE_ParCSRDiagScale_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_IJParCSRMatrix_h
#include "bHYPRE_IJParCSRMatrix.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_ParCSRDiagScale_h
#include "bHYPRE_ParCSRDiagScale.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._includes) */
/* Put additional include files here... */


#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "bHYPRE_IJParCSRMatrix.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._includes) */

/*
 * Private data for class bHYPRE.ParCSRDiagScale
 */

struct bHYPRE_ParCSRDiagScale__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._data) */
  /* Put private data members here... */
   MPI_Comm comm;
   bHYPRE_IJParCSRMatrix matrix;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_ParCSRDiagScale__data*
bHYPRE_ParCSRDiagScale__get_data(
  bHYPRE_ParCSRDiagScale);

extern void
bHYPRE_ParCSRDiagScale__set_data(
  bHYPRE_ParCSRDiagScale,
  struct bHYPRE_ParCSRDiagScale__data*);

extern
void
impl_bHYPRE_ParCSRDiagScale__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_ParCSRDiagScale__ctor(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_ParCSRDiagScale__ctor2(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_ParCSRDiagScale__dtor(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_ParCSRDiagScale
impl_bHYPRE_ParCSRDiagScale_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_IJParCSRMatrix A,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ParCSRDiagScale__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_ParCSRDiagScale(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ParCSRDiagScale__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_ParCSRDiagScale(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Solver(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetOperator(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetTolerance(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetMaxIterations(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetLogging(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetPrintLevel(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_GetNumIterations(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_GetRelResidualNorm(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetCommunicator(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_ParCSRDiagScale_Destroy(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntParameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleParameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetStringParameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray1Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray2Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_GetIntValue(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_GetDoubleValue(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_Setup(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_Apply(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_ApplyAdjoint(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ParCSRDiagScale__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_ParCSRDiagScale(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ParCSRDiagScale__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_ParCSRDiagScale(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Solver(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
