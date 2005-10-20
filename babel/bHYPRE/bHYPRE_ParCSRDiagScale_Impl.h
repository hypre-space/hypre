/*
 * File:          bHYPRE_ParCSRDiagScale_Impl.h
 * Symbol:        bHYPRE.ParCSRDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for bHYPRE.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_ParCSRDiagScale_Impl_h
#define included_bHYPRE_ParCSRDiagScale_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_ParCSRDiagScale_h
#include "bHYPRE_ParCSRDiagScale.h"
#endif
#ifndef included_bHYPRE_Solver_h
#include "bHYPRE_Solver.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
#include "utilities.h"
#include "bHYPRE_IJParCSRMatrix.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._includes) */

/*
 * Private data for class bHYPRE.ParCSRDiagScale
 */

struct bHYPRE_ParCSRDiagScale__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._data) */
  /* Put private data members here... */
   MPI_Comm comm;
   bHYPRE_Operator matrix;
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
  void);

extern
void
impl_bHYPRE_ParCSRDiagScale__ctor(
  /* in */ bHYPRE_ParCSRDiagScale self);

extern
void
impl_bHYPRE_ParCSRDiagScale__dtor(
  /* in */ bHYPRE_ParCSRDiagScale self);

/*
 * User-defined object methods
 */

extern
bHYPRE_ParCSRDiagScale
impl_bHYPRE_ParCSRDiagScale_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern struct bHYPRE_ParCSRDiagScale__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_ParCSRDiagScale(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_bHYPRE_ParCSRDiagScale(struct 
  bHYPRE_ParCSRDiagScale__object* obj);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetCommunicator(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntParameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleParameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetStringParameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray1Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray2Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_GetIntValue(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_GetDoubleValue(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_Setup(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_Apply(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetOperator(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetTolerance(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetMaxIterations(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetLogging(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_SetPrintLevel(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_GetNumIterations(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_ParCSRDiagScale_GetRelResidualNorm(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ double* norm);

extern struct bHYPRE_ParCSRDiagScale__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_ParCSRDiagScale(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_bHYPRE_ParCSRDiagScale(struct 
  bHYPRE_ParCSRDiagScale__object* obj);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParCSRDiagScale_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
