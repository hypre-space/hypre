/*
 * File:          bHYPRE_SStructDiagScale_Impl.h
 * Symbol:        bHYPRE.SStructDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for bHYPRE.SStructDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_SStructDiagScale_Impl_h
#define included_bHYPRE_SStructDiagScale_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
#ifndef included_bHYPRE_SStructDiagScale_h
#include "bHYPRE_SStructDiagScale.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale._includes) */
/* Insert-Code-Here {bHYPRE.SStructDiagScale._includes} (include files) */

#include "HYPRE.h"
#include "utilities.h"
#include "bHYPRE_SStructMatrix.h"

/* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale._includes) */

/*
 * Private data for class bHYPRE.SStructDiagScale
 */

struct bHYPRE_SStructDiagScale__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructDiagScale._data) */
  /* Insert-Code-Here {bHYPRE.SStructDiagScale._data} (private data members) */

   MPI_Comm comm;
   bHYPRE_Operator matrix;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructDiagScale._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructDiagScale__data*
bHYPRE_SStructDiagScale__get_data(
  bHYPRE_SStructDiagScale);

extern void
bHYPRE_SStructDiagScale__set_data(
  bHYPRE_SStructDiagScale,
  struct bHYPRE_SStructDiagScale__data*);

extern
void
impl_bHYPRE_SStructDiagScale__load(
  void);

extern
void
impl_bHYPRE_SStructDiagScale__ctor(
  /* in */ bHYPRE_SStructDiagScale self);

extern
void
impl_bHYPRE_SStructDiagScale__dtor(
  /* in */ bHYPRE_SStructDiagScale self);

/*
 * User-defined object methods
 */

extern
bHYPRE_SStructDiagScale
impl_bHYPRE_SStructDiagScale_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_SStructDiagScale__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_SStructDiagScale(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructDiagScale_fgetURL_bHYPRE_SStructDiagScale(struct 
  bHYPRE_SStructDiagScale__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_SStructDiagScale_SetCommunicator(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetIntParameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetDoubleParameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetStringParameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_SStructDiagScale_GetIntValue(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_SStructDiagScale_GetDoubleValue(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_SStructDiagScale_Setup(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_SStructDiagScale_Apply(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_SStructDiagScale_ApplyAdjoint(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetOperator(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetTolerance(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetMaxIterations(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetLogging(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_SStructDiagScale_SetPrintLevel(
  /* in */ bHYPRE_SStructDiagScale self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_SStructDiagScale_GetNumIterations(
  /* in */ bHYPRE_SStructDiagScale self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_SStructDiagScale_GetRelResidualNorm(
  /* in */ bHYPRE_SStructDiagScale self,
  /* out */ double* norm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_SStructDiagScale__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_bHYPRE_SStructDiagScale(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructDiagScale_fgetURL_bHYPRE_SStructDiagScale(struct 
  bHYPRE_SStructDiagScale__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructDiagScale_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructDiagScale_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
