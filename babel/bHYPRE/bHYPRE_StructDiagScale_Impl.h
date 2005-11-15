/*
 * File:          bHYPRE_StructDiagScale_Impl.h
 * Symbol:        bHYPRE.StructDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for bHYPRE.StructDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_StructDiagScale_Impl_h
#define included_bHYPRE_StructDiagScale_Impl_h

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
#ifndef included_bHYPRE_StructDiagScale_h
#include "bHYPRE_StructDiagScale.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale._includes) */
/* Insert-Code-Here {bHYPRE.StructDiagScale._includes} (include files) */

#include "HYPRE.h"
#include "utilities.h"
#include "bHYPRE_StructMatrix.h"

/* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale._includes) */

/*
 * Private data for class bHYPRE.StructDiagScale
 */

struct bHYPRE_StructDiagScale__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructDiagScale._data) */
  /* Insert-Code-Here {bHYPRE.StructDiagScale._data} (private data members) */

   MPI_Comm comm;
   bHYPRE_Operator matrix;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructDiagScale._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructDiagScale__data*
bHYPRE_StructDiagScale__get_data(
  bHYPRE_StructDiagScale);

extern void
bHYPRE_StructDiagScale__set_data(
  bHYPRE_StructDiagScale,
  struct bHYPRE_StructDiagScale__data*);

extern
void
impl_bHYPRE_StructDiagScale__load(
  void);

extern
void
impl_bHYPRE_StructDiagScale__ctor(
  /* in */ bHYPRE_StructDiagScale self);

extern
void
impl_bHYPRE_StructDiagScale__dtor(
  /* in */ bHYPRE_StructDiagScale self);

/*
 * User-defined object methods
 */

extern
bHYPRE_StructDiagScale
impl_bHYPRE_StructDiagScale_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct bHYPRE_StructDiagScale__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructDiagScale(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_StructDiagScale(struct 
  bHYPRE_StructDiagScale__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_StructDiagScale_SetCommunicator(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetIntParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetStringParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetIntArray1Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetIntArray2Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetIntValue(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetDoubleValue(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_StructDiagScale_Setup(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_StructDiagScale_Apply(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetOperator(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetTolerance(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetMaxIterations(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetLogging(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetPrintLevel(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetNumIterations(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetRelResidualNorm(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ double* norm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct bHYPRE_StructDiagScale__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructDiagScale(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_StructDiagScale(struct 
  bHYPRE_StructDiagScale__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
