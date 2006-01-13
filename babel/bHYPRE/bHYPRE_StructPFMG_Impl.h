/*
 * File:          bHYPRE_StructPFMG_Impl.h
 * Symbol:        bHYPRE.StructPFMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.StructPFMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_StructPFMG_Impl_h
#define included_bHYPRE_StructPFMG_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_Solver_h
#include "bHYPRE_Solver.h"
#endif
#ifndef included_bHYPRE_StructMatrix_h
#include "bHYPRE_StructMatrix.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_StructPFMG_h
#include "bHYPRE_StructPFMG.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
#include "HYPRE_struct_ls.h"
#include "utilities.h"
#include "bHYPRE_StructMatrix.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG._includes) */

/*
 * Private data for class bHYPRE.StructPFMG
 */

struct bHYPRE_StructPFMG__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG._data) */
  /* Put private data members here... */
   MPI_Comm comm;   /* may not be needed if we use the static Create function exclusively */
   HYPRE_StructSolver solver;
   bHYPRE_StructMatrix matrix;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructPFMG__data*
bHYPRE_StructPFMG__get_data(
  bHYPRE_StructPFMG);

extern void
bHYPRE_StructPFMG__set_data(
  bHYPRE_StructPFMG,
  struct bHYPRE_StructPFMG__data*);

extern
void
impl_bHYPRE_StructPFMG__load(
  void);

extern
void
impl_bHYPRE_StructPFMG__ctor(
  /* in */ bHYPRE_StructPFMG self);

extern
void
impl_bHYPRE_StructPFMG__dtor(
  /* in */ bHYPRE_StructPFMG self);

/*
 * User-defined object methods
 */

extern
bHYPRE_StructPFMG
impl_bHYPRE_StructPFMG_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructMatrix A);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_StructMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_StructMatrix(struct 
  bHYPRE_StructMatrix__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_StructPFMG__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_StructPFMG(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_StructPFMG(struct 
  bHYPRE_StructPFMG__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructPFMG_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructPFMG_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructPFMG_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_StructPFMG_SetCommunicator(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_StructPFMG_SetIntParameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_StructPFMG_SetDoubleParameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_StructPFMG_SetStringParameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_StructPFMG_SetIntArray1Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructPFMG_SetIntArray2Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_StructPFMG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructPFMG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_StructPFMG_GetIntValue(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_StructPFMG_GetDoubleValue(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_StructPFMG_Setup(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_StructPFMG_Apply(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_StructPFMG_ApplyAdjoint(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_StructPFMG_SetOperator(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_StructPFMG_SetTolerance(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_StructPFMG_SetMaxIterations(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_StructPFMG_SetLogging(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_StructPFMG_SetPrintLevel(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_StructPFMG_GetNumIterations(
  /* in */ bHYPRE_StructPFMG self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_StructPFMG_GetRelResidualNorm(
  /* in */ bHYPRE_StructPFMG self,
  /* out */ double* norm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_StructMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_StructMatrix(struct 
  bHYPRE_StructMatrix__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_StructPFMG__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_StructPFMG(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_StructPFMG(struct 
  bHYPRE_StructPFMG__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructPFMG_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructPFMG_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructPFMG_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructPFMG_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
