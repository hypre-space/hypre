/*
 * File:          bHYPRE_StructJacobi_Impl.h
 * Symbol:        bHYPRE.StructJacobi-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.StructJacobi
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_StructJacobi_Impl_h
#define included_bHYPRE_StructJacobi_Impl_h

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
#ifndef included_bHYPRE_StructJacobi_h
#include "bHYPRE_StructJacobi.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi._includes) */
/* Insert-Code-Here {bHYPRE.StructJacobi._includes} (include files) */
#include "HYPRE.h"
#include "HYPRE_struct_ls.h"
#include "utilities.h"
#include "bHYPRE_StructMatrix.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi._includes) */

/*
 * Private data for class bHYPRE.StructJacobi
 */

struct bHYPRE_StructJacobi__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi._data) */
  /* Insert-Code-Here {bHYPRE.StructJacobi._data} (private data members) */
   MPI_Comm comm;   /* may not be needed if we use the static Create function exclusively */
   HYPRE_StructSolver solver;
   bHYPRE_StructMatrix matrix;
   double rel_resid_norm;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructJacobi__data*
bHYPRE_StructJacobi__get_data(
  bHYPRE_StructJacobi);

extern void
bHYPRE_StructJacobi__set_data(
  bHYPRE_StructJacobi,
  struct bHYPRE_StructJacobi__data*);

extern
void
impl_bHYPRE_StructJacobi__load(
  void);

extern
void
impl_bHYPRE_StructJacobi__ctor(
  /* in */ bHYPRE_StructJacobi self);

extern
void
impl_bHYPRE_StructJacobi__dtor(
  /* in */ bHYPRE_StructJacobi self);

/*
 * User-defined object methods
 */

extern
bHYPRE_StructJacobi
impl_bHYPRE_StructJacobi_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructMatrix A);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_StructMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_StructMatrix(struct 
  bHYPRE_StructMatrix__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructJacobi_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructJacobi_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_StructJacobi__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_StructJacobi(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_StructJacobi(struct 
  bHYPRE_StructJacobi__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructJacobi_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_StructJacobi_SetCommunicator(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_StructJacobi_SetIntParameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_StructJacobi_SetDoubleParameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_StructJacobi_SetStringParameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_StructJacobi_SetIntArray1Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructJacobi_SetIntArray2Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_StructJacobi_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructJacobi_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_StructJacobi_GetIntValue(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_StructJacobi_GetDoubleValue(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_StructJacobi_Setup(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_StructJacobi_Apply(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_StructJacobi_ApplyAdjoint(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_StructJacobi_SetOperator(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_StructJacobi_SetTolerance(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_StructJacobi_SetMaxIterations(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_StructJacobi_SetLogging(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_StructJacobi_SetPrintLevel(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_StructJacobi_GetNumIterations(
  /* in */ bHYPRE_StructJacobi self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_StructJacobi_GetRelResidualNorm(
  /* in */ bHYPRE_StructJacobi self,
  /* out */ double* norm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_StructMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_StructMatrix(struct 
  bHYPRE_StructMatrix__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructJacobi_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructJacobi_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_StructJacobi__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_StructJacobi(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_StructJacobi(struct 
  bHYPRE_StructJacobi__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructJacobi_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructJacobi_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
