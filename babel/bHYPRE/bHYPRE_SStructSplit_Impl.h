/*
 * File:          bHYPRE_SStructSplit_Impl.h
 * Symbol:        bHYPRE.SStructSplit-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.SStructSplit
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_SStructSplit_Impl_h
#define included_bHYPRE_SStructSplit_Impl_h

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
#ifndef included_bHYPRE_SStructSplit_h
#include "bHYPRE_SStructSplit.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit._includes) */
/* Insert-Code-Here {bHYPRE.SStructSplit._includes} (include files) */
#include "HYPRE.h"
#include "HYPRE_sstruct_ls.h"
#include "utilities.h"
#include "bHYPRE_SStructMatrix.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit._includes) */

/*
 * Private data for class bHYPRE.SStructSplit
 */

struct bHYPRE_SStructSplit__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit._data) */
  /* Insert-Code-Here {bHYPRE.SStructSplit._data} (private data members) */

   MPI_Comm comm;   /* may not be needed if we use the static Create function exclusively */
   HYPRE_SStructSolver solver;
   bHYPRE_SStructMatrix matrix;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructSplit__data*
bHYPRE_SStructSplit__get_data(
  bHYPRE_SStructSplit);

extern void
bHYPRE_SStructSplit__set_data(
  bHYPRE_SStructSplit,
  struct bHYPRE_SStructSplit__data*);

extern
void
impl_bHYPRE_SStructSplit__load(
  void);

extern
void
impl_bHYPRE_SStructSplit__ctor(
  /* in */ bHYPRE_SStructSplit self);

extern
void
impl_bHYPRE_SStructSplit__dtor(
  /* in */ bHYPRE_SStructSplit self);

/*
 * User-defined object methods
 */

extern
bHYPRE_SStructSplit
impl_bHYPRE_SStructSplit_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_SStructSplit__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_SStructSplit(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_SStructSplit(struct 
  bHYPRE_SStructSplit__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_SStructSplit_SetCommunicator(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_SStructSplit_SetIntParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_SStructSplit_SetDoubleParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_SStructSplit_SetStringParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_SStructSplit_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructSplit_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_SStructSplit_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructSplit_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_SStructSplit_GetIntValue(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_SStructSplit_GetDoubleValue(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_SStructSplit_Setup(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_SStructSplit_Apply(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_SStructSplit_ApplyAdjoint(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_SStructSplit_SetOperator(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_SStructSplit_SetTolerance(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_SStructSplit_SetMaxIterations(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_SStructSplit_SetLogging(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_SStructSplit_SetPrintLevel(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_SStructSplit_GetNumIterations(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_SStructSplit_GetRelResidualNorm(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ double* norm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_SStructSplit__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_SStructSplit(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_SStructSplit(struct 
  bHYPRE_SStructSplit__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
