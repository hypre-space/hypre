/*
 * File:          bHYPRE_BiCGSTAB_Impl.h
 * Symbol:        bHYPRE.BiCGSTAB-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for bHYPRE.BiCGSTAB
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_BiCGSTAB_Impl_h
#define included_bHYPRE_BiCGSTAB_Impl_h

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
#ifndef included_bHYPRE_BiCGSTAB_h
#include "bHYPRE_BiCGSTAB.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_bHYPRE_PreconditionedSolver_h
#include "bHYPRE_PreconditionedSolver.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB._includes) */
/* Insert-Code-Here {bHYPRE.BiCGSTAB._includes} (include files) */
/* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB._includes) */

/*
 * Private data for class bHYPRE.BiCGSTAB
 */

struct bHYPRE_BiCGSTAB__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.BiCGSTAB._data) */
  /* Insert-Code-Here {bHYPRE.BiCGSTAB._data} (private data members) */

   bHYPRE_MPICommunicator mpicomm;
   bHYPRE_Operator matrix;
   bHYPRE_Solver precond;

   double   tol;
   double   cf_tol;
   double   rel_residual_norm;
   int      min_iter;
   int      max_iter;
   int      stop_crit;
   int      converged;
   int      num_iterations;

   bHYPRE_Vector p;
   bHYPRE_Vector q;
   bHYPRE_Vector r;
   bHYPRE_Vector r0;
   bHYPRE_Vector s;
   bHYPRE_Vector v;

   /* additional log info (logged when `logging' > 0) */
   int      print_level;
   int      logging;
   double * norms;
   const char   * log_file_name;

  /* DO-NOT-DELETE splicer.end(bHYPRE.BiCGSTAB._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_BiCGSTAB__data*
bHYPRE_BiCGSTAB__get_data(
  bHYPRE_BiCGSTAB);

extern void
bHYPRE_BiCGSTAB__set_data(
  bHYPRE_BiCGSTAB,
  struct bHYPRE_BiCGSTAB__data*);

extern
void
impl_bHYPRE_BiCGSTAB__load(
  void);

extern
void
impl_bHYPRE_BiCGSTAB__ctor(
  /* in */ bHYPRE_BiCGSTAB self);

extern
void
impl_bHYPRE_BiCGSTAB__dtor(
  /* in */ bHYPRE_BiCGSTAB self);

/*
 * User-defined object methods
 */

extern
bHYPRE_BiCGSTAB
impl_bHYPRE_BiCGSTAB_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_BiCGSTAB__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_BiCGSTAB(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_BiCGSTAB(struct 
  bHYPRE_BiCGSTAB__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
extern
int32_t
impl_bHYPRE_BiCGSTAB_SetCommunicator(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetIntParameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetDoubleParameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetStringParameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetIntArray1Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetIntArray2Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetDoubleArray1Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetDoubleArray2Parameter(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_BiCGSTAB_GetIntValue(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_BiCGSTAB_GetDoubleValue(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_BiCGSTAB_Setup(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_BiCGSTAB_Apply(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_BiCGSTAB_ApplyAdjoint(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetOperator(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetTolerance(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetMaxIterations(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetLogging(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetPrintLevel(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_BiCGSTAB_GetNumIterations(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_BiCGSTAB_GetRelResidualNorm(
  /* in */ bHYPRE_BiCGSTAB self,
  /* out */ double* norm);

extern
int32_t
impl_bHYPRE_BiCGSTAB_SetPreconditioner(
  /* in */ bHYPRE_BiCGSTAB self,
  /* in */ bHYPRE_Solver s);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_BiCGSTAB__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_BiCGSTAB(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_BiCGSTAB(struct 
  bHYPRE_BiCGSTAB__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_BiCGSTAB_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BiCGSTAB_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
#ifdef __cplusplus
}
#endif
#endif
