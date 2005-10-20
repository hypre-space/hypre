/*
 * File:          bHYPRE_PCG_Impl.h
 * Symbol:        bHYPRE.PCG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for bHYPRE.PCG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_PCG_Impl_h
#define included_bHYPRE_PCG_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_PCG_h
#include "bHYPRE_PCG.h"
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
#ifndef included_bHYPRE_PreconditionedSolver_h
#include "bHYPRE_PreconditionedSolver.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._includes) */
/* Put additional include files here... */
#include "HYPRE.h"
#include "utilities.h"
#include "krylov.h"
#include "HYPRE_parcsr_ls.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.PCG._includes) */

/*
 * Private data for class bHYPRE.PCG
 */

struct bHYPRE_PCG__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.PCG._data) */
  /* Put private data members here... */

   MPI_Comm comm;
   HYPRE_Solver solver;
   bHYPRE_Operator matrix;
   char * vector_type;

   /* parameter cache, to save in Set*Parameter functions and copy in Apply: */
   double tol;
   double atolf;
   double cf_tol;
   int maxiter;
   int relchange;
   int twonorm;
   int log_level;
   int printlevel;
   int stop_crit;

   /* preconditioner cache, to save in SetPreconditioner and apply in Apply:*/
   char * precond_name;
   HYPRE_Solver * solverprecond;
   HYPRE_PtrToSolverFcn precond; /* function */
   HYPRE_PtrToSolverFcn precond_setup; /* function */

  /* DO-NOT-DELETE splicer.end(bHYPRE.PCG._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_PCG__data*
bHYPRE_PCG__get_data(
  bHYPRE_PCG);

extern void
bHYPRE_PCG__set_data(
  bHYPRE_PCG,
  struct bHYPRE_PCG__data*);

extern
void
impl_bHYPRE_PCG__load(
  void);

extern
void
impl_bHYPRE_PCG__ctor(
  /* in */ bHYPRE_PCG self);

extern
void
impl_bHYPRE_PCG__dtor(
  /* in */ bHYPRE_PCG self);

/*
 * User-defined object methods
 */

extern
bHYPRE_PCG
impl_bHYPRE_PCG_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern struct bHYPRE_PCG__object* impl_bHYPRE_PCG_fconnect_bHYPRE_PCG(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_PCG(struct bHYPRE_PCG__object* obj);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_Operator(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_PCG_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_PCG_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_PCG_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
extern
int32_t
impl_bHYPRE_PCG_SetCommunicator(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_PCG_SetIntParameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_PCG_SetDoubleParameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_PCG_SetStringParameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_PCG_SetIntArray1Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_PCG_SetIntArray2Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_PCG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_PCG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_PCG_GetIntValue(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_PCG_GetDoubleValue(
  /* in */ bHYPRE_PCG self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_PCG_Setup(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_PCG_Apply(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_PCG_SetOperator(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_PCG_SetTolerance(
  /* in */ bHYPRE_PCG self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_PCG_SetMaxIterations(
  /* in */ bHYPRE_PCG self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_PCG_SetLogging(
  /* in */ bHYPRE_PCG self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_PCG_SetPrintLevel(
  /* in */ bHYPRE_PCG self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_PCG_GetNumIterations(
  /* in */ bHYPRE_PCG self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_PCG_GetRelResidualNorm(
  /* in */ bHYPRE_PCG self,
  /* out */ double* norm);

extern
int32_t
impl_bHYPRE_PCG_SetPreconditioner(
  /* in */ bHYPRE_PCG self,
  /* in */ bHYPRE_Solver s);

extern struct bHYPRE_PCG__object* impl_bHYPRE_PCG_fconnect_bHYPRE_PCG(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_PCG(struct bHYPRE_PCG__object* obj);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_Operator(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_PCG_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_PCG_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_PCG_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_PCG_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_PCG_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
#ifdef __cplusplus
}
#endif
#endif
