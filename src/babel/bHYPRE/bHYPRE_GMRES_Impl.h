/*
 * File:          bHYPRE_GMRES_Impl.h
 * Symbol:        bHYPRE.GMRES-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.GMRES
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_GMRES_Impl_h
#define included_bHYPRE_GMRES_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_GMRES_h
#include "bHYPRE_GMRES.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_Operator_h
#include "bHYPRE_Operator.h"
#endif
#ifndef included_bHYPRE_PreconditionedSolver_h
#include "bHYPRE_PreconditionedSolver.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.GMRES._includes) */
/* Insert-Code-Here {bHYPRE.GMRES._includes} (include files) */


/* DO-NOT-DELETE splicer.end(bHYPRE.GMRES._includes) */

/*
 * Private data for class bHYPRE.GMRES
 */

struct bHYPRE_GMRES__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.GMRES._data) */
  /* Insert-Code-Here {bHYPRE.GMRES._data} (private data members) */

   bHYPRE_MPICommunicator bmpicomm;
   bHYPRE_Operator matrix;
   bHYPRE_Solver precond;
   int      k_dim;
   int      min_iter;
   int      max_iter;
   int      rel_change;
   int      stop_crit;
   int      converged;
   double   tol;
   double   cf_tol;
   double   rel_residual_norm;

   bHYPRE_Vector r;
   bHYPRE_Vector w;
   bHYPRE_Vector * p;

   /* log info (always logged) */
   int      num_iterations;
 
   int     print_level; /* printing when print_level>0 */
   int     logging;  /* extra computations for logging when logging>0 */
   double  *norms;
   char    *log_file_name;

  /* DO-NOT-DELETE splicer.end(bHYPRE.GMRES._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_GMRES__data*
bHYPRE_GMRES__get_data(
  bHYPRE_GMRES);

extern void
bHYPRE_GMRES__set_data(
  bHYPRE_GMRES,
  struct bHYPRE_GMRES__data*);

extern
void
impl_bHYPRE_GMRES__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_GMRES__ctor(
  /* in */ bHYPRE_GMRES self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_GMRES__ctor2(
  /* in */ bHYPRE_GMRES self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_GMRES__dtor(
  /* in */ bHYPRE_GMRES self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_GMRES
impl_bHYPRE_GMRES_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_GMRES__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_GMRES(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_GMRES__object* impl_bHYPRE_GMRES_fcast_bHYPRE_GMRES(void* 
  bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_GMRES_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_Operator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_GMRES_fcast_bHYPRE_Operator(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_PreconditionedSolver(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_GMRES_fcast_bHYPRE_PreconditionedSolver(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_Solver(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_GMRES_fcast_bHYPRE_Solver(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_GMRES_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_GMRES_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_GMRES_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_GMRES_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_GMRES_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_GMRES_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_GMRES_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_GMRES_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_GMRES_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_GMRES_SetPreconditioner(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_Solver s,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_GetPreconditioner(
  /* in */ bHYPRE_GMRES self,
  /* out */ bHYPRE_Solver* s,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_Clone(
  /* in */ bHYPRE_GMRES self,
  /* out */ bHYPRE_PreconditionedSolver* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetOperator(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetTolerance(
  /* in */ bHYPRE_GMRES self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetMaxIterations(
  /* in */ bHYPRE_GMRES self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetLogging(
  /* in */ bHYPRE_GMRES self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetPrintLevel(
  /* in */ bHYPRE_GMRES self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_GetNumIterations(
  /* in */ bHYPRE_GMRES self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_GetRelResidualNorm(
  /* in */ bHYPRE_GMRES self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetCommunicator(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_GMRES_Destroy(
  /* in */ bHYPRE_GMRES self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetIntParameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetDoubleParameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetStringParameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetIntArray1Parameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetIntArray2Parameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetDoubleArray1Parameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_SetDoubleArray2Parameter(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_GetIntValue(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_GetDoubleValue(
  /* in */ bHYPRE_GMRES self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_Setup(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_Apply(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_GMRES_ApplyAdjoint(
  /* in */ bHYPRE_GMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_GMRES__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_GMRES(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_GMRES__object* impl_bHYPRE_GMRES_fcast_bHYPRE_GMRES(void* 
  bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_GMRES_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_Operator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_GMRES_fcast_bHYPRE_Operator(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_PreconditionedSolver(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_GMRES_fcast_bHYPRE_PreconditionedSolver(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_Solver(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_GMRES_fcast_bHYPRE_Solver(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_GMRES_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_GMRES_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_GMRES_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_GMRES_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_GMRES_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_GMRES_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_GMRES_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_GMRES_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_GMRES_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_GMRES_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
