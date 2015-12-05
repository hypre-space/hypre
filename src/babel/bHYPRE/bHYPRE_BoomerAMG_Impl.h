/*
 * File:          bHYPRE_BoomerAMG_Impl.h
 * Symbol:        bHYPRE.BoomerAMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.BoomerAMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_BoomerAMG_Impl_h
#define included_bHYPRE_BoomerAMG_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_BoomerAMG_h
#include "bHYPRE_BoomerAMG.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG._includes) */
/* Put additional include files here... */


#include "HYPRE_parcsr_ls.h"
#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "bHYPRE_IJParCSRMatrix.h"
#include "bHYPRE_IJParCSRVector.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG._includes) */

/*
 * Private data for class bHYPRE.BoomerAMG
 */

struct bHYPRE_BoomerAMG__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG._data) */
  /* Put private data members here... */
   MPI_Comm comm;
   HYPRE_Solver solver;
   bHYPRE_IJParCSRMatrix matrix;
  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_BoomerAMG__data*
bHYPRE_BoomerAMG__get_data(
  bHYPRE_BoomerAMG);

extern void
bHYPRE_BoomerAMG__set_data(
  bHYPRE_BoomerAMG,
  struct bHYPRE_BoomerAMG__data*);

extern
void
impl_bHYPRE_BoomerAMG__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_BoomerAMG__ctor(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_BoomerAMG__ctor2(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_BoomerAMG__dtor(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_BoomerAMG
impl_bHYPRE_BoomerAMG_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_IJParCSRMatrix A,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_BoomerAMG__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_BoomerAMG(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_BoomerAMG__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_BoomerAMG(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Operator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Solver(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_Solver(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BoomerAMG_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BoomerAMG_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BoomerAMG_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_BoomerAMG_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_BoomerAMG_SetLevelRelaxWt(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ double relax_wt,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_InitGridRelaxation(
  /* in */ bHYPRE_BoomerAMG self,
  /* out array<int,column-major> */ struct sidl_int__array** num_grid_sweeps,
  /* out array<int,column-major> */ struct sidl_int__array** grid_relax_type,
  /* out array<int,2,
    column-major> */ struct sidl_int__array** grid_relax_points,
  /* in */ int32_t coarsen_type,
  /* out array<double,
    column-major> */ struct sidl_double__array** relax_weights,
  /* in */ int32_t max_levels,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetOperator(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetTolerance(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetMaxIterations(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetLogging(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetPrintLevel(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_GetNumIterations(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_GetRelResidualNorm(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetCommunicator(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_BoomerAMG_Destroy(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetIntParameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetDoubleParameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetStringParameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetIntArray1Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetIntArray2Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_GetIntValue(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_GetDoubleValue(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_Setup(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_Apply(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_BoomerAMG_ApplyAdjoint(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_BoomerAMG__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_BoomerAMG(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_BoomerAMG__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_BoomerAMG(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Operator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Solver(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_Solver(void* bi, sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BoomerAMG_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BoomerAMG_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BoomerAMG_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_BoomerAMG_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
