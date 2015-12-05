/*
 * File:          bHYPRE_IJParCSRVector_Impl.h
 * Symbol:        bHYPRE.IJParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.IJParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_IJParCSRVector_Impl_h
#define included_bHYPRE_IJParCSRVector_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_IJParCSRVector_h
#include "bHYPRE_IJParCSRVector.h"
#endif
#ifndef included_bHYPRE_IJVectorView_h
#include "bHYPRE_IJVectorView.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_h
#include "bHYPRE_MatrixVectorView.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_h
#include "bHYPRE_ProblemDefinition.h"
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

/* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector._includes) */
/* Insert-Code-Here {bHYPRE.IJParCSRVector._includes} (include files) */

#include "HYPRE_IJ_mv.h"
#include "HYPRE.h"
#include "_hypre_utilities.h"

/* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector._includes) */

/*
 * Private data for class bHYPRE.IJParCSRVector
 */

struct bHYPRE_IJParCSRVector__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector._data) */
  /* Put private data members here... */
  HYPRE_IJVector ij_b;
  MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_IJParCSRVector__data*
bHYPRE_IJParCSRVector__get_data(
  bHYPRE_IJParCSRVector);

extern void
bHYPRE_IJParCSRVector__set_data(
  bHYPRE_IJParCSRVector,
  struct bHYPRE_IJParCSRVector__data*);

extern
void
impl_bHYPRE_IJParCSRVector__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IJParCSRVector__ctor(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IJParCSRVector__ctor2(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IJParCSRVector__dtor(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_IJParCSRVector
impl_bHYPRE_IJParCSRVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_IJParCSRVector__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJParCSRVector(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJParCSRVector__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_IJParCSRVector(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJVectorView(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_IJVectorView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MatrixVectorView(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_MatrixVectorView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_ProblemDefinition(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_ProblemDefinition(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_IJParCSRVector_SetLocalRange(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_SetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in rarray[nvalues] */ int32_t* indices,
  /* in rarray[nvalues] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_AddToValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in rarray[nvalues] */ int32_t* indices,
  /* in rarray[nvalues] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_GetLocalRange(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_GetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in rarray[nvalues] */ int32_t* indices,
  /* inout rarray[nvalues] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Print(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ const char* filename,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Read(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ const char* filename,
  /* in */ bHYPRE_MPICommunicator comm,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_SetCommunicator(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IJParCSRVector_Destroy(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Initialize(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Assemble(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Clear(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Copy(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Clone(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Scale(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ double a,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Dot(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Axpy(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_IJParCSRVector__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJParCSRVector(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJParCSRVector__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_IJParCSRVector(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJVectorView(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_IJVectorView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MatrixVectorView(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_MatrixVectorView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_ProblemDefinition(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_ProblemDefinition(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);

#ifdef __cplusplus
}
#endif
#endif
