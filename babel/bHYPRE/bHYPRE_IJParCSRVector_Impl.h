/*
 * File:          bHYPRE_IJParCSRVector_Impl.h
 * Symbol:        bHYPRE.IJParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for bHYPRE.IJParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_IJParCSRVector_Impl_h
#define included_bHYPRE_IJParCSRVector_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_IJParCSRVector_h
#include "bHYPRE_IJParCSRVector.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_bHYPRE_IJVectorView_h
#include "bHYPRE_IJVectorView.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_h
#include "bHYPRE_ProblemDefinition.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_h
#include "bHYPRE_MatrixVectorView.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector._includes) */
/* Put additional include files here... */
#include "HYPRE_IJ_mv.h"
#include "HYPRE.h"
#include "utilities.h"
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
  void);

extern
void
impl_bHYPRE_IJParCSRVector__ctor(
  /* in */ bHYPRE_IJParCSRVector self);

extern
void
impl_bHYPRE_IJParCSRVector__dtor(
  /* in */ bHYPRE_IJParCSRVector self);

/*
 * User-defined object methods
 */

extern
bHYPRE_IJParCSRVector
impl_bHYPRE_IJParCSRVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper);

extern struct bHYPRE_IJParCSRVector__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJParCSRVector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJParCSRVector(struct 
  bHYPRE_IJParCSRVector__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_IJVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJVectorView(struct 
  bHYPRE_IJVectorView__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_IJParCSRVector_SetCommunicator(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Initialize(
  /* in */ bHYPRE_IJParCSRVector self);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Assemble(
  /* in */ bHYPRE_IJParCSRVector self);

extern
int32_t
impl_bHYPRE_IJParCSRVector_SetLocalRange(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper);

extern
int32_t
impl_bHYPRE_IJParCSRVector_SetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in */ int32_t* indices,
  /* in */ double* values);

extern
int32_t
impl_bHYPRE_IJParCSRVector_AddToValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in */ int32_t* indices,
  /* in */ double* values);

extern
int32_t
impl_bHYPRE_IJParCSRVector_GetLocalRange(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper);

extern
int32_t
impl_bHYPRE_IJParCSRVector_GetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in */ int32_t* indices,
  /* inout */ double* values);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Print(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ const char* filename);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Read(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ const char* filename,
  /* in */ bHYPRE_MPICommunicator comm);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Clear(
  /* in */ bHYPRE_IJParCSRVector self);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Copy(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Clone(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Scale(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ double a);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Dot(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d);

extern
int32_t
impl_bHYPRE_IJParCSRVector_Axpy(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x);

extern struct bHYPRE_IJParCSRVector__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJParCSRVector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJParCSRVector(struct 
  bHYPRE_IJParCSRVector__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_IJVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJVectorView(struct 
  bHYPRE_IJVectorView__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
