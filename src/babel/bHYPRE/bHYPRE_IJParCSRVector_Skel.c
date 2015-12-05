/*
 * File:          bHYPRE_IJParCSRVector_Skel.c
 * Symbol:        bHYPRE.IJParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for bHYPRE.IJParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_IJParCSRVector_IOR.h"
#include "bHYPRE_IJParCSRVector.h"
#include <stddef.h>

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
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
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
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
static int32_t
skel_bHYPRE_IJParCSRVector_SetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in rarray[nvalues] */ struct sidl_int__array* indices,
  /* in rarray[nvalues] */ struct sidl_double__array* values,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* indices_proxy = sidl_int__array_ensure(indices, 1,
    sidl_column_major_order);
  int32_t* indices_tmp = indices_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nvalues = sidlLength(indices_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRVector_SetValues(
      self,
      nvalues,
      indices_tmp,
      values_tmp,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRVector_AddToValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in rarray[nvalues] */ struct sidl_int__array* indices,
  /* in rarray[nvalues] */ struct sidl_double__array* values,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* indices_proxy = sidl_int__array_ensure(indices, 1,
    sidl_column_major_order);
  int32_t* indices_tmp = indices_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nvalues = sidlLength(indices_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRVector_AddToValues(
      self,
      nvalues,
      indices_tmp,
      values_tmp,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRVector_GetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in rarray[nvalues] */ struct sidl_int__array* indices,
  /* inout rarray[nvalues] */ struct sidl_double__array** values,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* indices_proxy = sidl_int__array_ensure(indices, 1,
    sidl_column_major_order);
  int32_t* indices_tmp = indices_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(*values,
    1, sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nvalues = sidlLength(indices_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRVector_GetValues(
      self,
      nvalues,
      indices_tmp,
      values_tmp,
      _ex);
  sidl_double__array_init(values_tmp, *values, 1, (*values)->d_metadata.d_lower,
    (*values)->d_metadata.d_upper, (*values)->d_metadata.d_stride);

  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_IJParCSRVector__set_epv(struct bHYPRE_IJParCSRVector__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_IJParCSRVector__ctor;
  epv->f__ctor2 = impl_bHYPRE_IJParCSRVector__ctor2;
  epv->f__dtor = impl_bHYPRE_IJParCSRVector__dtor;
  epv->f_SetLocalRange = impl_bHYPRE_IJParCSRVector_SetLocalRange;
  epv->f_SetValues = skel_bHYPRE_IJParCSRVector_SetValues;
  epv->f_AddToValues = skel_bHYPRE_IJParCSRVector_AddToValues;
  epv->f_GetLocalRange = impl_bHYPRE_IJParCSRVector_GetLocalRange;
  epv->f_GetValues = skel_bHYPRE_IJParCSRVector_GetValues;
  epv->f_Print = impl_bHYPRE_IJParCSRVector_Print;
  epv->f_Read = impl_bHYPRE_IJParCSRVector_Read;
  epv->f_SetCommunicator = impl_bHYPRE_IJParCSRVector_SetCommunicator;
  epv->f_Destroy = impl_bHYPRE_IJParCSRVector_Destroy;
  epv->f_Initialize = impl_bHYPRE_IJParCSRVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_IJParCSRVector_Assemble;
  epv->f_Clear = impl_bHYPRE_IJParCSRVector_Clear;
  epv->f_Copy = impl_bHYPRE_IJParCSRVector_Copy;
  epv->f_Clone = impl_bHYPRE_IJParCSRVector_Clone;
  epv->f_Scale = impl_bHYPRE_IJParCSRVector_Scale;
  epv->f_Dot = impl_bHYPRE_IJParCSRVector_Dot;
  epv->f_Axpy = impl_bHYPRE_IJParCSRVector_Axpy;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_IJParCSRVector__set_sepv(struct bHYPRE_IJParCSRVector__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_IJParCSRVector_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_IJParCSRVector__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_bHYPRE_IJParCSRVector__load(&_throwaway_exception);
}
struct bHYPRE_IJParCSRVector__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJParCSRVector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJParCSRVector(url, ar,
    _ex);
}

struct bHYPRE_IJParCSRVector__object* 
  skel_bHYPRE_IJParCSRVector_fcast_bHYPRE_IJParCSRVector(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_IJParCSRVector(bi, _ex);
}

struct bHYPRE_IJVectorView__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJVectorView(url, ar, _ex);
}

struct bHYPRE_IJVectorView__object* 
  skel_bHYPRE_IJParCSRVector_fcast_bHYPRE_IJVectorView(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_IJVectorView(bi, _ex);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MPICommunicator(url, ar,
    _ex);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_IJParCSRVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_MPICommunicator(bi, _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MatrixVectorView(url, ar,
    _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_IJParCSRVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_MatrixVectorView(bi, _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_ProblemDefinition(url, ar,
    _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_IJParCSRVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_ProblemDefinition(bi, _ex);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(url, ar, _ex);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_IJParCSRVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fcast_bHYPRE_Vector(bi, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_IJParCSRVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_IJParCSRVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_IJParCSRVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_sidl_RuntimeException(url, ar,
    _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_IJParCSRVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fcast_sidl_RuntimeException(bi, _ex);
}

struct bHYPRE_IJParCSRVector__data*
bHYPRE_IJParCSRVector__get_data(bHYPRE_IJParCSRVector self)
{
  return (struct bHYPRE_IJParCSRVector__data*)(self ? self->d_data : NULL);
}

void bHYPRE_IJParCSRVector__set_data(
  bHYPRE_IJParCSRVector self,
  struct bHYPRE_IJParCSRVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
