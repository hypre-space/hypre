/*
 * File:          bHYPRE_IJParCSRVector_Skel.c
 * Symbol:        bHYPRE.IJParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for bHYPRE.IJParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_IJParCSRVector_IOR.h"
#include "bHYPRE_IJParCSRVector.h"
#include <stddef.h>

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
static int32_t
skel_bHYPRE_IJParCSRVector_SetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ struct sidl_int__array* indices,
/* in */ struct sidl_double__array* values)
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
      values_tmp);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRVector_AddToValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ struct sidl_int__array* indices,
/* in */ struct sidl_double__array* values)
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
      values_tmp);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRVector_GetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ struct sidl_int__array* indices,
/* inout */ struct sidl_double__array** values)
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
      values_tmp);
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
  epv->f__dtor = impl_bHYPRE_IJParCSRVector__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_IJParCSRVector_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_IJParCSRVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_IJParCSRVector_Assemble;
  epv->f_SetLocalRange = impl_bHYPRE_IJParCSRVector_SetLocalRange;
  epv->f_SetValues = skel_bHYPRE_IJParCSRVector_SetValues;
  epv->f_AddToValues = skel_bHYPRE_IJParCSRVector_AddToValues;
  epv->f_GetLocalRange = impl_bHYPRE_IJParCSRVector_GetLocalRange;
  epv->f_GetValues = skel_bHYPRE_IJParCSRVector_GetValues;
  epv->f_Print = impl_bHYPRE_IJParCSRVector_Print;
  epv->f_Read = impl_bHYPRE_IJParCSRVector_Read;
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
  impl_bHYPRE_IJParCSRVector__load();
}
struct bHYPRE_IJParCSRVector__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJParCSRVector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJParCSRVector(url, _ex);
}

char* skel_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJParCSRVector(struct 
  bHYPRE_IJParCSRVector__object* obj) { 
  return impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJParCSRVector(obj);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MPICommunicator(url, _ex);
}

char* skel_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_IJParCSRVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_IJParCSRVector_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) { 
  return impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_Vector(obj);
}

struct bHYPRE_IJVectorView__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJVectorView(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJVectorView(url, _ex);
}

char* skel_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJVectorView(struct 
  bHYPRE_IJVectorView__object* obj) { 
  return impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJVectorView(obj);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_ProblemDefinition(url, _ex);
}

char* skel_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj) { 
  return impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_ProblemDefinition(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseInterface(obj);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MatrixVectorView(url, _ex);
}

char* skel_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj) { 
  return impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_MatrixVectorView(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseClass(obj);
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
