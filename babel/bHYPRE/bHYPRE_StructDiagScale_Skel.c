/*
 * File:          bHYPRE_StructDiagScale_Skel.c
 * Symbol:        bHYPRE.StructDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for bHYPRE.StructDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_StructDiagScale_IOR.h"
#include "bHYPRE_StructDiagScale.h"
#include <stddef.h>

extern
void
impl_bHYPRE_StructDiagScale__load(
  void);

extern
void
impl_bHYPRE_StructDiagScale__ctor(
  /* in */ bHYPRE_StructDiagScale self);

extern
void
impl_bHYPRE_StructDiagScale__dtor(
  /* in */ bHYPRE_StructDiagScale self);

extern
bHYPRE_StructDiagScale
impl_bHYPRE_StructDiagScale_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct bHYPRE_StructDiagScale__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructDiagScale(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_StructDiagScale(struct 
  bHYPRE_StructDiagScale__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_StructDiagScale_SetCommunicator(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetIntParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetStringParameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetIntArray1Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetIntArray2Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetIntValue(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetDoubleValue(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_StructDiagScale_Setup(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_StructDiagScale_Apply(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetOperator(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetTolerance(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetMaxIterations(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetLogging(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_StructDiagScale_SetPrintLevel(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetNumIterations(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_StructDiagScale_GetRelResidualNorm(
  /* in */ bHYPRE_StructDiagScale self,
  /* out */ double* norm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct bHYPRE_StructDiagScale__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructDiagScale(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_StructDiagScale(struct 
  bHYPRE_StructDiagScale__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructDiagScale_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_StructDiagScale_SetIntArray1Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  int32_t* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_StructDiagScale_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_StructDiagScale_SetIntArray2Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructDiagScale_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructDiagScale_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  double* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_StructDiagScale_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_StructDiagScale_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructDiagScale self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructDiagScale_SetDoubleArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructDiagScale__set_epv(struct bHYPRE_StructDiagScale__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructDiagScale__ctor;
  epv->f__dtor = impl_bHYPRE_StructDiagScale__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructDiagScale_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_StructDiagScale_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_StructDiagScale_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_StructDiagScale_SetStringParameter;
  epv->f_SetIntArray1Parameter = 
    skel_bHYPRE_StructDiagScale_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = 
    skel_bHYPRE_StructDiagScale_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_StructDiagScale_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_StructDiagScale_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_StructDiagScale_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_StructDiagScale_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_StructDiagScale_Setup;
  epv->f_Apply = impl_bHYPRE_StructDiagScale_Apply;
  epv->f_SetOperator = impl_bHYPRE_StructDiagScale_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_StructDiagScale_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_StructDiagScale_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_StructDiagScale_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_StructDiagScale_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_StructDiagScale_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_StructDiagScale_GetRelResidualNorm;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructDiagScale__set_sepv(struct bHYPRE_StructDiagScale__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_StructDiagScale_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_StructDiagScale__call_load(void) { 
  impl_bHYPRE_StructDiagScale__load();
}
struct bHYPRE_Solver__object* 
  skel_bHYPRE_StructDiagScale_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Solver(url, _ex);
}

char* skel_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj) { 
  return impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Solver(obj);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_StructDiagScale_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_MPICommunicator(url, _ex);
}

char* skel_bHYPRE_StructDiagScale_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_StructDiagScale_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Operator(url, _ex);
}

char* skel_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) { 
  return impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Operator(obj);
}

struct bHYPRE_StructDiagScale__object* 
  skel_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructDiagScale(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_StructDiagScale(url, _ex);
}

char* skel_bHYPRE_StructDiagScale_fgetURL_bHYPRE_StructDiagScale(struct 
  bHYPRE_StructDiagScale__object* obj) { 
  return impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_StructDiagScale(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructDiagScale_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructDiagScale_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_StructDiagScale_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_StructDiagScale_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_StructDiagScale_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructDiagScale_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) { 
  return impl_bHYPRE_StructDiagScale_fgetURL_bHYPRE_Vector(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructDiagScale_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_StructDiagScale_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_StructDiagScale_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructDiagScale_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructDiagScale_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_StructDiagScale_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_StructDiagScale_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_StructDiagScale__data*
bHYPRE_StructDiagScale__get_data(bHYPRE_StructDiagScale self)
{
  return (struct bHYPRE_StructDiagScale__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructDiagScale__set_data(
  bHYPRE_StructDiagScale self,
  struct bHYPRE_StructDiagScale__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
