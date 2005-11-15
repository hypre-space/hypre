/*
 * File:          bHYPRE_ParaSails_Skel.c
 * Symbol:        bHYPRE.ParaSails-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for bHYPRE.ParaSails
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_ParaSails_IOR.h"
#include "bHYPRE_ParaSails.h"
#include <stddef.h>

extern
void
impl_bHYPRE_ParaSails__load(
  void);

extern
void
impl_bHYPRE_ParaSails__ctor(
  /* in */ bHYPRE_ParaSails self);

extern
void
impl_bHYPRE_ParaSails__dtor(
  /* in */ bHYPRE_ParaSails self);

extern
bHYPRE_ParaSails
impl_bHYPRE_ParaSails_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_ParaSails__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_ParaSails(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_bHYPRE_ParaSails(struct 
  bHYPRE_ParaSails__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_ParaSails_SetCommunicator(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_ParaSails_SetIntParameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_ParaSails_SetDoubleParameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_ParaSails_SetStringParameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_ParaSails_SetIntArray1Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_ParaSails_SetIntArray2Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_ParaSails_SetDoubleArray1Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_ParaSails_SetDoubleArray2Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_ParaSails_GetIntValue(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_ParaSails_GetDoubleValue(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_ParaSails_Setup(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_ParaSails_Apply(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_ParaSails_SetOperator(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_ParaSails_SetTolerance(
  /* in */ bHYPRE_ParaSails self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_ParaSails_SetMaxIterations(
  /* in */ bHYPRE_ParaSails self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_ParaSails_SetLogging(
  /* in */ bHYPRE_ParaSails self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_ParaSails_SetPrintLevel(
  /* in */ bHYPRE_ParaSails self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_ParaSails_GetNumIterations(
  /* in */ bHYPRE_ParaSails self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_ParaSails_GetRelResidualNorm(
  /* in */ bHYPRE_ParaSails self,
  /* out */ double* norm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_ParaSails__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_ParaSails(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_bHYPRE_ParaSails(struct 
  bHYPRE_ParaSails__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_ParaSails_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_ParaSails_SetIntArray1Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  int32_t* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_ParaSails_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_ParaSails_SetIntArray2Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_ParaSails_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_ParaSails_SetDoubleArray1Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  double* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_ParaSails_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_ParaSails_SetDoubleArray2Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_ParaSails_SetDoubleArray2Parameter(
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
bHYPRE_ParaSails__set_epv(struct bHYPRE_ParaSails__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_ParaSails__ctor;
  epv->f__dtor = impl_bHYPRE_ParaSails__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_ParaSails_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_ParaSails_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_ParaSails_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_ParaSails_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_ParaSails_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_ParaSails_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_ParaSails_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_ParaSails_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_ParaSails_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_ParaSails_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_ParaSails_Setup;
  epv->f_Apply = impl_bHYPRE_ParaSails_Apply;
  epv->f_SetOperator = impl_bHYPRE_ParaSails_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_ParaSails_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_ParaSails_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_ParaSails_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_ParaSails_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_ParaSails_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_ParaSails_GetRelResidualNorm;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_ParaSails__set_sepv(struct bHYPRE_ParaSails__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_ParaSails_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_ParaSails__call_load(void) { 
  impl_bHYPRE_ParaSails__load();
}
struct bHYPRE_Solver__object* 
  skel_bHYPRE_ParaSails_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ParaSails_fconnect_bHYPRE_Solver(url, _ex);
}

char* skel_bHYPRE_ParaSails_fgetURL_bHYPRE_Solver(struct bHYPRE_Solver__object* 
  obj) { 
  return impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Solver(obj);
}

struct bHYPRE_ParaSails__object* 
  skel_bHYPRE_ParaSails_fconnect_bHYPRE_ParaSails(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ParaSails_fconnect_bHYPRE_ParaSails(url, _ex);
}

char* skel_bHYPRE_ParaSails_fgetURL_bHYPRE_ParaSails(struct 
  bHYPRE_ParaSails__object* obj) { 
  return impl_bHYPRE_ParaSails_fgetURL_bHYPRE_ParaSails(obj);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_ParaSails_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ParaSails_fconnect_bHYPRE_MPICommunicator(url, _ex);
}

char* skel_bHYPRE_ParaSails_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_ParaSails_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_ParaSails_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ParaSails_fconnect_bHYPRE_Operator(url, _ex);
}

char* skel_bHYPRE_ParaSails_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) { 
  return impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Operator(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_ParaSails_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ParaSails_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_ParaSails_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_ParaSails_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_ParaSails_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ParaSails_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_ParaSails_fgetURL_bHYPRE_Vector(struct bHYPRE_Vector__object* 
  obj) { 
  return impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Vector(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_ParaSails_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ParaSails_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_ParaSails_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_ParaSails_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_ParaSails_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ParaSails_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_ParaSails_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_ParaSails_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_ParaSails__data*
bHYPRE_ParaSails__get_data(bHYPRE_ParaSails self)
{
  return (struct bHYPRE_ParaSails__data*)(self ? self->d_data : NULL);
}

void bHYPRE_ParaSails__set_data(
  bHYPRE_ParaSails self,
  struct bHYPRE_ParaSails__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
