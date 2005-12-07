/*
 * File:          bHYPRE_SStructSplit_Skel.c
 * Symbol:        bHYPRE.SStructSplit-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for bHYPRE.SStructSplit
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_SStructSplit_IOR.h"
#include "bHYPRE_SStructSplit.h"
#include <stddef.h>

extern
void
impl_bHYPRE_SStructSplit__load(
  void);

extern
void
impl_bHYPRE_SStructSplit__ctor(
  /* in */ bHYPRE_SStructSplit self);

extern
void
impl_bHYPRE_SStructSplit__dtor(
  /* in */ bHYPRE_SStructSplit self);

extern
bHYPRE_SStructSplit
impl_bHYPRE_SStructSplit_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_SStructSplit__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_SStructSplit(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_SStructSplit(struct 
  bHYPRE_SStructSplit__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_SStructSplit_SetCommunicator(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_SStructSplit_SetIntParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_SStructSplit_SetDoubleParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_SStructSplit_SetStringParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_SStructSplit_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructSplit_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_SStructSplit_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructSplit_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_SStructSplit_GetIntValue(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_SStructSplit_GetDoubleValue(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_SStructSplit_Setup(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_SStructSplit_Apply(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_SStructSplit_ApplyAdjoint(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_SStructSplit_SetOperator(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_SStructSplit_SetTolerance(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_SStructSplit_SetMaxIterations(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_SStructSplit_SetLogging(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_SStructSplit_SetPrintLevel(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_SStructSplit_GetNumIterations(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_SStructSplit_GetRelResidualNorm(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ double* norm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_SStructSplit__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_SStructSplit(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_SStructSplit(struct 
  bHYPRE_SStructSplit__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_SStructSplit_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  int32_t* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_SStructSplit_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_SStructSplit_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructSplit_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructSplit_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  double* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_SStructSplit_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_SStructSplit_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructSplit_SetDoubleArray2Parameter(
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
bHYPRE_SStructSplit__set_epv(struct bHYPRE_SStructSplit__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructSplit__ctor;
  epv->f__dtor = impl_bHYPRE_SStructSplit__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_SStructSplit_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_SStructSplit_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_SStructSplit_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_SStructSplit_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_SStructSplit_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_SStructSplit_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_SStructSplit_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_SStructSplit_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_SStructSplit_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_SStructSplit_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_SStructSplit_Setup;
  epv->f_Apply = impl_bHYPRE_SStructSplit_Apply;
  epv->f_ApplyAdjoint = impl_bHYPRE_SStructSplit_ApplyAdjoint;
  epv->f_SetOperator = impl_bHYPRE_SStructSplit_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_SStructSplit_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_SStructSplit_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_SStructSplit_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_SStructSplit_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_SStructSplit_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_SStructSplit_GetRelResidualNorm;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructSplit__set_sepv(struct bHYPRE_SStructSplit__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_SStructSplit_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_SStructSplit__call_load(void) { 
  impl_bHYPRE_SStructSplit__load();
}
struct bHYPRE_Solver__object* 
  skel_bHYPRE_SStructSplit_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Solver(url, _ex);
}

char* skel_bHYPRE_SStructSplit_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj) { 
  return impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Solver(obj);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_SStructSplit_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructSplit_fconnect_bHYPRE_MPICommunicator(url, _ex);
}

char* skel_bHYPRE_SStructSplit_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_SStructSplit_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Operator(url, _ex);
}

char* skel_bHYPRE_SStructSplit_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) { 
  return impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Operator(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructSplit_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructSplit_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_SStructSplit_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_SStructSplit_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_SStructSplit_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_SStructSplit_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) { 
  return impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Vector(obj);
}

struct bHYPRE_SStructSplit__object* 
  skel_bHYPRE_SStructSplit_fconnect_bHYPRE_SStructSplit(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructSplit_fconnect_bHYPRE_SStructSplit(url, _ex);
}

char* skel_bHYPRE_SStructSplit_fgetURL_bHYPRE_SStructSplit(struct 
  bHYPRE_SStructSplit__object* obj) { 
  return impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_SStructSplit(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructSplit_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructSplit_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_SStructSplit_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructSplit_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructSplit_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_SStructSplit_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_SStructSplit__data*
bHYPRE_SStructSplit__get_data(bHYPRE_SStructSplit self)
{
  return (struct bHYPRE_SStructSplit__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructSplit__set_data(
  bHYPRE_SStructSplit self,
  struct bHYPRE_SStructSplit__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
