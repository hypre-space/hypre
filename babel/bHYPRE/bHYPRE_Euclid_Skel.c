/*
 * File:          bHYPRE_Euclid_Skel.c
 * Symbol:        bHYPRE.Euclid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side glue code for bHYPRE.Euclid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "bHYPRE_Euclid_IOR.h"
#include "bHYPRE_Euclid.h"
#include <stddef.h>

extern
void
impl_bHYPRE_Euclid__load(
  void);

extern
void
impl_bHYPRE_Euclid__ctor(
  /* in */ bHYPRE_Euclid self);

extern
void
impl_bHYPRE_Euclid__dtor(
  /* in */ bHYPRE_Euclid self);

extern
bHYPRE_Euclid
impl_bHYPRE_Euclid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_IJParCSRMatrix A);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_Euclid_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_Euclid__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_Euclid(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_Euclid(struct 
  bHYPRE_Euclid__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_Euclid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_Euclid_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_Euclid_SetParameters(
  /* in */ bHYPRE_Euclid self,
  /* in */ int32_t argc,
  /* inout */ char** argv);

extern
int32_t
impl_bHYPRE_Euclid_SetCommunicator(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_Euclid_SetIntParameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_Euclid_SetDoubleParameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_Euclid_SetStringParameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_Euclid_SetIntArray1Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_Euclid_SetIntArray2Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_Euclid_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_Euclid_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_Euclid_GetIntValue(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_Euclid_GetDoubleValue(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_Euclid_Setup(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_Euclid_Apply(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_Euclid_ApplyAdjoint(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_Euclid_SetOperator(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_Euclid_SetTolerance(
  /* in */ bHYPRE_Euclid self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_Euclid_SetMaxIterations(
  /* in */ bHYPRE_Euclid self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_Euclid_SetLogging(
  /* in */ bHYPRE_Euclid self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_Euclid_SetPrintLevel(
  /* in */ bHYPRE_Euclid self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_Euclid_GetNumIterations(
  /* in */ bHYPRE_Euclid self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_Euclid_GetRelResidualNorm(
  /* in */ bHYPRE_Euclid self,
  /* out */ double* norm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_Euclid_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_Euclid__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_Euclid(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_bHYPRE_Euclid(struct 
  bHYPRE_Euclid__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_Euclid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_Euclid_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_Euclid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_Euclid_SetIntArray1Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
/* in rarray[nvalues] */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  int32_t* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_Euclid_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_Euclid_SetIntArray2Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
/* in array<int,2,column-major> */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_Euclid_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_Euclid_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
/* in rarray[nvalues] */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  double* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_Euclid_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_Euclid_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
/* in array<double,2,column-major> */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_Euclid_SetDoubleArray2Parameter(
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
bHYPRE_Euclid__set_epv(struct bHYPRE_Euclid__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_Euclid__ctor;
  epv->f__dtor = impl_bHYPRE_Euclid__dtor;
  epv->f_SetParameters = impl_bHYPRE_Euclid_SetParameters;
  epv->f_SetCommunicator = impl_bHYPRE_Euclid_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_Euclid_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_Euclid_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_Euclid_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_Euclid_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_Euclid_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = skel_bHYPRE_Euclid_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = skel_bHYPRE_Euclid_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_Euclid_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_Euclid_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_Euclid_Setup;
  epv->f_Apply = impl_bHYPRE_Euclid_Apply;
  epv->f_ApplyAdjoint = impl_bHYPRE_Euclid_ApplyAdjoint;
  epv->f_SetOperator = impl_bHYPRE_Euclid_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_Euclid_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_Euclid_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_Euclid_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_Euclid_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_Euclid_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_Euclid_GetRelResidualNorm;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_Euclid__set_sepv(struct bHYPRE_Euclid__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_Euclid_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_Euclid__call_load(void) { 
  impl_bHYPRE_Euclid__load();
}
struct bHYPRE_Solver__object* skel_bHYPRE_Euclid_fconnect_bHYPRE_Solver(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_Euclid_fconnect_bHYPRE_Solver(url, _ex);
}

char* skel_bHYPRE_Euclid_fgetURL_bHYPRE_Solver(struct bHYPRE_Solver__object* 
  obj) { 
  return impl_bHYPRE_Euclid_fgetURL_bHYPRE_Solver(obj);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_Euclid_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_Euclid_fconnect_bHYPRE_MPICommunicator(url, _ex);
}

char* skel_bHYPRE_Euclid_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_Euclid_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_Euclid_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_Euclid_fconnect_bHYPRE_Operator(url, _ex);
}

char* skel_bHYPRE_Euclid_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) { 
  return impl_bHYPRE_Euclid_fgetURL_bHYPRE_Operator(obj);
}

struct bHYPRE_IJParCSRMatrix__object* 
  skel_bHYPRE_Euclid_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_Euclid_fconnect_bHYPRE_IJParCSRMatrix(url, _ex);
}

char* skel_bHYPRE_Euclid_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj) { 
  return impl_bHYPRE_Euclid_fgetURL_bHYPRE_IJParCSRMatrix(obj);
}

struct sidl_ClassInfo__object* skel_bHYPRE_Euclid_fconnect_sidl_ClassInfo(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_Euclid_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_Euclid_fgetURL_sidl_ClassInfo(struct sidl_ClassInfo__object* 
  obj) { 
  return impl_bHYPRE_Euclid_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* skel_bHYPRE_Euclid_fconnect_bHYPRE_Vector(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_Euclid_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_Euclid_fgetURL_bHYPRE_Vector(struct bHYPRE_Vector__object* 
  obj) { 
  return impl_bHYPRE_Euclid_fgetURL_bHYPRE_Vector(obj);
}

struct bHYPRE_Euclid__object* skel_bHYPRE_Euclid_fconnect_bHYPRE_Euclid(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_Euclid_fconnect_bHYPRE_Euclid(url, _ex);
}

char* skel_bHYPRE_Euclid_fgetURL_bHYPRE_Euclid(struct bHYPRE_Euclid__object* 
  obj) { 
  return impl_bHYPRE_Euclid_fgetURL_bHYPRE_Euclid(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_Euclid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_Euclid_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_Euclid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_Euclid_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* skel_bHYPRE_Euclid_fconnect_sidl_BaseClass(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_Euclid_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_Euclid_fgetURL_sidl_BaseClass(struct sidl_BaseClass__object* 
  obj) { 
  return impl_bHYPRE_Euclid_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_Euclid__data*
bHYPRE_Euclid__get_data(bHYPRE_Euclid self)
{
  return (struct bHYPRE_Euclid__data*)(self ? self->d_data : NULL);
}

void bHYPRE_Euclid__set_data(
  bHYPRE_Euclid self,
  struct bHYPRE_Euclid__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
