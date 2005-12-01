/*
 * File:          bHYPRE_CGNR_Skel.c
 * Symbol:        bHYPRE.CGNR-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for bHYPRE.CGNR
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_CGNR_IOR.h"
#include "bHYPRE_CGNR.h"
#include <stddef.h>

extern
void
impl_bHYPRE_CGNR__load(
  void);

extern
void
impl_bHYPRE_CGNR__ctor(
  /* in */ bHYPRE_CGNR self);

extern
void
impl_bHYPRE_CGNR__dtor(
  /* in */ bHYPRE_CGNR self);

extern
bHYPRE_CGNR
impl_bHYPRE_CGNR_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_CGNR__object* impl_bHYPRE_CGNR_fconnect_bHYPRE_CGNR(char* 
  url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_CGNR(struct bHYPRE_CGNR__object* 
  obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Operator(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
extern
int32_t
impl_bHYPRE_CGNR_SetCommunicator(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_CGNR_SetIntParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_CGNR_SetDoubleParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_CGNR_SetStringParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_CGNR_SetIntArray1Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_CGNR_SetIntArray2Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_CGNR_SetDoubleArray1Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_CGNR_SetDoubleArray2Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_CGNR_GetIntValue(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_CGNR_GetDoubleValue(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_CGNR_Setup(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_CGNR_Apply(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_CGNR_ApplyAdjoint(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_CGNR_SetOperator(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_CGNR_SetTolerance(
  /* in */ bHYPRE_CGNR self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_CGNR_SetMaxIterations(
  /* in */ bHYPRE_CGNR self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_CGNR_SetLogging(
  /* in */ bHYPRE_CGNR self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_CGNR_SetPrintLevel(
  /* in */ bHYPRE_CGNR self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_CGNR_GetNumIterations(
  /* in */ bHYPRE_CGNR self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_CGNR_GetRelResidualNorm(
  /* in */ bHYPRE_CGNR self,
  /* out */ double* norm);

extern
int32_t
impl_bHYPRE_CGNR_SetPreconditioner(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Solver s);

extern
int32_t
impl_bHYPRE_CGNR_GetPreconditioner(
  /* in */ bHYPRE_CGNR self,
  /* out */ bHYPRE_Solver* s);

extern
int32_t
impl_bHYPRE_CGNR_Clone(
  /* in */ bHYPRE_CGNR self,
  /* out */ bHYPRE_PreconditionedSolver* x);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_CGNR__object* impl_bHYPRE_CGNR_fconnect_bHYPRE_CGNR(char* 
  url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_CGNR(struct bHYPRE_CGNR__object* 
  obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Operator(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_ClassInfo(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_BaseClass(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_CGNR_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
static int32_t
skel_bHYPRE_CGNR_SetIntArray1Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  int32_t* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_CGNR_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_CGNR_SetIntArray2Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_CGNR_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_CGNR_SetDoubleArray1Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  double* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_CGNR_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_CGNR_SetDoubleArray2Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_CGNR_SetDoubleArray2Parameter(
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
bHYPRE_CGNR__set_epv(struct bHYPRE_CGNR__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_CGNR__ctor;
  epv->f__dtor = impl_bHYPRE_CGNR__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_CGNR_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_CGNR_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_CGNR_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_CGNR_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_CGNR_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_CGNR_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = skel_bHYPRE_CGNR_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = skel_bHYPRE_CGNR_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_CGNR_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_CGNR_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_CGNR_Setup;
  epv->f_Apply = impl_bHYPRE_CGNR_Apply;
  epv->f_ApplyAdjoint = impl_bHYPRE_CGNR_ApplyAdjoint;
  epv->f_SetOperator = impl_bHYPRE_CGNR_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_CGNR_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_CGNR_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_CGNR_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_CGNR_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_CGNR_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_CGNR_GetRelResidualNorm;
  epv->f_SetPreconditioner = impl_bHYPRE_CGNR_SetPreconditioner;
  epv->f_GetPreconditioner = impl_bHYPRE_CGNR_GetPreconditioner;
  epv->f_Clone = impl_bHYPRE_CGNR_Clone;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_CGNR__set_sepv(struct bHYPRE_CGNR__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_CGNR_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_CGNR__call_load(void) { 
  impl_bHYPRE_CGNR__load();
}
struct bHYPRE_Solver__object* skel_bHYPRE_CGNR_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_CGNR_fconnect_bHYPRE_Solver(url, _ex);
}

char* skel_bHYPRE_CGNR_fgetURL_bHYPRE_Solver(struct bHYPRE_Solver__object* obj) 
  { 
  return impl_bHYPRE_CGNR_fgetURL_bHYPRE_Solver(obj);
}

struct bHYPRE_CGNR__object* skel_bHYPRE_CGNR_fconnect_bHYPRE_CGNR(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_CGNR_fconnect_bHYPRE_CGNR(url, _ex);
}

char* skel_bHYPRE_CGNR_fgetURL_bHYPRE_CGNR(struct bHYPRE_CGNR__object* obj) { 
  return impl_bHYPRE_CGNR_fgetURL_bHYPRE_CGNR(obj);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_CGNR_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_CGNR_fconnect_bHYPRE_MPICommunicator(url, _ex);
}

char* skel_bHYPRE_CGNR_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_CGNR_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct bHYPRE_Operator__object* skel_bHYPRE_CGNR_fconnect_bHYPRE_Operator(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_CGNR_fconnect_bHYPRE_Operator(url, _ex);
}

char* skel_bHYPRE_CGNR_fgetURL_bHYPRE_Operator(struct bHYPRE_Operator__object* 
  obj) { 
  return impl_bHYPRE_CGNR_fgetURL_bHYPRE_Operator(obj);
}

struct sidl_ClassInfo__object* skel_bHYPRE_CGNR_fconnect_sidl_ClassInfo(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_CGNR_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_CGNR_fgetURL_sidl_ClassInfo(struct sidl_ClassInfo__object* 
  obj) { 
  return impl_bHYPRE_CGNR_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* skel_bHYPRE_CGNR_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_CGNR_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_CGNR_fgetURL_bHYPRE_Vector(struct bHYPRE_Vector__object* obj) 
  { 
  return impl_bHYPRE_CGNR_fgetURL_bHYPRE_Vector(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_CGNR_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_CGNR_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_CGNR_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_CGNR_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* skel_bHYPRE_CGNR_fconnect_sidl_BaseClass(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_CGNR_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_CGNR_fgetURL_sidl_BaseClass(struct sidl_BaseClass__object* 
  obj) { 
  return impl_bHYPRE_CGNR_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_PreconditionedSolver__object* 
  skel_bHYPRE_CGNR_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_CGNR_fconnect_bHYPRE_PreconditionedSolver(url, _ex);
}

char* skel_bHYPRE_CGNR_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj) { 
  return impl_bHYPRE_CGNR_fgetURL_bHYPRE_PreconditionedSolver(obj);
}

struct bHYPRE_CGNR__data*
bHYPRE_CGNR__get_data(bHYPRE_CGNR self)
{
  return (struct bHYPRE_CGNR__data*)(self ? self->d_data : NULL);
}

void bHYPRE_CGNR__set_data(
  bHYPRE_CGNR self,
  struct bHYPRE_CGNR__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
