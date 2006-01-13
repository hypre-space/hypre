/*
 * File:          bHYPRE_HGMRES_Skel.c
 * Symbol:        bHYPRE.HGMRES-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side glue code for bHYPRE.HGMRES
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "bHYPRE_HGMRES_IOR.h"
#include "bHYPRE_HGMRES.h"
#include <stddef.h>

extern
void
impl_bHYPRE_HGMRES__load(
  void);

extern
void
impl_bHYPRE_HGMRES__ctor(
  /* in */ bHYPRE_HGMRES self);

extern
void
impl_bHYPRE_HGMRES__dtor(
  /* in */ bHYPRE_HGMRES self);

extern
bHYPRE_HGMRES
impl_bHYPRE_HGMRES_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_HGMRES__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_HGMRES(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_HGMRES(struct 
  bHYPRE_HGMRES__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
extern
int32_t
impl_bHYPRE_HGMRES_SetCommunicator(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_HGMRES_SetIntParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_HGMRES_SetDoubleParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_HGMRES_SetStringParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_HGMRES_SetIntArray1Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_HGMRES_SetIntArray2Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_HGMRES_SetDoubleArray1Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_HGMRES_SetDoubleArray2Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_HGMRES_GetIntValue(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_HGMRES_GetDoubleValue(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_HGMRES_Setup(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_HGMRES_Apply(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_HGMRES_ApplyAdjoint(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_HGMRES_SetOperator(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_HGMRES_SetTolerance(
  /* in */ bHYPRE_HGMRES self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_HGMRES_SetMaxIterations(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_HGMRES_SetLogging(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_HGMRES_SetPrintLevel(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_HGMRES_GetNumIterations(
  /* in */ bHYPRE_HGMRES self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_HGMRES_GetRelResidualNorm(
  /* in */ bHYPRE_HGMRES self,
  /* out */ double* norm);

extern
int32_t
impl_bHYPRE_HGMRES_SetPreconditioner(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Solver s);

extern
int32_t
impl_bHYPRE_HGMRES_GetPreconditioner(
  /* in */ bHYPRE_HGMRES self,
  /* out */ bHYPRE_Solver* s);

extern
int32_t
impl_bHYPRE_HGMRES_Clone(
  /* in */ bHYPRE_HGMRES self,
  /* out */ bHYPRE_PreconditionedSolver* x);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Solver(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_Vector(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_HGMRES__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_HGMRES(char* url, sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_HGMRES(struct 
  bHYPRE_HGMRES__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_HGMRES_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj);
static int32_t
skel_bHYPRE_HGMRES_SetIntArray1Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
/* in rarray[nvalues] */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  int32_t* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_HGMRES_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_HGMRES_SetIntArray2Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
/* in array<int,2,column-major> */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_HGMRES_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_HGMRES_SetDoubleArray1Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
/* in rarray[nvalues] */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  double* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_HGMRES_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_HGMRES_SetDoubleArray2Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
/* in array<double,2,column-major> */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_HGMRES_SetDoubleArray2Parameter(
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
bHYPRE_HGMRES__set_epv(struct bHYPRE_HGMRES__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_HGMRES__ctor;
  epv->f__dtor = impl_bHYPRE_HGMRES__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_HGMRES_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_HGMRES_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_HGMRES_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_HGMRES_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_HGMRES_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_HGMRES_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = skel_bHYPRE_HGMRES_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = skel_bHYPRE_HGMRES_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_HGMRES_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_HGMRES_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_HGMRES_Setup;
  epv->f_Apply = impl_bHYPRE_HGMRES_Apply;
  epv->f_ApplyAdjoint = impl_bHYPRE_HGMRES_ApplyAdjoint;
  epv->f_SetOperator = impl_bHYPRE_HGMRES_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_HGMRES_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_HGMRES_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_HGMRES_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_HGMRES_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_HGMRES_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_HGMRES_GetRelResidualNorm;
  epv->f_SetPreconditioner = impl_bHYPRE_HGMRES_SetPreconditioner;
  epv->f_GetPreconditioner = impl_bHYPRE_HGMRES_GetPreconditioner;
  epv->f_Clone = impl_bHYPRE_HGMRES_Clone;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_HGMRES__set_sepv(struct bHYPRE_HGMRES__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_HGMRES_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_HGMRES__call_load(void) { 
  impl_bHYPRE_HGMRES__load();
}
struct bHYPRE_Solver__object* skel_bHYPRE_HGMRES_fconnect_bHYPRE_Solver(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_HGMRES_fconnect_bHYPRE_Solver(url, _ex);
}

char* skel_bHYPRE_HGMRES_fgetURL_bHYPRE_Solver(struct bHYPRE_Solver__object* 
  obj) { 
  return impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Solver(obj);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_HGMRES_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_HGMRES_fconnect_bHYPRE_MPICommunicator(url, _ex);
}

char* skel_bHYPRE_HGMRES_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_HGMRES_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_HGMRES_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_HGMRES_fconnect_bHYPRE_Operator(url, _ex);
}

char* skel_bHYPRE_HGMRES_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) { 
  return impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Operator(obj);
}

struct sidl_ClassInfo__object* skel_bHYPRE_HGMRES_fconnect_sidl_ClassInfo(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_HGMRES_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_HGMRES_fgetURL_sidl_ClassInfo(struct sidl_ClassInfo__object* 
  obj) { 
  return impl_bHYPRE_HGMRES_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* skel_bHYPRE_HGMRES_fconnect_bHYPRE_Vector(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_HGMRES_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_HGMRES_fgetURL_bHYPRE_Vector(struct bHYPRE_Vector__object* 
  obj) { 
  return impl_bHYPRE_HGMRES_fgetURL_bHYPRE_Vector(obj);
}

struct bHYPRE_HGMRES__object* skel_bHYPRE_HGMRES_fconnect_bHYPRE_HGMRES(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_HGMRES_fconnect_bHYPRE_HGMRES(url, _ex);
}

char* skel_bHYPRE_HGMRES_fgetURL_bHYPRE_HGMRES(struct bHYPRE_HGMRES__object* 
  obj) { 
  return impl_bHYPRE_HGMRES_fgetURL_bHYPRE_HGMRES(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_HGMRES_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_HGMRES_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_HGMRES_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_HGMRES_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* skel_bHYPRE_HGMRES_fconnect_sidl_BaseClass(char* 
  url, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_HGMRES_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_HGMRES_fgetURL_sidl_BaseClass(struct sidl_BaseClass__object* 
  obj) { 
  return impl_bHYPRE_HGMRES_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_PreconditionedSolver__object* 
  skel_bHYPRE_HGMRES_fconnect_bHYPRE_PreconditionedSolver(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_HGMRES_fconnect_bHYPRE_PreconditionedSolver(url, _ex);
}

char* skel_bHYPRE_HGMRES_fgetURL_bHYPRE_PreconditionedSolver(struct 
  bHYPRE_PreconditionedSolver__object* obj) { 
  return impl_bHYPRE_HGMRES_fgetURL_bHYPRE_PreconditionedSolver(obj);
}

struct bHYPRE_HGMRES__data*
bHYPRE_HGMRES__get_data(bHYPRE_HGMRES self)
{
  return (struct bHYPRE_HGMRES__data*)(self ? self->d_data : NULL);
}

void bHYPRE_HGMRES__set_data(
  bHYPRE_HGMRES self,
  struct bHYPRE_HGMRES__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
