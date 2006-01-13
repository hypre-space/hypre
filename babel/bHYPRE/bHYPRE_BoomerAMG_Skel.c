/*
 * File:          bHYPRE_BoomerAMG_Skel.c
 * Symbol:        bHYPRE.BoomerAMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side glue code for bHYPRE.BoomerAMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "bHYPRE_BoomerAMG_IOR.h"
#include "bHYPRE_BoomerAMG.h"
#include <stddef.h>

extern
void
impl_bHYPRE_BoomerAMG__load(
  void);

extern
void
impl_bHYPRE_BoomerAMG__ctor(
  /* in */ bHYPRE_BoomerAMG self);

extern
void
impl_bHYPRE_BoomerAMG__dtor(
  /* in */ bHYPRE_BoomerAMG self);

extern
bHYPRE_BoomerAMG
impl_bHYPRE_BoomerAMG_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_IJParCSRMatrix A);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_BoomerAMG__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_BoomerAMG(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_BoomerAMG(struct 
  bHYPRE_BoomerAMG__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_BoomerAMG_SetLevelRelaxWt(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ double relax_wt,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_BoomerAMG_InitGridRelaxation(
  /* in */ bHYPRE_BoomerAMG self,
  /* out array<int,column-major> */ struct sidl_int__array** num_grid_sweeps,
  /* out array<int,column-major> */ struct sidl_int__array** grid_relax_type,
  /* out array<int,2,
    column-major> */ struct sidl_int__array** grid_relax_points,
  /* in */ int32_t coarsen_type,
  /* out array<double,
    column-major> */ struct sidl_double__array** relax_weights,
  /* in */ int32_t max_levels);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetCommunicator(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetIntParameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetDoubleParameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetStringParameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetIntArray1Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetIntArray2Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_BoomerAMG_GetIntValue(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_BoomerAMG_GetDoubleValue(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_BoomerAMG_Setup(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_BoomerAMG_Apply(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_BoomerAMG_ApplyAdjoint(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetOperator(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Operator A);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetTolerance(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ double tolerance);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetMaxIterations(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t max_iterations);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetLogging(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_BoomerAMG_SetPrintLevel(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t level);

extern
int32_t
impl_bHYPRE_BoomerAMG_GetNumIterations(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ int32_t* num_iterations);

extern
int32_t
impl_bHYPRE_BoomerAMG_GetRelResidualNorm(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ double* norm);

extern struct bHYPRE_Solver__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj);
extern struct bHYPRE_BoomerAMG__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_BoomerAMG(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_BoomerAMG(struct 
  bHYPRE_BoomerAMG__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_BoomerAMG_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_BoomerAMG_InitGridRelaxation(
  /* in */ bHYPRE_BoomerAMG self,
  /* out array<int,column-major> */ struct sidl_int__array** num_grid_sweeps,
  /* out array<int,column-major> */ struct sidl_int__array** grid_relax_type,
  /* out array<int,2,
    column-major> */ struct sidl_int__array** grid_relax_points,
  /* in */ int32_t coarsen_type,
  /* out array<double,
    column-major> */ struct sidl_double__array** relax_weights,
/* in */ int32_t max_levels)
{
  int32_t _return;
  struct sidl_int__array* num_grid_sweeps_proxy = NULL;
  struct sidl_int__array* grid_relax_type_proxy = NULL;
  struct sidl_int__array* grid_relax_points_proxy = NULL;
  struct sidl_double__array* relax_weights_proxy = NULL;
  _return =
    impl_bHYPRE_BoomerAMG_InitGridRelaxation(
      self,
      &num_grid_sweeps_proxy,
      &grid_relax_type_proxy,
      &grid_relax_points_proxy,
      coarsen_type,
      &relax_weights_proxy,
      max_levels);
  *num_grid_sweeps = sidl_int__array_ensure(num_grid_sweeps_proxy, 1,
    sidl_column_major_order);
  sidl_int__array_deleteRef(num_grid_sweeps_proxy);
  *grid_relax_type = sidl_int__array_ensure(grid_relax_type_proxy, 1,
    sidl_column_major_order);
  sidl_int__array_deleteRef(grid_relax_type_proxy);
  *grid_relax_points = sidl_int__array_ensure(grid_relax_points_proxy, 2,
    sidl_column_major_order);
  sidl_int__array_deleteRef(grid_relax_points_proxy);
  *relax_weights = sidl_double__array_ensure(relax_weights_proxy, 1,
    sidl_column_major_order);
  sidl_double__array_deleteRef(relax_weights_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_BoomerAMG_SetIntArray1Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
/* in rarray[nvalues] */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  int32_t* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_BoomerAMG_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_BoomerAMG_SetIntArray2Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
/* in array<int,2,column-major> */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_BoomerAMG_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_BoomerAMG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
/* in rarray[nvalues] */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  double* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_BoomerAMG_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_BoomerAMG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
/* in array<double,2,column-major> */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_BoomerAMG_SetDoubleArray2Parameter(
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
bHYPRE_BoomerAMG__set_epv(struct bHYPRE_BoomerAMG__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_BoomerAMG__ctor;
  epv->f__dtor = impl_bHYPRE_BoomerAMG__dtor;
  epv->f_SetLevelRelaxWt = impl_bHYPRE_BoomerAMG_SetLevelRelaxWt;
  epv->f_InitGridRelaxation = skel_bHYPRE_BoomerAMG_InitGridRelaxation;
  epv->f_SetCommunicator = impl_bHYPRE_BoomerAMG_SetCommunicator;
  epv->f_SetIntParameter = impl_bHYPRE_BoomerAMG_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_BoomerAMG_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_BoomerAMG_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_BoomerAMG_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_BoomerAMG_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_BoomerAMG_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_BoomerAMG_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_BoomerAMG_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_BoomerAMG_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_BoomerAMG_Setup;
  epv->f_Apply = impl_bHYPRE_BoomerAMG_Apply;
  epv->f_ApplyAdjoint = impl_bHYPRE_BoomerAMG_ApplyAdjoint;
  epv->f_SetOperator = impl_bHYPRE_BoomerAMG_SetOperator;
  epv->f_SetTolerance = impl_bHYPRE_BoomerAMG_SetTolerance;
  epv->f_SetMaxIterations = impl_bHYPRE_BoomerAMG_SetMaxIterations;
  epv->f_SetLogging = impl_bHYPRE_BoomerAMG_SetLogging;
  epv->f_SetPrintLevel = impl_bHYPRE_BoomerAMG_SetPrintLevel;
  epv->f_GetNumIterations = impl_bHYPRE_BoomerAMG_GetNumIterations;
  epv->f_GetRelResidualNorm = impl_bHYPRE_BoomerAMG_GetRelResidualNorm;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_BoomerAMG__set_sepv(struct bHYPRE_BoomerAMG__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_BoomerAMG_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_BoomerAMG__call_load(void) { 
  impl_bHYPRE_BoomerAMG__load();
}
struct bHYPRE_Solver__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Solver(url, _ex);
}

char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Solver(struct bHYPRE_Solver__object* 
  obj) { 
  return impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Solver(obj);
}

struct bHYPRE_BoomerAMG__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_BoomerAMG(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_BoomerAMG(url, _ex);
}

char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_BoomerAMG(struct 
  bHYPRE_BoomerAMG__object* obj) { 
  return impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_BoomerAMG(obj);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_MPICommunicator(url, _ex);
}

char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Operator(url, _ex);
}

char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) { 
  return impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Operator(obj);
}

struct bHYPRE_IJParCSRMatrix__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_IJParCSRMatrix(url, _ex);
}

char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj) { 
  return impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_IJParCSRMatrix(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_BoomerAMG_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_BoomerAMG_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_BoomerAMG_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_BoomerAMG_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Vector(struct bHYPRE_Vector__object* 
  obj) { 
  return impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Vector(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_BoomerAMG_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_BoomerAMG_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_BoomerAMG_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_BoomerAMG_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_BoomerAMG_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_BoomerAMG_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_BoomerAMG__data*
bHYPRE_BoomerAMG__get_data(bHYPRE_BoomerAMG self)
{
  return (struct bHYPRE_BoomerAMG__data*)(self ? self->d_data : NULL);
}

void bHYPRE_BoomerAMG__set_data(
  bHYPRE_BoomerAMG self,
  struct bHYPRE_BoomerAMG__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
