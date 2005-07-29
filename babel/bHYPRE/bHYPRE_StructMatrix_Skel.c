/*
 * File:          bHYPRE_StructMatrix_Skel.c
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_StructMatrix_IOR.h"
#include "bHYPRE_StructMatrix.h"
#include <stddef.h>

extern
void
impl_bHYPRE_StructMatrix__load(
  void);

extern
void
impl_bHYPRE_StructMatrix__ctor(
  /* in */ bHYPRE_StructMatrix self);

extern
void
impl_bHYPRE_StructMatrix__dtor(
  /* in */ bHYPRE_StructMatrix self);

extern
bHYPRE_StructMatrix
impl_bHYPRE_StructMatrix_Create(
  /* in */ void* mpi_comm,
  /* in */ bHYPRE_StructGrid grid,
  /* in */ bHYPRE_StructStencil stencil);

extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructMatrix(struct 
  bHYPRE_StructMatrix__object* obj);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct bHYPRE_StructBuildMatrix__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructBuildMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructBuildMatrix(struct 
  bHYPRE_StructBuildMatrix__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructStencil(struct 
  bHYPRE_StructStencil__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_StructMatrix_SetCommunicator(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ void* mpi_comm);

extern
int32_t
impl_bHYPRE_StructMatrix_Initialize(
  /* in */ bHYPRE_StructMatrix self);

extern
int32_t
impl_bHYPRE_StructMatrix_Assemble(
  /* in */ bHYPRE_StructMatrix self);

extern
int32_t
impl_bHYPRE_StructMatrix_GetObject(
  /* in */ bHYPRE_StructMatrix self,
  /* out */ sidl_BaseInterface* A);

extern
int32_t
impl_bHYPRE_StructMatrix_SetGrid(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_StructGrid grid);

extern
int32_t
impl_bHYPRE_StructMatrix_SetStencil(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_StructStencil stencil);

extern
int32_t
impl_bHYPRE_StructMatrix_SetValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in */ int32_t* stencil_indices,
  /* in */ double* values);

extern
int32_t
impl_bHYPRE_StructMatrix_SetBoxValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t* ilower,
  /* in */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in */ int32_t* stencil_indices,
  /* in */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructMatrix_SetNumGhost(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t* num_ghost,
  /* in */ int32_t dim2);

extern
int32_t
impl_bHYPRE_StructMatrix_SetSymmetric(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t symmetric);

extern
int32_t
impl_bHYPRE_StructMatrix_SetConstantEntries(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t num_stencil_constant_points,
  /* in */ int32_t* stencil_constant_points);

extern
int32_t
impl_bHYPRE_StructMatrix_SetConstantValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t num_stencil_indices,
  /* in */ int32_t* stencil_indices,
  /* in */ double* values);

extern
int32_t
impl_bHYPRE_StructMatrix_SetIntParameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_StructMatrix_SetDoubleParameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_StructMatrix_SetStringParameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_StructMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_StructMatrix_GetIntValue(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_StructMatrix_GetDoubleValue(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_StructMatrix_Setup(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_StructMatrix_Apply(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructMatrix(struct 
  bHYPRE_StructMatrix__object* obj);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct bHYPRE_StructBuildMatrix__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructBuildMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructBuildMatrix(struct 
  bHYPRE_StructBuildMatrix__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructStencil(struct 
  bHYPRE_StructStencil__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_StructMatrix_SetValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ struct sidl_int__array* index,
  /* in */ struct sidl_int__array* stencil_indices,
/* in */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  struct sidl_int__array* stencil_indices_proxy = 
    sidl_int__array_ensure(stencil_indices, 1, sidl_column_major_order);
  int32_t* stencil_indices_tmp = stencil_indices_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t num_stencil_indices = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetValues(
      self,
      index_tmp,
      dim,
      num_stencil_indices,
      stencil_indices_tmp,
      values_tmp);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetBoxValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ struct sidl_int__array* ilower,
  /* in */ struct sidl_int__array* iupper,
  /* in */ struct sidl_int__array* stencil_indices,
/* in */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  int32_t* ilower_tmp = ilower_proxy->d_firstElement;
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  int32_t* iupper_tmp = iupper_proxy->d_firstElement;
  struct sidl_int__array* stencil_indices_proxy = 
    sidl_int__array_ensure(stencil_indices, 1, sidl_column_major_order);
  int32_t* stencil_indices_tmp = stencil_indices_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t num_stencil_indices = sidlLength(stencil_indices_proxy,0);
  int32_t nvalues = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(iupper_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetBoxValues(
      self,
      ilower_tmp,
      iupper_tmp,
      dim,
      num_stencil_indices,
      stencil_indices_tmp,
      values_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetNumGhost(
  /* in */ bHYPRE_StructMatrix self,
/* in */ struct sidl_int__array* num_ghost)
{
  int32_t _return;
  struct sidl_int__array* num_ghost_proxy = sidl_int__array_ensure(num_ghost, 1,
    sidl_column_major_order);
  int32_t* num_ghost_tmp = num_ghost_proxy->d_firstElement;
  int32_t dim2 = sidlLength(num_ghost_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetNumGhost(
      self,
      num_ghost_tmp,
      dim2);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetConstantEntries(
  /* in */ bHYPRE_StructMatrix self,
/* in */ struct sidl_int__array* stencil_constant_points)
{
  int32_t _return;
  struct sidl_int__array* stencil_constant_points_proxy = 
    sidl_int__array_ensure(stencil_constant_points, 1, sidl_column_major_order);
  int32_t* stencil_constant_points_tmp = 
    stencil_constant_points_proxy->d_firstElement;
  int32_t num_stencil_constant_points = 
    sidlLength(stencil_constant_points_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetConstantEntries(
      self,
      num_stencil_constant_points,
      stencil_constant_points_tmp);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetConstantValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ struct sidl_int__array* stencil_indices,
/* in */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* stencil_indices_proxy = 
    sidl_int__array_ensure(stencil_indices, 1, sidl_column_major_order);
  int32_t* stencil_indices_tmp = stencil_indices_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t num_stencil_indices = sidlLength(values_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetConstantValues(
      self,
      num_stencil_indices,
      stencil_indices_tmp,
      values_tmp);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  int32_t* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  double* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
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
bHYPRE_StructMatrix__set_epv(struct bHYPRE_StructMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructMatrix__ctor;
  epv->f__dtor = impl_bHYPRE_StructMatrix__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructMatrix_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_StructMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_StructMatrix_Assemble;
  epv->f_GetObject = impl_bHYPRE_StructMatrix_GetObject;
  epv->f_SetGrid = impl_bHYPRE_StructMatrix_SetGrid;
  epv->f_SetStencil = impl_bHYPRE_StructMatrix_SetStencil;
  epv->f_SetValues = skel_bHYPRE_StructMatrix_SetValues;
  epv->f_SetBoxValues = skel_bHYPRE_StructMatrix_SetBoxValues;
  epv->f_SetNumGhost = skel_bHYPRE_StructMatrix_SetNumGhost;
  epv->f_SetSymmetric = impl_bHYPRE_StructMatrix_SetSymmetric;
  epv->f_SetConstantEntries = skel_bHYPRE_StructMatrix_SetConstantEntries;
  epv->f_SetConstantValues = skel_bHYPRE_StructMatrix_SetConstantValues;
  epv->f_SetIntParameter = impl_bHYPRE_StructMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_StructMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_StructMatrix_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_StructMatrix_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_StructMatrix_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_StructMatrix_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_StructMatrix_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_StructMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_StructMatrix_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_StructMatrix_Setup;
  epv->f_Apply = impl_bHYPRE_StructMatrix_Apply;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructMatrix__set_sepv(struct bHYPRE_StructMatrix__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_StructMatrix_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_StructMatrix__call_load(void) { 
  impl_bHYPRE_StructMatrix__load();
}
struct bHYPRE_StructMatrix__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrix(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrix(url, _ex);
}

char* skel_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructMatrix(struct 
  bHYPRE_StructMatrix__object* obj) { 
  return impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructMatrix(obj);
}

struct bHYPRE_StructGrid__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructGrid(url, _ex);
}

char* skel_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj) { 
  return impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructGrid(obj);
}

struct bHYPRE_StructBuildMatrix__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructBuildMatrix(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructBuildMatrix(url, _ex);
}

char* skel_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructBuildMatrix(struct 
  bHYPRE_StructBuildMatrix__object* obj) { 
  return impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructBuildMatrix(obj);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Operator(url, _ex);
}

char* skel_bHYPRE_StructMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) { 
  return impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_Operator(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_StructMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_StructMatrix_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_StructMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) { 
  return impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_Vector(obj);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_ProblemDefinition(url, _ex);
}

char* skel_bHYPRE_StructMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj) { 
  return impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_ProblemDefinition(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_StructMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_StructMatrix_fgetURL_sidl_BaseInterface(obj);
}

struct bHYPRE_StructStencil__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructStencil(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructStencil(url, _ex);
}

char* skel_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructStencil(struct 
  bHYPRE_StructStencil__object* obj) { 
  return impl_bHYPRE_StructMatrix_fgetURL_bHYPRE_StructStencil(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_StructMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_StructMatrix_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_StructMatrix__data*
bHYPRE_StructMatrix__get_data(bHYPRE_StructMatrix self)
{
  return (struct bHYPRE_StructMatrix__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructMatrix__set_data(
  bHYPRE_StructMatrix self,
  struct bHYPRE_StructMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
