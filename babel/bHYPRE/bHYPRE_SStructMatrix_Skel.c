/*
 * File:          bHYPRE_SStructMatrix_Skel.c
 * Symbol:        bHYPRE.SStructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for bHYPRE.SStructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_SStructMatrix_IOR.h"
#include "bHYPRE_SStructMatrix.h"
#include <stddef.h>

extern
void
impl_bHYPRE_SStructMatrix__load(
  void);

extern
void
impl_bHYPRE_SStructMatrix__ctor(
  /* in */ bHYPRE_SStructMatrix self);

extern
void
impl_bHYPRE_SStructMatrix__dtor(
  /* in */ bHYPRE_SStructMatrix self);

extern
bHYPRE_SStructMatrix
impl_bHYPRE_SStructMatrix_Create(
  /* in */ void* mpi_comm,
  /* in */ bHYPRE_SStructGraph graph);

extern struct bHYPRE_SStructMatrix__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructMatrix(struct 
  bHYPRE_SStructMatrix__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructGraph(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_SStructBuildMatrix__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructBuildMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructBuildMatrix(struct 
  bHYPRE_SStructBuildMatrix__object* obj);
extern
int32_t
impl_bHYPRE_SStructMatrix_SetCommunicator(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ void* mpi_comm);

extern
int32_t
impl_bHYPRE_SStructMatrix_Initialize(
  /* in */ bHYPRE_SStructMatrix self);

extern
int32_t
impl_bHYPRE_SStructMatrix_Assemble(
  /* in */ bHYPRE_SStructMatrix self);

extern
int32_t
impl_bHYPRE_SStructMatrix_GetObject(
  /* in */ bHYPRE_SStructMatrix self,
  /* out */ sidl_BaseInterface* A);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetGraph(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ bHYPRE_SStructGraph graph);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ int32_t* entries,
  /* in */ double* values);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetBoxValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t* ilower,
  /* in */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ int32_t* entries,
  /* in */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructMatrix_AddToValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ int32_t* entries,
  /* in */ double* values);

extern
int32_t
impl_bHYPRE_SStructMatrix_AddToBoxValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t* ilower,
  /* in */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ int32_t* entries,
  /* in */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetSymmetric(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t to_var,
  /* in */ int32_t symmetric);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetNSSymmetric(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t symmetric);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetComplex(
  /* in */ bHYPRE_SStructMatrix self);

extern
int32_t
impl_bHYPRE_SStructMatrix_Print(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* filename,
  /* in */ int32_t all);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetIntParameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetDoubleParameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetStringParameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_SStructMatrix_GetIntValue(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_SStructMatrix_GetDoubleValue(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_SStructMatrix_Setup(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_SStructMatrix_Apply(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern struct bHYPRE_SStructMatrix__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructMatrix(struct 
  bHYPRE_SStructMatrix__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructGraph(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_SStructBuildMatrix__object* 
  impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructBuildMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructBuildMatrix(struct 
  bHYPRE_SStructBuildMatrix__object* obj);
static int32_t
skel_bHYPRE_SStructMatrix_SetValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ struct sidl_int__array* entries,
/* in */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1,
    sidl_column_major_order);
  int32_t* entries_tmp = entries_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nentries = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_SStructMatrix_SetValues(
      self,
      part,
      index_tmp,
      dim,
      var,
      nentries,
      entries_tmp,
      values_tmp);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_SetBoxValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* ilower,
  /* in */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in */ struct sidl_int__array* entries,
/* in */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  int32_t* ilower_tmp = ilower_proxy->d_firstElement;
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  int32_t* iupper_tmp = iupper_proxy->d_firstElement;
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1,
    sidl_column_major_order);
  int32_t* entries_tmp = entries_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nentries = sidlLength(entries_proxy,0);
  int32_t nvalues = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(iupper_proxy,0);
  _return =
    impl_bHYPRE_SStructMatrix_SetBoxValues(
      self,
      part,
      ilower_tmp,
      iupper_tmp,
      dim,
      var,
      nentries,
      entries_tmp,
      values_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_AddToValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ struct sidl_int__array* entries,
/* in */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1,
    sidl_column_major_order);
  int32_t* entries_tmp = entries_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nentries = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_SStructMatrix_AddToValues(
      self,
      part,
      index_tmp,
      dim,
      var,
      nentries,
      entries_tmp,
      values_tmp);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_AddToBoxValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* ilower,
  /* in */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in */ struct sidl_int__array* entries,
/* in */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  int32_t* ilower_tmp = ilower_proxy->d_firstElement;
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  int32_t* iupper_tmp = iupper_proxy->d_firstElement;
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1,
    sidl_column_major_order);
  int32_t* entries_tmp = entries_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nentries = sidlLength(entries_proxy,0);
  int32_t nvalues = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(iupper_proxy,0);
  _return =
    impl_bHYPRE_SStructMatrix_AddToBoxValues(
      self,
      part,
      ilower_tmp,
      iupper_tmp,
      dim,
      var,
      nentries,
      entries_tmp,
      values_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  int32_t* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_SStructMatrix_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructMatrix_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  double* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_SStructMatrix_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_SStructMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructMatrix_SetDoubleArray2Parameter(
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
bHYPRE_SStructMatrix__set_epv(struct bHYPRE_SStructMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructMatrix__ctor;
  epv->f__dtor = impl_bHYPRE_SStructMatrix__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_SStructMatrix_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_SStructMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructMatrix_Assemble;
  epv->f_GetObject = impl_bHYPRE_SStructMatrix_GetObject;
  epv->f_SetGraph = impl_bHYPRE_SStructMatrix_SetGraph;
  epv->f_SetValues = skel_bHYPRE_SStructMatrix_SetValues;
  epv->f_SetBoxValues = skel_bHYPRE_SStructMatrix_SetBoxValues;
  epv->f_AddToValues = skel_bHYPRE_SStructMatrix_AddToValues;
  epv->f_AddToBoxValues = skel_bHYPRE_SStructMatrix_AddToBoxValues;
  epv->f_SetSymmetric = impl_bHYPRE_SStructMatrix_SetSymmetric;
  epv->f_SetNSSymmetric = impl_bHYPRE_SStructMatrix_SetNSSymmetric;
  epv->f_SetComplex = impl_bHYPRE_SStructMatrix_SetComplex;
  epv->f_Print = impl_bHYPRE_SStructMatrix_Print;
  epv->f_SetIntParameter = impl_bHYPRE_SStructMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_SStructMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_SStructMatrix_SetStringParameter;
  epv->f_SetIntArray1Parameter = 
    skel_bHYPRE_SStructMatrix_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = 
    skel_bHYPRE_SStructMatrix_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_SStructMatrix_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_SStructMatrix_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_SStructMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_SStructMatrix_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_SStructMatrix_Setup;
  epv->f_Apply = impl_bHYPRE_SStructMatrix_Apply;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructMatrix__set_sepv(struct bHYPRE_SStructMatrix__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_SStructMatrix_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_SStructMatrix__call_load(void) { 
  impl_bHYPRE_SStructMatrix__load();
}
struct bHYPRE_SStructMatrix__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructMatrix(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructMatrix(url, _ex);
}

char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructMatrix(struct 
  bHYPRE_SStructMatrix__object* obj) { 
  return impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructMatrix(obj);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_Operator(url, _ex);
}

char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) { 
  return impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Operator(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructMatrix_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_SStructMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_SStructMatrix_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) { 
  return impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_Vector(obj);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_ProblemDefinition(url, _ex);
}

char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj) { 
  return impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_ProblemDefinition(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructMatrix_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_SStructMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_SStructMatrix_fgetURL_sidl_BaseInterface(obj);
}

struct bHYPRE_SStructGraph__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructGraph(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructGraph(url, _ex);
}

char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj) { 
  return impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructGraph(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructMatrix_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_SStructMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_SStructMatrix_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_SStructBuildMatrix__object* 
  skel_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructBuildMatrix(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructMatrix_fconnect_bHYPRE_SStructBuildMatrix(url, _ex);
}

char* skel_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructBuildMatrix(struct 
  bHYPRE_SStructBuildMatrix__object* obj) { 
  return impl_bHYPRE_SStructMatrix_fgetURL_bHYPRE_SStructBuildMatrix(obj);
}

struct bHYPRE_SStructMatrix__data*
bHYPRE_SStructMatrix__get_data(bHYPRE_SStructMatrix self)
{
  return (struct bHYPRE_SStructMatrix__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructMatrix__set_data(
  bHYPRE_SStructMatrix self,
  struct bHYPRE_SStructMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
