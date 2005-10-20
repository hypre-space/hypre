/*
 * File:          bHYPRE_SStructParCSRVector_Skel.c
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side glue code for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#include "bHYPRE_SStructParCSRVector_IOR.h"
#include "bHYPRE_SStructParCSRVector.h"
#include <stddef.h>

extern
void
impl_bHYPRE_SStructParCSRVector__load(
  void);

extern
void
impl_bHYPRE_SStructParCSRVector__ctor(
  /* in */ bHYPRE_SStructParCSRVector self);

extern
void
impl_bHYPRE_SStructParCSRVector__dtor(
  /* in */ bHYPRE_SStructParCSRVector self);

extern
bHYPRE_SStructParCSRVector
impl_bHYPRE_SStructParCSRVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid);

extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj);
extern struct bHYPRE_SStruct_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStruct_MatrixVectorView(
  char* url, sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStruct_MatrixVectorView(
  struct bHYPRE_SStruct_MatrixVectorView__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructParCSRVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructParCSRVector(struct 
  bHYPRE_SStructParCSRVector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructParCSRVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructParCSRVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructVectorView(struct 
  bHYPRE_SStructVectorView__object* obj);
extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetCommunicator(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Initialize(
  /* in */ bHYPRE_SStructParCSRVector self);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Assemble(
  /* in */ bHYPRE_SStructParCSRVector self);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_GetObject(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface* A);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetGrid(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_SStructGrid grid);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[one] */ double* values,
  /* in */ int32_t one);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_AddToValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[one] */ double* values,
  /* in */ int32_t one);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_AddToBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Gather(
  /* in */ bHYPRE_SStructParCSRVector self);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_GetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_GetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* inout rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetComplex(
  /* in */ bHYPRE_SStructParCSRVector self);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Print(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ const char* filename,
  /* in */ int32_t all);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Clear(
  /* in */ bHYPRE_SStructParCSRVector self);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Copy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Clone(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Scale(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ double a);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Dot(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Axpy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x);

extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj);
extern struct bHYPRE_SStruct_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStruct_MatrixVectorView(
  char* url, sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStruct_MatrixVectorView(
  struct bHYPRE_SStruct_MatrixVectorView__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructParCSRVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructParCSRVector(struct 
  bHYPRE_SStructParCSRVector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructParCSRVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructParCSRVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructVectorView(struct 
  bHYPRE_SStructVectorView__object* obj);
static int32_t
skel_bHYPRE_SStructParCSRVector_SetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
/* in rarray[one] */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t one = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRVector_SetValues(
      self,
      part,
      index_tmp,
      dim,
      var,
      values_tmp,
      one);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRVector_SetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
/* in rarray[nvalues] */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  int32_t* ilower_tmp = ilower_proxy->d_firstElement;
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  int32_t* iupper_tmp = iupper_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nvalues = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(iupper_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRVector_SetBoxValues(
      self,
      part,
      ilower_tmp,
      iupper_tmp,
      dim,
      var,
      values_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRVector_AddToValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
/* in rarray[one] */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t one = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRVector_AddToValues(
      self,
      part,
      index_tmp,
      dim,
      var,
      values_tmp,
      one);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRVector_AddToBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
/* in rarray[nvalues] */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  int32_t* ilower_tmp = ilower_proxy->d_firstElement;
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  int32_t* iupper_tmp = iupper_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nvalues = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(iupper_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRVector_AddToBoxValues(
      self,
      part,
      ilower_tmp,
      iupper_tmp,
      dim,
      var,
      values_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRVector_GetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
/* out */ double* value)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRVector_GetValues(
      self,
      part,
      index_tmp,
      dim,
      var,
      value);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRVector_GetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
/* inout rarray[nvalues] */ struct sidl_double__array** values)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  int32_t* ilower_tmp = ilower_proxy->d_firstElement;
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  int32_t* iupper_tmp = iupper_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(*values,
    1, sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nvalues = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(iupper_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRVector_GetBoxValues(
      self,
      part,
      ilower_tmp,
      iupper_tmp,
      dim,
      var,
      values_tmp,
      nvalues);
  sidl_double__array_init(values_tmp, *values, 1, (*values)->d_metadata.d_lower,
    (*values)->d_metadata.d_upper, (*values)->d_metadata.d_stride);

  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructParCSRVector__set_epv(struct bHYPRE_SStructParCSRVector__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructParCSRVector__ctor;
  epv->f__dtor = impl_bHYPRE_SStructParCSRVector__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_SStructParCSRVector_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_SStructParCSRVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructParCSRVector_Assemble;
  epv->f_GetObject = impl_bHYPRE_SStructParCSRVector_GetObject;
  epv->f_SetGrid = impl_bHYPRE_SStructParCSRVector_SetGrid;
  epv->f_SetValues = skel_bHYPRE_SStructParCSRVector_SetValues;
  epv->f_SetBoxValues = skel_bHYPRE_SStructParCSRVector_SetBoxValues;
  epv->f_AddToValues = skel_bHYPRE_SStructParCSRVector_AddToValues;
  epv->f_AddToBoxValues = skel_bHYPRE_SStructParCSRVector_AddToBoxValues;
  epv->f_Gather = impl_bHYPRE_SStructParCSRVector_Gather;
  epv->f_GetValues = skel_bHYPRE_SStructParCSRVector_GetValues;
  epv->f_GetBoxValues = skel_bHYPRE_SStructParCSRVector_GetBoxValues;
  epv->f_SetComplex = impl_bHYPRE_SStructParCSRVector_SetComplex;
  epv->f_Print = impl_bHYPRE_SStructParCSRVector_Print;
  epv->f_Clear = impl_bHYPRE_SStructParCSRVector_Clear;
  epv->f_Copy = impl_bHYPRE_SStructParCSRVector_Copy;
  epv->f_Clone = impl_bHYPRE_SStructParCSRVector_Clone;
  epv->f_Scale = impl_bHYPRE_SStructParCSRVector_Scale;
  epv->f_Dot = impl_bHYPRE_SStructParCSRVector_Dot;
  epv->f_Axpy = impl_bHYPRE_SStructParCSRVector_Axpy;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructParCSRVector__set_sepv(struct bHYPRE_SStructParCSRVector__sepv 
  *sepv)
{
  sepv->f_Create = impl_bHYPRE_SStructParCSRVector_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_SStructParCSRVector__call_load(void) { 
  impl_bHYPRE_SStructParCSRVector__load();
}
struct bHYPRE_SStructGrid__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(url, _ex);
}

char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj) { 
  return impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructGrid(obj);
}

struct bHYPRE_SStruct_MatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStruct_MatrixVectorView(
  char* url, sidl_BaseInterface *_ex) { 
  return 
    impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStruct_MatrixVectorView(
    url, _ex);
}

char* 
  skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStruct_MatrixVectorView(
  struct bHYPRE_SStruct_MatrixVectorView__object* obj) { 
  return 
    impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStruct_MatrixVectorView(
    obj);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(url,
    _ex);
}

char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_SStructParCSRVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_SStructParCSRVector_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) { 
  return impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_Vector(obj);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(url,
    _ex);
}

char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj) { 
  return impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_ProblemDefinition(obj);
}

struct bHYPRE_SStructParCSRVector__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(char* url,
  sidl_BaseInterface *_ex) { 
  return 
    impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(url,
    _ex);
}

char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructParCSRVector(struct 
  bHYPRE_SStructParCSRVector__object* obj) { 
  return 
    impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructParCSRVector(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_SStructParCSRVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_SStructParCSRVector_fgetURL_sidl_BaseInterface(obj);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(url,
    _ex);
}

char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj) { 
  return impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_MatrixVectorView(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_SStructParCSRVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_SStructParCSRVector_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_SStructVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(url,
    _ex);
}

char* skel_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructVectorView(struct 
  bHYPRE_SStructVectorView__object* obj) { 
  return impl_bHYPRE_SStructParCSRVector_fgetURL_bHYPRE_SStructVectorView(obj);
}

struct bHYPRE_SStructParCSRVector__data*
bHYPRE_SStructParCSRVector__get_data(bHYPRE_SStructParCSRVector self)
{
  return (struct bHYPRE_SStructParCSRVector__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructParCSRVector__set_data(
  bHYPRE_SStructParCSRVector self,
  struct bHYPRE_SStructParCSRVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
