/*
 * File:          bHYPRE_SStructParCSRVector_Skel.c
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_SStructParCSRVector_IOR.h"
#include "bHYPRE_SStructParCSRVector.h"
#include <stddef.h>

extern
void
impl_bHYPRE_SStructParCSRVector__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRVector__ctor(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRVector__ctor2(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRVector__dtor(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_SStructParCSRVector
impl_bHYPRE_SStructParCSRVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructMatrixVectorView(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructParCSRVector(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetGrid(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

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
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_AddToValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

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
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Gather(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_GetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

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
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetComplex(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Print(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ const char* filename,
  /* in */ int32_t all,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_GetObject(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface* A,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_SetCommunicator(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRVector_Destroy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Initialize(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Assemble(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Clear(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Copy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Clone(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Scale(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ double a,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Dot(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRVector_Axpy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructMatrixVectorView(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructParCSRVector(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
static int32_t
skel_bHYPRE_SStructParCSRVector_SetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ double value,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRVector_SetValues(
      self,
      part,
      index_tmp,
      dim,
      var,
      value,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRVector_SetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ struct sidl_double__array* values,
/* out */ sidl_BaseInterface *_ex)
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
      nvalues,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRVector_AddToValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ double value,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRVector_AddToValues(
      self,
      part,
      index_tmp,
      dim,
      var,
      value,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRVector_AddToBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ struct sidl_double__array* values,
/* out */ sidl_BaseInterface *_ex)
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
      nvalues,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRVector_GetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* out */ double* value,
/* out */ sidl_BaseInterface *_ex)
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
      value,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRVector_GetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* inout rarray[nvalues] */ struct sidl_double__array** values,
/* out */ sidl_BaseInterface *_ex)
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
      nvalues,
      _ex);
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
  epv->f__ctor2 = impl_bHYPRE_SStructParCSRVector__ctor2;
  epv->f__dtor = impl_bHYPRE_SStructParCSRVector__dtor;
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
  epv->f_GetObject = impl_bHYPRE_SStructParCSRVector_GetObject;
  epv->f_SetCommunicator = impl_bHYPRE_SStructParCSRVector_SetCommunicator;
  epv->f_Destroy = impl_bHYPRE_SStructParCSRVector_Destroy;
  epv->f_Initialize = impl_bHYPRE_SStructParCSRVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructParCSRVector_Assemble;
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
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_bHYPRE_SStructParCSRVector__load(&_throwaway_exception);
}
struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(url,
    ar, _ex);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MPICommunicator(bi, _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(url,
    ar, _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MatrixVectorView(bi, _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(url,
    ar, _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_ProblemDefinition(bi,
    _ex);
}

struct bHYPRE_SStructGrid__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(url, ar,
    _ex);
}

struct bHYPRE_SStructGrid__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructGrid(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructGrid(bi, _ex);
}

struct bHYPRE_SStructMatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructMatrixVectorView(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return 
    impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructMatrixVectorView(url,
    ar, _ex);
}

struct bHYPRE_SStructMatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface *_ex) { 
  return 
    impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructMatrixVectorView(bi,
    _ex);
}

struct bHYPRE_SStructParCSRVector__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return 
    impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(url, ar,
    _ex);
}

struct bHYPRE_SStructParCSRVector__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructParCSRVector(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructParCSRVector(bi,
    _ex);
}

struct bHYPRE_SStructVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(url,
    ar, _ex);
}

struct bHYPRE_SStructVectorView__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructVectorView(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructVectorView(bi,
    _ex);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(url, ar, _ex);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_Vector(bi, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(url, ar,
    _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_SStructParCSRVector_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fconnect_sidl_RuntimeException(url, ar,
    _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_SStructParCSRVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRVector_fcast_sidl_RuntimeException(bi, _ex);
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
