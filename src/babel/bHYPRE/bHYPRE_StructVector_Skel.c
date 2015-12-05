/*
 * File:          bHYPRE_StructVector_Skel.c
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_StructVector_IOR.h"
#include "bHYPRE_StructVector.h"
#include <stddef.h>

extern
void
impl_bHYPRE_StructVector__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructVector__ctor(
  /* in */ bHYPRE_StructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructVector__ctor2(
  /* in */ bHYPRE_StructVector self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructVector__dtor(
  /* in */ bHYPRE_StructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_StructVector
impl_bHYPRE_StructVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_StructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructVector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructVector__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_StructVector(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructVectorView__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_StructVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_StructVector_SetGrid(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_StructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_SetNumGhost(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_SetValue(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ int32_t* grid_index,
  /* in */ int32_t dim,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_SetBoxValues(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_SetCommunicator(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructVector_Destroy(
  /* in */ bHYPRE_StructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_Initialize(
  /* in */ bHYPRE_StructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_Assemble(
  /* in */ bHYPRE_StructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_Clear(
  /* in */ bHYPRE_StructVector self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_Copy(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_Clone(
  /* in */ bHYPRE_StructVector self,
  /* out */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_Scale(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_Dot(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructVector_Axpy(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_StructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructVector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructVector__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_StructVector(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructVectorView__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_StructVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
static int32_t
skel_bHYPRE_StructVector_SetNumGhost(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim2] */ struct sidl_int__array* num_ghost,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* num_ghost_proxy = sidl_int__array_ensure(num_ghost, 1,
    sidl_column_major_order);
  int32_t* num_ghost_tmp = num_ghost_proxy->d_firstElement;
  int32_t dim2 = sidlLength(num_ghost_proxy,0);
  _return =
    impl_bHYPRE_StructVector_SetNumGhost(
      self,
      num_ghost_tmp,
      dim2,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_StructVector_SetValue(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ struct sidl_int__array* grid_index,
  /* in */ double value,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* grid_index_proxy = sidl_int__array_ensure(grid_index,
    1, sidl_column_major_order);
  int32_t* grid_index_tmp = grid_index_proxy->d_firstElement;
  int32_t dim = sidlLength(grid_index_proxy,0);
  _return =
    impl_bHYPRE_StructVector_SetValue(
      self,
      grid_index_tmp,
      dim,
      value,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_StructVector_SetBoxValues(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
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
    impl_bHYPRE_StructVector_SetBoxValues(
      self,
      ilower_tmp,
      iupper_tmp,
      dim,
      values_tmp,
      nvalues,
      _ex);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructVector__set_epv(struct bHYPRE_StructVector__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructVector__ctor;
  epv->f__ctor2 = impl_bHYPRE_StructVector__ctor2;
  epv->f__dtor = impl_bHYPRE_StructVector__dtor;
  epv->f_SetGrid = impl_bHYPRE_StructVector_SetGrid;
  epv->f_SetNumGhost = skel_bHYPRE_StructVector_SetNumGhost;
  epv->f_SetValue = skel_bHYPRE_StructVector_SetValue;
  epv->f_SetBoxValues = skel_bHYPRE_StructVector_SetBoxValues;
  epv->f_SetCommunicator = impl_bHYPRE_StructVector_SetCommunicator;
  epv->f_Destroy = impl_bHYPRE_StructVector_Destroy;
  epv->f_Initialize = impl_bHYPRE_StructVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_StructVector_Assemble;
  epv->f_Clear = impl_bHYPRE_StructVector_Clear;
  epv->f_Copy = impl_bHYPRE_StructVector_Copy;
  epv->f_Clone = impl_bHYPRE_StructVector_Clone;
  epv->f_Scale = impl_bHYPRE_StructVector_Scale;
  epv->f_Dot = impl_bHYPRE_StructVector_Dot;
  epv->f_Axpy = impl_bHYPRE_StructVector_Axpy;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructVector__set_sepv(struct bHYPRE_StructVector__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_StructVector_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_StructVector__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_bHYPRE_StructVector__load(&_throwaway_exception);
}
struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(url, ar, _ex);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_StructVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fcast_bHYPRE_MPICommunicator(bi, _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(url, ar,
    _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_StructVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fcast_bHYPRE_MatrixVectorView(bi, _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(url, ar,
    _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_StructVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fcast_bHYPRE_ProblemDefinition(bi, _ex);
}

struct bHYPRE_StructGrid__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(url, ar, _ex);
}

struct bHYPRE_StructGrid__object* 
  skel_bHYPRE_StructVector_fcast_bHYPRE_StructGrid(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fcast_bHYPRE_StructGrid(bi, _ex);
}

struct bHYPRE_StructVector__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(url, ar, _ex);
}

struct bHYPRE_StructVector__object* 
  skel_bHYPRE_StructVector_fcast_bHYPRE_StructVector(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fcast_bHYPRE_StructVector(bi, _ex);
}

struct bHYPRE_StructVectorView__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(url, ar,
    _ex);
}

struct bHYPRE_StructVectorView__object* 
  skel_bHYPRE_StructVector_fcast_bHYPRE_StructVectorView(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fcast_bHYPRE_StructVectorView(bi, _ex);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_Vector(url, ar, _ex);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_StructVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fcast_bHYPRE_Vector(bi, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_StructVector_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_StructVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fcast_sidl_RuntimeException(bi, _ex);
}

struct bHYPRE_StructVector__data*
bHYPRE_StructVector__get_data(bHYPRE_StructVector self)
{
  return (struct bHYPRE_StructVector__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructVector__set_data(
  bHYPRE_StructVector self,
  struct bHYPRE_StructVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
