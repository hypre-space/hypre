/*
 * File:          bHYPRE_StructVector_Skel.c
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_StructVector_IOR.h"
#include "bHYPRE_StructVector.h"
#include <stddef.h>

extern
void
impl_bHYPRE_StructVector__load(
  void);

extern
void
impl_bHYPRE_StructVector__ctor(
  /* in */ bHYPRE_StructVector self);

extern
void
impl_bHYPRE_StructVector__dtor(
  /* in */ bHYPRE_StructVector self);

extern
bHYPRE_StructVector
impl_bHYPRE_StructVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructGrid grid);

extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct bHYPRE_StructVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVectorView(struct 
  bHYPRE_StructVectorView__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct bHYPRE_StructVector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVector(struct 
  bHYPRE_StructVector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_StructVector_Clear(
  /* in */ bHYPRE_StructVector self);

extern
int32_t
impl_bHYPRE_StructVector_Copy(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_StructVector_Clone(
  /* in */ bHYPRE_StructVector self,
  /* out */ bHYPRE_Vector* x);

extern
int32_t
impl_bHYPRE_StructVector_Scale(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a);

extern
int32_t
impl_bHYPRE_StructVector_Dot(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d);

extern
int32_t
impl_bHYPRE_StructVector_Axpy(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_StructVector_SetCommunicator(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_StructVector_Initialize(
  /* in */ bHYPRE_StructVector self);

extern
int32_t
impl_bHYPRE_StructVector_Assemble(
  /* in */ bHYPRE_StructVector self);

extern
int32_t
impl_bHYPRE_StructVector_SetGrid(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_StructGrid grid);

extern
int32_t
impl_bHYPRE_StructVector_SetNumGhost(
  /* in */ bHYPRE_StructVector self,
  /* in */ int32_t* num_ghost,
  /* in */ int32_t dim2);

extern
int32_t
impl_bHYPRE_StructVector_SetValue(
  /* in */ bHYPRE_StructVector self,
  /* in */ int32_t* grid_index,
  /* in */ int32_t dim,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_StructVector_SetBoxValues(
  /* in */ bHYPRE_StructVector self,
  /* in */ int32_t* ilower,
  /* in */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ double* values,
  /* in */ int32_t nvalues);

extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct bHYPRE_StructVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVectorView(struct 
  bHYPRE_StructVectorView__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct bHYPRE_StructVector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVector(struct 
  bHYPRE_StructVector__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_StructVector_SetNumGhost(
  /* in */ bHYPRE_StructVector self,
/* in */ struct sidl_int__array* num_ghost)
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
      dim2);
  return _return;
}

static int32_t
skel_bHYPRE_StructVector_SetValue(
  /* in */ bHYPRE_StructVector self,
  /* in */ struct sidl_int__array* grid_index,
/* in */ double value)
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
      value);
  return _return;
}

static int32_t
skel_bHYPRE_StructVector_SetBoxValues(
  /* in */ bHYPRE_StructVector self,
  /* in */ struct sidl_int__array* ilower,
  /* in */ struct sidl_int__array* iupper,
/* in */ struct sidl_double__array* values)
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
      nvalues);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructVector__set_epv(struct bHYPRE_StructVector__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructVector__ctor;
  epv->f__dtor = impl_bHYPRE_StructVector__dtor;
  epv->f_Clear = impl_bHYPRE_StructVector_Clear;
  epv->f_Copy = impl_bHYPRE_StructVector_Copy;
  epv->f_Clone = impl_bHYPRE_StructVector_Clone;
  epv->f_Scale = impl_bHYPRE_StructVector_Scale;
  epv->f_Dot = impl_bHYPRE_StructVector_Dot;
  epv->f_Axpy = impl_bHYPRE_StructVector_Axpy;
  epv->f_SetCommunicator = impl_bHYPRE_StructVector_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_StructVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_StructVector_Assemble;
  epv->f_SetGrid = impl_bHYPRE_StructVector_SetGrid;
  epv->f_SetNumGhost = skel_bHYPRE_StructVector_SetNumGhost;
  epv->f_SetValue = skel_bHYPRE_StructVector_SetValue;
  epv->f_SetBoxValues = skel_bHYPRE_StructVector_SetBoxValues;

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
  impl_bHYPRE_StructVector__load();
}
struct bHYPRE_StructGrid__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(url, _ex);
}

char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj) { 
  return impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructGrid(obj);
}

struct bHYPRE_StructVectorView__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVectorView(url, _ex);
}

char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_StructVectorView(struct 
  bHYPRE_StructVectorView__object* obj) { 
  return impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVectorView(obj);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(url, _ex);
}

char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_StructVector_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_StructVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_StructVector_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) { 
  return impl_bHYPRE_StructVector_fgetURL_bHYPRE_Vector(obj);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_ProblemDefinition(url, _ex);
}

char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj) { 
  return impl_bHYPRE_StructVector_fgetURL_bHYPRE_ProblemDefinition(obj);
}

struct bHYPRE_StructVector__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_StructVector(url, _ex);
}

char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_StructVector(struct 
  bHYPRE_StructVector__object* obj) { 
  return impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructVector(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_StructVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_StructVector_fgetURL_sidl_BaseInterface(obj);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_bHYPRE_MatrixVectorView(url, _ex);
}

char* skel_bHYPRE_StructVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj) { 
  return impl_bHYPRE_StructVector_fgetURL_bHYPRE_MatrixVectorView(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructVector_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_StructVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_StructVector_fgetURL_sidl_BaseClass(obj);
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
