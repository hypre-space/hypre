/*
 * File:          bHYPRE_SStructGrid_Skel.c
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side glue code for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#include "bHYPRE_SStructGrid_IOR.h"
#include "bHYPRE_SStructGrid.h"
#include <stddef.h>

extern
void
impl_bHYPRE_SStructGrid__load(
  void);

extern
void
impl_bHYPRE_SStructGrid__ctor(
  /* in */ bHYPRE_SStructGrid self);

extern
void
impl_bHYPRE_SStructGrid__dtor(
  /* in */ bHYPRE_SStructGrid self);

extern
bHYPRE_SStructGrid
impl_bHYPRE_SStructGrid_Create(
  /* in */ void* mpi_comm,
  /* in */ int32_t ndim,
  /* in */ int32_t nparts);

extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGrid_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGrid_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructGrid_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGrid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGrid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGrid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructGrid_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGrid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_SStructGrid_SetNumDimParts(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t ndim,
  /* in */ int32_t nparts);

extern
int32_t
impl_bHYPRE_SStructGrid_SetCommunicator(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ void* mpi_comm);

extern
int32_t
impl_bHYPRE_SStructGrid_SetExtents(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim);

extern
int32_t
impl_bHYPRE_SStructGrid_SetVariable(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t nvars,
  /* in */ enum bHYPRE_SStructVariable__enum vartype);

extern
int32_t
impl_bHYPRE_SStructGrid_AddVariable(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ enum bHYPRE_SStructVariable__enum vartype);

extern
int32_t
impl_bHYPRE_SStructGrid_SetNeighborBox(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t nbor_part,
  /* in rarray[dim] */ int32_t* nbor_ilower,
  /* in rarray[dim] */ int32_t* nbor_iupper,
  /* in rarray[dim] */ int32_t* index_map,
  /* in */ int32_t dim);

extern
int32_t
impl_bHYPRE_SStructGrid_AddUnstructuredPart(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper);

extern
int32_t
impl_bHYPRE_SStructGrid_SetPeriodic(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* periodic,
  /* in */ int32_t dim);

extern
int32_t
impl_bHYPRE_SStructGrid_SetNumGhost(
  /* in */ bHYPRE_SStructGrid self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2);

extern
int32_t
impl_bHYPRE_SStructGrid_Assemble(
  /* in */ bHYPRE_SStructGrid self);

extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGrid_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGrid_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructGrid_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGrid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGrid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGrid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructGrid_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGrid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_SStructGrid_SetExtents(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
/* in rarray[dim] */ struct sidl_int__array* iupper)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  int32_t* ilower_tmp = ilower_proxy->d_firstElement;
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  int32_t* iupper_tmp = iupper_proxy->d_firstElement;
  int32_t dim = sidlLength(iupper_proxy,0);
  _return =
    impl_bHYPRE_SStructGrid_SetExtents(
      self,
      part,
      ilower_tmp,
      iupper_tmp,
      dim);
  return _return;
}

static int32_t
skel_bHYPRE_SStructGrid_AddVariable(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
/* in */ enum bHYPRE_SStructVariable__enum vartype)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_SStructGrid_AddVariable(
      self,
      part,
      index_tmp,
      dim,
      var,
      vartype);
  return _return;
}

static int32_t
skel_bHYPRE_SStructGrid_SetNeighborBox(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t nbor_part,
  /* in rarray[dim] */ struct sidl_int__array* nbor_ilower,
  /* in rarray[dim] */ struct sidl_int__array* nbor_iupper,
/* in rarray[dim] */ struct sidl_int__array* index_map)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  int32_t* ilower_tmp = ilower_proxy->d_firstElement;
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  int32_t* iupper_tmp = iupper_proxy->d_firstElement;
  struct sidl_int__array* nbor_ilower_proxy = 
    sidl_int__array_ensure(nbor_ilower, 1, sidl_column_major_order);
  int32_t* nbor_ilower_tmp = nbor_ilower_proxy->d_firstElement;
  struct sidl_int__array* nbor_iupper_proxy = 
    sidl_int__array_ensure(nbor_iupper, 1, sidl_column_major_order);
  int32_t* nbor_iupper_tmp = nbor_iupper_proxy->d_firstElement;
  struct sidl_int__array* index_map_proxy = sidl_int__array_ensure(index_map, 1,
    sidl_column_major_order);
  int32_t* index_map_tmp = index_map_proxy->d_firstElement;
  int32_t dim = sidlLength(nbor_ilower_proxy,0);
  _return =
    impl_bHYPRE_SStructGrid_SetNeighborBox(
      self,
      part,
      ilower_tmp,
      iupper_tmp,
      nbor_part,
      nbor_ilower_tmp,
      nbor_iupper_tmp,
      index_map_tmp,
      dim);
  return _return;
}

static int32_t
skel_bHYPRE_SStructGrid_SetPeriodic(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
/* in rarray[dim] */ struct sidl_int__array* periodic)
{
  int32_t _return;
  struct sidl_int__array* periodic_proxy = sidl_int__array_ensure(periodic, 1,
    sidl_column_major_order);
  int32_t* periodic_tmp = periodic_proxy->d_firstElement;
  int32_t dim = sidlLength(periodic_proxy,0);
  _return =
    impl_bHYPRE_SStructGrid_SetPeriodic(
      self,
      part,
      periodic_tmp,
      dim);
  return _return;
}

static int32_t
skel_bHYPRE_SStructGrid_SetNumGhost(
  /* in */ bHYPRE_SStructGrid self,
/* in rarray[dim2] */ struct sidl_int__array* num_ghost)
{
  int32_t _return;
  struct sidl_int__array* num_ghost_proxy = sidl_int__array_ensure(num_ghost, 1,
    sidl_column_major_order);
  int32_t* num_ghost_tmp = num_ghost_proxy->d_firstElement;
  int32_t dim2 = sidlLength(num_ghost_proxy,0);
  _return =
    impl_bHYPRE_SStructGrid_SetNumGhost(
      self,
      num_ghost_tmp,
      dim2);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructGrid__set_epv(struct bHYPRE_SStructGrid__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructGrid__ctor;
  epv->f__dtor = impl_bHYPRE_SStructGrid__dtor;
  epv->f_SetNumDimParts = impl_bHYPRE_SStructGrid_SetNumDimParts;
  epv->f_SetCommunicator = impl_bHYPRE_SStructGrid_SetCommunicator;
  epv->f_SetExtents = skel_bHYPRE_SStructGrid_SetExtents;
  epv->f_SetVariable = impl_bHYPRE_SStructGrid_SetVariable;
  epv->f_AddVariable = skel_bHYPRE_SStructGrid_AddVariable;
  epv->f_SetNeighborBox = skel_bHYPRE_SStructGrid_SetNeighborBox;
  epv->f_AddUnstructuredPart = impl_bHYPRE_SStructGrid_AddUnstructuredPart;
  epv->f_SetPeriodic = skel_bHYPRE_SStructGrid_SetPeriodic;
  epv->f_SetNumGhost = skel_bHYPRE_SStructGrid_SetNumGhost;
  epv->f_Assemble = impl_bHYPRE_SStructGrid_Assemble;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructGrid__set_sepv(struct bHYPRE_SStructGrid__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_SStructGrid_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_SStructGrid__call_load(void) { 
  impl_bHYPRE_SStructGrid__load();
}
struct bHYPRE_SStructGrid__object* 
  skel_bHYPRE_SStructGrid_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGrid_fconnect_bHYPRE_SStructGrid(url, _ex);
}

char* skel_bHYPRE_SStructGrid_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj) { 
  return impl_bHYPRE_SStructGrid_fgetURL_bHYPRE_SStructGrid(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructGrid_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGrid_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_SStructGrid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_SStructGrid_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructGrid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGrid_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_SStructGrid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_SStructGrid_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructGrid_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGrid_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_SStructGrid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_SStructGrid_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_SStructGrid__data*
bHYPRE_SStructGrid__get_data(bHYPRE_SStructGrid self)
{
  return (struct bHYPRE_SStructGrid__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructGrid__set_data(
  bHYPRE_SStructGrid self,
  struct bHYPRE_SStructGrid__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
