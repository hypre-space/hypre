/*
 * File:          bHYPRE_StructGrid_Skel.c
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_StructGrid_IOR.h"
#include "bHYPRE_StructGrid.h"
#include <stddef.h>

extern
void
impl_bHYPRE_StructGrid__load(
  void);

extern
void
impl_bHYPRE_StructGrid__ctor(
  /* in */ bHYPRE_StructGrid self);

extern
void
impl_bHYPRE_StructGrid__dtor(
  /* in */ bHYPRE_StructGrid self);

extern
bHYPRE_StructGrid
impl_bHYPRE_StructGrid_Create(
  /* in */ void* mpi_comm,
  /* in */ int32_t dim);

extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_StructGrid_SetCommunicator(
  /* in */ bHYPRE_StructGrid self,
  /* in */ void* mpi_comm);

extern
int32_t
impl_bHYPRE_StructGrid_SetDimension(
  /* in */ bHYPRE_StructGrid self,
  /* in */ int32_t dim);

extern
int32_t
impl_bHYPRE_StructGrid_SetExtents(
  /* in */ bHYPRE_StructGrid self,
  /* in */ int32_t* ilower,
  /* in */ int32_t* iupper,
  /* in */ int32_t dim);

extern
int32_t
impl_bHYPRE_StructGrid_SetPeriodic(
  /* in */ bHYPRE_StructGrid self,
  /* in */ int32_t* periodic,
  /* in */ int32_t dim);

extern
int32_t
impl_bHYPRE_StructGrid_SetNumGhost(
  /* in */ bHYPRE_StructGrid self,
  /* in */ int32_t* num_ghost,
  /* in */ int32_t dim2);

extern
int32_t
impl_bHYPRE_StructGrid_Assemble(
  /* in */ bHYPRE_StructGrid self);

extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructGrid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_StructGrid_SetExtents(
  /* in */ bHYPRE_StructGrid self,
  /* in */ struct sidl_int__array* ilower,
/* in */ struct sidl_int__array* iupper)
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
    impl_bHYPRE_StructGrid_SetExtents(
      self,
      ilower_tmp,
      iupper_tmp,
      dim);
  return _return;
}

static int32_t
skel_bHYPRE_StructGrid_SetPeriodic(
  /* in */ bHYPRE_StructGrid self,
/* in */ struct sidl_int__array* periodic)
{
  int32_t _return;
  struct sidl_int__array* periodic_proxy = sidl_int__array_ensure(periodic, 1,
    sidl_column_major_order);
  int32_t* periodic_tmp = periodic_proxy->d_firstElement;
  int32_t dim = sidlLength(periodic_proxy,0);
  _return =
    impl_bHYPRE_StructGrid_SetPeriodic(
      self,
      periodic_tmp,
      dim);
  return _return;
}

static int32_t
skel_bHYPRE_StructGrid_SetNumGhost(
  /* in */ bHYPRE_StructGrid self,
/* in */ struct sidl_int__array* num_ghost)
{
  int32_t _return;
  struct sidl_int__array* num_ghost_proxy = sidl_int__array_ensure(num_ghost, 1,
    sidl_column_major_order);
  int32_t* num_ghost_tmp = num_ghost_proxy->d_firstElement;
  int32_t dim2 = sidlLength(num_ghost_proxy,0);
  _return =
    impl_bHYPRE_StructGrid_SetNumGhost(
      self,
      num_ghost_tmp,
      dim2);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructGrid__set_epv(struct bHYPRE_StructGrid__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructGrid__ctor;
  epv->f__dtor = impl_bHYPRE_StructGrid__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructGrid_SetCommunicator;
  epv->f_SetDimension = impl_bHYPRE_StructGrid_SetDimension;
  epv->f_SetExtents = skel_bHYPRE_StructGrid_SetExtents;
  epv->f_SetPeriodic = skel_bHYPRE_StructGrid_SetPeriodic;
  epv->f_SetNumGhost = skel_bHYPRE_StructGrid_SetNumGhost;
  epv->f_Assemble = impl_bHYPRE_StructGrid_Assemble;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructGrid__set_sepv(struct bHYPRE_StructGrid__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_StructGrid_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_StructGrid__call_load(void) { 
  impl_bHYPRE_StructGrid__load();
}
struct bHYPRE_StructGrid__object* 
  skel_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(url, _ex);
}

char* skel_bHYPRE_StructGrid_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj) { 
  return impl_bHYPRE_StructGrid_fgetURL_bHYPRE_StructGrid(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_StructGrid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_StructGrid_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_StructGrid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_StructGrid_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructGrid_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_StructGrid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_StructGrid_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_StructGrid__data*
bHYPRE_StructGrid__get_data(bHYPRE_StructGrid self)
{
  return (struct bHYPRE_StructGrid__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructGrid__set_data(
  bHYPRE_StructGrid self,
  struct bHYPRE_StructGrid__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
