/*
 * File:          bHYPRE_SStructGraph_Skel.c
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side glue code for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#include "bHYPRE_SStructGraph_IOR.h"
#include "bHYPRE_SStructGraph.h"
#include <stddef.h>

extern
void
impl_bHYPRE_SStructGraph__load(
  void);

extern
void
impl_bHYPRE_SStructGraph__ctor(
  /* in */ bHYPRE_SStructGraph self);

extern
void
impl_bHYPRE_SStructGraph__dtor(
  /* in */ bHYPRE_SStructGraph self);

extern
bHYPRE_SStructGraph
impl_bHYPRE_SStructGraph_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid);

extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj);
extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_SStructGraph_SetCommGrid(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid);

extern
int32_t
impl_bHYPRE_SStructGraph_SetStencil(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ bHYPRE_SStructStencil stencil);

extern
int32_t
impl_bHYPRE_SStructGraph_AddEntries(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t to_part,
  /* in rarray[dim] */ int32_t* to_index,
  /* in */ int32_t to_var);

extern
int32_t
impl_bHYPRE_SStructGraph_SetObjectType(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t type);

extern
int32_t
impl_bHYPRE_SStructGraph_SetCommunicator(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_SStructGraph_Initialize(
  /* in */ bHYPRE_SStructGraph self);

extern
int32_t
impl_bHYPRE_SStructGraph_Assemble(
  /* in */ bHYPRE_SStructGraph self);

extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj);
extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_SStructGraph_AddEntries(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ int32_t to_part,
  /* in rarray[dim] */ struct sidl_int__array* to_index,
/* in */ int32_t to_var)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1,
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  struct sidl_int__array* to_index_proxy = sidl_int__array_ensure(to_index, 1,
    sidl_column_major_order);
  int32_t* to_index_tmp = to_index_proxy->d_firstElement;
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_SStructGraph_AddEntries(
      self,
      part,
      index_tmp,
      dim,
      var,
      to_part,
      to_index_tmp,
      to_var);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructGraph__set_epv(struct bHYPRE_SStructGraph__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructGraph__ctor;
  epv->f__dtor = impl_bHYPRE_SStructGraph__dtor;
  epv->f_SetCommGrid = impl_bHYPRE_SStructGraph_SetCommGrid;
  epv->f_SetStencil = impl_bHYPRE_SStructGraph_SetStencil;
  epv->f_AddEntries = skel_bHYPRE_SStructGraph_AddEntries;
  epv->f_SetObjectType = impl_bHYPRE_SStructGraph_SetObjectType;
  epv->f_SetCommunicator = impl_bHYPRE_SStructGraph_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_SStructGraph_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructGraph_Assemble;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructGraph__set_sepv(struct bHYPRE_SStructGraph__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_SStructGraph_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_SStructGraph__call_load(void) { 
  impl_bHYPRE_SStructGraph__load();
}
struct bHYPRE_SStructGrid__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(url, _ex);
}

char* skel_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj) { 
  return impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGrid(obj);
}

struct bHYPRE_SStructStencil__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(url, _ex);
}

char* skel_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj) { 
  return impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructStencil(obj);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(url, _ex);
}

char* skel_bHYPRE_SStructGraph_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_SStructGraph_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_SStructGraph_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(url, _ex);
}

char* skel_bHYPRE_SStructGraph_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj) { 
  return impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_ProblemDefinition(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_SStructGraph_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseInterface(obj);
}

struct bHYPRE_SStructGraph__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(url, _ex);
}

char* skel_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj) { 
  return impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGraph(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_SStructGraph_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_SStructGraph__data*
bHYPRE_SStructGraph__get_data(bHYPRE_SStructGraph self)
{
  return (struct bHYPRE_SStructGraph__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructGraph__set_data(
  bHYPRE_SStructGraph self,
  struct bHYPRE_SStructGraph__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
