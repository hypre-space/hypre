/*
 * File:          bHYPRE_SStructGraph_Skel.c
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_SStructGraph_IOR.h"
#include "bHYPRE_SStructGraph.h"
#include <stddef.h>

extern
void
impl_bHYPRE_SStructGraph__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructGraph__ctor(
  /* in */ bHYPRE_SStructGraph self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructGraph__ctor2(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructGraph__dtor(
  /* in */ bHYPRE_SStructGraph self,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_SStructGraph
impl_bHYPRE_SStructGraph_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGraph_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructGraph_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructGraph_fcast_bHYPRE_SStructGraph(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGraph_fcast_bHYPRE_SStructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructGraph_fcast_bHYPRE_SStructStencil(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructGraph_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGraph_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructGraph_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructGraph_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_SStructGraph_SetCommGrid(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructGraph_SetStencil(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ bHYPRE_SStructStencil stencil,
  /* out */ sidl_BaseInterface *_ex);

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
  /* in */ int32_t to_var,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructGraph_SetObjectType(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t type,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructGraph_SetCommunicator(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructGraph_Initialize(
  /* in */ bHYPRE_SStructGraph self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructGraph_Assemble(
  /* in */ bHYPRE_SStructGraph self,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGraph_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructGraph_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructGraph_fcast_bHYPRE_SStructGraph(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGraph_fcast_bHYPRE_SStructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructGraph_fcast_bHYPRE_SStructStencil(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructGraph_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGraph_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructGraph_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructGraph_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
static int32_t
skel_bHYPRE_SStructGraph_AddEntries(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ int32_t to_part,
  /* in rarray[dim] */ struct sidl_int__array* to_index,
  /* in */ int32_t to_var,
/* out */ sidl_BaseInterface *_ex)
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
      to_var,
      _ex);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructGraph__set_epv(struct bHYPRE_SStructGraph__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructGraph__ctor;
  epv->f__ctor2 = impl_bHYPRE_SStructGraph__ctor2;
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
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_bHYPRE_SStructGraph__load(&_throwaway_exception);
}
struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(url, ar, _ex);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_SStructGraph_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fcast_bHYPRE_MPICommunicator(bi, _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(url, ar,
    _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructGraph_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fcast_bHYPRE_ProblemDefinition(bi, _ex);
}

struct bHYPRE_SStructGraph__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(url, ar, _ex);
}

struct bHYPRE_SStructGraph__object* 
  skel_bHYPRE_SStructGraph_fcast_bHYPRE_SStructGraph(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fcast_bHYPRE_SStructGraph(bi, _ex);
}

struct bHYPRE_SStructGrid__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(url, ar, _ex);
}

struct bHYPRE_SStructGrid__object* 
  skel_bHYPRE_SStructGraph_fcast_bHYPRE_SStructGrid(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fcast_bHYPRE_SStructGrid(bi, _ex);
}

struct bHYPRE_SStructStencil__object* 
  skel_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(url, ar, _ex);
}

struct bHYPRE_SStructStencil__object* 
  skel_bHYPRE_SStructGraph_fcast_bHYPRE_SStructStencil(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fcast_bHYPRE_SStructStencil(bi, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructGraph_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructGraph_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructGraph_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_SStructGraph_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_SStructGraph_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructGraph_fcast_sidl_RuntimeException(bi, _ex);
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
