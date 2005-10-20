/*
 * File:          bHYPRE_SStructGrid_Impl.h
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_SStructGrid_Impl_h
#define included_bHYPRE_SStructGrid_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._includes) */
/* Put additional include files here... */
#include "HYPRE_sstruct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._includes) */

/*
 * Private data for class bHYPRE.SStructGrid
 */

struct bHYPRE_SStructGrid__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._data) */
  /* Put private data members here... */
   HYPRE_SStructGrid grid;
   MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructGrid__data*
bHYPRE_SStructGrid__get_data(
  bHYPRE_SStructGrid);

extern void
bHYPRE_SStructGrid__set_data(
  bHYPRE_SStructGrid,
  struct bHYPRE_SStructGrid__data*);

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

/*
 * User-defined object methods
 */

extern
bHYPRE_SStructGrid
impl_bHYPRE_SStructGrid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t ndim,
  /* in */ int32_t nparts);

extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGrid_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGrid_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGrid_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGrid_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
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
  /* in */ bHYPRE_MPICommunicator mpi_comm);

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
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGrid_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGrid_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
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
#ifdef __cplusplus
}
#endif
#endif
