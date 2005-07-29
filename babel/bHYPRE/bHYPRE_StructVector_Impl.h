/*
 * File:          bHYPRE_StructVector_Impl.h
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_StructVector_Impl_h
#define included_bHYPRE_StructVector_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif
#ifndef included_bHYPRE_StructBuildVector_h
#include "bHYPRE_StructBuildVector.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_h
#include "bHYPRE_ProblemDefinition.h"
#endif
#ifndef included_bHYPRE_StructVector_h
#include "bHYPRE_StructVector.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._includes) */
/* Put additional include files here... */
#include "HYPRE_struct_mv.h"
#include "HYPRE.h"
#include "utilities.h"
#include "bHYPRE_StructBuildVector.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._includes) */

/*
 * Private data for class bHYPRE.StructVector
 */

struct bHYPRE_StructVector__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._data) */
  /* Put private data members here... */
   HYPRE_StructVector vec;
   MPI_Comm comm;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_StructVector__data*
bHYPRE_StructVector__get_data(
  bHYPRE_StructVector);

extern void
bHYPRE_StructVector__set_data(
  bHYPRE_StructVector,
  struct bHYPRE_StructVector__data*);

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

/*
 * User-defined object methods
 */

extern
bHYPRE_StructVector
impl_bHYPRE_StructVector_Create(
  /* in */ void* mpi_comm,
  /* in */ bHYPRE_StructGrid grid);

extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct bHYPRE_StructBuildVector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructBuildVector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructBuildVector(struct 
  bHYPRE_StructBuildVector__object* obj);
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
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_StructVector_SetCommunicator(
  /* in */ bHYPRE_StructVector self,
  /* in */ void* mpi_comm);

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
impl_bHYPRE_StructVector_GetObject(
  /* in */ bHYPRE_StructVector self,
  /* out */ sidl_BaseInterface* A);

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

extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj);
extern struct bHYPRE_StructBuildVector__object* 
  impl_bHYPRE_StructVector_fconnect_bHYPRE_StructBuildVector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_bHYPRE_StructBuildVector(struct 
  bHYPRE_StructBuildVector__object* obj);
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
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
