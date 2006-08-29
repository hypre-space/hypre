/*
 * File:          bHYPRE_StructVectorView.h
 * Symbol:        bHYPRE.StructVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.StructVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructVectorView_h
#define included_bHYPRE_StructVectorView_h

/**
 * Symbol "bHYPRE.StructVectorView" (version 1.0.0)
 */
struct bHYPRE_StructVectorView__object;
struct bHYPRE_StructVectorView__array;
typedef struct bHYPRE_StructVectorView__object* bHYPRE_StructVectorView;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif
#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif

#ifndef included_sidl_rmi_Call_h
#include "sidl_rmi_Call.h"
#endif
#ifndef included_sidl_rmi_Return_h
#include "sidl_rmi_Return.h"
#endif
#ifdef SIDL_C_HAS_INLINE
#ifndef included_bHYPRE_StructVectorView_IOR_h
#include "bHYPRE_StructVectorView_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_StructVectorView
bHYPRE_StructVectorView__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Method:  SetGrid[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVectorView_SetGrid(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ bHYPRE_StructGrid grid,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetGrid)(
    self->d_object,
    grid,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  SetNumGhost[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVectorView_SetNumGhost(
  /* in */ bHYPRE_StructVectorView self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t num_ghost_lower[1], num_ghost_upper[1], num_ghost_stride[1]; 
  struct sidl_int__array num_ghost_real;
  struct sidl_int__array*num_ghost_tmp = &num_ghost_real;
  num_ghost_upper[0] = dim2-1;
  sidl_int__array_init(num_ghost, num_ghost_tmp, 1, num_ghost_lower,
    num_ghost_upper, num_ghost_stride);
  return (*self->d_epv->f_SetNumGhost)(
    self->d_object,
    num_ghost_tmp,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  SetValue[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVectorView_SetValue(
  /* in */ bHYPRE_StructVectorView self,
  /* in rarray[dim] */ int32_t* grid_index,
  /* in */ int32_t dim,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t grid_index_lower[1], grid_index_upper[1], grid_index_stride[1]; 
  struct sidl_int__array grid_index_real;
  struct sidl_int__array*grid_index_tmp = &grid_index_real;
  grid_index_upper[0] = dim-1;
  sidl_int__array_init(grid_index, grid_index_tmp, 1, grid_index_lower,
    grid_index_upper, grid_index_stride);
  return (*self->d_epv->f_SetValue)(
    self->d_object,
    grid_index_tmp,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  SetBoxValues[]
 */
int32_t
bHYPRE_StructVectorView_SetBoxValues(
  /* in */ bHYPRE_StructVectorView self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVectorView_SetCommunicator(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetCommunicator)(
    self->d_object,
    mpi_comm,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVectorView_Initialize(
  /* in */ bHYPRE_StructVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Initialize)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVectorView_Assemble(
  /* in */ bHYPRE_StructVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Assemble)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_StructVectorView_addRef(
  /* in */ bHYPRE_StructVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_addRef)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_StructVectorView_deleteRef(
  /* in */ bHYPRE_StructVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_deleteRef)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_StructVectorView_isSame(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isSame)(
    self->d_object,
    iobj,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_StructVectorView_isType(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isType)(
    self->d_object,
    name,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_ClassInfo
bHYPRE_StructVectorView_getClassInfo(
  /* in */ bHYPRE_StructVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_getClassInfo)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructVectorView__object*
bHYPRE_StructVectorView__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructVectorView__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_StructVectorView__exec(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ const char* methodName,
  /* in */ sidl_rmi_Call inArgs,
  /* in */ sidl_rmi_Return outArgs,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__exec)(
    self->d_object,
    methodName,
    inArgs,
    outArgs,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * Get the URL of the Implementation of this object (for RMI)
 */
SIDL_C_INLINE_DECL
char*
bHYPRE_StructVectorView__getURL(
  /* in */ bHYPRE_StructVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__getURL)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * On a remote object, addrefs the remote instance.
 */
SIDL_C_INLINE_DECL
void
bHYPRE_StructVectorView__raddRef(
  /* in */ bHYPRE_StructVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__raddRef)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_StructVectorView__isRemote(
  /* in */ bHYPRE_StructVectorView self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__isRemote)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
sidl_bool
bHYPRE_StructVectorView__isLocal(
  /* in */ bHYPRE_StructVectorView self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_create1d(int32_t len);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_create1dInit(
  int32_t len, 
  bHYPRE_StructVectorView* data);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_borrow(
  bHYPRE_StructVectorView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_smartCopy(
  struct bHYPRE_StructVectorView__array *array);

void
bHYPRE_StructVectorView__array_addRef(
  struct bHYPRE_StructVectorView__array* array);

void
bHYPRE_StructVectorView__array_deleteRef(
  struct bHYPRE_StructVectorView__array* array);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get1(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get2(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get3(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get4(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get5(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get6(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get7(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t indices[]);

void
bHYPRE_StructVectorView__array_set1(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set2(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set3(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set4(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set5(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set6(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set7(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t indices[],
  bHYPRE_StructVectorView const value);

int32_t
bHYPRE_StructVectorView__array_dimen(
  const struct bHYPRE_StructVectorView__array* array);

int32_t
bHYPRE_StructVectorView__array_lower(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVectorView__array_upper(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVectorView__array_length(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVectorView__array_stride(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t ind);

int
bHYPRE_StructVectorView__array_isColumnOrder(
  const struct bHYPRE_StructVectorView__array* array);

int
bHYPRE_StructVectorView__array_isRowOrder(
  const struct bHYPRE_StructVectorView__array* array);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_slice(
  struct bHYPRE_StructVectorView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructVectorView__array_copy(
  const struct bHYPRE_StructVectorView__array* src,
  struct bHYPRE_StructVectorView__array* dest);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_ensure(
  struct bHYPRE_StructVectorView__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_StructVectorView__connectI

#pragma weak bHYPRE_StructVectorView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructVectorView__object*
bHYPRE_StructVectorView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructVectorView__object*
bHYPRE_StructVectorView__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
