/*
 * File:          bHYPRE_StructMatrixView.h
 * Symbol:        bHYPRE.StructMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.StructMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructMatrixView_h
#define included_bHYPRE_StructMatrixView_h

/**
 * Symbol "bHYPRE.StructMatrixView" (version 1.0.0)
 */
struct bHYPRE_StructMatrixView__object;
struct bHYPRE_StructMatrixView__array;
typedef struct bHYPRE_StructMatrixView__object* bHYPRE_StructMatrixView;

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
#ifndef included_bHYPRE_StructStencil_h
#include "bHYPRE_StructStencil.h"
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
#ifndef included_bHYPRE_StructMatrixView_IOR_h
#include "bHYPRE_StructMatrixView_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Method:  SetGrid[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructMatrixView_SetGrid(
  /* in */ bHYPRE_StructMatrixView self,
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
 * Method:  SetStencil[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructMatrixView_SetStencil(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ bHYPRE_StructStencil stencil,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetStencil)(
    self->d_object,
    stencil,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  SetValues[]
 */
int32_t
bHYPRE_StructMatrixView_SetValues(
  /* in */ bHYPRE_StructMatrixView self,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  SetBoxValues[]
 */
int32_t
bHYPRE_StructMatrixView_SetBoxValues(
  /* in */ bHYPRE_StructMatrixView self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  SetNumGhost[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructMatrixView_SetNumGhost(
  /* in */ bHYPRE_StructMatrixView self,
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
 * Method:  SetSymmetric[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructMatrixView_SetSymmetric(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ int32_t symmetric,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetSymmetric)(
    self->d_object,
    symmetric,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  SetConstantEntries[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructMatrixView_SetConstantEntries(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ int32_t num_stencil_constant_points,
  /* in rarray[num_stencil_constant_points] */ int32_t* stencil_constant_points,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t stencil_constant_points_lower[1], stencil_constant_points_upper[1],
    stencil_constant_points_stride[1]; 
  struct sidl_int__array stencil_constant_points_real;
  struct sidl_int__array*stencil_constant_points_tmp = 
    &stencil_constant_points_real;
  stencil_constant_points_upper[0] = num_stencil_constant_points-1;
  sidl_int__array_init(stencil_constant_points, stencil_constant_points_tmp, 1,
    stencil_constant_points_lower, stencil_constant_points_upper,
    stencil_constant_points_stride);
  return (*self->d_epv->f_SetConstantEntries)(
    self->d_object,
    stencil_constant_points_tmp,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  SetConstantValues[]
 */
int32_t
bHYPRE_StructMatrixView_SetConstantValues(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructMatrixView_SetCommunicator(
  /* in */ bHYPRE_StructMatrixView self,
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
bHYPRE_StructMatrixView_Initialize(
  /* in */ bHYPRE_StructMatrixView self,
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
bHYPRE_StructMatrixView_Assemble(
  /* in */ bHYPRE_StructMatrixView self,
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
bHYPRE_StructMatrixView_addRef(
  /* in */ bHYPRE_StructMatrixView self,
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
bHYPRE_StructMatrixView_deleteRef(
  /* in */ bHYPRE_StructMatrixView self,
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
bHYPRE_StructMatrixView_isSame(
  /* in */ bHYPRE_StructMatrixView self,
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
bHYPRE_StructMatrixView_isType(
  /* in */ bHYPRE_StructMatrixView self,
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
bHYPRE_StructMatrixView_getClassInfo(
  /* in */ bHYPRE_StructMatrixView self,
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
struct bHYPRE_StructMatrixView__object*
bHYPRE_StructMatrixView__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructMatrixView__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_StructMatrixView__exec(
  /* in */ bHYPRE_StructMatrixView self,
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
bHYPRE_StructMatrixView__getURL(
  /* in */ bHYPRE_StructMatrixView self,
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
bHYPRE_StructMatrixView__raddRef(
  /* in */ bHYPRE_StructMatrixView self,
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
bHYPRE_StructMatrixView__isRemote(
  /* in */ bHYPRE_StructMatrixView self,
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
bHYPRE_StructMatrixView__isLocal(
  /* in */ bHYPRE_StructMatrixView self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create1d(int32_t len);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create1dInit(
  int32_t len, 
  bHYPRE_StructMatrixView* data);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_borrow(
  bHYPRE_StructMatrixView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_smartCopy(
  struct bHYPRE_StructMatrixView__array *array);

void
bHYPRE_StructMatrixView__array_addRef(
  struct bHYPRE_StructMatrixView__array* array);

void
bHYPRE_StructMatrixView__array_deleteRef(
  struct bHYPRE_StructMatrixView__array* array);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get1(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get2(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get3(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get4(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get5(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get6(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get7(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t indices[]);

void
bHYPRE_StructMatrixView__array_set1(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set2(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set3(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set4(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set5(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set6(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set7(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t indices[],
  bHYPRE_StructMatrixView const value);

int32_t
bHYPRE_StructMatrixView__array_dimen(
  const struct bHYPRE_StructMatrixView__array* array);

int32_t
bHYPRE_StructMatrixView__array_lower(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructMatrixView__array_upper(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructMatrixView__array_length(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructMatrixView__array_stride(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind);

int
bHYPRE_StructMatrixView__array_isColumnOrder(
  const struct bHYPRE_StructMatrixView__array* array);

int
bHYPRE_StructMatrixView__array_isRowOrder(
  const struct bHYPRE_StructMatrixView__array* array);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_slice(
  struct bHYPRE_StructMatrixView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructMatrixView__array_copy(
  const struct bHYPRE_StructMatrixView__array* src,
  struct bHYPRE_StructMatrixView__array* dest);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_ensure(
  struct bHYPRE_StructMatrixView__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_StructMatrixView__connectI

#pragma weak bHYPRE_StructMatrixView__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructMatrixView__object*
bHYPRE_StructMatrixView__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructMatrixView__object*
bHYPRE_StructMatrixView__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
