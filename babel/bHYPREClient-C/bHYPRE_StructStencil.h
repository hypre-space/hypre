/*
 * File:          bHYPRE_StructStencil.h
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructStencil_h
#define included_bHYPRE_StructStencil_h

/**
 * Symbol "bHYPRE.StructStencil" (version 1.0.0)
 * 
 * Define a structured stencil for a structured problem
 * description.  More than one implementation is not envisioned,
 * thus the decision has been made to make this a class rather than
 * an interface.
 */
struct bHYPRE_StructStencil__object;
struct bHYPRE_StructStencil__array;
typedef struct bHYPRE_StructStencil__object* bHYPRE_StructStencil;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
#ifndef included_bHYPRE_StructStencil_IOR_h
#include "bHYPRE_StructStencil_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__create(sidl_BaseInterface* _ex);

/**
 * RMI constructor function for the class.
 */
bHYPRE_StructStencil
bHYPRE_StructStencil__createRemote(const char * url, sidl_BaseInterface *_ex);

/**
 * Wraps up the private data struct pointer (struct bHYPRE_StructStencil__data) passed in rather than running the constructor.
 */
bHYPRE_StructStencil
bHYPRE_StructStencil__wrapObj(void * data, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_StructStencil
bHYPRE_StructStencil__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Method:  Create[]
 */
bHYPRE_StructStencil
bHYPRE_StructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  SetDimension[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructStencil_SetDimension(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetDimension)(
    self,
    dim,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  SetSize[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructStencil_SetSize(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetSize)(
    self,
    size,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  SetElement[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructStencil_SetElement(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t index,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t offset_lower[1], offset_upper[1], offset_stride[1]; 
  struct sidl_int__array offset_real;
  struct sidl_int__array*offset_tmp = &offset_real;
  offset_upper[0] = dim-1;
  sidl_int__array_init(offset, offset_tmp, 1, offset_lower, offset_upper,
    offset_stride);
  return (*self->d_epv->f_SetElement)(
    self,
    index,
    offset_tmp,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_StructStencil_addRef(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_addRef)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_StructStencil_deleteRef(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_deleteRef)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_StructStencil_isSame(
  /* in */ bHYPRE_StructStencil self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_StructStencil_isType(
  /* in */ bHYPRE_StructStencil self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isType)(
    self,
    name,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_ClassInfo
bHYPRE_StructStencil_getClassInfo(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_getClassInfo)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructStencil__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_StructStencil__exec(
  /* in */ bHYPRE_StructStencil self,
  /* in */ const char* methodName,
  /* in */ sidl_rmi_Call inArgs,
  /* in */ sidl_rmi_Return outArgs,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__exec)(
    self,
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
bHYPRE_StructStencil__getURL(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__getURL)(
    self,
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
bHYPRE_StructStencil__raddRef(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__raddRef)(
    self,
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
bHYPRE_StructStencil__isRemote(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__isRemote)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
sidl_bool
bHYPRE_StructStencil__isLocal(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create1d(int32_t len);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create1dInit(
  int32_t len, 
  bHYPRE_StructStencil* data);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_borrow(
  bHYPRE_StructStencil* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_smartCopy(
  struct bHYPRE_StructStencil__array *array);

void
bHYPRE_StructStencil__array_addRef(
  struct bHYPRE_StructStencil__array* array);

void
bHYPRE_StructStencil__array_deleteRef(
  struct bHYPRE_StructStencil__array* array);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get1(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get2(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get3(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get4(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get5(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get6(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get7(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t indices[]);

void
bHYPRE_StructStencil__array_set1(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  bHYPRE_StructStencil const value);

void
bHYPRE_StructStencil__array_set2(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructStencil const value);

void
bHYPRE_StructStencil__array_set3(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructStencil const value);

void
bHYPRE_StructStencil__array_set4(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructStencil const value);

void
bHYPRE_StructStencil__array_set5(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructStencil const value);

void
bHYPRE_StructStencil__array_set6(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructStencil const value);

void
bHYPRE_StructStencil__array_set7(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructStencil const value);

void
bHYPRE_StructStencil__array_set(
  struct bHYPRE_StructStencil__array* array,
  const int32_t indices[],
  bHYPRE_StructStencil const value);

int32_t
bHYPRE_StructStencil__array_dimen(
  const struct bHYPRE_StructStencil__array* array);

int32_t
bHYPRE_StructStencil__array_lower(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructStencil__array_upper(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructStencil__array_length(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructStencil__array_stride(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind);

int
bHYPRE_StructStencil__array_isColumnOrder(
  const struct bHYPRE_StructStencil__array* array);

int
bHYPRE_StructStencil__array_isRowOrder(
  const struct bHYPRE_StructStencil__array* array);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_slice(
  struct bHYPRE_StructStencil__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructStencil__array_copy(
  const struct bHYPRE_StructStencil__array* src,
  struct bHYPRE_StructStencil__array* dest);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_ensure(
  struct bHYPRE_StructStencil__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_StructStencil__connectI

#pragma weak bHYPRE_StructStencil__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
