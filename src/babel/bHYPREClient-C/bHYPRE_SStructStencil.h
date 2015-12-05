/*
 * File:          bHYPRE_SStructStencil.h
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructStencil_h
#define included_bHYPRE_SStructStencil_h

/**
 * Symbol "bHYPRE.SStructStencil" (version 1.0.0)
 * 
 * The semi-structured grid stencil class.
 */
struct bHYPRE_SStructStencil__object;
struct bHYPRE_SStructStencil__array;
typedef struct bHYPRE_SStructStencil__object* bHYPRE_SStructStencil;

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
#ifndef included_bHYPRE_SStructStencil_IOR_h
#include "bHYPRE_SStructStencil_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__create(sidl_BaseInterface* _ex);

/**
 * RMI constructor function for the class.
 */
bHYPRE_SStructStencil
bHYPRE_SStructStencil__createRemote(const char * url, sidl_BaseInterface *_ex);

/**
 * Wraps up the private data struct pointer (struct bHYPRE\_SStructStencil\_\_data) passed in rather than running the constructor.
 */
bHYPRE_SStructStencil
bHYPRE_SStructStencil__wrapObj(void * data, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_SStructStencil
bHYPRE_SStructStencil__connect(const char *, sidl_BaseInterface *_ex);

/**
 *  This function is the preferred way to create a SStruct Stencil. 
 */
bHYPRE_SStructStencil
bHYPRE_SStructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex);

/**
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */
SIDL_C_INLINE_DECL
void
bHYPRE_SStructStencil_Destroy(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_Destroy)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the number of spatial dimensions and stencil entries.
 * DEPRECATED, use Create:
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructStencil_SetNumDimSize(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetNumDimSize)(
    self,
    ndim,
    size,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set a stencil entry.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructStencil_SetEntry(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t entry,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t offset_lower[1], offset_upper[1], offset_stride[1]; 
  struct sidl_int__array offset_real;
  struct sidl_int__array*offset_tmp = &offset_real;
  offset_upper[0] = dim-1;
  sidl_int__array_init(offset, offset_tmp, 1, offset_lower, offset_upper,
    offset_stride);
  return (*self->d_epv->f_SetEntry)(
    self,
    entry,
    offset_tmp,
    var,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_SStructStencil_addRef(
  /* in */ bHYPRE_SStructStencil self,
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
bHYPRE_SStructStencil_deleteRef(
  /* in */ bHYPRE_SStructStencil self,
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
bHYPRE_SStructStencil_isSame(
  /* in */ bHYPRE_SStructStencil self,
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
bHYPRE_SStructStencil_isType(
  /* in */ bHYPRE_SStructStencil self,
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
bHYPRE_SStructStencil_getClassInfo(
  /* in */ bHYPRE_SStructStencil self,
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
struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructStencil__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_SStructStencil__exec(
  /* in */ bHYPRE_SStructStencil self,
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
bHYPRE_SStructStencil__getURL(
  /* in */ bHYPRE_SStructStencil self,
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
bHYPRE_SStructStencil__raddRef(
  /* in */ bHYPRE_SStructStencil self,
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
bHYPRE_SStructStencil__isRemote(
  /* in */ bHYPRE_SStructStencil self,
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
bHYPRE_SStructStencil__isLocal(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_create1d(int32_t len);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructStencil* data);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_borrow(
  bHYPRE_SStructStencil* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_smartCopy(
  struct bHYPRE_SStructStencil__array *array);

void
bHYPRE_SStructStencil__array_addRef(
  struct bHYPRE_SStructStencil__array* array);

void
bHYPRE_SStructStencil__array_deleteRef(
  struct bHYPRE_SStructStencil__array* array);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get1(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get2(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get3(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get4(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get5(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get6(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get7(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructStencil__array_set1(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  bHYPRE_SStructStencil const value);

void
bHYPRE_SStructStencil__array_set2(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructStencil const value);

void
bHYPRE_SStructStencil__array_set3(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructStencil const value);

void
bHYPRE_SStructStencil__array_set4(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructStencil const value);

void
bHYPRE_SStructStencil__array_set5(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructStencil const value);

void
bHYPRE_SStructStencil__array_set6(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructStencil const value);

void
bHYPRE_SStructStencil__array_set7(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructStencil const value);

void
bHYPRE_SStructStencil__array_set(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t indices[],
  bHYPRE_SStructStencil const value);

int32_t
bHYPRE_SStructStencil__array_dimen(
  const struct bHYPRE_SStructStencil__array* array);

int32_t
bHYPRE_SStructStencil__array_lower(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructStencil__array_upper(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructStencil__array_length(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructStencil__array_stride(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t ind);

int
bHYPRE_SStructStencil__array_isColumnOrder(
  const struct bHYPRE_SStructStencil__array* array);

int
bHYPRE_SStructStencil__array_isRowOrder(
  const struct bHYPRE_SStructStencil__array* array);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_slice(
  struct bHYPRE_SStructStencil__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructStencil__array_copy(
  const struct bHYPRE_SStructStencil__array* src,
  struct bHYPRE_SStructStencil__array* dest);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_ensure(
  struct bHYPRE_SStructStencil__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_SStructStencil__connectI

#pragma weak bHYPRE_SStructStencil__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
