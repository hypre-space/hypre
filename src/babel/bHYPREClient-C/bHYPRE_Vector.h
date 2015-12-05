/*
 * File:          bHYPRE_Vector.h
 * Symbol:        bHYPRE.Vector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.Vector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_Vector_h
#define included_bHYPRE_Vector_h

/**
 * Symbol "bHYPRE.Vector" (version 1.0.0)
 */
struct bHYPRE_Vector__object;
struct bHYPRE_Vector__array;
typedef struct bHYPRE_Vector__object* bHYPRE_Vector;

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
#ifndef included_bHYPRE_Vector_IOR_h
#include "bHYPRE_Vector_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_Vector
bHYPRE_Vector__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Set {\tt self} to 0.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Vector_Clear(
  /* in */ bHYPRE_Vector self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Clear)(
    self->d_object,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Copy data from x into {\tt self}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Vector_Copy(
  /* in */ bHYPRE_Vector self,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Copy)(
    self->d_object,
    x,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Create an {\tt x} compatible with {\tt self}.
 * The new vector's data is not specified.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Vector_Clone(
  /* in */ bHYPRE_Vector self,
  /* out */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Clone)(
    self->d_object,
    x,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Scale {\tt self} by {\tt a}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Vector_Scale(
  /* in */ bHYPRE_Vector self,
  /* in */ double a,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Scale)(
    self->d_object,
    a,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Vector_Dot(
  /* in */ bHYPRE_Vector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Dot)(
    self->d_object,
    x,
    d,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Add {\tt a}{\tt x} to {\tt self}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_Vector_Axpy(
  /* in */ bHYPRE_Vector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Axpy)(
    self->d_object,
    a,
    x,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_Vector_addRef(
  /* in */ bHYPRE_Vector self,
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
bHYPRE_Vector_deleteRef(
  /* in */ bHYPRE_Vector self,
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
bHYPRE_Vector_isSame(
  /* in */ bHYPRE_Vector self,
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
bHYPRE_Vector_isType(
  /* in */ bHYPRE_Vector self,
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
bHYPRE_Vector_getClassInfo(
  /* in */ bHYPRE_Vector self,
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
struct bHYPRE_Vector__object*
bHYPRE_Vector__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_Vector__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_Vector__exec(
  /* in */ bHYPRE_Vector self,
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
bHYPRE_Vector__getURL(
  /* in */ bHYPRE_Vector self,
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
bHYPRE_Vector__raddRef(
  /* in */ bHYPRE_Vector self,
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
bHYPRE_Vector__isRemote(
  /* in */ bHYPRE_Vector self,
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
bHYPRE_Vector__isLocal(
  /* in */ bHYPRE_Vector self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_Vector__array*
bHYPRE_Vector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_create1d(int32_t len);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_create1dInit(
  int32_t len, 
  bHYPRE_Vector* data);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_borrow(
  bHYPRE_Vector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_smartCopy(
  struct bHYPRE_Vector__array *array);

void
bHYPRE_Vector__array_addRef(
  struct bHYPRE_Vector__array* array);

void
bHYPRE_Vector__array_deleteRef(
  struct bHYPRE_Vector__array* array);

bHYPRE_Vector
bHYPRE_Vector__array_get1(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1);

bHYPRE_Vector
bHYPRE_Vector__array_get2(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_Vector
bHYPRE_Vector__array_get3(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_Vector
bHYPRE_Vector__array_get4(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_Vector
bHYPRE_Vector__array_get5(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_Vector
bHYPRE_Vector__array_get6(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_Vector
bHYPRE_Vector__array_get7(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_Vector
bHYPRE_Vector__array_get(
  const struct bHYPRE_Vector__array* array,
  const int32_t indices[]);

void
bHYPRE_Vector__array_set1(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  bHYPRE_Vector const value);

void
bHYPRE_Vector__array_set2(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_Vector const value);

void
bHYPRE_Vector__array_set3(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_Vector const value);

void
bHYPRE_Vector__array_set4(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_Vector const value);

void
bHYPRE_Vector__array_set5(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_Vector const value);

void
bHYPRE_Vector__array_set6(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_Vector const value);

void
bHYPRE_Vector__array_set7(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_Vector const value);

void
bHYPRE_Vector__array_set(
  struct bHYPRE_Vector__array* array,
  const int32_t indices[],
  bHYPRE_Vector const value);

int32_t
bHYPRE_Vector__array_dimen(
  const struct bHYPRE_Vector__array* array);

int32_t
bHYPRE_Vector__array_lower(
  const struct bHYPRE_Vector__array* array,
  const int32_t ind);

int32_t
bHYPRE_Vector__array_upper(
  const struct bHYPRE_Vector__array* array,
  const int32_t ind);

int32_t
bHYPRE_Vector__array_length(
  const struct bHYPRE_Vector__array* array,
  const int32_t ind);

int32_t
bHYPRE_Vector__array_stride(
  const struct bHYPRE_Vector__array* array,
  const int32_t ind);

int
bHYPRE_Vector__array_isColumnOrder(
  const struct bHYPRE_Vector__array* array);

int
bHYPRE_Vector__array_isRowOrder(
  const struct bHYPRE_Vector__array* array);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_slice(
  struct bHYPRE_Vector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_Vector__array_copy(
  const struct bHYPRE_Vector__array* src,
  struct bHYPRE_Vector__array* dest);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_ensure(
  struct bHYPRE_Vector__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_Vector__connectI

#pragma weak bHYPRE_Vector__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Vector__object*
bHYPRE_Vector__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_Vector__object*
bHYPRE_Vector__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
