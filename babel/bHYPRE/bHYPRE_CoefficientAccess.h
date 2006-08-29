/*
 * File:          bHYPRE_CoefficientAccess.h
 * Symbol:        bHYPRE.CoefficientAccess-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.CoefficientAccess
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_CoefficientAccess_h
#define included_bHYPRE_CoefficientAccess_h

/**
 * Symbol "bHYPRE.CoefficientAccess" (version 1.0.0)
 */
struct bHYPRE_CoefficientAccess__object;
struct bHYPRE_CoefficientAccess__array;
typedef struct bHYPRE_CoefficientAccess__object* bHYPRE_CoefficientAccess;

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
#ifndef included_bHYPRE_CoefficientAccess_IOR_h
#include "bHYPRE_CoefficientAccess_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__connect(const char *, sidl_BaseInterface *_ex);

/**
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_CoefficientAccess_GetRow(
  /* in */ bHYPRE_CoefficientAccess self,
  /* in */ int32_t row,
  /* out */ int32_t* size,
  /* out array<int,column-major> */ struct sidl_int__array** col_ind,
  /* out array<double,column-major> */ struct sidl_double__array** values,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_GetRow)(
    self->d_object,
    row,
    size,
    col_ind,
    values,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_CoefficientAccess_addRef(
  /* in */ bHYPRE_CoefficientAccess self,
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
bHYPRE_CoefficientAccess_deleteRef(
  /* in */ bHYPRE_CoefficientAccess self,
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
bHYPRE_CoefficientAccess_isSame(
  /* in */ bHYPRE_CoefficientAccess self,
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
bHYPRE_CoefficientAccess_isType(
  /* in */ bHYPRE_CoefficientAccess self,
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
bHYPRE_CoefficientAccess_getClassInfo(
  /* in */ bHYPRE_CoefficientAccess self,
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
struct bHYPRE_CoefficientAccess__object*
bHYPRE_CoefficientAccess__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_CoefficientAccess__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_CoefficientAccess__exec(
  /* in */ bHYPRE_CoefficientAccess self,
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
bHYPRE_CoefficientAccess__getURL(
  /* in */ bHYPRE_CoefficientAccess self,
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
bHYPRE_CoefficientAccess__raddRef(
  /* in */ bHYPRE_CoefficientAccess self,
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
bHYPRE_CoefficientAccess__isRemote(
  /* in */ bHYPRE_CoefficientAccess self,
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
bHYPRE_CoefficientAccess__isLocal(
  /* in */ bHYPRE_CoefficientAccess self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_create1d(int32_t len);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_create1dInit(
  int32_t len, 
  bHYPRE_CoefficientAccess* data);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_borrow(
  bHYPRE_CoefficientAccess* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_smartCopy(
  struct bHYPRE_CoefficientAccess__array *array);

void
bHYPRE_CoefficientAccess__array_addRef(
  struct bHYPRE_CoefficientAccess__array* array);

void
bHYPRE_CoefficientAccess__array_deleteRef(
  struct bHYPRE_CoefficientAccess__array* array);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get1(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get2(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get3(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get4(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get5(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get6(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get7(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t indices[]);

void
bHYPRE_CoefficientAccess__array_set1(
  struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  bHYPRE_CoefficientAccess const value);

void
bHYPRE_CoefficientAccess__array_set2(
  struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_CoefficientAccess const value);

void
bHYPRE_CoefficientAccess__array_set3(
  struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_CoefficientAccess const value);

void
bHYPRE_CoefficientAccess__array_set4(
  struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_CoefficientAccess const value);

void
bHYPRE_CoefficientAccess__array_set5(
  struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_CoefficientAccess const value);

void
bHYPRE_CoefficientAccess__array_set6(
  struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_CoefficientAccess const value);

void
bHYPRE_CoefficientAccess__array_set7(
  struct bHYPRE_CoefficientAccess__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_CoefficientAccess const value);

void
bHYPRE_CoefficientAccess__array_set(
  struct bHYPRE_CoefficientAccess__array* array,
  const int32_t indices[],
  bHYPRE_CoefficientAccess const value);

int32_t
bHYPRE_CoefficientAccess__array_dimen(
  const struct bHYPRE_CoefficientAccess__array* array);

int32_t
bHYPRE_CoefficientAccess__array_lower(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t ind);

int32_t
bHYPRE_CoefficientAccess__array_upper(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t ind);

int32_t
bHYPRE_CoefficientAccess__array_length(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t ind);

int32_t
bHYPRE_CoefficientAccess__array_stride(
  const struct bHYPRE_CoefficientAccess__array* array,
  const int32_t ind);

int
bHYPRE_CoefficientAccess__array_isColumnOrder(
  const struct bHYPRE_CoefficientAccess__array* array);

int
bHYPRE_CoefficientAccess__array_isRowOrder(
  const struct bHYPRE_CoefficientAccess__array* array);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_slice(
  struct bHYPRE_CoefficientAccess__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_CoefficientAccess__array_copy(
  const struct bHYPRE_CoefficientAccess__array* src,
  struct bHYPRE_CoefficientAccess__array* dest);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_ensure(
  struct bHYPRE_CoefficientAccess__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_CoefficientAccess__connectI

#pragma weak bHYPRE_CoefficientAccess__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_CoefficientAccess__object*
bHYPRE_CoefficientAccess__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_CoefficientAccess__object*
bHYPRE_CoefficientAccess__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
