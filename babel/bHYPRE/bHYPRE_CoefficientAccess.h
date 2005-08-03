/*
 * File:          bHYPRE_CoefficientAccess.h
 * Symbol:        bHYPRE.CoefficientAccess-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.8
 * Description:   Client-side glue code for bHYPRE.CoefficientAccess
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
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
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif

#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.
 */
bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_CoefficientAccess_addRef(
  /* in */ bHYPRE_CoefficientAccess self);

void
bHYPRE_CoefficientAccess_deleteRef(
  /* in */ bHYPRE_CoefficientAccess self);

sidl_bool
bHYPRE_CoefficientAccess_isSame(
  /* in */ bHYPRE_CoefficientAccess self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_CoefficientAccess_queryInt(
  /* in */ bHYPRE_CoefficientAccess self,
  /* in */ const char* name);

sidl_bool
bHYPRE_CoefficientAccess_isType(
  /* in */ bHYPRE_CoefficientAccess self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_CoefficientAccess_getClassInfo(
  /* in */ bHYPRE_CoefficientAccess self);

/**
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 * 
 */
int32_t
bHYPRE_CoefficientAccess_GetRow(
  /* in */ bHYPRE_CoefficientAccess self,
  /* in */ int32_t row,
  /* out */ int32_t* size,
  /* out array<int,column-major> */ struct sidl_int__array** col_ind,
  /* out array<double,column-major> */ struct sidl_double__array** values);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_CoefficientAccess__object*
bHYPRE_CoefficientAccess__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_CoefficientAccess__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_CoefficientAccess__exec(
  /* in */ bHYPRE_CoefficientAccess self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_CoefficientAccess__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_CoefficientAccess__getURL(
  /* in */ bHYPRE_CoefficientAccess self);
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

#ifdef __cplusplus
}
#endif
#endif
