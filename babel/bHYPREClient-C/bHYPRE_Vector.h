/*
 * File:          bHYPRE_Vector.h
 * Symbol:        bHYPRE.Vector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.10
 * Description:   Client-side glue code for bHYPRE.Vector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
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
bHYPRE_Vector
bHYPRE_Vector__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_Vector_addRef(
  /* in */ bHYPRE_Vector self);

void
bHYPRE_Vector_deleteRef(
  /* in */ bHYPRE_Vector self);

sidl_bool
bHYPRE_Vector_isSame(
  /* in */ bHYPRE_Vector self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_Vector_queryInt(
  /* in */ bHYPRE_Vector self,
  /* in */ const char* name);

sidl_bool
bHYPRE_Vector_isType(
  /* in */ bHYPRE_Vector self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_Vector_getClassInfo(
  /* in */ bHYPRE_Vector self);

/**
 * Set {\tt self} to 0.
 * 
 */
int32_t
bHYPRE_Vector_Clear(
  /* in */ bHYPRE_Vector self);

/**
 * Copy x into {\tt self}.
 * 
 */
int32_t
bHYPRE_Vector_Copy(
  /* in */ bHYPRE_Vector self,
  /* in */ bHYPRE_Vector x);

/**
 * Create an {\tt x} compatible with {\tt self}.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 * 
 */
int32_t
bHYPRE_Vector_Clone(
  /* in */ bHYPRE_Vector self,
  /* out */ bHYPRE_Vector* x);

/**
 * Scale {\tt self} by {\tt a}.
 * 
 */
int32_t
bHYPRE_Vector_Scale(
  /* in */ bHYPRE_Vector self,
  /* in */ double a);

/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */
int32_t
bHYPRE_Vector_Dot(
  /* in */ bHYPRE_Vector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d);

/**
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */
int32_t
bHYPRE_Vector_Axpy(
  /* in */ bHYPRE_Vector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_Vector__object*
bHYPRE_Vector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_Vector__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_Vector__exec(
  /* in */ bHYPRE_Vector self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_Vector__getURL(
  /* in */ bHYPRE_Vector self);
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

#ifdef __cplusplus
}
#endif
#endif
