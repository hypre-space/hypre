/*
 * File:          bHYPRE_StructStencil.h
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Client-side glue code for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
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
 * 
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
 * Constructor function for the class.
 */
struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_StructStencil
bHYPRE_StructStencil__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_StructStencil
bHYPRE_StructStencil__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_StructStencil_addRef(
  /* in */ bHYPRE_StructStencil self);

void
bHYPRE_StructStencil_deleteRef(
  /* in */ bHYPRE_StructStencil self);

sidl_bool
bHYPRE_StructStencil_isSame(
  /* in */ bHYPRE_StructStencil self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_StructStencil_queryInt(
  /* in */ bHYPRE_StructStencil self,
  /* in */ const char* name);

sidl_bool
bHYPRE_StructStencil_isType(
  /* in */ bHYPRE_StructStencil self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_StructStencil_getClassInfo(
  /* in */ bHYPRE_StructStencil self);

/**
 * Method:  Create[]
 */
bHYPRE_StructStencil
bHYPRE_StructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size);

/**
 * Method:  SetDimension[]
 */
int32_t
bHYPRE_StructStencil_SetDimension(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t dim);

/**
 * Method:  SetSize[]
 */
int32_t
bHYPRE_StructStencil_SetSize(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t size);

/**
 * Method:  SetElement[]
 */
int32_t
bHYPRE_StructStencil_SetElement(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t index,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructStencil__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_StructStencil__exec(
  /* in */ bHYPRE_StructStencil self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_StructStencil__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_StructStencil__getURL(
  /* in */ bHYPRE_StructStencil self);
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

#ifdef __cplusplus
}
#endif
#endif
