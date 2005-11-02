/*
 * File:          bHYPRE_MatrixVectorView.h
 * Symbol:        bHYPRE.MatrixVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.10
 * Description:   Client-side glue code for bHYPRE.MatrixVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#ifndef included_bHYPRE_MatrixVectorView_h
#define included_bHYPRE_MatrixVectorView_h

/**
 * Symbol "bHYPRE.MatrixVectorView" (version 1.0.0)
 * 
 * This interface is defined to express the conceptual structure of the object
 * system.  Derived interfaces and classes have similar functions such as
 * SetValues and Print, but the functions are not declared here because the
 * function argument lists vary
 * 
 */
struct bHYPRE_MatrixVectorView__object;
struct bHYPRE_MatrixVectorView__array;
typedef struct bHYPRE_MatrixVectorView__object* bHYPRE_MatrixVectorView;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
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
bHYPRE_MatrixVectorView
bHYPRE_MatrixVectorView__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_MatrixVectorView_addRef(
  /* in */ bHYPRE_MatrixVectorView self);

void
bHYPRE_MatrixVectorView_deleteRef(
  /* in */ bHYPRE_MatrixVectorView self);

sidl_bool
bHYPRE_MatrixVectorView_isSame(
  /* in */ bHYPRE_MatrixVectorView self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_MatrixVectorView_queryInt(
  /* in */ bHYPRE_MatrixVectorView self,
  /* in */ const char* name);

sidl_bool
bHYPRE_MatrixVectorView_isType(
  /* in */ bHYPRE_MatrixVectorView self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_MatrixVectorView_getClassInfo(
  /* in */ bHYPRE_MatrixVectorView self);

int32_t
bHYPRE_MatrixVectorView_SetCommunicator(
  /* in */ bHYPRE_MatrixVectorView self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

int32_t
bHYPRE_MatrixVectorView_Initialize(
  /* in */ bHYPRE_MatrixVectorView self);

int32_t
bHYPRE_MatrixVectorView_Assemble(
  /* in */ bHYPRE_MatrixVectorView self);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_MatrixVectorView__object*
bHYPRE_MatrixVectorView__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_MatrixVectorView__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_MatrixVectorView__exec(
  /* in */ bHYPRE_MatrixVectorView self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_MatrixVectorView__getURL(
  /* in */ bHYPRE_MatrixVectorView self);
struct bHYPRE_MatrixVectorView__array*
bHYPRE_MatrixVectorView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_MatrixVectorView__array*
bHYPRE_MatrixVectorView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_MatrixVectorView__array*
bHYPRE_MatrixVectorView__array_create1d(int32_t len);

struct bHYPRE_MatrixVectorView__array*
bHYPRE_MatrixVectorView__array_create1dInit(
  int32_t len, 
  bHYPRE_MatrixVectorView* data);

struct bHYPRE_MatrixVectorView__array*
bHYPRE_MatrixVectorView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_MatrixVectorView__array*
bHYPRE_MatrixVectorView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_MatrixVectorView__array*
bHYPRE_MatrixVectorView__array_borrow(
  bHYPRE_MatrixVectorView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_MatrixVectorView__array*
bHYPRE_MatrixVectorView__array_smartCopy(
  struct bHYPRE_MatrixVectorView__array *array);

void
bHYPRE_MatrixVectorView__array_addRef(
  struct bHYPRE_MatrixVectorView__array* array);

void
bHYPRE_MatrixVectorView__array_deleteRef(
  struct bHYPRE_MatrixVectorView__array* array);

bHYPRE_MatrixVectorView
bHYPRE_MatrixVectorView__array_get1(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1);

bHYPRE_MatrixVectorView
bHYPRE_MatrixVectorView__array_get2(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_MatrixVectorView
bHYPRE_MatrixVectorView__array_get3(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_MatrixVectorView
bHYPRE_MatrixVectorView__array_get4(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_MatrixVectorView
bHYPRE_MatrixVectorView__array_get5(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_MatrixVectorView
bHYPRE_MatrixVectorView__array_get6(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_MatrixVectorView
bHYPRE_MatrixVectorView__array_get7(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_MatrixVectorView
bHYPRE_MatrixVectorView__array_get(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t indices[]);

void
bHYPRE_MatrixVectorView__array_set1(
  struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  bHYPRE_MatrixVectorView const value);

void
bHYPRE_MatrixVectorView__array_set2(
  struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_MatrixVectorView const value);

void
bHYPRE_MatrixVectorView__array_set3(
  struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_MatrixVectorView const value);

void
bHYPRE_MatrixVectorView__array_set4(
  struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_MatrixVectorView const value);

void
bHYPRE_MatrixVectorView__array_set5(
  struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_MatrixVectorView const value);

void
bHYPRE_MatrixVectorView__array_set6(
  struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_MatrixVectorView const value);

void
bHYPRE_MatrixVectorView__array_set7(
  struct bHYPRE_MatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_MatrixVectorView const value);

void
bHYPRE_MatrixVectorView__array_set(
  struct bHYPRE_MatrixVectorView__array* array,
  const int32_t indices[],
  bHYPRE_MatrixVectorView const value);

int32_t
bHYPRE_MatrixVectorView__array_dimen(
  const struct bHYPRE_MatrixVectorView__array* array);

int32_t
bHYPRE_MatrixVectorView__array_lower(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_MatrixVectorView__array_upper(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_MatrixVectorView__array_length(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_MatrixVectorView__array_stride(
  const struct bHYPRE_MatrixVectorView__array* array,
  const int32_t ind);

int
bHYPRE_MatrixVectorView__array_isColumnOrder(
  const struct bHYPRE_MatrixVectorView__array* array);

int
bHYPRE_MatrixVectorView__array_isRowOrder(
  const struct bHYPRE_MatrixVectorView__array* array);

struct bHYPRE_MatrixVectorView__array*
bHYPRE_MatrixVectorView__array_slice(
  struct bHYPRE_MatrixVectorView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_MatrixVectorView__array_copy(
  const struct bHYPRE_MatrixVectorView__array* src,
  struct bHYPRE_MatrixVectorView__array* dest);

struct bHYPRE_MatrixVectorView__array*
bHYPRE_MatrixVectorView__array_ensure(
  struct bHYPRE_MatrixVectorView__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
