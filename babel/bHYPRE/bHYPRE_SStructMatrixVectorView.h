/*
 * File:          bHYPRE_SStructMatrixVectorView.h
 * Symbol:        bHYPRE.SStructMatrixVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for bHYPRE.SStructMatrixVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_SStructMatrixVectorView_h
#define included_bHYPRE_SStructMatrixVectorView_h

/**
 * Symbol "bHYPRE.SStructMatrixVectorView" (version 1.0.0)
 */
struct bHYPRE_SStructMatrixVectorView__object;
struct bHYPRE_SStructMatrixVectorView__array;
typedef struct bHYPRE_SStructMatrixVectorView__object* 
  bHYPRE_SStructMatrixVectorView;

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
bHYPRE_SStructMatrixVectorView
bHYPRE_SStructMatrixVectorView__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_SStructMatrixVectorView_addRef(
  /* in */ bHYPRE_SStructMatrixVectorView self);

void
bHYPRE_SStructMatrixVectorView_deleteRef(
  /* in */ bHYPRE_SStructMatrixVectorView self);

sidl_bool
bHYPRE_SStructMatrixVectorView_isSame(
  /* in */ bHYPRE_SStructMatrixVectorView self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructMatrixVectorView_queryInt(
  /* in */ bHYPRE_SStructMatrixVectorView self,
  /* in */ const char* name);

sidl_bool
bHYPRE_SStructMatrixVectorView_isType(
  /* in */ bHYPRE_SStructMatrixVectorView self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_SStructMatrixVectorView_getClassInfo(
  /* in */ bHYPRE_SStructMatrixVectorView self);

int32_t
bHYPRE_SStructMatrixVectorView_SetCommunicator(
  /* in */ bHYPRE_SStructMatrixVectorView self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

int32_t
bHYPRE_SStructMatrixVectorView_Initialize(
  /* in */ bHYPRE_SStructMatrixVectorView self);

int32_t
bHYPRE_SStructMatrixVectorView_Assemble(
  /* in */ bHYPRE_SStructMatrixVectorView self);

/**
 *  A semi-structured matrix or vector contains a Struct or IJ matrix
 *  or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
bHYPRE_SStructMatrixVectorView_GetObject(
  /* in */ bHYPRE_SStructMatrixVectorView self,
  /* out */ sidl_BaseInterface* A);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructMatrixVectorView__object*
bHYPRE_SStructMatrixVectorView__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructMatrixVectorView__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_SStructMatrixVectorView__exec(
  /* in */ bHYPRE_SStructMatrixVectorView self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_SStructMatrixVectorView__getURL(
  /* in */ bHYPRE_SStructMatrixVectorView self);
struct bHYPRE_SStructMatrixVectorView__array*
bHYPRE_SStructMatrixVectorView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructMatrixVectorView__array*
bHYPRE_SStructMatrixVectorView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructMatrixVectorView__array*
bHYPRE_SStructMatrixVectorView__array_create1d(int32_t len);

struct bHYPRE_SStructMatrixVectorView__array*
bHYPRE_SStructMatrixVectorView__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructMatrixVectorView* data);

struct bHYPRE_SStructMatrixVectorView__array*
bHYPRE_SStructMatrixVectorView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructMatrixVectorView__array*
bHYPRE_SStructMatrixVectorView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructMatrixVectorView__array*
bHYPRE_SStructMatrixVectorView__array_borrow(
  bHYPRE_SStructMatrixVectorView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructMatrixVectorView__array*
bHYPRE_SStructMatrixVectorView__array_smartCopy(
  struct bHYPRE_SStructMatrixVectorView__array *array);

void
bHYPRE_SStructMatrixVectorView__array_addRef(
  struct bHYPRE_SStructMatrixVectorView__array* array);

void
bHYPRE_SStructMatrixVectorView__array_deleteRef(
  struct bHYPRE_SStructMatrixVectorView__array* array);

bHYPRE_SStructMatrixVectorView
bHYPRE_SStructMatrixVectorView__array_get1(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1);

bHYPRE_SStructMatrixVectorView
bHYPRE_SStructMatrixVectorView__array_get2(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructMatrixVectorView
bHYPRE_SStructMatrixVectorView__array_get3(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructMatrixVectorView
bHYPRE_SStructMatrixVectorView__array_get4(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructMatrixVectorView
bHYPRE_SStructMatrixVectorView__array_get5(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructMatrixVectorView
bHYPRE_SStructMatrixVectorView__array_get6(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructMatrixVectorView
bHYPRE_SStructMatrixVectorView__array_get7(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructMatrixVectorView
bHYPRE_SStructMatrixVectorView__array_get(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructMatrixVectorView__array_set1(
  struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  bHYPRE_SStructMatrixVectorView const value);

void
bHYPRE_SStructMatrixVectorView__array_set2(
  struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructMatrixVectorView const value);

void
bHYPRE_SStructMatrixVectorView__array_set3(
  struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructMatrixVectorView const value);

void
bHYPRE_SStructMatrixVectorView__array_set4(
  struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructMatrixVectorView const value);

void
bHYPRE_SStructMatrixVectorView__array_set5(
  struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructMatrixVectorView const value);

void
bHYPRE_SStructMatrixVectorView__array_set6(
  struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructMatrixVectorView const value);

void
bHYPRE_SStructMatrixVectorView__array_set7(
  struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructMatrixVectorView const value);

void
bHYPRE_SStructMatrixVectorView__array_set(
  struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t indices[],
  bHYPRE_SStructMatrixVectorView const value);

int32_t
bHYPRE_SStructMatrixVectorView__array_dimen(
  const struct bHYPRE_SStructMatrixVectorView__array* array);

int32_t
bHYPRE_SStructMatrixVectorView__array_lower(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructMatrixVectorView__array_upper(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructMatrixVectorView__array_length(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructMatrixVectorView__array_stride(
  const struct bHYPRE_SStructMatrixVectorView__array* array,
  const int32_t ind);

int
bHYPRE_SStructMatrixVectorView__array_isColumnOrder(
  const struct bHYPRE_SStructMatrixVectorView__array* array);

int
bHYPRE_SStructMatrixVectorView__array_isRowOrder(
  const struct bHYPRE_SStructMatrixVectorView__array* array);

struct bHYPRE_SStructMatrixVectorView__array*
bHYPRE_SStructMatrixVectorView__array_slice(
  struct bHYPRE_SStructMatrixVectorView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructMatrixVectorView__array_copy(
  const struct bHYPRE_SStructMatrixVectorView__array* src,
  struct bHYPRE_SStructMatrixVectorView__array* dest);

struct bHYPRE_SStructMatrixVectorView__array*
bHYPRE_SStructMatrixVectorView__array_ensure(
  struct bHYPRE_SStructMatrixVectorView__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
