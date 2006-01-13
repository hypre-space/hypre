/*
 * File:          bHYPRE_MPICommunicator.h
 * Symbol:        bHYPRE.MPICommunicator-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for bHYPRE.MPICommunicator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_MPICommunicator_h
#define included_bHYPRE_MPICommunicator_h

/**
 * Symbol "bHYPRE.MPICommunicator" (version 1.0.0)
 * 
 * MPICommunicator class
 *  two Create functions: use CreateC if called from C code,
 *  CreateF if called from Fortran code
 * 
 * 
 */
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_MPICommunicator__array;
typedef struct bHYPRE_MPICommunicator__object* bHYPRE_MPICommunicator;

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
struct bHYPRE_MPICommunicator__object*
bHYPRE_MPICommunicator__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_MPICommunicator
bHYPRE_MPICommunicator__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_MPICommunicator
bHYPRE_MPICommunicator__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_MPICommunicator_addRef(
  /* in */ bHYPRE_MPICommunicator self);

void
bHYPRE_MPICommunicator_deleteRef(
  /* in */ bHYPRE_MPICommunicator self);

sidl_bool
bHYPRE_MPICommunicator_isSame(
  /* in */ bHYPRE_MPICommunicator self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_MPICommunicator_queryInt(
  /* in */ bHYPRE_MPICommunicator self,
  /* in */ const char* name);

sidl_bool
bHYPRE_MPICommunicator_isType(
  /* in */ bHYPRE_MPICommunicator self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_MPICommunicator_getClassInfo(
  /* in */ bHYPRE_MPICommunicator self);

/**
 * Method:  CreateC[]
 */
bHYPRE_MPICommunicator
bHYPRE_MPICommunicator_CreateC(
  /* in */ void* mpi_comm);

/**
 * Method:  CreateF[]
 */
bHYPRE_MPICommunicator
bHYPRE_MPICommunicator_CreateF(
  /* in */ void* mpi_comm);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_MPICommunicator__object*
bHYPRE_MPICommunicator__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_MPICommunicator__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_MPICommunicator__exec(
  /* in */ bHYPRE_MPICommunicator self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_MPICommunicator__getURL(
  /* in */ bHYPRE_MPICommunicator self);
struct bHYPRE_MPICommunicator__array*
bHYPRE_MPICommunicator__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_MPICommunicator__array*
bHYPRE_MPICommunicator__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_MPICommunicator__array*
bHYPRE_MPICommunicator__array_create1d(int32_t len);

struct bHYPRE_MPICommunicator__array*
bHYPRE_MPICommunicator__array_create1dInit(
  int32_t len, 
  bHYPRE_MPICommunicator* data);

struct bHYPRE_MPICommunicator__array*
bHYPRE_MPICommunicator__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_MPICommunicator__array*
bHYPRE_MPICommunicator__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_MPICommunicator__array*
bHYPRE_MPICommunicator__array_borrow(
  bHYPRE_MPICommunicator* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_MPICommunicator__array*
bHYPRE_MPICommunicator__array_smartCopy(
  struct bHYPRE_MPICommunicator__array *array);

void
bHYPRE_MPICommunicator__array_addRef(
  struct bHYPRE_MPICommunicator__array* array);

void
bHYPRE_MPICommunicator__array_deleteRef(
  struct bHYPRE_MPICommunicator__array* array);

bHYPRE_MPICommunicator
bHYPRE_MPICommunicator__array_get1(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1);

bHYPRE_MPICommunicator
bHYPRE_MPICommunicator__array_get2(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_MPICommunicator
bHYPRE_MPICommunicator__array_get3(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_MPICommunicator
bHYPRE_MPICommunicator__array_get4(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_MPICommunicator
bHYPRE_MPICommunicator__array_get5(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_MPICommunicator
bHYPRE_MPICommunicator__array_get6(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_MPICommunicator
bHYPRE_MPICommunicator__array_get7(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_MPICommunicator
bHYPRE_MPICommunicator__array_get(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t indices[]);

void
bHYPRE_MPICommunicator__array_set1(
  struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  bHYPRE_MPICommunicator const value);

void
bHYPRE_MPICommunicator__array_set2(
  struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_MPICommunicator const value);

void
bHYPRE_MPICommunicator__array_set3(
  struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_MPICommunicator const value);

void
bHYPRE_MPICommunicator__array_set4(
  struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_MPICommunicator const value);

void
bHYPRE_MPICommunicator__array_set5(
  struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_MPICommunicator const value);

void
bHYPRE_MPICommunicator__array_set6(
  struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_MPICommunicator const value);

void
bHYPRE_MPICommunicator__array_set7(
  struct bHYPRE_MPICommunicator__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_MPICommunicator const value);

void
bHYPRE_MPICommunicator__array_set(
  struct bHYPRE_MPICommunicator__array* array,
  const int32_t indices[],
  bHYPRE_MPICommunicator const value);

int32_t
bHYPRE_MPICommunicator__array_dimen(
  const struct bHYPRE_MPICommunicator__array* array);

int32_t
bHYPRE_MPICommunicator__array_lower(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t ind);

int32_t
bHYPRE_MPICommunicator__array_upper(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t ind);

int32_t
bHYPRE_MPICommunicator__array_length(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t ind);

int32_t
bHYPRE_MPICommunicator__array_stride(
  const struct bHYPRE_MPICommunicator__array* array,
  const int32_t ind);

int
bHYPRE_MPICommunicator__array_isColumnOrder(
  const struct bHYPRE_MPICommunicator__array* array);

int
bHYPRE_MPICommunicator__array_isRowOrder(
  const struct bHYPRE_MPICommunicator__array* array);

struct bHYPRE_MPICommunicator__array*
bHYPRE_MPICommunicator__array_slice(
  struct bHYPRE_MPICommunicator__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_MPICommunicator__array_copy(
  const struct bHYPRE_MPICommunicator__array* src,
  struct bHYPRE_MPICommunicator__array* dest);

struct bHYPRE_MPICommunicator__array*
bHYPRE_MPICommunicator__array_ensure(
  struct bHYPRE_MPICommunicator__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
