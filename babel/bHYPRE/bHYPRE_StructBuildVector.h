/*
 * File:          bHYPRE_StructBuildVector.h
 * Symbol:        bHYPRE.StructBuildVector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.8
 * Description:   Client-side glue code for bHYPRE.StructBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_StructBuildVector_h
#define included_bHYPRE_StructBuildVector_h

/**
 * Symbol "bHYPRE.StructBuildVector" (version 1.0.0)
 */
struct bHYPRE_StructBuildVector__object;
struct bHYPRE_StructBuildVector__array;
typedef struct bHYPRE_StructBuildVector__object* bHYPRE_StructBuildVector;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
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
bHYPRE_StructBuildVector
bHYPRE_StructBuildVector__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_StructBuildVector_addRef(
  /* in */ bHYPRE_StructBuildVector self);

void
bHYPRE_StructBuildVector_deleteRef(
  /* in */ bHYPRE_StructBuildVector self);

sidl_bool
bHYPRE_StructBuildVector_isSame(
  /* in */ bHYPRE_StructBuildVector self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_StructBuildVector_queryInt(
  /* in */ bHYPRE_StructBuildVector self,
  /* in */ const char* name);

sidl_bool
bHYPRE_StructBuildVector_isType(
  /* in */ bHYPRE_StructBuildVector self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_StructBuildVector_getClassInfo(
  /* in */ bHYPRE_StructBuildVector self);

int32_t
bHYPRE_StructBuildVector_SetCommunicator(
  /* in */ bHYPRE_StructBuildVector self,
  /* in */ void* mpi_comm);

int32_t
bHYPRE_StructBuildVector_Initialize(
  /* in */ bHYPRE_StructBuildVector self);

int32_t
bHYPRE_StructBuildVector_Assemble(
  /* in */ bHYPRE_StructBuildVector self);

int32_t
bHYPRE_StructBuildVector_GetObject(
  /* in */ bHYPRE_StructBuildVector self,
  /* out */ sidl_BaseInterface* A);

/**
 * Method:  SetGrid[]
 */
int32_t
bHYPRE_StructBuildVector_SetGrid(
  /* in */ bHYPRE_StructBuildVector self,
  /* in */ bHYPRE_StructGrid grid);

/**
 * Method:  SetNumGhost[]
 */
int32_t
bHYPRE_StructBuildVector_SetNumGhost(
  /* in */ bHYPRE_StructBuildVector self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2);

/**
 * Method:  SetValue[]
 */
int32_t
bHYPRE_StructBuildVector_SetValue(
  /* in */ bHYPRE_StructBuildVector self,
  /* in rarray[dim] */ int32_t* grid_index,
  /* in */ int32_t dim,
  /* in */ double value);

/**
 * Method:  SetBoxValues[]
 */
int32_t
bHYPRE_StructBuildVector_SetBoxValues(
  /* in */ bHYPRE_StructBuildVector self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructBuildVector__object*
bHYPRE_StructBuildVector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructBuildVector__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_StructBuildVector__exec(
  /* in */ bHYPRE_StructBuildVector self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_StructBuildVector__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_StructBuildVector__getURL(
  /* in */ bHYPRE_StructBuildVector self);
struct bHYPRE_StructBuildVector__array*
bHYPRE_StructBuildVector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructBuildVector__array*
bHYPRE_StructBuildVector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructBuildVector__array*
bHYPRE_StructBuildVector__array_create1d(int32_t len);

struct bHYPRE_StructBuildVector__array*
bHYPRE_StructBuildVector__array_create1dInit(
  int32_t len, 
  bHYPRE_StructBuildVector* data);

struct bHYPRE_StructBuildVector__array*
bHYPRE_StructBuildVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructBuildVector__array*
bHYPRE_StructBuildVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructBuildVector__array*
bHYPRE_StructBuildVector__array_borrow(
  bHYPRE_StructBuildVector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructBuildVector__array*
bHYPRE_StructBuildVector__array_smartCopy(
  struct bHYPRE_StructBuildVector__array *array);

void
bHYPRE_StructBuildVector__array_addRef(
  struct bHYPRE_StructBuildVector__array* array);

void
bHYPRE_StructBuildVector__array_deleteRef(
  struct bHYPRE_StructBuildVector__array* array);

bHYPRE_StructBuildVector
bHYPRE_StructBuildVector__array_get1(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1);

bHYPRE_StructBuildVector
bHYPRE_StructBuildVector__array_get2(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructBuildVector
bHYPRE_StructBuildVector__array_get3(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructBuildVector
bHYPRE_StructBuildVector__array_get4(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructBuildVector
bHYPRE_StructBuildVector__array_get5(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructBuildVector
bHYPRE_StructBuildVector__array_get6(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructBuildVector
bHYPRE_StructBuildVector__array_get7(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructBuildVector
bHYPRE_StructBuildVector__array_get(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t indices[]);

void
bHYPRE_StructBuildVector__array_set1(
  struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  bHYPRE_StructBuildVector const value);

void
bHYPRE_StructBuildVector__array_set2(
  struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructBuildVector const value);

void
bHYPRE_StructBuildVector__array_set3(
  struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructBuildVector const value);

void
bHYPRE_StructBuildVector__array_set4(
  struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructBuildVector const value);

void
bHYPRE_StructBuildVector__array_set5(
  struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructBuildVector const value);

void
bHYPRE_StructBuildVector__array_set6(
  struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructBuildVector const value);

void
bHYPRE_StructBuildVector__array_set7(
  struct bHYPRE_StructBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructBuildVector const value);

void
bHYPRE_StructBuildVector__array_set(
  struct bHYPRE_StructBuildVector__array* array,
  const int32_t indices[],
  bHYPRE_StructBuildVector const value);

int32_t
bHYPRE_StructBuildVector__array_dimen(
  const struct bHYPRE_StructBuildVector__array* array);

int32_t
bHYPRE_StructBuildVector__array_lower(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructBuildVector__array_upper(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructBuildVector__array_length(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructBuildVector__array_stride(
  const struct bHYPRE_StructBuildVector__array* array,
  const int32_t ind);

int
bHYPRE_StructBuildVector__array_isColumnOrder(
  const struct bHYPRE_StructBuildVector__array* array);

int
bHYPRE_StructBuildVector__array_isRowOrder(
  const struct bHYPRE_StructBuildVector__array* array);

struct bHYPRE_StructBuildVector__array*
bHYPRE_StructBuildVector__array_slice(
  struct bHYPRE_StructBuildVector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructBuildVector__array_copy(
  const struct bHYPRE_StructBuildVector__array* src,
  struct bHYPRE_StructBuildVector__array* dest);

struct bHYPRE_StructBuildVector__array*
bHYPRE_StructBuildVector__array_ensure(
  struct bHYPRE_StructBuildVector__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
