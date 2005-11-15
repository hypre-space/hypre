/*
 * File:          bHYPRE_StructVectorView.h
 * Symbol:        bHYPRE.StructVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.StructVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_StructVectorView_h
#define included_bHYPRE_StructVectorView_h

/**
 * Symbol "bHYPRE.StructVectorView" (version 1.0.0)
 */
struct bHYPRE_StructVectorView__object;
struct bHYPRE_StructVectorView__array;
typedef struct bHYPRE_StructVectorView__object* bHYPRE_StructVectorView;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
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
bHYPRE_StructVectorView
bHYPRE_StructVectorView__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_StructVectorView_addRef(
  /* in */ bHYPRE_StructVectorView self);

void
bHYPRE_StructVectorView_deleteRef(
  /* in */ bHYPRE_StructVectorView self);

sidl_bool
bHYPRE_StructVectorView_isSame(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_StructVectorView_queryInt(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ const char* name);

sidl_bool
bHYPRE_StructVectorView_isType(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_StructVectorView_getClassInfo(
  /* in */ bHYPRE_StructVectorView self);

int32_t
bHYPRE_StructVectorView_SetCommunicator(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

int32_t
bHYPRE_StructVectorView_Initialize(
  /* in */ bHYPRE_StructVectorView self);

int32_t
bHYPRE_StructVectorView_Assemble(
  /* in */ bHYPRE_StructVectorView self);

/**
 * Method:  SetGrid[]
 */
int32_t
bHYPRE_StructVectorView_SetGrid(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ bHYPRE_StructGrid grid);

/**
 * Method:  SetNumGhost[]
 */
int32_t
bHYPRE_StructVectorView_SetNumGhost(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ int32_t* num_ghost,
  /* in */ int32_t dim2);

/**
 * Method:  SetValue[]
 */
int32_t
bHYPRE_StructVectorView_SetValue(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ int32_t* grid_index,
  /* in */ int32_t dim,
  /* in */ double value);

/**
 * Method:  SetBoxValues[]
 */
int32_t
bHYPRE_StructVectorView_SetBoxValues(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ int32_t* ilower,
  /* in */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ double* values,
  /* in */ int32_t nvalues);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructVectorView__object*
bHYPRE_StructVectorView__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructVectorView__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_StructVectorView__exec(
  /* in */ bHYPRE_StructVectorView self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_StructVectorView__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_StructVectorView__getURL(
  /* in */ bHYPRE_StructVectorView self);
struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_create1d(int32_t len);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_create1dInit(
  int32_t len, 
  bHYPRE_StructVectorView* data);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_borrow(
  bHYPRE_StructVectorView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_smartCopy(
  struct bHYPRE_StructVectorView__array *array);

void
bHYPRE_StructVectorView__array_addRef(
  struct bHYPRE_StructVectorView__array* array);

void
bHYPRE_StructVectorView__array_deleteRef(
  struct bHYPRE_StructVectorView__array* array);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get1(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get2(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get3(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get4(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get5(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get6(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get7(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructVectorView
bHYPRE_StructVectorView__array_get(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t indices[]);

void
bHYPRE_StructVectorView__array_set1(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set2(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set3(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set4(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set5(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set6(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set7(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructVectorView const value);

void
bHYPRE_StructVectorView__array_set(
  struct bHYPRE_StructVectorView__array* array,
  const int32_t indices[],
  bHYPRE_StructVectorView const value);

int32_t
bHYPRE_StructVectorView__array_dimen(
  const struct bHYPRE_StructVectorView__array* array);

int32_t
bHYPRE_StructVectorView__array_lower(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVectorView__array_upper(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVectorView__array_length(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVectorView__array_stride(
  const struct bHYPRE_StructVectorView__array* array,
  const int32_t ind);

int
bHYPRE_StructVectorView__array_isColumnOrder(
  const struct bHYPRE_StructVectorView__array* array);

int
bHYPRE_StructVectorView__array_isRowOrder(
  const struct bHYPRE_StructVectorView__array* array);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_slice(
  struct bHYPRE_StructVectorView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructVectorView__array_copy(
  const struct bHYPRE_StructVectorView__array* src,
  struct bHYPRE_StructVectorView__array* dest);

struct bHYPRE_StructVectorView__array*
bHYPRE_StructVectorView__array_ensure(
  struct bHYPRE_StructVectorView__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
