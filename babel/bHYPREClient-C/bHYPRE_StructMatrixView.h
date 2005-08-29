/*
 * File:          bHYPRE_StructMatrixView.h
 * Symbol:        bHYPRE.StructMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.8
 * Description:   Client-side glue code for bHYPRE.StructMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_StructMatrixView_h
#define included_bHYPRE_StructMatrixView_h

/**
 * Symbol "bHYPRE.StructMatrixView" (version 1.0.0)
 */
struct bHYPRE_StructMatrixView__object;
struct bHYPRE_StructMatrixView__array;
typedef struct bHYPRE_StructMatrixView__object* bHYPRE_StructMatrixView;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif
#ifndef included_bHYPRE_StructStencil_h
#include "bHYPRE_StructStencil.h"
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
bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_StructMatrixView_addRef(
  /* in */ bHYPRE_StructMatrixView self);

void
bHYPRE_StructMatrixView_deleteRef(
  /* in */ bHYPRE_StructMatrixView self);

sidl_bool
bHYPRE_StructMatrixView_isSame(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_StructMatrixView_queryInt(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ const char* name);

sidl_bool
bHYPRE_StructMatrixView_isType(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_StructMatrixView_getClassInfo(
  /* in */ bHYPRE_StructMatrixView self);

int32_t
bHYPRE_StructMatrixView_SetCommunicator(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ void* mpi_comm);

int32_t
bHYPRE_StructMatrixView_Initialize(
  /* in */ bHYPRE_StructMatrixView self);

int32_t
bHYPRE_StructMatrixView_Assemble(
  /* in */ bHYPRE_StructMatrixView self);

/**
 * Method:  SetGrid[]
 */
int32_t
bHYPRE_StructMatrixView_SetGrid(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ bHYPRE_StructGrid grid);

/**
 * Method:  SetStencil[]
 */
int32_t
bHYPRE_StructMatrixView_SetStencil(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ bHYPRE_StructStencil stencil);

/**
 * Method:  SetValues[]
 */
int32_t
bHYPRE_StructMatrixView_SetValues(
  /* in */ bHYPRE_StructMatrixView self,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values);

/**
 * Method:  SetBoxValues[]
 */
int32_t
bHYPRE_StructMatrixView_SetBoxValues(
  /* in */ bHYPRE_StructMatrixView self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

/**
 * Method:  SetNumGhost[]
 */
int32_t
bHYPRE_StructMatrixView_SetNumGhost(
  /* in */ bHYPRE_StructMatrixView self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2);

/**
 * Method:  SetSymmetric[]
 */
int32_t
bHYPRE_StructMatrixView_SetSymmetric(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ int32_t symmetric);

/**
 * Method:  SetConstantEntries[]
 */
int32_t
bHYPRE_StructMatrixView_SetConstantEntries(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ int32_t num_stencil_constant_points,
  /* in rarray[num_stencil_constant_points] */ int32_t* 
    stencil_constant_points);

/**
 * Method:  SetConstantValues[]
 */
int32_t
bHYPRE_StructMatrixView_SetConstantValues(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructMatrixView__object*
bHYPRE_StructMatrixView__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructMatrixView__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_StructMatrixView__exec(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_StructMatrixView__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_StructMatrixView__getURL(
  /* in */ bHYPRE_StructMatrixView self);
struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create1d(int32_t len);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create1dInit(
  int32_t len, 
  bHYPRE_StructMatrixView* data);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_borrow(
  bHYPRE_StructMatrixView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_smartCopy(
  struct bHYPRE_StructMatrixView__array *array);

void
bHYPRE_StructMatrixView__array_addRef(
  struct bHYPRE_StructMatrixView__array* array);

void
bHYPRE_StructMatrixView__array_deleteRef(
  struct bHYPRE_StructMatrixView__array* array);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get1(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get2(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get3(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get4(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get5(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get6(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get7(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t indices[]);

void
bHYPRE_StructMatrixView__array_set1(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set2(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set3(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set4(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set5(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set6(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set7(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructMatrixView const value);

void
bHYPRE_StructMatrixView__array_set(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t indices[],
  bHYPRE_StructMatrixView const value);

int32_t
bHYPRE_StructMatrixView__array_dimen(
  const struct bHYPRE_StructMatrixView__array* array);

int32_t
bHYPRE_StructMatrixView__array_lower(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructMatrixView__array_upper(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructMatrixView__array_length(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructMatrixView__array_stride(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind);

int
bHYPRE_StructMatrixView__array_isColumnOrder(
  const struct bHYPRE_StructMatrixView__array* array);

int
bHYPRE_StructMatrixView__array_isRowOrder(
  const struct bHYPRE_StructMatrixView__array* array);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_slice(
  struct bHYPRE_StructMatrixView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructMatrixView__array_copy(
  const struct bHYPRE_StructMatrixView__array* src,
  struct bHYPRE_StructMatrixView__array* dest);

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_ensure(
  struct bHYPRE_StructMatrixView__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
