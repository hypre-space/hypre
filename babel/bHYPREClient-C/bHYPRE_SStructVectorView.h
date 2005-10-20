/*
 * File:          bHYPRE_SStructVectorView.h
 * Symbol:        bHYPRE.SStructVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.8
 * Description:   Client-side glue code for bHYPRE.SStructVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_SStructVectorView_h
#define included_bHYPRE_SStructVectorView_h

/**
 * Symbol "bHYPRE.SStructVectorView" (version 1.0.0)
 */
struct bHYPRE_SStructVectorView__object;
struct bHYPRE_SStructVectorView__array;
typedef struct bHYPRE_SStructVectorView__object* bHYPRE_SStructVectorView;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
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
bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_SStructVectorView_addRef(
  /* in */ bHYPRE_SStructVectorView self);

void
bHYPRE_SStructVectorView_deleteRef(
  /* in */ bHYPRE_SStructVectorView self);

sidl_bool
bHYPRE_SStructVectorView_isSame(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructVectorView_queryInt(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ const char* name);

sidl_bool
bHYPRE_SStructVectorView_isType(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_SStructVectorView_getClassInfo(
  /* in */ bHYPRE_SStructVectorView self);

int32_t
bHYPRE_SStructVectorView_SetCommunicator(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

int32_t
bHYPRE_SStructVectorView_Initialize(
  /* in */ bHYPRE_SStructVectorView self);

int32_t
bHYPRE_SStructVectorView_Assemble(
  /* in */ bHYPRE_SStructVectorView self);

int32_t
bHYPRE_SStructVectorView_GetObject(
  /* in */ bHYPRE_SStructVectorView self,
  /* out */ sidl_BaseInterface* A);

/**
 * Set the vector grid.
 * 
 */
int32_t
bHYPRE_SStructVectorView_SetGrid(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ bHYPRE_SStructGrid grid);

/**
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructVectorView_SetValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[one] */ double* values,
  /* in */ int32_t one);

/**
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructVectorView_SetBoxValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

/**
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructVectorView_AddToValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[one] */ double* values,
  /* in */ int32_t one);

/**
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructVectorView_AddToBoxValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

/**
 * Gather vector data before calling {\tt GetValues}.
 * 
 */
int32_t
bHYPRE_SStructVectorView_Gather(
  /* in */ bHYPRE_SStructVectorView self);

/**
 * Get vector coefficients index by index.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructVectorView_GetValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ double* value);

/**
 * Get vector coefficients a box at a time.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructVectorView_GetBoxValues(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* inout rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

/**
 * Set the vector to be complex.
 * 
 */
int32_t
bHYPRE_SStructVectorView_SetComplex(
  /* in */ bHYPRE_SStructVectorView self);

/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_SStructVectorView_Print(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ const char* filename,
  /* in */ int32_t all);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructVectorView__object*
bHYPRE_SStructVectorView__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructVectorView__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_SStructVectorView__exec(
  /* in */ bHYPRE_SStructVectorView self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_SStructVectorView__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_SStructVectorView__getURL(
  /* in */ bHYPRE_SStructVectorView self);
struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_create1d(int32_t len);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructVectorView* data);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_borrow(
  bHYPRE_SStructVectorView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_smartCopy(
  struct bHYPRE_SStructVectorView__array *array);

void
bHYPRE_SStructVectorView__array_addRef(
  struct bHYPRE_SStructVectorView__array* array);

void
bHYPRE_SStructVectorView__array_deleteRef(
  struct bHYPRE_SStructVectorView__array* array);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get1(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get2(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get3(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get4(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get5(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get6(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get7(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructVectorView
bHYPRE_SStructVectorView__array_get(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructVectorView__array_set1(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set2(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set3(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set4(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set5(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set6(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set7(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructVectorView const value);

void
bHYPRE_SStructVectorView__array_set(
  struct bHYPRE_SStructVectorView__array* array,
  const int32_t indices[],
  bHYPRE_SStructVectorView const value);

int32_t
bHYPRE_SStructVectorView__array_dimen(
  const struct bHYPRE_SStructVectorView__array* array);

int32_t
bHYPRE_SStructVectorView__array_lower(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructVectorView__array_upper(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructVectorView__array_length(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructVectorView__array_stride(
  const struct bHYPRE_SStructVectorView__array* array,
  const int32_t ind);

int
bHYPRE_SStructVectorView__array_isColumnOrder(
  const struct bHYPRE_SStructVectorView__array* array);

int
bHYPRE_SStructVectorView__array_isRowOrder(
  const struct bHYPRE_SStructVectorView__array* array);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_slice(
  struct bHYPRE_SStructVectorView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructVectorView__array_copy(
  const struct bHYPRE_SStructVectorView__array* src,
  struct bHYPRE_SStructVectorView__array* dest);

struct bHYPRE_SStructVectorView__array*
bHYPRE_SStructVectorView__array_ensure(
  struct bHYPRE_SStructVectorView__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
