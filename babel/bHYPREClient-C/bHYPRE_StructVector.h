/*
 * File:          bHYPRE_StructVector.h
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_StructVector_h
#define included_bHYPRE_StructVector_h

/**
 * Symbol "bHYPRE.StructVector" (version 1.0.0)
 */
struct bHYPRE_StructVector__object;
struct bHYPRE_StructVector__array;
typedef struct bHYPRE_StructVector__object* bHYPRE_StructVector;

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
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
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
struct bHYPRE_StructVector__object*
bHYPRE_StructVector__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_StructVector
bHYPRE_StructVector__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_StructVector
bHYPRE_StructVector__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_StructVector_addRef(
  /* in */ bHYPRE_StructVector self);

void
bHYPRE_StructVector_deleteRef(
  /* in */ bHYPRE_StructVector self);

sidl_bool
bHYPRE_StructVector_isSame(
  /* in */ bHYPRE_StructVector self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_StructVector_queryInt(
  /* in */ bHYPRE_StructVector self,
  /* in */ const char* name);

sidl_bool
bHYPRE_StructVector_isType(
  /* in */ bHYPRE_StructVector self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_StructVector_getClassInfo(
  /* in */ bHYPRE_StructVector self);

/**
 * Method:  Create[]
 */
bHYPRE_StructVector
bHYPRE_StructVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructGrid grid);

/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */
int32_t
bHYPRE_StructVector_SetCommunicator(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_StructVector_Initialize(
  /* in */ bHYPRE_StructVector self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
bHYPRE_StructVector_Assemble(
  /* in */ bHYPRE_StructVector self);

/**
 * Method:  SetGrid[]
 */
int32_t
bHYPRE_StructVector_SetGrid(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_StructGrid grid);

/**
 * Method:  SetNumGhost[]
 */
int32_t
bHYPRE_StructVector_SetNumGhost(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2);

/**
 * Method:  SetValue[]
 */
int32_t
bHYPRE_StructVector_SetValue(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ int32_t* grid_index,
  /* in */ int32_t dim,
  /* in */ double value);

/**
 * Method:  SetBoxValues[]
 */
int32_t
bHYPRE_StructVector_SetBoxValues(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

/**
 * Set {\tt self} to 0.
 * 
 */
int32_t
bHYPRE_StructVector_Clear(
  /* in */ bHYPRE_StructVector self);

/**
 * Copy x into {\tt self}.
 * 
 */
int32_t
bHYPRE_StructVector_Copy(
  /* in */ bHYPRE_StructVector self,
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
bHYPRE_StructVector_Clone(
  /* in */ bHYPRE_StructVector self,
  /* out */ bHYPRE_Vector* x);

/**
 * Scale {\tt self} by {\tt a}.
 * 
 */
int32_t
bHYPRE_StructVector_Scale(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a);

/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */
int32_t
bHYPRE_StructVector_Dot(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d);

/**
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */
int32_t
bHYPRE_StructVector_Axpy(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructVector__object*
bHYPRE_StructVector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructVector__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_StructVector__exec(
  /* in */ bHYPRE_StructVector self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_StructVector__getURL(
  /* in */ bHYPRE_StructVector self);
struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_create1d(int32_t len);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_create1dInit(
  int32_t len, 
  bHYPRE_StructVector* data);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_borrow(
  bHYPRE_StructVector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_smartCopy(
  struct bHYPRE_StructVector__array *array);

void
bHYPRE_StructVector__array_addRef(
  struct bHYPRE_StructVector__array* array);

void
bHYPRE_StructVector__array_deleteRef(
  struct bHYPRE_StructVector__array* array);

bHYPRE_StructVector
bHYPRE_StructVector__array_get1(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1);

bHYPRE_StructVector
bHYPRE_StructVector__array_get2(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructVector
bHYPRE_StructVector__array_get3(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructVector
bHYPRE_StructVector__array_get4(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructVector
bHYPRE_StructVector__array_get5(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructVector
bHYPRE_StructVector__array_get6(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructVector
bHYPRE_StructVector__array_get7(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructVector
bHYPRE_StructVector__array_get(
  const struct bHYPRE_StructVector__array* array,
  const int32_t indices[]);

void
bHYPRE_StructVector__array_set1(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set2(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set3(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set4(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set5(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set6(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set7(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set(
  struct bHYPRE_StructVector__array* array,
  const int32_t indices[],
  bHYPRE_StructVector const value);

int32_t
bHYPRE_StructVector__array_dimen(
  const struct bHYPRE_StructVector__array* array);

int32_t
bHYPRE_StructVector__array_lower(
  const struct bHYPRE_StructVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVector__array_upper(
  const struct bHYPRE_StructVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVector__array_length(
  const struct bHYPRE_StructVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVector__array_stride(
  const struct bHYPRE_StructVector__array* array,
  const int32_t ind);

int
bHYPRE_StructVector__array_isColumnOrder(
  const struct bHYPRE_StructVector__array* array);

int
bHYPRE_StructVector__array_isRowOrder(
  const struct bHYPRE_StructVector__array* array);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_slice(
  struct bHYPRE_StructVector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructVector__array_copy(
  const struct bHYPRE_StructVector__array* src,
  struct bHYPRE_StructVector__array* dest);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_ensure(
  struct bHYPRE_StructVector__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
