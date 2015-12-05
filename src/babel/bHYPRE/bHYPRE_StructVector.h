/*
 * File:          bHYPRE_StructVector.h
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:28 PST
 * Description:   Client-side glue code for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1129
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
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

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_SIDL_ClassInfo_h
#include "SIDL_ClassInfo.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif
#ifndef included_bHYPRE_StructStencil_h
#include "bHYPRE_StructStencil.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
bHYPRE_StructVector
bHYPRE_StructVector__create(void);

void
bHYPRE_StructVector_addRef(
  bHYPRE_StructVector self);

void
bHYPRE_StructVector_deleteRef(
  bHYPRE_StructVector self);

SIDL_bool
bHYPRE_StructVector_isSame(
  bHYPRE_StructVector self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_StructVector_queryInt(
  bHYPRE_StructVector self,
  const char* name);

SIDL_bool
bHYPRE_StructVector_isType(
  bHYPRE_StructVector self,
  const char* name);

SIDL_ClassInfo
bHYPRE_StructVector_getClassInfo(
  bHYPRE_StructVector self);

/**
 * Set {\tt self} to 0.
 * 
 */
int32_t
bHYPRE_StructVector_Clear(
  bHYPRE_StructVector self);

/**
 * Copy x into {\tt self}.
 * 
 */
int32_t
bHYPRE_StructVector_Copy(
  bHYPRE_StructVector self,
  bHYPRE_Vector x);

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
  bHYPRE_StructVector self,
  bHYPRE_Vector* x);

/**
 * Scale {\tt self} by {\tt a}.
 * 
 */
int32_t
bHYPRE_StructVector_Scale(
  bHYPRE_StructVector self,
  double a);

/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */
int32_t
bHYPRE_StructVector_Dot(
  bHYPRE_StructVector self,
  bHYPRE_Vector x,
  double* d);

/**
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */
int32_t
bHYPRE_StructVector_Axpy(
  bHYPRE_StructVector self,
  double a,
  bHYPRE_Vector x);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_StructVector_SetCommunicator(
  bHYPRE_StructVector self,
  void* mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_StructVector_Initialize(
  bHYPRE_StructVector self);

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
  bHYPRE_StructVector self);

/**
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
bHYPRE_StructVector_GetObject(
  bHYPRE_StructVector self,
  SIDL_BaseInterface* A);

/**
 * Method:  SetGrid[]
 */
int32_t
bHYPRE_StructVector_SetGrid(
  bHYPRE_StructVector self,
  bHYPRE_StructGrid grid);

/**
 * Method:  SetStencil[]
 */
int32_t
bHYPRE_StructVector_SetStencil(
  bHYPRE_StructVector self,
  bHYPRE_StructStencil stencil);

/**
 * Method:  SetValue[]
 */
int32_t
bHYPRE_StructVector_SetValue(
  bHYPRE_StructVector self,
  struct SIDL_int__array* grid_index,
  double value);

/**
 * Method:  SetBoxValues[]
 */
int32_t
bHYPRE_StructVector_SetBoxValues(
  bHYPRE_StructVector self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  struct SIDL_double__array* values);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_StructVector
bHYPRE_StructVector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructVector__cast2(
  void* obj,
  const char* type);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_createCol(int32_t        dimen,
                                     const int32_t lower[],
                                     const int32_t upper[]);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_createRow(int32_t        dimen,
                                     const int32_t lower[],
                                     const int32_t upper[]);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_create1d(int32_t len);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_borrow(bHYPRE_StructVector*firstElement,
                                  int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_smartCopy(struct bHYPRE_StructVector__array *array);

void
bHYPRE_StructVector__array_addRef(struct bHYPRE_StructVector__array* array);

void
bHYPRE_StructVector__array_deleteRef(struct bHYPRE_StructVector__array* array);

bHYPRE_StructVector
bHYPRE_StructVector__array_get1(const struct bHYPRE_StructVector__array* array,
                                const int32_t i1);

bHYPRE_StructVector
bHYPRE_StructVector__array_get2(const struct bHYPRE_StructVector__array* array,
                                const int32_t i1,
                                const int32_t i2);

bHYPRE_StructVector
bHYPRE_StructVector__array_get3(const struct bHYPRE_StructVector__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3);

bHYPRE_StructVector
bHYPRE_StructVector__array_get4(const struct bHYPRE_StructVector__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                const int32_t i4);

bHYPRE_StructVector
bHYPRE_StructVector__array_get(const struct bHYPRE_StructVector__array* array,
                               const int32_t indices[]);

void
bHYPRE_StructVector__array_set1(struct bHYPRE_StructVector__array* array,
                                const int32_t i1,
                                bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set2(struct bHYPRE_StructVector__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set3(struct bHYPRE_StructVector__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set4(struct bHYPRE_StructVector__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                const int32_t i4,
                                bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set(struct bHYPRE_StructVector__array* array,
                               const int32_t indices[],
                               bHYPRE_StructVector const value);

int32_t
bHYPRE_StructVector__array_dimen(const struct bHYPRE_StructVector__array* 
  array);

int32_t
bHYPRE_StructVector__array_lower(const struct bHYPRE_StructVector__array* array,
                                 const int32_t ind);

int32_t
bHYPRE_StructVector__array_upper(const struct bHYPRE_StructVector__array* array,
                                 const int32_t ind);

int32_t
bHYPRE_StructVector__array_stride(const struct bHYPRE_StructVector__array* 
  array,
                                  const int32_t ind);

int
bHYPRE_StructVector__array_isColumnOrder(const struct 
  bHYPRE_StructVector__array* array);

int
bHYPRE_StructVector__array_isRowOrder(const struct bHYPRE_StructVector__array* 
  array);

void
bHYPRE_StructVector__array_slice(const struct bHYPRE_StructVector__array* src,
                                       int32_t        dimen,
                                       const int32_t  numElem[],
                                       const int32_t  *srcStart,
                                       const int32_t  *srcStride,
                                       const int32_t  *newStart);

void
bHYPRE_StructVector__array_copy(const struct bHYPRE_StructVector__array* src,
                                      struct bHYPRE_StructVector__array* dest);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_ensure(struct bHYPRE_StructVector__array* src,
                                  int32_t dimen,
                                  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
