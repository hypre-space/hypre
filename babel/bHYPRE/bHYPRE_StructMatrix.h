/*
 * File:          bHYPRE_StructMatrix.h
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:29 PST
 * Description:   Client-side glue code for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1124
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructMatrix_h
#define included_bHYPRE_StructMatrix_h

/**
 * Symbol "bHYPRE.StructMatrix" (version 1.0.0)
 * 
 * A single class that implements both a build interface and an
 * operator interface. It returns itself for GetConstructedObject.
 * 
 */
struct bHYPRE_StructMatrix__object;
struct bHYPRE_StructMatrix__array;
typedef struct bHYPRE_StructMatrix__object* bHYPRE_StructMatrix;

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
bHYPRE_StructMatrix
bHYPRE_StructMatrix__create(void);

void
bHYPRE_StructMatrix_addRef(
  bHYPRE_StructMatrix self);

void
bHYPRE_StructMatrix_deleteRef(
  bHYPRE_StructMatrix self);

SIDL_bool
bHYPRE_StructMatrix_isSame(
  bHYPRE_StructMatrix self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_StructMatrix_queryInt(
  bHYPRE_StructMatrix self,
  const char* name);

SIDL_bool
bHYPRE_StructMatrix_isType(
  bHYPRE_StructMatrix self,
  const char* name);

SIDL_ClassInfo
bHYPRE_StructMatrix_getClassInfo(
  bHYPRE_StructMatrix self);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_StructMatrix_SetCommunicator(
  bHYPRE_StructMatrix self,
  void* mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructMatrix_SetIntParameter(
  bHYPRE_StructMatrix self,
  const char* name,
  int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructMatrix_SetDoubleParameter(
  bHYPRE_StructMatrix self,
  const char* name,
  double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructMatrix_SetStringParameter(
  bHYPRE_StructMatrix self,
  const char* name,
  const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructMatrix_SetIntArray1Parameter(
  bHYPRE_StructMatrix self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructMatrix_SetIntArray2Parameter(
  bHYPRE_StructMatrix self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructMatrix_SetDoubleArray1Parameter(
  bHYPRE_StructMatrix self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructMatrix_SetDoubleArray2Parameter(
  bHYPRE_StructMatrix self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructMatrix_GetIntValue(
  bHYPRE_StructMatrix self,
  const char* name,
  int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_StructMatrix_GetDoubleValue(
  bHYPRE_StructMatrix self,
  const char* name,
  double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_StructMatrix_Setup(
  bHYPRE_StructMatrix self,
  bHYPRE_Vector b,
  bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_StructMatrix_Apply(
  bHYPRE_StructMatrix self,
  bHYPRE_Vector b,
  bHYPRE_Vector* x);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_StructMatrix_Initialize(
  bHYPRE_StructMatrix self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
bHYPRE_StructMatrix_Assemble(
  bHYPRE_StructMatrix self);

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
bHYPRE_StructMatrix_GetObject(
  bHYPRE_StructMatrix self,
  SIDL_BaseInterface* A);

/**
 * Method:  SetGrid[]
 */
int32_t
bHYPRE_StructMatrix_SetGrid(
  bHYPRE_StructMatrix self,
  bHYPRE_StructGrid grid);

/**
 * Method:  SetStencil[]
 */
int32_t
bHYPRE_StructMatrix_SetStencil(
  bHYPRE_StructMatrix self,
  bHYPRE_StructStencil stencil);

/**
 * Method:  SetValues[]
 */
int32_t
bHYPRE_StructMatrix_SetValues(
  bHYPRE_StructMatrix self,
  struct SIDL_int__array* index,
  int32_t num_stencil_indices,
  struct SIDL_int__array* stencil_indices,
  struct SIDL_double__array* values);

/**
 * Method:  SetBoxValues[]
 */
int32_t
bHYPRE_StructMatrix_SetBoxValues(
  bHYPRE_StructMatrix self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t num_stencil_indices,
  struct SIDL_int__array* stencil_indices,
  struct SIDL_double__array* values);

/**
 * Method:  SetNumGhost[]
 */
int32_t
bHYPRE_StructMatrix_SetNumGhost(
  bHYPRE_StructMatrix self,
  struct SIDL_int__array* num_ghost);

/**
 * Method:  SetSymmetric[]
 */
int32_t
bHYPRE_StructMatrix_SetSymmetric(
  bHYPRE_StructMatrix self,
  int32_t symmetric);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_StructMatrix
bHYPRE_StructMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructMatrix__cast2(
  void* obj,
  const char* type);

struct bHYPRE_StructMatrix__array*
bHYPRE_StructMatrix__array_createCol(int32_t        dimen,
                                     const int32_t lower[],
                                     const int32_t upper[]);

struct bHYPRE_StructMatrix__array*
bHYPRE_StructMatrix__array_createRow(int32_t        dimen,
                                     const int32_t lower[],
                                     const int32_t upper[]);

struct bHYPRE_StructMatrix__array*
bHYPRE_StructMatrix__array_create1d(int32_t len);

struct bHYPRE_StructMatrix__array*
bHYPRE_StructMatrix__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructMatrix__array*
bHYPRE_StructMatrix__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructMatrix__array*
bHYPRE_StructMatrix__array_borrow(bHYPRE_StructMatrix*firstElement,
                                  int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_StructMatrix__array*
bHYPRE_StructMatrix__array_smartCopy(struct bHYPRE_StructMatrix__array *array);

void
bHYPRE_StructMatrix__array_addRef(struct bHYPRE_StructMatrix__array* array);

void
bHYPRE_StructMatrix__array_deleteRef(struct bHYPRE_StructMatrix__array* array);

bHYPRE_StructMatrix
bHYPRE_StructMatrix__array_get1(const struct bHYPRE_StructMatrix__array* array,
                                const int32_t i1);

bHYPRE_StructMatrix
bHYPRE_StructMatrix__array_get2(const struct bHYPRE_StructMatrix__array* array,
                                const int32_t i1,
                                const int32_t i2);

bHYPRE_StructMatrix
bHYPRE_StructMatrix__array_get3(const struct bHYPRE_StructMatrix__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3);

bHYPRE_StructMatrix
bHYPRE_StructMatrix__array_get4(const struct bHYPRE_StructMatrix__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                const int32_t i4);

bHYPRE_StructMatrix
bHYPRE_StructMatrix__array_get(const struct bHYPRE_StructMatrix__array* array,
                               const int32_t indices[]);

void
bHYPRE_StructMatrix__array_set1(struct bHYPRE_StructMatrix__array* array,
                                const int32_t i1,
                                bHYPRE_StructMatrix const value);

void
bHYPRE_StructMatrix__array_set2(struct bHYPRE_StructMatrix__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                bHYPRE_StructMatrix const value);

void
bHYPRE_StructMatrix__array_set3(struct bHYPRE_StructMatrix__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                bHYPRE_StructMatrix const value);

void
bHYPRE_StructMatrix__array_set4(struct bHYPRE_StructMatrix__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                const int32_t i4,
                                bHYPRE_StructMatrix const value);

void
bHYPRE_StructMatrix__array_set(struct bHYPRE_StructMatrix__array* array,
                               const int32_t indices[],
                               bHYPRE_StructMatrix const value);

int32_t
bHYPRE_StructMatrix__array_dimen(const struct bHYPRE_StructMatrix__array* 
  array);

int32_t
bHYPRE_StructMatrix__array_lower(const struct bHYPRE_StructMatrix__array* array,
                                 const int32_t ind);

int32_t
bHYPRE_StructMatrix__array_upper(const struct bHYPRE_StructMatrix__array* array,
                                 const int32_t ind);

int32_t
bHYPRE_StructMatrix__array_stride(const struct bHYPRE_StructMatrix__array* 
  array,
                                  const int32_t ind);

int
bHYPRE_StructMatrix__array_isColumnOrder(const struct 
  bHYPRE_StructMatrix__array* array);

int
bHYPRE_StructMatrix__array_isRowOrder(const struct bHYPRE_StructMatrix__array* 
  array);

void
bHYPRE_StructMatrix__array_slice(const struct bHYPRE_StructMatrix__array* src,
                                       int32_t        dimen,
                                       const int32_t  numElem[],
                                       const int32_t  *srcStart,
                                       const int32_t  *srcStride,
                                       const int32_t  *newStart);

void
bHYPRE_StructMatrix__array_copy(const struct bHYPRE_StructMatrix__array* src,
                                      struct bHYPRE_StructMatrix__array* dest);

struct bHYPRE_StructMatrix__array*
bHYPRE_StructMatrix__array_ensure(struct bHYPRE_StructMatrix__array* src,
                                  int32_t dimen,
                                  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
