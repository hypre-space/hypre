/*
 * File:          bHYPRE_SStructParCSRMatrix.h
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:34 PST
 * Generated:     20030401 14:47:41 PST
 * Description:   Client-side glue code for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 827
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructParCSRMatrix_h
#define included_bHYPRE_SStructParCSRMatrix_h

/**
 * Symbol "bHYPRE.SStructParCSRMatrix" (version 1.0.0)
 * 
 * The SStructParCSR matrix class.
 * 
 * Objects of this type can be cast to SStructBuildMatrix or
 * Operator objects using the {\tt \_\_cast} methods.
 * 
 */
struct bHYPRE_SStructParCSRMatrix__object;
struct bHYPRE_SStructParCSRMatrix__array;
typedef struct bHYPRE_SStructParCSRMatrix__object* bHYPRE_SStructParCSRMatrix;

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
#ifndef included_bHYPRE_SStructGraph_h
#include "bHYPRE_SStructGraph.h"
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
bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__create(void);

void
bHYPRE_SStructParCSRMatrix_addRef(
  bHYPRE_SStructParCSRMatrix self);

void
bHYPRE_SStructParCSRMatrix_deleteRef(
  bHYPRE_SStructParCSRMatrix self);

SIDL_bool
bHYPRE_SStructParCSRMatrix_isSame(
  bHYPRE_SStructParCSRMatrix self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_SStructParCSRMatrix_queryInt(
  bHYPRE_SStructParCSRMatrix self,
  const char* name);

SIDL_bool
bHYPRE_SStructParCSRMatrix_isType(
  bHYPRE_SStructParCSRMatrix self,
  const char* name);

SIDL_ClassInfo
bHYPRE_SStructParCSRMatrix_getClassInfo(
  bHYPRE_SStructParCSRMatrix self);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetCommunicator(
  bHYPRE_SStructParCSRMatrix self,
  void* mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Initialize(
  bHYPRE_SStructParCSRMatrix self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Assemble(
  bHYPRE_SStructParCSRMatrix self);

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
bHYPRE_SStructParCSRMatrix_GetObject(
  bHYPRE_SStructParCSRMatrix self,
  SIDL_BaseInterface* A);

/**
 * Set the matrix graph.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetGraph(
  bHYPRE_SStructParCSRMatrix self,
  bHYPRE_SStructGraph graph);

/**
 * Set matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetValues(
  bHYPRE_SStructParCSRMatrix self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values);

/**
 * Set matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetBoxValues(
  bHYPRE_SStructParCSRMatrix self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values);

/**
 * Add to matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_AddToValues(
  bHYPRE_SStructParCSRMatrix self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values);

/**
 * Add to matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of stencil
 * type.  Also, they must all represent couplings to the same
 * variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  bHYPRE_SStructParCSRMatrix self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  int32_t nentries,
  struct SIDL_int__array* entries,
  struct SIDL_double__array* values);

/**
 * Define symmetry properties for the stencil entries in the
 * matrix.  The boolean argument {\tt symmetric} is applied to
 * stencil entries on part {\tt part} that couple variable {\tt
 * var} to variable {\tt to\_var}.  A value of -1 may be used
 * for {\tt part}, {\tt var}, or {\tt to\_var} to specify
 * ``all''.  For example, if {\tt part} and {\tt to\_var} are
 * set to -1, then the boolean is applied to stencil entries on
 * all parts that couple variable {\tt var} to all other
 * variables.
 * 
 * By default, matrices are assumed to be nonsymmetric.
 * Significant storage savings can be made if the matrix is
 * symmetric.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetSymmetric(
  bHYPRE_SStructParCSRMatrix self,
  int32_t part,
  int32_t var,
  int32_t to_var,
  int32_t symmetric);

/**
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  bHYPRE_SStructParCSRMatrix self,
  int32_t symmetric);

/**
 * Set the matrix to be complex.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetComplex(
  bHYPRE_SStructParCSRMatrix self);

/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Print(
  bHYPRE_SStructParCSRMatrix self,
  const char* filename,
  int32_t all);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetIntParameter(
  bHYPRE_SStructParCSRMatrix self,
  const char* name,
  int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  bHYPRE_SStructParCSRMatrix self,
  const char* name,
  double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetStringParameter(
  bHYPRE_SStructParCSRMatrix self,
  const char* name,
  const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  bHYPRE_SStructParCSRMatrix self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  bHYPRE_SStructParCSRMatrix self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  bHYPRE_SStructParCSRMatrix self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  bHYPRE_SStructParCSRMatrix self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_GetIntValue(
  bHYPRE_SStructParCSRMatrix self,
  const char* name,
  int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  bHYPRE_SStructParCSRMatrix self,
  const char* name,
  double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Setup(
  bHYPRE_SStructParCSRMatrix self,
  bHYPRE_Vector b,
  bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Apply(
  bHYPRE_SStructParCSRMatrix self,
  bHYPRE_Vector b,
  bHYPRE_Vector* x);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructParCSRMatrix__cast2(
  void* obj,
  const char* type);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_createCol(int32_t        dimen,
                                            const int32_t lower[],
                                            const int32_t upper[]);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_createRow(int32_t        dimen,
                                            const int32_t lower[],
                                            const int32_t upper[]);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_create1d(int32_t len);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_borrow(
  bHYPRE_SStructParCSRMatrix*firstElement,
                                         int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_smartCopy(struct 
  bHYPRE_SStructParCSRMatrix__array *array);

void
bHYPRE_SStructParCSRMatrix__array_addRef(struct 
  bHYPRE_SStructParCSRMatrix__array* array);

void
bHYPRE_SStructParCSRMatrix__array_deleteRef(struct 
  bHYPRE_SStructParCSRMatrix__array* array);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get1(const struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                       const int32_t i1);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get2(const struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                       const int32_t i1,
                                       const int32_t i2);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get3(const struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                       const int32_t i1,
                                       const int32_t i2,
                                       const int32_t i3);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get4(const struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                       const int32_t i1,
                                       const int32_t i2,
                                       const int32_t i3,
                                       const int32_t i4);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get(const struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                      const int32_t indices[]);

void
bHYPRE_SStructParCSRMatrix__array_set1(struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                       const int32_t i1,
                                       bHYPRE_SStructParCSRMatrix const value);

void
bHYPRE_SStructParCSRMatrix__array_set2(struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                       const int32_t i1,
                                       const int32_t i2,
                                       bHYPRE_SStructParCSRMatrix const value);

void
bHYPRE_SStructParCSRMatrix__array_set3(struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                       const int32_t i1,
                                       const int32_t i2,
                                       const int32_t i3,
                                       bHYPRE_SStructParCSRMatrix const value);

void
bHYPRE_SStructParCSRMatrix__array_set4(struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                       const int32_t i1,
                                       const int32_t i2,
                                       const int32_t i3,
                                       const int32_t i4,
                                       bHYPRE_SStructParCSRMatrix const value);

void
bHYPRE_SStructParCSRMatrix__array_set(struct bHYPRE_SStructParCSRMatrix__array* 
  array,
                                      const int32_t indices[],
                                      bHYPRE_SStructParCSRMatrix const value);

int32_t
bHYPRE_SStructParCSRMatrix__array_dimen(const struct 
  bHYPRE_SStructParCSRMatrix__array* array);

int32_t
bHYPRE_SStructParCSRMatrix__array_lower(const struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                        const int32_t ind);

int32_t
bHYPRE_SStructParCSRMatrix__array_upper(const struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                        const int32_t ind);

int32_t
bHYPRE_SStructParCSRMatrix__array_stride(const struct 
  bHYPRE_SStructParCSRMatrix__array* array,
                                         const int32_t ind);

int
bHYPRE_SStructParCSRMatrix__array_isColumnOrder(const struct 
  bHYPRE_SStructParCSRMatrix__array* array);

int
bHYPRE_SStructParCSRMatrix__array_isRowOrder(const struct 
  bHYPRE_SStructParCSRMatrix__array* array);

void
bHYPRE_SStructParCSRMatrix__array_slice(const struct 
  bHYPRE_SStructParCSRMatrix__array* src,
                                              int32_t        dimen,
                                              const int32_t  numElem[],
                                              const int32_t  *srcStart,
                                              const int32_t  *srcStride,
                                              const int32_t  *newStart);

void
bHYPRE_SStructParCSRMatrix__array_copy(const struct 
  bHYPRE_SStructParCSRMatrix__array* src,
                                             struct 
  bHYPRE_SStructParCSRMatrix__array* dest);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_ensure(struct 
  bHYPRE_SStructParCSRMatrix__array* src,
                                         int32_t dimen,
                                         int     ordering);

#ifdef __cplusplus
}
#endif
#endif
