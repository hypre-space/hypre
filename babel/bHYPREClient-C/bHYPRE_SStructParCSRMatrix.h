/*
 * File:          bHYPRE_SStructParCSRMatrix.h
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:41 PST
 * Generated:     20050225 15:45:43 PST
 * Description:   Client-side glue code for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
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

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructGraph_h
#include "bHYPRE_SStructGraph.h"
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
  /*in*/ bHYPRE_SStructParCSRMatrix self);

void
bHYPRE_SStructParCSRMatrix_deleteRef(
  /*in*/ bHYPRE_SStructParCSRMatrix self);

sidl_bool
bHYPRE_SStructParCSRMatrix_isSame(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructParCSRMatrix_queryInt(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name);

sidl_bool
bHYPRE_SStructParCSRMatrix_isType(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name);

sidl_ClassInfo
bHYPRE_SStructParCSRMatrix_getClassInfo(
  /*in*/ bHYPRE_SStructParCSRMatrix self);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetCommunicator(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ void* mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetIntParameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*in*/ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*in*/ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetStringParameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*in*/ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*in*/ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_GetIntValue(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*out*/ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* name,
  /*out*/ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Setup(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ bHYPRE_Vector b,
  /*in*/ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Apply(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ bHYPRE_Vector b,
  /*inout*/ bHYPRE_Vector* x);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Initialize(
  /*in*/ bHYPRE_SStructParCSRMatrix self);

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
  /*in*/ bHYPRE_SStructParCSRMatrix self);

/**
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_GetObject(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*out*/ sidl_BaseInterface* A);

/**
 * Set the matrix graph.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetGraph(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ bHYPRE_SStructGraph graph);

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
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values);

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
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values);

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
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values);

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
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values);

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
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ int32_t part,
  /*in*/ int32_t var,
  /*in*/ int32_t to_var,
  /*in*/ int32_t symmetric);

/**
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ int32_t symmetric);

/**
 * Set the matrix to be complex.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetComplex(
  /*in*/ bHYPRE_SStructParCSRMatrix self);

/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Print(
  /*in*/ bHYPRE_SStructParCSRMatrix self,
  /*in*/ const char* filename,
  /*in*/ int32_t all);

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
bHYPRE_SStructParCSRMatrix__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_create1d(int32_t len);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructParCSRMatrix* data);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_borrow(
  bHYPRE_SStructParCSRMatrix* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_smartCopy(
  struct bHYPRE_SStructParCSRMatrix__array *array);

void
bHYPRE_SStructParCSRMatrix__array_addRef(
  struct bHYPRE_SStructParCSRMatrix__array* array);

void
bHYPRE_SStructParCSRMatrix__array_deleteRef(
  struct bHYPRE_SStructParCSRMatrix__array* array);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get1(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get2(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get3(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get4(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get5(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get6(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get7(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructParCSRMatrix__array_set1(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  bHYPRE_SStructParCSRMatrix const value);

void
bHYPRE_SStructParCSRMatrix__array_set2(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructParCSRMatrix const value);

void
bHYPRE_SStructParCSRMatrix__array_set3(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructParCSRMatrix const value);

void
bHYPRE_SStructParCSRMatrix__array_set4(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructParCSRMatrix const value);

void
bHYPRE_SStructParCSRMatrix__array_set5(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructParCSRMatrix const value);

void
bHYPRE_SStructParCSRMatrix__array_set6(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructParCSRMatrix const value);

void
bHYPRE_SStructParCSRMatrix__array_set7(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructParCSRMatrix const value);

void
bHYPRE_SStructParCSRMatrix__array_set(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t indices[],
  bHYPRE_SStructParCSRMatrix const value);

int32_t
bHYPRE_SStructParCSRMatrix__array_dimen(
  const struct bHYPRE_SStructParCSRMatrix__array* array);

int32_t
bHYPRE_SStructParCSRMatrix__array_lower(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructParCSRMatrix__array_upper(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructParCSRMatrix__array_length(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructParCSRMatrix__array_stride(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t ind);

int
bHYPRE_SStructParCSRMatrix__array_isColumnOrder(
  const struct bHYPRE_SStructParCSRMatrix__array* array);

int
bHYPRE_SStructParCSRMatrix__array_isRowOrder(
  const struct bHYPRE_SStructParCSRMatrix__array* array);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_slice(
  struct bHYPRE_SStructParCSRMatrix__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructParCSRMatrix__array_copy(
  const struct bHYPRE_SStructParCSRMatrix__array* src,
  struct bHYPRE_SStructParCSRMatrix__array* dest);

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_ensure(
  struct bHYPRE_SStructParCSRMatrix__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
