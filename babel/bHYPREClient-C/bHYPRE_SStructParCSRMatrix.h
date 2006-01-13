/*
 * File:          bHYPRE_SStructParCSRMatrix.h
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_SStructParCSRMatrix_h
#define included_bHYPRE_SStructParCSRMatrix_h

/**
 * Symbol "bHYPRE.SStructParCSRMatrix" (version 1.0.0)
 * 
 * The SStructParCSR matrix class.
 * 
 * Objects of this type can be cast to SStructMatrixView or
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
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
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
struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_SStructParCSRMatrix_addRef(
  /* in */ bHYPRE_SStructParCSRMatrix self);

void
bHYPRE_SStructParCSRMatrix_deleteRef(
  /* in */ bHYPRE_SStructParCSRMatrix self);

sidl_bool
bHYPRE_SStructParCSRMatrix_isSame(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructParCSRMatrix_queryInt(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name);

sidl_bool
bHYPRE_SStructParCSRMatrix_isType(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_SStructParCSRMatrix_getClassInfo(
  /* in */ bHYPRE_SStructParCSRMatrix self);

/**
 * Method:  Create[]
 */
bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGraph graph);

/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetCommunicator(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Initialize(
  /* in */ bHYPRE_SStructParCSRMatrix self);

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
  /* in */ bHYPRE_SStructParCSRMatrix self);

/**
 *  A semi-structured matrix or vector contains a Struct or IJ matrix
 *  or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_GetObject(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface* A);

/**
 * Set the matrix graph.
 * DEPRECATED     Use Create
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetGraph(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_SStructGraph graph);

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
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nentries] */ double* values);

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
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

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
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nentries] */ double* values);

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
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

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
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t to_var,
  /* in */ int32_t symmetric);

/**
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t symmetric);

/**
 * Set the matrix to be complex.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetComplex(
  /* in */ bHYPRE_SStructParCSRMatrix self);

/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Print(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* filename,
  /* in */ int32_t all);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetIntParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetStringParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_GetIntValue(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Setup(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_Apply(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructParCSRMatrix_ApplyAdjoint(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructParCSRMatrix__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_SStructParCSRMatrix__exec(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_SStructParCSRMatrix__getURL(
  /* in */ bHYPRE_SStructParCSRMatrix self);
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
