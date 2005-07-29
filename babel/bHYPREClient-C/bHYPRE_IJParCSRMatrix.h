/*
 * File:          bHYPRE_IJParCSRMatrix.h
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_IJParCSRMatrix_h
#define included_bHYPRE_IJParCSRMatrix_h

/**
 * Symbol "bHYPRE.IJParCSRMatrix" (version 1.0.0)
 * 
 * The IJParCSR matrix class.
 * 
 * Objects of this type can be cast to IJBuildMatrix, Operator, or
 * CoefficientAccess objects using the {\tt \_\_cast} methods.
 * 
 */
struct bHYPRE_IJParCSRMatrix__object;
struct bHYPRE_IJParCSRMatrix__array;
typedef struct bHYPRE_IJParCSRMatrix__object* bHYPRE_IJParCSRMatrix;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
struct bHYPRE_IJParCSRMatrix__object*
bHYPRE_IJParCSRMatrix__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_IJParCSRMatrix_addRef(
  /* in */ bHYPRE_IJParCSRMatrix self);

void
bHYPRE_IJParCSRMatrix_deleteRef(
  /* in */ bHYPRE_IJParCSRMatrix self);

sidl_bool
bHYPRE_IJParCSRMatrix_isSame(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_IJParCSRMatrix_queryInt(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name);

sidl_bool
bHYPRE_IJParCSRMatrix_isType(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_IJParCSRMatrix_getClassInfo(
  /* in */ bHYPRE_IJParCSRMatrix self);

/**
 * Method:  Create[]
 */
bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix_Create(
  /* in */ void* mpi_comm,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper);

/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row of the diagonal and off-diagonal blocks.  The diagonal
 * block is the submatrix whose column numbers correspond to
 * rows owned by this process, and the off-diagonal block is
 * everything else.  The arrays {\tt diag\_sizes} and {\tt
 * offdiag\_sizes} contain estimated sizes for each row of the
 * diagonal and off-diagonal blocks, respectively.  This routine
 * can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t* diag_sizes,
  /* in */ int32_t* offdiag_sizes,
  /* in */ int32_t local_nrows);

/**
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_GetRow(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t row,
  /* out */ int32_t* size,
  /* out */ struct sidl_int__array** col_ind,
  /* out */ struct sidl_double__array** values);

/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetCommunicator(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ void* mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Initialize(
  /* in */ bHYPRE_IJParCSRMatrix self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Assemble(
  /* in */ bHYPRE_IJParCSRMatrix self);

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
bHYPRE_IJParCSRMatrix_GetObject(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ sidl_BaseInterface* A);

/**
 * Set the local range for a matrix object.  Each process owns
 * some unique consecutive range of rows, indicated by the
 * global row indices {\tt ilower} and {\tt iupper}.  The row
 * data is required to be such that the value of {\tt ilower} on
 * any process $p$ be exactly one more than the value of {\tt
 * iupper} on process $p-1$.  Note that the first row of the
 * global matrix may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * For square matrices, {\tt jlower} and {\tt jupper} typically
 * should match {\tt ilower} and {\tt iupper}, respectively.
 * For rectangular matrices, {\tt jlower} and {\tt jupper}
 * should define a partitioning of the columns.  This
 * partitioning must be used for any vector $v$ that will be
 * used in matrix-vector products with the rectangular matrix.
 * The matrix data structure may use {\tt jlower} and {\tt
 * jupper} to store the diagonal blocks (rectangular in general)
 * of the matrix separately from the rest of the matrix.
 * 
 * Collective.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetLocalRange(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper);

/**
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
 * ncols} and {\tt rows} are of dimension {\tt nrows} and
 * contain the number of columns in each row and the row
 * indices, respectively.  The array {\tt cols} contains the
 * column indices for each of the {\tt rows}, and is ordered by
 * rows.  The data in the {\tt values} array corresponds
 * directly to the column entries in {\tt cols}.  The last argument
 * is the size of the cols and values arrays, i.e. the total number
 * of nonzeros being provided, i.e. the sum of all values in ncols.
 * This functin erases any previous values at the specified locations and
 * replaces them with new ones, or, if there was no value there before,
 * inserts a new one.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in */ int32_t* ncols,
  /* in */ int32_t* rows,
  /* in */ int32_t* cols,
  /* in */ double* values,
  /* in */ int32_t nnonzeros);

/**
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_AddToValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in */ int32_t* ncols,
  /* in */ int32_t* rows,
  /* in */ int32_t* cols,
  /* in */ double* values,
  /* in */ int32_t nnonzeros);

/**
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_GetLocalRange(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ int32_t* ilower,
  /* out */ int32_t* iupper,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper);

/**
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_GetRowCounts(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in */ int32_t* rows,
  /* inout */ int32_t* ncols);

/**
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_GetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in */ int32_t* ncols,
  /* in */ int32_t* rows,
  /* in */ int32_t* cols,
  /* inout */ double* values,
  /* in */ int32_t nnonzeros);

/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  The integer nrows is the number of rows in
 * the local matrix.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetRowSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t* sizes,
  /* in */ int32_t nrows);

/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Print(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* filename);

/**
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Read(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* filename,
  /* in */ void* comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetIntParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetDoubleParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetStringParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_GetIntValue(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_GetDoubleValue(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Setup(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_IJParCSRMatrix_Apply(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_IJParCSRMatrix__object*
bHYPRE_IJParCSRMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_IJParCSRMatrix__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_IJParCSRMatrix__exec(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_IJParCSRMatrix__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_IJParCSRMatrix__getURL(
  /* in */ bHYPRE_IJParCSRMatrix self);
struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_create1d(int32_t len);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_create1dInit(
  int32_t len, 
  bHYPRE_IJParCSRMatrix* data);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_borrow(
  bHYPRE_IJParCSRMatrix* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_smartCopy(
  struct bHYPRE_IJParCSRMatrix__array *array);

void
bHYPRE_IJParCSRMatrix__array_addRef(
  struct bHYPRE_IJParCSRMatrix__array* array);

void
bHYPRE_IJParCSRMatrix__array_deleteRef(
  struct bHYPRE_IJParCSRMatrix__array* array);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get1(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get2(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get3(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get4(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get5(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get6(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get7(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_IJParCSRMatrix
bHYPRE_IJParCSRMatrix__array_get(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t indices[]);

void
bHYPRE_IJParCSRMatrix__array_set1(
  struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  bHYPRE_IJParCSRMatrix const value);

void
bHYPRE_IJParCSRMatrix__array_set2(
  struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_IJParCSRMatrix const value);

void
bHYPRE_IJParCSRMatrix__array_set3(
  struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_IJParCSRMatrix const value);

void
bHYPRE_IJParCSRMatrix__array_set4(
  struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_IJParCSRMatrix const value);

void
bHYPRE_IJParCSRMatrix__array_set5(
  struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_IJParCSRMatrix const value);

void
bHYPRE_IJParCSRMatrix__array_set6(
  struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_IJParCSRMatrix const value);

void
bHYPRE_IJParCSRMatrix__array_set7(
  struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_IJParCSRMatrix const value);

void
bHYPRE_IJParCSRMatrix__array_set(
  struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t indices[],
  bHYPRE_IJParCSRMatrix const value);

int32_t
bHYPRE_IJParCSRMatrix__array_dimen(
  const struct bHYPRE_IJParCSRMatrix__array* array);

int32_t
bHYPRE_IJParCSRMatrix__array_lower(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJParCSRMatrix__array_upper(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJParCSRMatrix__array_length(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJParCSRMatrix__array_stride(
  const struct bHYPRE_IJParCSRMatrix__array* array,
  const int32_t ind);

int
bHYPRE_IJParCSRMatrix__array_isColumnOrder(
  const struct bHYPRE_IJParCSRMatrix__array* array);

int
bHYPRE_IJParCSRMatrix__array_isRowOrder(
  const struct bHYPRE_IJParCSRMatrix__array* array);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_slice(
  struct bHYPRE_IJParCSRMatrix__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_IJParCSRMatrix__array_copy(
  const struct bHYPRE_IJParCSRMatrix__array* src,
  struct bHYPRE_IJParCSRMatrix__array* dest);

struct bHYPRE_IJParCSRMatrix__array*
bHYPRE_IJParCSRMatrix__array_ensure(
  struct bHYPRE_IJParCSRMatrix__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
