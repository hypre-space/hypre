/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header file for HYPRE_IJMatrix and HYPRE_IJVector libraries
 *
 *****************************************************************************/

#ifndef _HYPRE_IJ_MV_H
#define _HYPRE_IJ_MV_H

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name IJ Matrix Building Interface
 *
 * "IJ" represents the "linear algebraic" conceptual view of a matrix.
The "I" and "J" in the name are meant to be mnemonic for the
traditional matrix notation A(I,J).

 *
 * @memo A linear-algebraic abstract matrix building interface
 * @version 0.3
 * @author Andrew J. Cleary
 **/
/*@{*/


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

/**
 * The {\tt HYPRE\_IJMatrix} object: This object follows the classic
builder pattern. An IJMatrix is *not* a matrix data structure itself, it *builds*
one. This allows users to use the same interface to build different
underlying data structures.

 **/
struct hypre_IJMatrix_struct;
typedef struct hypre_IJMatrix_struct *HYPRE_IJMatrix;
/**
 * The {\tt HYPRE\_IJVector} object: This object follows the classic
builder pattern. An IJVector is *not* a vector data structure itself, it *builds*
one. This allows users to use the same interface to build different
underlying data structures.

 **/
struct hypre_IJVector_struct;
typedef struct hypre_IJVector_struct *HYPRE_IJVector;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/


/* HYPRE_IJMatrix.c */
/** 
Create a matrix builder that implements the IJ interface.

Collective.

{\bf Note:} Must be the first function called using "matrix" as an actual argument.
@return integer error code
@param HYPRE_IJMatrix &IJmatrix: the matrix to be initialized.
@param MPI_Comm comm: a single MPI_Communicator that contains exactly the MPI processes that are to
participate in any collective operations.
@param int global_m, global_n: the dimensions of the entire, global matrix.
@param int local_m, local_n: the dimensions of the locally stored matrix.
*/
int HYPRE_IJMatrixCreate( MPI_Comm comm , HYPRE_IJMatrix *in_matrix_ptr , int global_m , int global_n );

/**
 * Destroy a matrix builder object. An object should be explicitly destroyed using
 * this destructor when the user's code no longer needs direct access
 * to the grid description.  Once destroyed, the object must not be
 * referenced again.  Note that the object may not be
 * deallocated at the completion of this call, since there may be
 * internal package references to the object.  The object will then be
 * destroyed when all internal reference counts go to zero.
 *
@return integer error code
@param HYPRE_IJMatrix IJmatrix: the matrix to be destroyed.
 **/
int HYPRE_IJMatrixDestroy( HYPRE_IJMatrix IJmatrix );

/**
 * Initialize an IJ_Matrix object.
   Must be called before any of the Insert or add routines are called.
 *
@return integer error code
@param HYPRE_IJMatrix IJmatrix: the matrix to be initialized.
 **/
int HYPRE_IJMatrixInitialize( HYPRE_IJMatrix IJmatrix );

/**
 * Assemble an IJ_Matrix object.
   Must be called after all of the Insert or add routines are called.
   After assemble completes, the builder has constructed a matrix that
   can be used to multiply vectors, define linear systems, build solvers
   and preconditioners, etc.
 *
@return integer error code
@param HYPRE_IJMatrix IJmatrix: the matrix to be assembled.
 **/
int HYPRE_IJMatrixAssemble( HYPRE_IJMatrix IJmatrix );

/**
 * Used only in test drivers.
 *
 **/
int HYPRE_IJMatrixDistribute( HYPRE_IJMatrix IJmatrix , const int *row_starts , const int *col_starts );

/**
Tells the builder which "storage type" it should build from the IJMatrix interface
calls. Conceptually, "IJMatrix" is just an "interface" (an abstract base class in C++-speak),
with no implementation. Specific concrete classes implement this interface. One way for a user
to choose which concrete class to use is by *instantiating* the class of choice (in an OO
implementation) or by *linking* in the implementation of choice (a "compile time" choice). An
intermediate method is through the use of HYPRE_SetIJMatrixStorageType that *may* be provided
by specific implementations. For example, it is possible to build either a PETSc MPIAIJ matrix
or a ParCSR matrix from the IJ matrix interface, and this choice can be controlled through this
function.

Not collective, but must be same on all processors in group that stores matrix.
@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
 the matrix to be operated on. 
@param int type [IN]
possible types should be documented with each implementation. The first HYPRE implementation
will list its possible values in HYPRE.h (at least for now). Note that this function is optional,
as all implementations must have a default. Eventually the plan is to let solvers choose a storage
type that they can work with automatically, but this is in the future.
*/
int HYPRE_IJMatrixSetLocalStorageType( HYPRE_IJMatrix IJmatrix , int type );

/**
Tells "matrix" how many rows and columns are locally owned.
Not collective.
REQUIREMENTS: HYPRE_IJMatrixSetLocalStorageType needs to be called before this routine

@return integer error code
@param HYPRE_IJMatrix matrix [IN]
 the matrix to be operated on. 
@param int local_m [IN]
 local number of rows
@param int local_n [IN]
 local number of columns
*/
int HYPRE_IJMatrixSetLocalSize( HYPRE_IJMatrix IJmatrix , int local_m , int local_n );

/**
* HYPRE_IJMatrixSetRowSizes( HYPRE_IJMatrix IJmatrix, const int *sizes); 

Not collective.
Tells "matrix" how many nonzeros to expect in each row.
Knowing this quantity apriori may have a significant impact on the time needed for the Assemble phase,
and this option should always be utilized if the information is available.
@return integer error code
@param HYPRE_IJMatrix &matrix [IN] 
the matrix to be operated on.
@param int *sizes [IN]
a vector of length = local_m giving the estimated sizes for the diagonal parts of
all local_m rows, in order from lowest globally numbered local row to highest.
*/
int HYPRE_IJMatrixSetRowSizes( HYPRE_IJMatrix IJmatrix , const int *sizes );

/**
* HYPRE_IJMatrixSetDiagRowSizes( HYPRE_IJMatrix IJmatrix, const int *sizes); 

Not collective.
Tells "matrix" how many nonzeros to expect in each row corresponding to other rows also
on this processor, also known as the "diagonal block" corresponding to this processor. Knowing
this quantity apriori may have a significant impact on the time needed for the Assemble phase,
and this option should always be utilized if the information is available.It is most useful in conjunction
with the next function.
@return integer error code
@param HYPRE_IJMatrix &matrix [IN] 
the matrix to be operated on.
@param int *sizes [IN]
a vector of length = local_m giving the exact sizes for the diagonal parts of
all local_m rows, in order from lowest globally numbered local row to highest.
*/
int HYPRE_IJMatrixSetDiagRowSizes( HYPRE_IJMatrix IJmatrix , const int *sizes );

/**
* HYPRE_IJMatrixSetOffDiagRowSizes( HYPRE_IJMatrix matrix, const int *sizes);

Tells "matrix" how many nonzeros to expect in each corresponding to rows NOT
on this processor, i.e. the off-diagonal block corresponding to this processor. As above, this option
should always be utilized if the information is available, as the setting of coefficients and assemble
operations can be very expensive without them.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on
@param int *sizes [IN]
a vector of length >= local_m giving the exact sizes for the off-diagonal parts of
all local_m rows, in order from lowest globally numbered local row to highest
*/
int HYPRE_IJMatrixSetOffDiagRowSizes( HYPRE_IJMatrix IJmatrix , const int *sizes );

/** 
{\bf Note:} It is probably best to ignore  HYPRE_IJMatrixQueryInsertionSemantics for the time being as it is really
planned as a possible future contingency and is confusing as it is now. -AJC, 6/99

There are three possible semantic "levels" in the integer parameter "row" used in Set and Add
functions, and each implementation supports one of them. This function returns a value indicating
the semantics supported by the instantiated IJMatrix. 
level = -1: processors may include values for "row" for any row number in the global matrix.
level = 0: processors may only include values for row representing locally stored rows.
level = > 0: in addition to the above, processors may also include values for "row" representing
rows within the set of locally stored rows plus the next "level" levels of nearest neighbors, 
also known as "level sets" or "nearest neighbors".
Since level 0 is the most restrictive, it is also the easiest to implement, and the safest to use, as
all IJ matrices MUST support level 0.
In contrast, level 1 is the most general and most difficult to implement, least safe to use, and
potentially least efficient.
Levels greater than 0 represent a compromise that is appropriate for many engineering applications,
like finite element applications, where contributions to a particular matrix row may be made from
more than one processor, typically in some small neighborhood around that row. 

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be initialized.
@param int *level [OUT]
level of off-processor value-setting that this implementation supports
*/
int HYPRE_IJMatrixQueryInsertionSemantics( HYPRE_IJMatrix IJmatrix , int *level );

/** 
Inserts a block of coefficients into an IJMatrix, overwriting any coefficients
in the event of a collision.
See "QueryInsertionSemantics" for discussion on which processor 
this routine can be called from.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int m, n [IN]
the size of the block of values to be added.
@param int *rows [IN]
an integer vector of length m giving the indices in the global matrix
corresponding to the rows in "values".
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be inserted into the matrix, stored in a dense, row-major
block of size m X n.

*/
int HYPRE_IJMatrixInsertBlock( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );

/** 
Modifies the values stored in matrix by adding in a block.
If there is no value already in a particular matrix position, the structure is augmented
with a new entry. In the event of a collision, the corresponding values are summed.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int m, n [IN]
the size of the block of values to be added.
@param int *rows [IN]
an integer vector of length m giving the indices in the global matrix
corresponding to the rows in "values".
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be inserted into the matrix, stored in a dense, row-major
block of size m X n.

*/
int HYPRE_IJMatrixAddToBlock( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );

/** 
Inserts a row into the matrix. This is generally a high-speed but inflexible method to build the
matrix. This call replaces any previously existing row structure with the structure represented by
indices and coeffs.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int n [IN]
the number of values in the row to be inserted.
@param int row [IN]
index of row to be inserted.
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be inserted into the matrix.
*/
int HYPRE_IJMatrixInsertRow( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );

/** 
Adds a row to the row of a matrix before assembly. 

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int n [IN]
the number of values in the row to be added.
@param int row [IN]
index of row to be added.
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be added to the matrix.
*/
int HYPRE_IJMatrixAddToRow( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );

/** 
Adds a row to the row of a matrix after assembly.
Note: Adds only to already existing elements.

DOES ASSEMBLE NEED TO BE RECALLED?

IMHO, this routine should not be a standard part of the IJ interface... AJC.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int n [IN]
the number of values in the row to be added.
@param int row [IN]
index of row to be added.
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be added to the matrix.
*/
int HYPRE_IJMatrixAddToRowAfter( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );

/** 
Inserts values in a particular row of the matrix.
Erases any previous values at the specified locations and replaces them with
new ones, or, if there was no value there before, inserts the value.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int n [IN]
the number of values in the row to be added.
@param int row [IN]
index of row to be added.
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be added to the matrix.
*/
int HYPRE_IJMatrixSetValues( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );

/** 
Adds to values in a particular row of the matrix.
Adds to any previous values at the specified locations, 
or, if there was no value there before, inserts the value.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int n [IN]
the number of values in the row to be added.
@param int row [IN]
index of row to be added.
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be added to the matrix.
*/
int HYPRE_IJMatrixAddToValues( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );

/** 
Inserts values in one or more rows of the matrix.
Erases any previous values at the specified locations and replaces them with
new ones, or, if there was no value there before, inserts the value.
Equivalent to SetValues except can span more than one row at a time.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int n [IN]
the number of values in the row to be added.
@param int * rows [IN]
m indexes of rows to be added.
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be added to the matrix.
*/
int HYPRE_IJMatrixSetBlockValues( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );

/** 
Adds to values in a particular row of the matrix.
Adds to any previous values at the specified locations, 
or, if there was no value there before, inserts the value.
Equivalent to AddValues except can span more than one row at a time.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int n [IN]
the number of values in the row to be added.
@param int * rows [IN]
m indexes of rows to be added.
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be added to the matrix.
*/
int HYPRE_IJMatrixAddToBlockValues( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );

/** 
Internal routine only.
*/
int hypre_RefIJMatrix( HYPRE_IJMatrix IJmatrix , HYPRE_IJMatrix *reference );

/** 
Internal routine only.
*/
void *HYPRE_IJMatrixGetLocalStorage( HYPRE_IJMatrix IJmatrix );

/** 
Internal routine only.
*/
int HYPRE_IJMatrixGetRowPartitioning( HYPRE_IJMatrix IJmatrix , const int **row_partitioning );

/** 
Internal routine only.
*/
int HYPRE_IJMatrixGetColPartitioning( HYPRE_IJMatrix IJmatrix , const int **col_partitioning );

/* HYPRE_IJVector.c */

int HYPRE_IJVectorCreate( MPI_Comm comm , HYPRE_IJVector *in_vector_ptr , int global_n );

int HYPRE_IJVectorDestroy( HYPRE_IJVector IJvector );

int HYPRE_IJVectorSetPartitioning( HYPRE_IJVector IJvector , const int *partitioning );

int HYPRE_IJVectorSetLocalPartitioning( HYPRE_IJVector IJvector , int vec_start_this_proc , int vec_start_next_proc );

int HYPRE_IJVectorInitialize( HYPRE_IJVector IJvector );

int HYPRE_IJVectorDistribute( HYPRE_IJVector IJvector , const int *vec_starts );

int HYPRE_IJVectorSetLocalStorageType( HYPRE_IJVector IJvector , int type );

int HYPRE_IJVectorZeroLocalComponents( HYPRE_IJVector IJvector );

int HYPRE_IJVectorSetLocalComponents( HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values );

int HYPRE_IJVectorSetLocalComponentsInBlock( HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values );

int HYPRE_IJVectorAddToLocalComponents( HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values );

int HYPRE_IJVectorAddToLocalComponentsInBlock( HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values );

int HYPRE_IJVectorAssemble( HYPRE_IJVector IJvector );

int HYPRE_IJVectorGetLocalComponents( HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , double *values );

int HYPRE_IJVectorGetLocalComponentsInBlock( HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , double *values );

int HYPRE_IJVectorGetLocalStorageType( HYPRE_IJVector IJvector , int *type );

void *HYPRE_IJVectorGetLocalStorage( HYPRE_IJVector IJvector );

int hypre_RefIJVector( HYPRE_IJVector IJvector , HYPRE_IJVector *reference );

#ifdef __cplusplus
}
#endif

#endif
