/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef HYPRE_IJ_MV_HEADER
#define HYPRE_IJ_MV_HEADER

#include "HYPRE_config.h"
#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ System Interface
 *
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 *
 * @memo A linear-algebraic conceptual interface
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ Matrices
 **/
/*@{*/

struct hypre_IJMatrix_struct;
/**
 * The matrix object.
 **/
typedef struct hypre_IJMatrix_struct *HYPRE_IJMatrix;

/**
 * Create a matrix object.  Each process owns some unique consecutive
 * range of rows, indicated by the global row indices {\tt ilower} and
 * {\tt iupper}.  The row data is required to be such that the value
 * of {\tt ilower} on any process $p$ be exactly one more than the
 * value of {\tt iupper} on process $p-1$.  Note that the first row of
 * the global matrix may start with any integer value.  In particular,
 * one may use zero- or one-based indexing.
 *
 * For square matrices, {\tt jlower} and {\tt jupper} typically should
 * match {\tt ilower} and {\tt iupper}, respectively.  For rectangular
 * matrices, {\tt jlower} and {\tt jupper} should define a
 * partitioning of the columns.  This partitioning must be used for
 * any vector $v$ that will be used in matrix-vector products with the
 * rectangular matrix.  The matrix data structure may use {\tt jlower}
 * and {\tt jupper} to store the diagonal blocks (rectangular in
 * general) of the matrix separately from the rest of the matrix.
 *
 * Collective.
 *
 * DEVELOPER NOTES: The arg {\tt matrix} was moved to the end; The
 * args {\tt ilower} and {\tt iupper} replace the args {\tt global\_m}
 * and {\tt global\_n}.
 **/
int HYPRE_IJMatrixCreate(MPI_Comm        comm,
                         int             ilower,
                         int             iupper,
                         int             jlower,
                         int             jupper,
                         HYPRE_IJMatrix *matrix);

/**
 * Destroy a matrix object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 *
 * DEVELOPER NOTES: None.
 **/
int HYPRE_IJMatrixDestroy(HYPRE_IJMatrix matrix);

/**
 * Prepare a matrix object for setting coefficient values.
 *
 * DEVELOPER NOTES: This should also re-initialize; Use in conjunction
 * with {\tt AddToValues} and {\tt Assemble} to take the place of the
 * previous routine {\tt HYPRE\_IJMatrixAddToRowAfter}.
 **/
int HYPRE_IJMatrixInitialize(HYPRE_IJMatrix matrix);

/**
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt ncols}
 * and {\tt rows} are of dimension {\tt nrows} and contain the number
 * of columns in each row and the row indices, respectively.  The
 * array {\tt cols} contains the column indices for each of the {\tt
 * rows}, and is ordered by rows.  The data in the {\tt values} array
 * corresponds directly to the column entries in {\tt cols}.  Erases
 * any previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before, inserts a
 * new one.
 *
 * Not collective.
 *
 * DEVELOPER NOTES: This routine is essentially new; it does not do
 * the same thing as it previous did.
 **/
int HYPRE_IJMatrixSetValues(HYPRE_IJMatrix  matrix,
                            int             nrows,
                            int            *ncols,
                            const int      *rows,
                            const int      *cols,
                            const double   *values);

/**
 * Adds to values for {\tt nrows} of the matrix.  Usage details are
 * analogous to \Ref{HYPRE_IJMatrixSetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value there
 * before, inserts a new one.
 *
 * Not collective.
 *
 * DEVELOPER NOTES: This routine is essentially new; it does not do
 * the same thing as it previous did.
 **/
int HYPRE_IJMatrixAddToValues(HYPRE_IJMatrix  matrix,
                              int             nrows,
                              int            *ncols,
                              const int      *rows,
                              const int      *cols,
                              const double   *values);

/**
 * Finalize the construction of the matrix before using.
 *
 * DEVELOPER NOTES: This should also re-assemble; Use in conjunction
 * with {\tt Initialize} and {\tt AddToValues} to take the place of
 * the previous routine {\tt HYPRE\_IJMatrixAddToRowAfter}.
 **/
int HYPRE_IJMatrixAssemble(HYPRE_IJMatrix matrix);

/**
 * Gets values for {\tt nrows} of the matrix.  Usage details are
 * analogous to \Ref{HYPRE_IJMatrixSetValues}.
 *
 * DEVELOPER NOTES: New routine.
 **/
int HYPRE_IJMatrixGetValues(HYPRE_IJMatrix  matrix,
                            int             nrows,
                            int            *ncols,
                            int            *rows,
                            int            *cols,
                            double         *values);

/**
 * Set the storage type of the matrix object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR}.
 *
 * Not collective, but must be the same on all processes.
 *
 * DEVELOPER NOTES: Changed function name from {\tt LocalStorage} to
 * {\tt Object}; New return type; Returned object is now the last arg.
 *
 * @see HYPRE_IJMatrixGetObject
 **/
int HYPRE_IJMatrixSetObjectType(HYPRE_IJMatrix matrix,
                                int            type);

/**
 * Get the storage type of the constructed matrix object.
 *
 * DEVELOPER NOTES: New routine.
 **/
int HYPRE_IJMatrixGetObjectType(HYPRE_IJMatrix  matrix,
                                int            *type);

/**
 * Get a reference to the constructed matrix object.
 *
 * DEVELOPER NOTES: Changed function name from {\tt LocalStorage} to
 * {\tt Object}; New return type; Returned object is now the last arg.
 *
 * @see HYPRE_IJMatrixSetObjectType
 **/
int HYPRE_IJMatrixGetObject(HYPRE_IJMatrix   matrix,
                            void           **object);

/**
 * (Optional) Set the max number of nonzeros to expect in each row.
 * The array {\tt sizes} contains estimated sizes for each row on this
 * process.  This call can significantly improve the efficiency of
 * matrix construction, and should always be utilized if possible.
 *
 * Not collective.
 *
 * DEVELOPER NOTES: None.
 **/
int HYPRE_IJMatrixSetRowSizes(HYPRE_IJMatrix  matrix,
                              const int      *sizes);

/**
 * (Optional) Set the max number of nonzeros to expect in each row of
 * the diagonal and off-diagonal blocks.  The diagonal block is the
 * submatrix whose column numbers correspond to rows owned by this
 * process, and the off-diagonal block is everything else.  The arrays
 * {\tt diag\_sizes} and {\tt offdiag\_sizes} contain estimated sizes
 * for each row of the diagonal and off-diagonal blocks, respectively.
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 *
 * Not collective.
 *
 * DEVELOPER NOTES: This routine now combines the functionality of the
 * previous {\tt SetDiag} and {\tt SetOffDiag} routines.
 **/
int HYPRE_IJMatrixSetDiagOffdSizes(HYPRE_IJMatrix  matrix,
                                   const int      *diag_sizes,
                                   const int      *offdiag_sizes);

/**
 * Read the matrix from file.  This is mainly for debugging purposes.
 *
 * DEVELOPER NOTES: New routine.
 **/
int HYPRE_IJMatrixRead(char           *filename,
		       MPI_Comm        comm,
		       int             type,
		       HYPRE_IJMatrix *matrix);

/*@}*/

/**
 * Print the matrix to file.  This is mainly for debugging purposes.
 *
 * DEVELOPER NOTES: New routine.
 **/
int HYPRE_IJMatrixPrint(HYPRE_IJMatrix  matrix,
                        char           *filename);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ Vectors
 **/
/*@{*/

struct hypre_IJVector_struct;
/**
 * The vector object.
 **/
typedef struct hypre_IJVector_struct *HYPRE_IJVector;

/**
 * Create a vector object.  Each process owns some unique consecutive
 * range of vector unknowns, indicated by the global indices {\tt
 * jlower} and {\tt jupper}.  The data is required to be such that the
 * value of {\tt jlower} on any process $p$ be exactly one more than
 * the value of {\tt jupper} on process $p-1$.  Note that the first
 * index of the global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 *
 * Collective.
 *
 * DEVELOPER NOTES: The arg {\tt vector} was moved to the end; The
 * args {\tt jlower} and {\tt jupper} replace the arg {\tt global\_n}.
 **/
int HYPRE_IJVectorCreate(MPI_Comm        comm,
                         int             jlower,
                         int             jupper,
                         HYPRE_IJVector *vector);

/**
 * Destroy a vector object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 *
 * DEVELOPER NOTES: None.
 **/
int HYPRE_IJVectorDestroy(HYPRE_IJVector vector);

/**
 * Prepare a vector object for setting coefficient values.
 *
 * DEVELOPER NOTES: This should also re-initialize.
 **/
int HYPRE_IJVectorInitialize(HYPRE_IJVector vector);

/**
 * Sets values in vector.  The arrays {\tt values} and {\tt indices}
 * are of dimension {\tt nvalues} and contain the vector values to be
 * set and the corresponding global vector indices, respectively.
 * Erases any previous values at the specified locations and replaces
 * them with new ones.
 *
 * Not collective.
 *
 * DEVELOPER NOTES: Changed name from {\tt SetLocalComponents};
 * Removed arg {\tt value\_indices}.
 **/
int HYPRE_IJVectorSetValues(HYPRE_IJVector  vector,
                            int             nvalues,
                            const int      *indices,
                            const double   *values);

/**
 * Adds to values in vector.  Usage details are analogous to
 * \Ref{HYPRE_IJVectorSetValues}.
 *
 * Not collective.
 *
 * DEVELOPER NOTES: Changed name from {\tt AddToLocalComponents};
 * Removed arg {\tt value\_indices}.
 **/
int HYPRE_IJVectorAddToValues(HYPRE_IJVector  vector,
                              int             nvalues,
                              const int      *indices,
                              const double   *values);

/**
 * Finalize the construction of the vector before using.
 *
 * DEVELOPER NOTES: This should also re-assemble.
 **/
int HYPRE_IJVectorAssemble(HYPRE_IJVector vector);

/**
 * Gets values in vector.  Usage details are analogous to
 * \Ref{HYPRE_IJVectorSetValues}.
 *
 * Not collective.
 *
 * DEVELOPER NOTES: Changed name from {\tt GetLocalComponents};
 * Removed arg {\tt value\_indices}.
 **/
int HYPRE_IJVectorGetValues(HYPRE_IJVector  vector,
                            int             nvalues,
                            const int      *indices,
                            double         *values);

/**
 * Set the storage type of the vector object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR}.
 *
 * Not collective, but must be the same on all processes.
 *
 * DEVELOPER NOTES: Changed function name from {\tt LocalStorage} to
 * {\tt Object}; New return type; Returned object is now the last arg.
 *
 * @see HYPRE_IJVectorGetObject
 **/
int HYPRE_IJVectorSetObjectType(HYPRE_IJVector vector,
                                int            type);

/**
 * Get the storage type of the constructed vector object.
 *
 * DEVELOPER NOTES: This routine was not advertised previously.
 **/
int HYPRE_IJVectorGetObjectType(HYPRE_IJVector  vector,
                                int            *type);

/**
 * Get a reference to the constructed vector object.
 *
 * DEVELOPER NOTES: Changed function name from {\tt LocalStorage} to
 * {\tt Object}; New return type; Returned object is now the last arg.
 *
 * @see HYPRE_IJVectorSetObjectType
 **/
int HYPRE_IJVectorGetObject(HYPRE_IJVector   vector,
                            void           **object);

/**
 * Read the vector from file.  This is mainly for debugging purposes.
 *
 * DEVELOPER NOTES: New routine.
 **/
int HYPRE_IJVectorRead(char           *filename,
		       MPI_Comm        comm,
		       int             type,
                       HYPRE_IJVector *vector);

/**
 * Print the vector to file.  This is mainly for debugging purposes.
 *
 * DEVELOPER NOTES: New routine.
 **/
int HYPRE_IJVectorPrint(HYPRE_IJVector  vector,
                        char           *filename);

/*@}*/
/*@}*/

#ifdef __cplusplus
}
#endif

#endif
