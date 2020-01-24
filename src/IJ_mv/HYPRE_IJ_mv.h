/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
 **/
HYPRE_Int HYPRE_IJMatrixCreate(MPI_Comm        comm,
                               HYPRE_BigInt    ilower,
                               HYPRE_BigInt    iupper,
                               HYPRE_BigInt    jlower,
                               HYPRE_BigInt    jupper,
                               HYPRE_IJMatrix *matrix);

/**
 * Destroy a matrix object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
HYPRE_Int HYPRE_IJMatrixDestroy(HYPRE_IJMatrix matrix);

/**
 * Prepare a matrix object for setting coefficient values.  This
 * routine will also re-initialize an already assembled matrix,
 * allowing users to modify coefficient values.
 **/
HYPRE_Int HYPRE_IJMatrixInitialize(HYPRE_IJMatrix matrix);

/**
 * Sets values for {\tt nrows} rows or partial rows of the matrix.  
 * The arrays {\tt ncols}
 * and {\tt rows} are of dimension {\tt nrows} and contain the number
 * of columns in each row and the row indices, respectively.  The
 * array {\tt cols} contains the column indices for each of the {\tt
 * rows}, and is ordered by rows.  The data in the {\tt values} array
 * corresponds directly to the column entries in {\tt cols}.  Erases
 * any previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before, inserts a
 * new one if set locally. Note that it is not possible to set values
 * on other processors. If one tries to set a value from proc i on proc j,
 * proc i will erase all previous occurrences of this value in its stack
 * (including values generated with AddToValues), and treat it like
 * a zero value. The actual value needs to be set on proc j.
 *
 * Note that a threaded version (threaded over the number of rows)
 * will be called if 
 * HYPRE_IJMatrixSetOMPFlag is set to a value != 0. 
 * This requires that rows[i] != rows[j] for i!= j
 * and is only efficient if a large number of rows is set in one call
 * to HYPRE_IJMatrixSetValues.
 *
 * Not collective.
 *
 **/
HYPRE_Int HYPRE_IJMatrixSetValues(HYPRE_IJMatrix       matrix,
                                  HYPRE_Int            nrows,
                                  HYPRE_Int           *ncols,
                                  const HYPRE_BigInt  *rows,
                                  const HYPRE_BigInt  *cols,
                                  const HYPRE_Complex *values);

/**
 * Sets all  matrix coefficients of an already assembled matrix to
 * {\tt value}
 **/
HYPRE_Int HYPRE_IJMatrixSetConstantValues(HYPRE_IJMatrix matrix,
                                          HYPRE_Complex value);

/**
 * Adds to values for {\tt nrows} rows or partial rows of the matrix.  
 * Usage details are analogous to \Ref{HYPRE_IJMatrixSetValues}.  
 * Adds to any previous values at the specified locations, or, if 
 * there was no value there before, inserts a new one. 
 * AddToValues can be used to add to values on other processors.
 *
 * Note that a threaded version (threaded over the number of rows)
 * will be called if 
 * HYPRE_IJMatrixSetOMPFlag is set to a value != 0. 
 * This requires that rows[i] != rows[j] for i!= j
 * and is only efficient if a large number of rows is added in one call
 * to HYPRE_IJMatrixAddToValues.
 *
 * Not collective.
 *
 **/
HYPRE_Int HYPRE_IJMatrixAddToValues(HYPRE_IJMatrix       matrix,
                                    HYPRE_Int            nrows,
                                    HYPRE_Int           *ncols,
                                    const HYPRE_BigInt  *rows,
                                    const HYPRE_BigInt  *cols,
                                    const HYPRE_Complex *values);

/**
 * Sets values for {\tt nrows} rows or partial rows of the matrix.
 *
 * Same as IJMatrixSetValues, but with an additional {\tt row_indexes} array
 * that provides indexes into the {\tt cols} and {\tt values} arrays.  Because
 * of this, there can be gaps between the row data in these latter two arrays.
 *
 **/
HYPRE_Int HYPRE_IJMatrixSetValues2(HYPRE_IJMatrix       matrix,
                                   HYPRE_Int            nrows,
                                   HYPRE_Int           *ncols,
                                   const HYPRE_BigInt  *rows,
                                   const HYPRE_Int     *row_indexes,
                                   const HYPRE_BigInt  *cols,
                                   const HYPRE_Complex *values);

/**
 * Adds to values for {\tt nrows} rows or partial rows of the matrix.  
 *
 * Same as IJMatrixAddToValues, but with an additional {\tt row_indexes} array
 * that provides indexes into the {\tt cols} and {\tt values} arrays.  Because
 * of this, there can be gaps between the row data in these latter two arrays.
 *
 **/
HYPRE_Int HYPRE_IJMatrixAddToValues2(HYPRE_IJMatrix       matrix,
                                     HYPRE_Int            nrows,
                                     HYPRE_Int           *ncols,
                                     const HYPRE_BigInt  *rows,
                                     const HYPRE_Int     *row_indexes,
                                     const HYPRE_BigInt  *cols,
                                     const HYPRE_Complex *values);

/**
 * Finalize the construction of the matrix before using.
 **/
HYPRE_Int HYPRE_IJMatrixAssemble(HYPRE_IJMatrix matrix);

/**
 * Gets number of nonzeros elements for {\tt nrows} rows specified in {\tt rows}
 * and returns them in {\tt ncols}, which needs to be allocated by the
 * user.
 **/
HYPRE_Int HYPRE_IJMatrixGetRowCounts(HYPRE_IJMatrix  matrix,
                                     HYPRE_Int       nrows,
                                     HYPRE_BigInt   *rows,
                                     HYPRE_Int      *ncols);

/**
 * Gets values for {\tt nrows} rows or partial rows of the matrix.  
 * Usage details are mostly
 * analogous to \Ref{HYPRE_IJMatrixSetValues}.
 * Note that if nrows is negative, the routine will return
 * the column_indices and matrix coefficients of the
 * (-nrows) rows contained in rows.
 **/
HYPRE_Int HYPRE_IJMatrixGetValues(HYPRE_IJMatrix  matrix,
                                  HYPRE_Int       nrows,
                                  HYPRE_Int      *ncols,
                                  HYPRE_BigInt   *rows,
                                  HYPRE_BigInt   *cols,
                                  HYPRE_Complex  *values);

/**
 * Set the storage type of the matrix object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR}.
 *
 * Not collective, but must be the same on all processes.
 *
 * @see HYPRE_IJMatrixGetObject
 **/
HYPRE_Int HYPRE_IJMatrixSetObjectType(HYPRE_IJMatrix matrix,
                                      HYPRE_Int      type);

/**
 * Get the storage type of the constructed matrix object.
 **/
HYPRE_Int HYPRE_IJMatrixGetObjectType(HYPRE_IJMatrix  matrix,
                                      HYPRE_Int      *type);

/**
 * Gets range of rows owned by this processor and range
 * of column partitioning for this processor.
 **/
HYPRE_Int HYPRE_IJMatrixGetLocalRange(HYPRE_IJMatrix  matrix,
                                      HYPRE_BigInt   *ilower,
                                      HYPRE_BigInt   *iupper,
                                      HYPRE_BigInt   *jlower,
                                      HYPRE_BigInt   *jupper);

/**
 * Get a reference to the constructed matrix object.
 *
 * @see HYPRE_IJMatrixSetObjectType
 **/
HYPRE_Int HYPRE_IJMatrixGetObject(HYPRE_IJMatrix   matrix,
                                  void           **object);

/**
 * (Optional) Set the max number of nonzeros to expect in each row.
 * The array {\tt sizes} contains estimated sizes for each row on this
 * process.  This call can significantly improve the efficiency of
 * matrix construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
HYPRE_Int HYPRE_IJMatrixSetRowSizes(HYPRE_IJMatrix   matrix,
                                    const HYPRE_Int *sizes);

/**
 * (Optional) Sets the exact number of nonzeros in each row of
 * the diagonal and off-diagonal blocks.  The diagonal block is the
 * submatrix whose column numbers correspond to rows owned by this
 * process, and the off-diagonal block is everything else.  The arrays
 * {\tt diag\_sizes} and {\tt offdiag\_sizes} contain estimated sizes
 * for each row of the diagonal and off-diagonal blocks, respectively.
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
HYPRE_Int HYPRE_IJMatrixSetDiagOffdSizes(HYPRE_IJMatrix   matrix,
                                         const HYPRE_Int *diag_sizes,
                                         const HYPRE_Int *offdiag_sizes);

/**
 * (Optional) Sets the maximum number of elements that are expected to be set
 * (or added) on other processors from this processor
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
HYPRE_Int HYPRE_IJMatrixSetMaxOffProcElmts(HYPRE_IJMatrix matrix,
                                           HYPRE_Int      max_off_proc_elmts);

/**
 * (Optional) Sets the print level, if the user wants to print
 * error messages. The default is 0, i.e. no error messages are printed.
 *
 **/
HYPRE_Int HYPRE_IJMatrixSetPrintLevel(HYPRE_IJMatrix matrix,
                                      HYPRE_Int      print_level);

/**
 * (Optional) if set, will use a threaded version of 
 * HYPRE_IJMatrixSetValues and HYPRE_IJMatrixAddToValues.
 * This is only useful if a large number of rows is set or added to
 * at once. 
 *
 * NOTE that the values in the rows array of HYPRE_IJMatrixSetValues 
 * or HYPRE_IJMatrixAddToValues must be different from each other !!!
 * 
 * This option is VERY inefficient if only a small number of rows 
 * is set or added at once and/or 
 * if reallocation of storage is required and/or 
 * if values are added to off processor values.
 *
 **/
HYPRE_Int HYPRE_IJMatrixSetOMPFlag(HYPRE_IJMatrix matrix,
                                   HYPRE_Int      omp_flag);

/**
 * Read the matrix from file.  This is mainly for debugging purposes.
 **/
HYPRE_Int HYPRE_IJMatrixRead(const char     *filename,
                             MPI_Comm        comm,
                             HYPRE_Int       type,
                             HYPRE_IJMatrix *matrix);

/**
 * Print the matrix to file.  This is mainly for debugging purposes.
 **/
HYPRE_Int HYPRE_IJMatrixPrint(HYPRE_IJMatrix  matrix,
                              const char     *filename);

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
 **/
HYPRE_Int HYPRE_IJVectorCreate(MPI_Comm        comm,
                               HYPRE_BigInt    jlower,
                               HYPRE_BigInt    jupper,
                               HYPRE_IJVector *vector);

/**
 * Destroy a vector object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
HYPRE_Int HYPRE_IJVectorDestroy(HYPRE_IJVector vector);

/**
 * Prepare a vector object for setting coefficient values.  This
 * routine will also re-initialize an already assembled vector,
 * allowing users to modify coefficient values.
 **/
HYPRE_Int HYPRE_IJVectorInitialize(HYPRE_IJVector vector);

/**
 * (Optional) Sets the maximum number of elements that are expected to be set
 * (or added) on other processors from this processor
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
HYPRE_Int HYPRE_IJVectorSetMaxOffProcElmts(HYPRE_IJVector vector,
                                           HYPRE_Int      max_off_proc_elmts);

/**
 * Sets values in vector.  The arrays {\tt values} and {\tt indices}
 * are of dimension {\tt nvalues} and contain the vector values to be
 * set and the corresponding global vector indices, respectively.
 * Erases any previous values at the specified locations and replaces
 * them with new ones.  Note that it is not possible to set values
 * on other processors. If one tries to set a value from proc i on proc j,
 * proc i will erase all previous occurrences of this value in its stack
 * (including values generated with AddToValues), and treat it like
 * a zero value. The actual value needs to be set on proc j.
 *
 * Not collective.
 **/
HYPRE_Int HYPRE_IJVectorSetValues(HYPRE_IJVector       vector,
                                  HYPRE_Int            nvalues,
                                  const HYPRE_BigInt  *indices,
                                  const HYPRE_Complex *values);

/**
 * Adds to values in vector.  Usage details are analogous to
 * \Ref{HYPRE_IJVectorSetValues}.
 * Adds to any previous values at the specified locations, or, if 
 * there was no value there before, inserts a new one. 
 * AddToValues can be used to add to values on other processors.
 *
 * Not collective.
 **/
HYPRE_Int HYPRE_IJVectorAddToValues(HYPRE_IJVector       vector,
                                    HYPRE_Int            nvalues,
                                    const HYPRE_BigInt  *indices,
                                    const HYPRE_Complex *values);

/**
 * Finalize the construction of the vector before using.
 **/
HYPRE_Int HYPRE_IJVectorAssemble(HYPRE_IJVector vector);

/**
 * Gets values in vector.  Usage details are analogous to
 * \Ref{HYPRE_IJVectorSetValues}.
 *
 * Not collective.
 **/
HYPRE_Int HYPRE_IJVectorGetValues(HYPRE_IJVector   vector,
                                  HYPRE_Int        nvalues,
                                  const HYPRE_BigInt *indices,
                                  HYPRE_Complex   *values);

/**
 * Set the storage type of the vector object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR}.
 *
 * Not collective, but must be the same on all processes.
 *
 * @see HYPRE_IJVectorGetObject
 **/
HYPRE_Int HYPRE_IJVectorSetObjectType(HYPRE_IJVector vector,
                                      HYPRE_Int      type);

/**
 * Get the storage type of the constructed vector object.
 **/
HYPRE_Int HYPRE_IJVectorGetObjectType(HYPRE_IJVector  vector,
                                      HYPRE_Int      *type);

/**
 * Returns range of the part of the vector owned by this processor.
 **/
HYPRE_Int HYPRE_IJVectorGetLocalRange(HYPRE_IJVector  vector,
                                      HYPRE_BigInt   *jlower,
                                      HYPRE_BigInt   *jupper);

/**
 * Get a reference to the constructed vector object.
 *
 * @see HYPRE_IJVectorSetObjectType
 **/
HYPRE_Int HYPRE_IJVectorGetObject(HYPRE_IJVector   vector,
                                  void           **object);

/**
 * (Optional) Sets the print level, if the user wants to print
 * error messages. The default is 0, i.e. no error messages are printed.
 *
 **/
HYPRE_Int HYPRE_IJVectorSetPrintLevel(HYPRE_IJVector vector,
                                      HYPRE_Int      print_level);

/**
 * Read the vector from file.  This is mainly for debugging purposes.
 **/
HYPRE_Int HYPRE_IJVectorRead(const char     *filename,
                             MPI_Comm        comm,
                             HYPRE_Int       type,
                             HYPRE_IJVector *vector);

/**
 * Print the vector to file.  This is mainly for debugging purposes.
 **/
HYPRE_Int HYPRE_IJVectorPrint(HYPRE_IJVector  vector,
                              const char     *filename);

/*@}*/
/*@}*/

#ifdef __cplusplus
}
#endif

#endif
