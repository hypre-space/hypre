/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef HYPRE_FEI_MV_HEADER
#define HYPRE_FEI_MV_HEADER

#include "HYPRE_config.h"
#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FEI System Interface
 *
 * This interface represents a FEI linear algebaric conceptual view of a
 * linear system.  
 *
 * @memo A FEI linear-algebraic conceptual interface
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FEI LSI
 **/
/*@{*/

struct hypre_FEILSI_struct;

/**
 * The matrix object.
 **/

typedef struct hypre_FEILSI_struct *HYPRE_FEILSI;

/**
 * Create a LSI object.  
 **/

int HYPRE_FEILSICreate(MPI_Comm comm, HYPRE_FEILSI *lsi);

/**
 * Destroy a LSI object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  
 **/

int HYPRE_FEILSIDestroy(HYPRE_FEILSI lsi);

/**
 * Set up the row allocation information (which processor owns which rows).
 * So my processor owns row and column eqnOffsets[mypid] to
 * eqnOffsets[mypid+1]-1.
 **/

int HYPRE_FEILSISetGlobalOffsets(HYPRE_FEILSI lsi, 
                                 int *eqnOffsets);

/**
 * Set up the sparsity pattern of the local matrix. 
 **/

int HYPRE_FEILSISetMatrixStructure(HYPRE_FEILSI lsi,
                                   int** colIndices, 
                                   int* rowLengths);

/**
 * sumIntoSystemMatrix:
 * The coefficients 'values' are to be accumumlated into (added to any 
 * values already in place) 0-based equation 'row' of the matrix.
 **/

int HYPRE_FEILSISumIntoSystemMatrix(HYPRE_FEILSI lsi,
                                    int nRows, const int* rowIndices,
                                    int nCols, const int* colIndices,
                                    const double* const* values);

/**
 * putIntoSystemMatrix:
 * The coefficients 'values' are to be put the given row.
 **/

int HYPRE_FEILSIPutIntoSystemMatrix(HYPRE_FEILSI lsi,
                                    int nRows, const int* rowIndices,
                                    int nCols, const int* colIndices,
                                    const double* const* values);

/**
 * sumIntoRHSVector:
 * The coefficients 'values' are to be accumumlated into (added to any 
 * values already in place) 0-based equation 'row' of the vector.
 **/

int HYPRE_FEILSISumIntoRHSVector(HYPRE_FEILSI lsi,
                                 int nRows, const double* values,
                                 const int* colIndices);

/**
 * putIntoRHSVector:
 * The coefficients 'values' are to put into corresponding rows
 **/

int HYPRE_FEILSIPutIntoRHSVector(HYPRE_FEILSI lsi,
                                 int nRows, const double* values,
                                 const int* colIndices);

/**
 * Finalize the construction of the matrix before using.
 **/

int HYPRE_FEILSIAssemble(HYPRE_FEILSI lsi);
   
#ifdef __cplusplus
}
#endif

#endif


