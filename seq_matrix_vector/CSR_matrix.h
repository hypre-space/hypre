/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for CSR Matrix data structures
 *
 *****************************************************************************/

#ifndef hypre_CSR_MATRIX_HEADER
#define hypre_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int     *ia;
   int     *ja;
   int      size;

} hypre_CSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRMatrixData(matrix)      ((matrix) -> data)
#define hypre_CSRMatrixIA(matrix)        ((matrix) -> ia)
#define hypre_CSRMatrixJA(matrix)        ((matrix) -> ja)
#define hypre_CSRMatrixSize(matrix)      ((matrix) -> size)
#define hypre_CSRMatrixNNZ(A)            (hypre_CSRMatrixIA(A)[hypre_CSRMatrixSize(A)]-1)


#endif
