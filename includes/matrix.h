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
 * Header info for Matrix data structures
 *
 *****************************************************************************/

#ifndef _MATRIX_HEADER
#define _MATRIX_HEADER


/*--------------------------------------------------------------------------
 * Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int     *ia;
   int     *ja;
   int      size;

} Matrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Matrix structure
 *--------------------------------------------------------------------------*/

#define MatrixData(matrix)      ((matrix) -> data)
#define MatrixIA(matrix)        ((matrix) -> ia)
#define MatrixJA(matrix)        ((matrix) -> ja)
#define MatrixSize(matrix)      ((matrix) -> size)
#define MatrixNNZ(A)            (MatrixIA(A)[MatrixSize(A)]-1)


#endif
