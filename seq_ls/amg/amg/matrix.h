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
 * Header info for hypre_Matrix data structures
 *
 *****************************************************************************/

#ifndef HYPRE_MATRIX_HEADER
#define HYPRE_MATRIX_HEADER


/*--------------------------------------------------------------------------
 * hypre_Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int     *ia;
   int     *ja;
   int      size;

} hypre_Matrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_MatrixData(matrix)      ((matrix) -> data)
#define hypre_MatrixIA(matrix)        ((matrix) -> ia)
#define hypre_MatrixJA(matrix)        ((matrix) -> ja)
#define hypre_MatrixSize(matrix)      ((matrix) -> size)


#endif
