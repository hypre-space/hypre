/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for the zzz_StructMatrix structures
 *
 *****************************************************************************/

#ifndef zzz_STRUCT_MATRIX_HEADER
#define zzz_STRUCT_MATRIX_HEADER


/*--------------------------------------------------------------------------
 * zzz_StructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_StructGrid     *grid;
   zzz_StructStencil  *stencil;

   zzz_SBoxArrayArray *stencil_space;
   zzz_BoxArray       *data_space;

   double             *data;         /* Pointer to matrix data */
   int               **data_indices; /* num-boxes by stencil-size array
                                        of indices into the data array.
                                        data_indices[b][s] is the starting
                                        index of matrix data corresponding
                                        to box b and stencil coefficient s. */

   int                 symmetric;    /* Is the matrix symmetric */

   int                 size;         /* Total number of nonzero coefficients */

} zzz_StructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_StructMatrix
 *--------------------------------------------------------------------------*/

#define zzz_StructMatrixStructGrid(matrix)    ((matrix) -> grid)
#define zzz_StructMatrixStructStencil(matrix) ((matrix) -> stencil)

#define zzz_StructMatrixStencilSpace(matrix)  ((matrix) -> stencil_space)
#define zzz_StructMatrixDataSpace(matrix)     ((matrix) -> data_space)

#define zzz_StructMatrixData(matrix)          ((matrix) -> data)
#define zzz_StructMatrixDataIndices(matrix)   ((matrix) -> data_indices)

#define zzz_StructMatrixSymmetric(matrix)     ((matrix) -> symmetric)

#define zzz_StructMatrixSize(matrix)          ((matrix) -> size)

#define zzz_StructMatrixBox(matrix, b) \
zzz_BoxArrayBox(zzz_StructMatrixDataSpace(matrix), b)

#define zzz_StructMatrixBoxData(matrix, b, s) \
(zzz_StructMatrixData(matrix) + zzz_StructMatrixDataIndices(matrix)[b][s])

#define zzz_StructMatrixBoxDataValue(matrix, b, s, index) \
(zzz_StructMatrixBoxData(matrix, b, s) + \
 zzz_BoxIndexRank(zzz_StructMatrixBox(matrix, b), index))

#define zzz_StructMatrixContext(matrix) \
StructGridContext(StructMatrixStructGrid(matrix))


#endif
