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
   MPI_Comm           *comm;

   zzz_StructGrid     *grid;
   zzz_StructStencil  *user_stencil;
   zzz_StructStencil  *stencil;
   int                 num_values;   /* Number of "stored" coefficients */

   zzz_BoxArray       *data_space;

   double             *data;         /* Pointer to matrix data */
   int                 data_size;    /* Size of matrix data */
   int               **data_indices; /* num-boxes by stencil-size array
                                        of indices into the data array.
                                        data_indices[b][s] is the starting
                                        index of matrix data corresponding
                                        to box b and stencil coefficient s. */

   int                 symmetric;    /* Is the matrix symmetric */
   int                *symm_coeff;   /* Which coeffs are "symmetric" */
   int                 num_ghost[6]; /* Num ghost layers in each direction */

   int                 global_size;  /* Total number of nonzero coefficients */

   zzz_CommPkg        *comm_pkg;     /* Info on how to update ghost data */

} zzz_StructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_StructMatrix
 *--------------------------------------------------------------------------*/

#define zzz_StructMatrixComm(matrix)          ((matrix) -> comm)
#define zzz_StructMatrixGrid(matrix)          ((matrix) -> grid)
#define zzz_StructMatrixUserStencil(matrix)   ((matrix) -> user_stencil)
#define zzz_StructMatrixStencil(matrix)       ((matrix) -> stencil)
#define zzz_StructMatrixNumValues(matrix)     ((matrix) -> num_values)
#define zzz_StructMatrixDataSpace(matrix)     ((matrix) -> data_space)
#define zzz_StructMatrixData(matrix)          ((matrix) -> data)
#define zzz_StructMatrixDataSize(matrix)      ((matrix) -> data_size)
#define zzz_StructMatrixDataIndices(matrix)   ((matrix) -> data_indices)
#define zzz_StructMatrixSymmetric(matrix)     ((matrix) -> symmetric)
#define zzz_StructMatrixSymmCoeff(matrix)     ((matrix) -> symm_coeff)
#define zzz_StructMatrixNumGhost(matrix)      ((matrix) -> num_ghost)
#define zzz_StructMatrixGlobalSize(matrix)    ((matrix) -> global_size)
#define zzz_StructMatrixCommPkg(matrix)       ((matrix) -> comm_pkg)

#define zzz_StructMatrixBox(matrix, b) \
zzz_BoxArrayBox(zzz_StructMatrixDataSpace(matrix), b)

#define zzz_StructMatrixBoxData(matrix, b, s) \
(zzz_StructMatrixData(matrix) + zzz_StructMatrixDataIndices(matrix)[b][s])

#define zzz_StructMatrixBoxDataValue(matrix, b, s, index) \
(zzz_StructMatrixBoxData(matrix, b, s) + \
 zzz_BoxIndexRank(zzz_StructMatrixBox(matrix, b), index))


#endif
