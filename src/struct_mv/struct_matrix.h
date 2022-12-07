/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the hypre_StructMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_MATRIX_HEADER
#define hypre_STRUCT_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * hypre_StructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructMatrix_struct
{
   MPI_Comm              comm;

   hypre_StructGrid     *grid;
   hypre_StructStencil  *user_stencil;
   hypre_StructStencil  *stencil;
   HYPRE_Int             num_values;                /* Number of "stored" coefficients */

   hypre_BoxArray       *data_space;

   HYPRE_MemoryLocation  memory_location;           /* memory location of data */
   HYPRE_Complex        *data;                      /* Pointer to variable matrix data */
   HYPRE_Complex        *data_const;                /* Pointer to constant matrix data */
   HYPRE_Complex       **stencil_data;              /* Pointer for each stencil */
   HYPRE_Int             data_alloced;              /* Boolean used for freeing data */
   HYPRE_Int             data_size;                 /* Size of variable matrix data */
   HYPRE_Int             data_const_size;           /* Size of constant matrix data */
   HYPRE_Int           **data_indices;              /* num-boxes by stencil-size array
                                                       of indices into the data array.
                                                       data_indices[b][s] is the starting
                                                       index of matrix data corresponding
                                                       to box b and stencil coefficient s */
   HYPRE_Int             constant_coefficient;      /* normally 0; set to 1 for
                                                       constant coefficient matrices
                                                       or 2 for constant coefficient
                                                       with variable diagonal */

   HYPRE_Int             symmetric;                 /* Is the matrix symmetric */
   HYPRE_Int            *symm_elements;             /* Which elements are "symmetric" */
   HYPRE_Int             num_ghost[2 * HYPRE_MAXDIM]; /* Num ghost layers in each direction */

   HYPRE_BigInt          global_size;               /* Total number of nonzero coeffs */

   hypre_CommPkg        *comm_pkg;                  /* Info on how to update ghost data */

   HYPRE_Int             ref_count;

} hypre_StructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_StructMatrixComm(matrix)                ((matrix) -> comm)
#define hypre_StructMatrixGrid(matrix)                ((matrix) -> grid)
#define hypre_StructMatrixUserStencil(matrix)         ((matrix) -> user_stencil)
#define hypre_StructMatrixStencil(matrix)             ((matrix) -> stencil)
#define hypre_StructMatrixNumValues(matrix)           ((matrix) -> num_values)
#define hypre_StructMatrixDataSpace(matrix)           ((matrix) -> data_space)
#define hypre_StructMatrixMemoryLocation(matrix)      ((matrix) -> memory_location)
#define hypre_StructMatrixData(matrix)                ((matrix) -> data)
#define hypre_StructMatrixDataConst(matrix)           ((matrix) -> data_const)
#define hypre_StructMatrixStencilData(matrix)         ((matrix) -> stencil_data)
#define hypre_StructMatrixDataAlloced(matrix)         ((matrix) -> data_alloced)
#define hypre_StructMatrixDataSize(matrix)            ((matrix) -> data_size)
#define hypre_StructMatrixDataConstSize(matrix)       ((matrix) -> data_const_size)
#define hypre_StructMatrixDataIndices(matrix)         ((matrix) -> data_indices)
#define hypre_StructMatrixConstantCoefficient(matrix) ((matrix) -> constant_coefficient)
#define hypre_StructMatrixSymmetric(matrix)           ((matrix) -> symmetric)
#define hypre_StructMatrixSymmElements(matrix)        ((matrix) -> symm_elements)
#define hypre_StructMatrixNumGhost(matrix)            ((matrix) -> num_ghost)
#define hypre_StructMatrixGlobalSize(matrix)          ((matrix) -> global_size)
#define hypre_StructMatrixCommPkg(matrix)             ((matrix) -> comm_pkg)
#define hypre_StructMatrixRefCount(matrix)            ((matrix) -> ref_count)

#define hypre_StructMatrixNDim(matrix) \
hypre_StructGridNDim(hypre_StructMatrixGrid(matrix))

#define hypre_StructMatrixBox(matrix, b) \
hypre_BoxArrayBox(hypre_StructMatrixDataSpace(matrix), b)

#define hypre_StructMatrixBoxData(matrix, b, s) \
(hypre_StructMatrixStencilData(matrix)[s] + hypre_StructMatrixDataIndices(matrix)[b][s])

#define hypre_StructMatrixBoxDataValue(matrix, b, s, index) \
(hypre_StructMatrixBoxData(matrix, b, s) + \
 hypre_BoxIndexRank(hypre_StructMatrixBox(matrix, b), index))

#define hypre_CCStructMatrixBoxDataValue(matrix, b, s, index) \
(hypre_StructMatrixBoxData(matrix, b, s) + \
 hypre_CCBoxIndexRank(hypre_StructMatrixBox(matrix, b), index))

#endif
