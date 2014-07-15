/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the hypre_StructMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_MATRIX_HEADER
#define hypre_STRUCT_MATRIX_HEADER

#include <assert.h>

/*--------------------------------------------------------------------------
 * hypre_StructMatrix:
 *
 * Rectangular matrices are supported by allowing different range and domain
 * grids.  Either the range is a coarsening of the domain or vice-versa, and
 * both the domain and range are coarsenings of some common index space with
 * coarsening factors 'rmap' and 'dmap' respectively.  The data storage is
 * dictated by the coarsest grid, and the boolean 'domain_is_coarse' indicates
 * whether that happens to be the domain or the range grid.  The stencil,
 * however, always represents a "row" stencil that operates on the domain grid
 * and produces a value on the range grid.  The data interface and accessor
 * macros are also row-stencil based, regardless of the underlying storage.
 * Each stencil entry can have either constant or variable coefficients as
 * indicated by the stencil-sized array 'constant'.
 *
 * The 'data' pointer below has space at the beginning for constant stencil
 * coefficient values followed by the stored variable coefficient values.
 * Accessing coefficients is done via 'data_indices' through the interface
 * routine hypre_StructMatrixBoxData().
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructMatrix_struct
{
   MPI_Comm              comm;

   hypre_StructGrid     *grid;
   hypre_StructGrid     *domain_grid;   /* Same as grid by default */
   hypre_StructStencil  *user_stencil;
   hypre_StructStencil  *stencil;
   HYPRE_Int            *constant;      /* Which stencil entries are constant? */
   hypre_Index           rmap, dmap;    /* Range and domain coarsening maps */

   hypre_BoxArray       *data_boxes;    /* Original fine data space */
   hypre_BoxArray       *data_space;    /* Mapped coarse data space */

   HYPRE_Complex        *data;          /* Pointer to matrix data */
   HYPRE_Int             data_alloced;  /* Boolean used for freeing data */
   HYPRE_Int             data_size;     /* Size of matrix data */
   HYPRE_Int           **data_indices;  /* Num boxes by stencil-size array of
                                           indices into the data array.
                                           data_indices[b][s] is the starting
                                           index of matrix data corresponding to
                                           box b and stencil coefficient s */
   HYPRE_Int             vdata_offset;  /* Offset to variable-coeff matrix data */
   HYPRE_Int             num_values;    /* Number of "stored" variable coeffs */
   HYPRE_Int             num_cvalues;   /* Number of "stored" constant coeffs */
   HYPRE_Int             domain_is_coarse;  /* 1 -> the domain is coarse */
   HYPRE_Int             constant_coefficient;  /* RDF: Phase this out in favor
                                                   of 'constant' array above.
                                                   Values can be {0, 1, 2} ->
                                                   {variable, constant, constant
                                                   with variable diagonal} */
   HYPRE_Int             symmetric;      /* Is the matrix symmetric */
   HYPRE_Int            *symm_elements;  /* Which elements are "symmetric" */
   HYPRE_Int             num_ghost[2*HYPRE_MAXDIM];  /* Num ghost layers in each
                                                      * direction */
                      
   HYPRE_Int             global_size;  /* Total number of nonzero coeffs */

   hypre_CommPkg        *comm_pkg;     /* Info on how to update ghost data */

   HYPRE_Int             ref_count;

} hypre_StructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_StructMatrixComm(matrix)          ((matrix) -> comm)
#define hypre_StructMatrixGrid(matrix)          ((matrix) -> grid)
#define hypre_StructMatrixDomainGrid(matrix)    ((matrix) -> domain_grid)
#define hypre_StructMatrixUserStencil(matrix)   ((matrix) -> user_stencil)
#define hypre_StructMatrixStencil(matrix)       ((matrix) -> stencil)
#define hypre_StructMatrixConstant(matrix)      ((matrix) -> constant)
#define hypre_StructMatrixConstEntry(matrix, s) ((matrix) -> constant[s])
#define hypre_StructMatrixRMap(matrix)          ((matrix) -> rmap)
#define hypre_StructMatrixDMap(matrix)          ((matrix) -> dmap)
#define hypre_StructMatrixDataBoxes(matrix)     ((matrix) -> data_boxes)
#define hypre_StructMatrixDataSpace(matrix)     ((matrix) -> data_space)
#define hypre_StructMatrixData(matrix)          ((matrix) -> data)
#define hypre_StructMatrixDataAlloced(matrix)   ((matrix) -> data_alloced)
#define hypre_StructMatrixDataSize(matrix)      ((matrix) -> data_size)
#define hypre_StructMatrixDataIndices(matrix)   ((matrix) -> data_indices)
#define hypre_StructMatrixVDataOffset(matrix)   ((matrix) -> vdata_offset)
#define hypre_StructMatrixNumValues(matrix)     ((matrix) -> num_values)
#define hypre_StructMatrixNumCValues(matrix)    ((matrix) -> num_cvalues)
#define hypre_StructMatrixDomainIsCoarse(matrix)((matrix) -> domain_is_coarse)
#define hypre_StructMatrixConstantCoefficient(matrix) ((matrix) -> constant_coefficient)
#define hypre_StructMatrixSymmetric(matrix)     ((matrix) -> symmetric)
#define hypre_StructMatrixSymmElements(matrix)  ((matrix) -> symm_elements)
#define hypre_StructMatrixNumGhost(matrix)      ((matrix) -> num_ghost)
#define hypre_StructMatrixGlobalSize(matrix)    ((matrix) -> global_size)
#define hypre_StructMatrixCommPkg(matrix)       ((matrix) -> comm_pkg)
#define hypre_StructMatrixRefCount(matrix)      ((matrix) -> ref_count)

#define hypre_StructMatrixNDim(matrix) \
hypre_StructGridNDim(hypre_StructMatrixGrid(matrix))

#define hypre_StructMatrixVData(matrix) \
(hypre_StructMatrixData(matrix) + hypre_StructMatrixVDataOffset(matrix))

#define hypre_StructMatrixBoxData(matrix, b, s) \
(hypre_StructMatrixData(matrix) + hypre_StructMatrixDataIndices(matrix)[b][s])

#endif
