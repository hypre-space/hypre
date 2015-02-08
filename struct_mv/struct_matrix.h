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
 * Rectangular matrices have different range and domain grids, which are defined
 * in terms of a common base grid and index space.  The range grid consists of a
 * coarsened subset of boxes in the base grid, as specified by the box numbers
 * in 'ran_boxnums' and the coarsening factor 'ran_stride'.  The domain grid is
 * similarly defined via 'dom_boxnums' and 'dom_stride'.  Either the range index
 * space is a coarsening of the domain index space or vice-versa.  The data
 * storage is dictated by the coarsest grid as indicated (for convenience) by
 * the two booleans 'range_is_coarse' and 'domain_is_coarse'.  The stencil
 * always represents a "row" stencil that operates on the domain grid and
 * produces a value on the range grid.  The data interface and accessor macros
 * are also row-stencil based, regardless of the underlying storage.  Each
 * stencil entry can have either constant or variable coefficients as indicated
 * by the stencil-sized array 'constant'.
 *
 * The 'data' pointer below has space at the beginning for constant stencil
 * coefficient values followed by the stored variable coefficient values.
 * Accessing coefficients is done via 'data_indices' through the interface
 * routine hypre_StructMatrixBoxData().  The number of boxes in data_boxes,
 * data_space, and data_indices is the same as in the base grid, even though
 * both ran_nboxes and dom_nboxes may be smaller.
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructMatrix_struct
{
   MPI_Comm              comm;

   hypre_StructGrid     *grid;          /* Base grid */
   HYPRE_Int             ran_nboxes;    /* Range grid number of boxes */
   HYPRE_Int            *ran_boxnums;   /* Range grid boxnums in base grid */
   hypre_Index           ran_stride;    /* Range grid coarsening stride */
   HYPRE_Int             dom_nboxes;    /* Domain grid number of boxes */
   HYPRE_Int            *dom_boxnums;   /* Domain grid boxnums in base grid */
   hypre_Index           dom_stride;    /* Domain grid coarsening stride */

   hypre_StructStencil  *user_stencil;
   hypre_StructStencil  *stencil;
   HYPRE_Int            *constant;      /* Which stencil entries are constant? */

   HYPRE_Complex        *data;          /* Pointer to matrix data */
   hypre_BoxArray       *data_space;    /* Layout of data (coarse index space) */
   hypre_BoxArray       *data_boxes;    /* Data extents on fine index space */
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
   HYPRE_Int             range_is_coarse;   /* 1 -> the range is coarse */
   HYPRE_Int             domain_is_coarse;  /* 1 -> the domain is coarse */
   HYPRE_Int             constant_coefficient;  /* RDF: Phase this out in favor
                                                   of 'constant' array above.
                                                   Values can be {0, 1, 2} ->
                                                   {variable, constant, constant
                                                   with variable diagonal} */
   HYPRE_Int             symmetric;      /* Is the matrix symmetric */
   HYPRE_Int            *symm_entries;   /* Which entries are "symmetric" */
   HYPRE_Int             num_ghost[2*HYPRE_MAXDIM];  /* Min num ghost layers */
   HYPRE_Int             add_ghost[2*HYPRE_MAXDIM];  /* Additional ghost layers */
                      
   HYPRE_Int             global_size;  /* Total number of nonzero coeffs */

   hypre_CommPkg        *comm_pkg;     /* Info on how to update ghost data */

   HYPRE_Int             ref_count;

   /* Information needed to Restore() after Resize() */
   HYPRE_Complex        *save_data;
   hypre_BoxArray       *save_data_space;
   HYPRE_Int             save_data_size;

} hypre_StructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_StructMatrixComm(matrix)          ((matrix) -> comm)
#define hypre_StructMatrixGrid(matrix)          ((matrix) -> grid)
#define hypre_StructMatrixRanNBoxes(matrix)     ((matrix) -> ran_nboxes)
#define hypre_StructMatrixRanBoxnums(matrix)    ((matrix) -> ran_boxnums)
#define hypre_StructMatrixRanStride(matrix)     ((matrix) -> ran_stride)
#define hypre_StructMatrixDomNBoxes(matrix)     ((matrix) -> dom_nboxes)
#define hypre_StructMatrixDomBoxnums(matrix)    ((matrix) -> dom_boxnums)
#define hypre_StructMatrixDomStride(matrix)     ((matrix) -> dom_stride)
#define hypre_StructMatrixUserStencil(matrix)   ((matrix) -> user_stencil)
#define hypre_StructMatrixStencil(matrix)       ((matrix) -> stencil)
#define hypre_StructMatrixConstant(matrix)      ((matrix) -> constant)
#define hypre_StructMatrixConstEntry(matrix, s) ((matrix) -> constant[s])
#define hypre_StructMatrixData(matrix)          ((matrix) -> data)
#define hypre_StructMatrixDataSpace(matrix)     ((matrix) -> data_space)
#define hypre_StructMatrixDataBoxes(matrix)     ((matrix) -> data_boxes)
#define hypre_StructMatrixDataAlloced(matrix)   ((matrix) -> data_alloced)
#define hypre_StructMatrixDataSize(matrix)      ((matrix) -> data_size)
#define hypre_StructMatrixDataIndices(matrix)   ((matrix) -> data_indices)
#define hypre_StructMatrixVDataOffset(matrix)   ((matrix) -> vdata_offset)
#define hypre_StructMatrixNumValues(matrix)     ((matrix) -> num_values)
#define hypre_StructMatrixNumCValues(matrix)    ((matrix) -> num_cvalues)
#define hypre_StructMatrixRangeIsCoarse(matrix) ((matrix) -> range_is_coarse)
#define hypre_StructMatrixDomainIsCoarse(matrix)((matrix) -> domain_is_coarse)
#define hypre_StructMatrixConstantCoefficient(matrix) ((matrix) -> constant_coefficient)
#define hypre_StructMatrixSymmetric(matrix)     ((matrix) -> symmetric)
#define hypre_StructMatrixSymmEntries(matrix)   ((matrix) -> symm_entries)
#define hypre_StructMatrixNumGhost(matrix)      ((matrix) -> num_ghost)
#define hypre_StructMatrixAddGhost(matrix)      ((matrix) -> add_ghost)
#define hypre_StructMatrixGlobalSize(matrix)    ((matrix) -> global_size)
#define hypre_StructMatrixCommPkg(matrix)       ((matrix) -> comm_pkg)
#define hypre_StructMatrixRefCount(matrix)      ((matrix) -> ref_count)

#define hypre_StructMatrixSaveData(matrix)      ((matrix) -> save_data)
#define hypre_StructMatrixSaveDataSpace(matrix) ((matrix) -> save_data_space)
#define hypre_StructMatrixSaveDataSize(matrix)  ((matrix) -> save_data_size)

#define hypre_StructMatrixNDim(matrix) \
hypre_StructGridNDim(hypre_StructMatrixGrid(matrix))

#define hypre_StructMatrixVData(matrix) \
(hypre_StructMatrixData(matrix) + hypre_StructMatrixVDataOffset(matrix))

#define hypre_StructMatrixBoxData(matrix, b, s) \
(hypre_StructMatrixData(matrix) + hypre_StructMatrixDataIndices(matrix)[b][s])

#endif
