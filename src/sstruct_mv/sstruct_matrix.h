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
 * Header info for the hypre_SStructMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_SSTRUCT_MATRIX_HEADER
#define hypre_SSTRUCT_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * hypre_SStructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructPMatrix_struct
{
   MPI_Comm                comm;
   hypre_SStructPGrid     *pgrid;
   hypre_SStructStencil  **stencils;     /* nvar array of stencils */

   HYPRE_Int               nvars;
   HYPRE_Int             **smaps;
   hypre_StructStencil  ***sstencils;    /* nvar x nvar array of sstencils */
   hypre_StructMatrix   ***smatrices;    /* nvar x nvar array of smatrices */
   HYPRE_Int             **symmetric;    /* Stencil entries symmetric?
                                          * (nvar x nvar array) */
   HYPRE_Int             **num_centries; /* (nvar x nvar) array */
   HYPRE_Int            ***centries;     /* (nvar x nvar x sentries_size) array constant entries */
   hypre_Index             dom_stride;   /* domain grid stride */
   hypre_Index             ran_stride;   /* range grid stride */

   /* temporary storage for SetValues routines */
   HYPRE_Int               sentries_size;
   HYPRE_Int              *sentries;

   HYPRE_Int               accumulated;  /* AddTo values accumulated? */

   HYPRE_Int               ref_count;

} hypre_SStructPMatrix;


/*--------------------------------------------------------------------------
 * hypre_SStructMatrix:
 *
 * - Storage of parts id dictated by the coarse grid
 *--------------------------------------------------------------------------*/
typedef struct hypre_SStructMatrix_struct
{
   MPI_Comm                comm;
   HYPRE_Int               ndim;
   HYPRE_Int            ***splits;        /* S/U-matrix split for each stencil */
   hypre_SStructGraph     *graph;

   /* S-matrix info */
   HYPRE_Int               nparts;
   HYPRE_Int              *part_ids;     /* (nparts) array */
   hypre_SStructPMatrix  **pmatrices;
   HYPRE_Int            ***symmetric;    /* Stencil entries symmetric?
                                          * (nparts x nvar x nvar array) */
   HYPRE_Int            ***num_centries; /* (nparts x nvar x nvar) array */
   HYPRE_Int           ****centries;     /* (nparts x nvar x nvar x entries_size) array */
   hypre_Index            *dom_stride;   /* (nparts) array of domain stride */
   hypre_Index            *ran_stride;   /* (nparts) array of range stride */

   /* U-matrix info */
   HYPRE_IJMatrix          ijmatrix;
   hypre_ParCSRMatrix     *parcsrmatrix;

   /* temporary storage for SetValues routines */
   HYPRE_Int               entries_size;
   HYPRE_Int              *Sentries;
   HYPRE_Int              *Uentries;
   HYPRE_Int              *tmp_col_coords;
   HYPRE_Complex          *tmp_coeffs;

   HYPRE_Int               ns_symmetric; /* Non-stencil entries symmetric? */
   HYPRE_Int               global_size;  /* Total number of nonzero coeffs */

   HYPRE_Int               ref_count;

   HYPRE_Int               dom_ghlocal_size; /* Number of unknowns in the domain grid
                                                including ghosts */
   HYPRE_Int               ran_ghlocal_size; /* Number of unknowns in the range grid
                                                including ghosts */
   HYPRE_Int               dom_ghstart_rank; /* Start rank in the domain grid
                                                including ghosts */
   HYPRE_Int               ran_ghstart_rank; /* Start rank in the range grid
                                                including ghosts */
   /* GEC0902   adding an object type to the matrix  */
   HYPRE_Int               object_type;
} hypre_SStructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_SStructMatrixComm(mat)           ((mat) -> comm)
#define hypre_SStructMatrixNDim(mat)           ((mat) -> ndim)
#define hypre_SStructMatrixGraph(mat)          ((mat) -> graph)
#define hypre_SStructMatrixSplits(mat)         ((mat) -> splits)
#define hypre_SStructMatrixSplit(mat, p, v)    ((mat) -> splits[p][v])
#define hypre_SStructMatrixNParts(mat)         ((mat) -> nparts)
#define hypre_SStructMatrixPartIDs(mat)        ((mat) -> part_ids)
#define hypre_SStructMatrixPartID(mat, p)      ((mat) -> part_ids[p])
#define hypre_SStructMatrixPMatrices(mat)      ((mat) -> pmatrices)
#define hypre_SStructMatrixPMatrix(mat, part)  ((mat) -> pmatrices[part])
#define hypre_SStructMatrixSymmetric(mat)      ((mat) -> symmetric)
#define hypre_SStructMatrixNumCEntries(mat)    ((mat) -> num_centries)
#define hypre_SStructMatrixCEntries(mat)       ((mat) -> centries)
#define hypre_SStructMatrixDomainStride(mat)   ((mat) -> dom_stride)
#define hypre_SStructMatrixRangeStride(mat)    ((mat) -> ran_stride)
#define hypre_SStructMatrixIJMatrix(mat)       ((mat) -> ijmatrix)
#define hypre_SStructMatrixParCSRMatrix(mat)   ((mat) -> parcsrmatrix)
#define hypre_SStructMatrixEntriesSize(mat)    ((mat) -> entries_size)
#define hypre_SStructMatrixSEntries(mat)       ((mat) -> Sentries)
#define hypre_SStructMatrixUEntries(mat)       ((mat) -> Uentries)
#define hypre_SStructMatrixTmpColCoords(mat)   ((mat) -> tmp_col_coords)
#define hypre_SStructMatrixTmpCoeffs(mat)      ((mat) -> tmp_coeffs)
#define hypre_SStructMatrixNSSymmetric(mat)    ((mat) -> ns_symmetric)
#define hypre_SStructMatrixGlobalSize(mat)     ((mat) -> global_size)
#define hypre_SStructMatrixRefCount(mat)       ((mat) -> ref_count)
#define hypre_SStructMatrixDomGhlocalSize(mat) ((mat) -> dom_ghlocal_size)
#define hypre_SStructMatrixRanGhlocalSize(mat) ((mat) -> ran_ghlocal_size)
#define hypre_SStructMatrixDomGhstartRank(mat) ((mat) -> dom_ghstart_rank)
#define hypre_SStructMatrixRanGhstartRank(mat) ((mat) -> ran_ghstart_rank)
#define hypre_SStructMatrixObjectType(mat)     ((mat) -> object_type)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructPMatrix
 *--------------------------------------------------------------------------*/

#define hypre_SStructPMatrixComm(pmat)              ((pmat) -> comm)
#define hypre_SStructPMatrixPGrid(pmat)             ((pmat) -> pgrid)
#define hypre_SStructPMatrixNDim(pmat) \
hypre_SStructPGridNDim(hypre_SStructPMatrixPGrid(pmat))
#define hypre_SStructPMatrixStencils(pmat)          ((pmat) -> stencils)
#define hypre_SStructPMatrixNVars(pmat)             ((pmat) -> nvars)
#define hypre_SStructPMatrixStencil(pmat, var)      ((pmat) -> stencils[var])
#define hypre_SStructPMatrixSMaps(pmat)             ((pmat) -> smaps)
#define hypre_SStructPMatrixSMap(pmat, var)         ((pmat) -> smaps[var])
#define hypre_SStructPMatrixSStencils(pmat)         ((pmat) -> sstencils)
#define hypre_SStructPMatrixSStencil(pmat, vi, vj) \
((pmat) -> sstencils[vi][vj])
#define hypre_SStructPMatrixSMatrices(pmat)         ((pmat) -> smatrices)
#define hypre_SStructPMatrixSMatrix(pmat, vi, vj)  \
((pmat) -> smatrices[vi][vj])
#define hypre_SStructPMatrixSymmetric(pmat)         ((pmat) -> symmetric)
#define hypre_SStructPMatrixNumCEntries(pmat)       ((pmat) -> num_centries)
#define hypre_SStructPMatrixCEntries(pmat)          ((pmat) -> centries)
#define hypre_SStructPMatrixDomainStride(pmat)      ((pmat) -> dom_stride)
#define hypre_SStructPMatrixRangeStride(pmat)       ((pmat) -> ran_stride)
#define hypre_SStructPMatrixSEntriesSize(pmat)      ((pmat) -> sentries_size)
#define hypre_SStructPMatrixSEntries(pmat)          ((pmat) -> sentries)
#define hypre_SStructPMatrixAccumulated(pmat)       ((pmat) -> accumulated)
#define hypre_SStructPMatrixRefCount(pmat)          ((pmat) -> ref_count)

#endif
