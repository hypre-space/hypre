/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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

typedef struct
{
   MPI_Comm                comm;
   hypre_SStructPGrid     *pgrid;
   hypre_SStructStencil  **stencils;     /* nvar array of stencils */

   int                     nvars;
   int                   **smaps;
   hypre_StructStencil  ***sstencils;    /* nvar x nvar array of sstencils */
   hypre_StructMatrix   ***smatrices;    /* nvar x nvar array of smatrices */

   /* temporary storage for SetValues routines */
   int                     sentries_size;
   int                    *sentries;

   int                     complex;      /* Is the matrix complex */

   int                     ref_count;

} hypre_SStructPMatrix;

typedef struct hypre_SStructMatrix_struct
{
   MPI_Comm                comm;
   int                     ndim;
   hypre_SStructGraph     *graph;
   int                  ***splits;   /* S/U-matrix split for each stencil */

   /* S-matrix info */
   int                     nparts;
   hypre_SStructPMatrix  **pmatrices;

   /* U-matrix info */
   HYPRE_IJMatrix          ijmatrix;
   hypre_ParCSRMatrix     *parcsrmatrix;
                         
   /* temporary storage for SetValues routines */
   int                     entries_size;
   int                    *Sentries;
   int                    *Uentries;
   int                    *tmp_col_coords;
   double                 *tmp_coeffs;

   int                     symmetric;    /* Is the matrix symmetric */
   int                     complex;      /* Is the matrix complex */
   int                     global_size;  /* Total number of nonzero coeffs */

   int                     ref_count;

} hypre_SStructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_SStructMatrixComm(mat)           ((mat) -> comm)
#define hypre_SStructMatrixNDim(mat)           ((mat) -> ndim)
#define hypre_SStructMatrixGraph(mat)          ((mat) -> graph)
#define hypre_SStructMatrixSplits(mat)         ((mat)-> splits)
#define hypre_SStructMatrixSplit(mat, p, v)    ((mat) -> splits[p][v])
#define hypre_SStructMatrixNParts(mat)         ((mat) -> nparts)
#define hypre_SStructMatrixPMatrices(mat)      ((mat) -> pmatrices)
#define hypre_SStructMatrixPMatrix(mat, part)  ((mat) -> pmatrices[part])
#define hypre_SStructMatrixIJMatrix(mat)       ((mat) -> ijmatrix)
#define hypre_SStructMatrixParCSRMatrix(mat)   ((mat) -> parcsrmatrix)
#define hypre_SStructMatrixEntriesSize(mat)    ((mat) -> entries_size)
#define hypre_SStructMatrixSEntries(mat)       ((mat) -> Sentries)
#define hypre_SStructMatrixUEntries(mat)       ((mat) -> Uentries)
#define hypre_SStructMatrixTmpColCoords(mat)   ((mat) -> tmp_col_coords)
#define hypre_SStructMatrixTmpCoeffs(mat)      ((mat) -> tmp_coeffs)
#define hypre_SStructMatrixSymmetric(mat)      ((mat) -> symmetric)
#define hypre_SStructMatrixComplex(mat)        ((mat) -> complex)
#define hypre_SStructMatrixGlobalSize(mat)     ((mat) -> global_size)
#define hypre_SStructMatrixRefCount(mat)       ((mat) -> ref_count)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructPMatrix
 *--------------------------------------------------------------------------*/

#define hypre_SStructPMatrixComm(pmat)              ((pmat) -> comm)
#define hypre_SStructPMatrixPGrid(pmat)             ((pmat) -> pgrid)
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
#define hypre_SStructPMatrixSEntriesSize(pmat)      ((pmat) -> sentries_size)
#define hypre_SStructPMatrixSEntries(pmat)          ((pmat) -> sentries)
#define hypre_SStructPMatrixComplex(pmat)           ((pmat) -> complex)
#define hypre_SStructPMatrixRefCount(pmat)          ((pmat) -> ref_count)

#endif
