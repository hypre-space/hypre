/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Auxiliary Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_AUX_PARCSR_MATRIX_HEADER
#define hypre_AUX_PARCSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int            local_num_rows;    /* defines number of rows on this processor */
   HYPRE_Int            local_num_rownnz;  /* defines number of nonzero rows on this processor */
   HYPRE_Int            local_num_cols;    /* defines number of cols of diag */

   HYPRE_Int            need_aux;                /* if need_aux = 1, aux_j, aux_data are used to
                                                    generate the parcsr matrix (default),
                                                    for need_aux = 0, data is put directly into
                                                    parcsr structure (requires the knowledge of
                                                    offd_i and diag_i ) */

   HYPRE_Int           *rownnz;                  /* row_nnz[i] contains the i-th nonzero row id */
   HYPRE_Int           *row_length;              /* row_length[i] contains number of stored
                                                    elements in i-th row */
   HYPRE_Int           *row_space;               /* row_space[i] contains space allocated to
                                                    i-th row */

   HYPRE_Int           *diag_sizes;              /* user input row lengths of diag */
   HYPRE_Int           *offd_sizes;              /* user input row lengths of diag */

   HYPRE_BigInt       **aux_j;                   /* contains collected column indices */
   HYPRE_Complex      **aux_data;                /* contains collected data */

   HYPRE_Int           *indx_diag;               /* indx_diag[i] points to first empty space of portion
                                                    in diag_j , diag_data assigned to row i */
   HYPRE_Int           *indx_offd;               /* indx_offd[i] points to first empty space of portion
                                                    in offd_j , offd_data assigned to row i */

   HYPRE_Int            max_off_proc_elmts;      /* length of off processor stash set for
                                                    SetValues and AddTOValues */
   HYPRE_Int            current_off_proc_elmts;  /* current no. of elements stored in stash */
   HYPRE_Int            off_proc_i_indx;         /* pointer to first empty space in
                                                    set_off_proc_i_set */
   HYPRE_BigInt        *off_proc_i;              /* length 2*num_off_procs_elmts, contains info pairs
                                                    (code, no. of elmts) where code contains global
                                                    row no. if  SetValues, and (-global row no. -1)
                                                    if  AddToValues */
   HYPRE_BigInt        *off_proc_j;              /* contains column indices
                                                  * ( global col id.)    if SetValues,
                                                  * (-global col id. -1) if AddToValues */
   HYPRE_Complex       *off_proc_data;           /* contains corresponding data */

   HYPRE_MemoryLocation memory_location;

#if defined(HYPRE_USING_GPU)
   HYPRE_BigInt         max_stack_elmts;
   HYPRE_BigInt         current_stack_elmts;
   HYPRE_BigInt        *stack_i;
   HYPRE_BigInt        *stack_j;
   HYPRE_Complex       *stack_data;
   char                *stack_sora;              /* Set (1) or Add (0) */
   HYPRE_Int            usr_on_proc_elmts;       /* user given num elmt on-proc */
   HYPRE_Int            usr_off_proc_elmts;      /* user given num elmt off-proc */
   HYPRE_BigInt         init_alloc_factor;
   HYPRE_BigInt         grow_factor;
#endif
} hypre_AuxParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParCSRMatrixLocalNumRows(matrix)         ((matrix) -> local_num_rows)
#define hypre_AuxParCSRMatrixLocalNumRownnz(matrix)       ((matrix) -> local_num_rownnz)
#define hypre_AuxParCSRMatrixLocalNumCols(matrix)         ((matrix) -> local_num_cols)

#define hypre_AuxParCSRMatrixNeedAux(matrix)              ((matrix) -> need_aux)
#define hypre_AuxParCSRMatrixRownnz(matrix)               ((matrix) -> rownnz)
#define hypre_AuxParCSRMatrixRowLength(matrix)            ((matrix) -> row_length)
#define hypre_AuxParCSRMatrixRowSpace(matrix)             ((matrix) -> row_space)
#define hypre_AuxParCSRMatrixAuxJ(matrix)                 ((matrix) -> aux_j)
#define hypre_AuxParCSRMatrixAuxData(matrix)              ((matrix) -> aux_data)

#define hypre_AuxParCSRMatrixIndxDiag(matrix)             ((matrix) -> indx_diag)
#define hypre_AuxParCSRMatrixIndxOffd(matrix)             ((matrix) -> indx_offd)

#define hypre_AuxParCSRMatrixDiagSizes(matrix)            ((matrix) -> diag_sizes)
#define hypre_AuxParCSRMatrixOffdSizes(matrix)            ((matrix) -> offd_sizes)

#define hypre_AuxParCSRMatrixMaxOffProcElmts(matrix)      ((matrix) -> max_off_proc_elmts)
#define hypre_AuxParCSRMatrixCurrentOffProcElmts(matrix)  ((matrix) -> current_off_proc_elmts)
#define hypre_AuxParCSRMatrixOffProcIIndx(matrix)         ((matrix) -> off_proc_i_indx)
#define hypre_AuxParCSRMatrixOffProcI(matrix)             ((matrix) -> off_proc_i)
#define hypre_AuxParCSRMatrixOffProcJ(matrix)             ((matrix) -> off_proc_j)
#define hypre_AuxParCSRMatrixOffProcData(matrix)          ((matrix) -> off_proc_data)

#define hypre_AuxParCSRMatrixMemoryLocation(matrix)       ((matrix) -> memory_location)

#if defined(HYPRE_USING_GPU)
#define hypre_AuxParCSRMatrixMaxStackElmts(matrix)        ((matrix) -> max_stack_elmts)
#define hypre_AuxParCSRMatrixCurrentStackElmts(matrix)    ((matrix) -> current_stack_elmts)
#define hypre_AuxParCSRMatrixStackI(matrix)               ((matrix) -> stack_i)
#define hypre_AuxParCSRMatrixStackJ(matrix)               ((matrix) -> stack_j)
#define hypre_AuxParCSRMatrixStackData(matrix)            ((matrix) -> stack_data)
#define hypre_AuxParCSRMatrixStackSorA(matrix)            ((matrix) -> stack_sora)
#define hypre_AuxParCSRMatrixUsrOnProcElmts(matrix)       ((matrix) -> usr_on_proc_elmts)
#define hypre_AuxParCSRMatrixUsrOffProcElmts(matrix)      ((matrix) -> usr_off_proc_elmts)
#define hypre_AuxParCSRMatrixInitAllocFactor(matrix)      ((matrix) -> init_alloc_factor)
#define hypre_AuxParCSRMatrixGrowFactor(matrix)           ((matrix) -> grow_factor)
#endif

#endif /* #ifndef hypre_AUX_PARCSR_MATRIX_HEADER */
