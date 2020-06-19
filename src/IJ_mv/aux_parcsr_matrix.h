/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

#if 0

#ifndef hypre_AUX_PARCSR_MATRIX_HEADER
#define hypre_AUX_PARCSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int       local_num_rows;   /* defines number of rows on this processors */
   HYPRE_Int       local_num_cols;   /* defines number of cols of diag */

   HYPRE_Int       need_aux; /* if need_aux = 1, aux_j, aux_data are used to
                                generate the parcsr matrix (default),
                                for need_aux = 0, data is put directly into
                                parcsr structure (requires the knowledge of
                                offd_i and diag_i ) */

   HYPRE_Int      *row_length; /* row_length_diag[i] contains number of stored
                                  elements in i-th row */
   HYPRE_Int      *row_space; /* row_space_diag[i] contains space allocated to
                                 i-th row */
   HYPRE_BigInt  **aux_j;	/* contains collected column indices */
   HYPRE_Complex **aux_data; /* contains collected data */

   HYPRE_Int      *indx_diag; /* indx_diag[i] points to first empty space of portion
                                 in diag_j , diag_data assigned to row i */  
   HYPRE_Int      *indx_offd; /* indx_offd[i] points to first empty space of portion
                                 in offd_j , offd_data assigned to row i */  
   HYPRE_Int	   max_off_proc_elmts; /* length of off processor stash set for
                                          SetValues and AddTOValues */
   HYPRE_Int	   current_num_elmts; /* current no. of elements stored in stash */
   HYPRE_Int	   off_proc_i_indx; /* pointer to first empty space in 
                                       set_off_proc_i_set */
   HYPRE_BigInt   *off_proc_i; /* length 2*num_off_procs_elmts, contains info pairs
                                  (code, no. of elmts) where code contains global
                                  row no., only used for AddToValues */
   HYPRE_BigInt   *off_proc_j; /* contains column indices */
   HYPRE_Complex  *off_proc_data; /* contains corresponding data */
} hypre_AuxParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParCSRMatrixLocalNumRows(matrix)  ((matrix) -> local_num_rows)
#define hypre_AuxParCSRMatrixLocalNumCols(matrix)  ((matrix) -> local_num_cols)

#define hypre_AuxParCSRMatrixNeedAux(matrix)   ((matrix) -> need_aux)
#define hypre_AuxParCSRMatrixRowLength(matrix) ((matrix) -> row_length)
#define hypre_AuxParCSRMatrixRowSpace(matrix)  ((matrix) -> row_space)
#define hypre_AuxParCSRMatrixAuxJ(matrix)      ((matrix) -> aux_j)
#define hypre_AuxParCSRMatrixAuxData(matrix)   ((matrix) -> aux_data)

#define hypre_AuxParCSRMatrixIndxDiag(matrix)  ((matrix) -> indx_diag)
#define hypre_AuxParCSRMatrixIndxOffd(matrix)  ((matrix) -> indx_offd)

#define hypre_AuxParCSRMatrixMaxOffProcElmts(matrix)  ((matrix) -> max_off_proc_elmts)
#define hypre_AuxParCSRMatrixCurrentNumElmts(matrix)  ((matrix) -> current_num_elmts)
#define hypre_AuxParCSRMatrixOffProcIIndx(matrix)  ((matrix) -> off_proc_i_indx)
#define hypre_AuxParCSRMatrixOffProcI(matrix)  ((matrix) -> off_proc_i)
#define hypre_AuxParCSRMatrixOffProcJ(matrix)  ((matrix) -> off_proc_j)
#define hypre_AuxParCSRMatrixOffProcData(matrix)  ((matrix) -> off_proc_data)
#define hypre_AuxParCSRMatrixAuxOffdJ(matrix)  ((matrix) -> aux_offd_j)
//#define hypre_AuxParCSRMatrixCancelIndx(matrix)  ((matrix) -> cancel_indx)

#endif

#endif

