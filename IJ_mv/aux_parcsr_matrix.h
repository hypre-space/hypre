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
   int      local_num_rows;   /* defines number of rows on this processors */
   int      local_num_cols;   /* defines number of cols of diag */

   int      need_aux; /* if need_aux = 1, aux_j, aux_data are used to
			generate the parcsr matrix (default),
			for need_aux = 0, data is put directly into
			parcsr structure (requires the knowledge of
			offd_i and diag_i ) */

   int     *row_length; /* row_length_diag[i] contains number of stored
				elements in i-th row */
   int     *row_space; /* row_space_diag[i] contains space allocated to
				i-th row */
   int    **aux_j;	/* contains collected column indices */
   double **aux_data; /* contains collected data */

   int     *indx_diag; /* indx_diag[i] points to first empty space of portion
			 in diag_j , diag_data assigned to row i */  
   int     *indx_offd; /* indx_offd[i] points to first empty space of portion
			 in offd_j , offd_data assigned to row i */  
   int	    max_off_proc_elmts_set; /* length of off processor stash for
					SetValues */
   int	    current_num_elmts_set; /* current no. of elements stored in stash */
   int	    off_proc_i_indx_set; /* pointer to first empty space in 
				set_off_proc_i_set */
   int     *off_proc_i_set; /* length 2*num_off_procs_elmts, contains info pairs
			(row no., no. of elmts) */
   int     *off_proc_j_set; /* contains column indices */
   double  *off_proc_data_set; /* contains corresponding data */
   int	    max_off_proc_elmts_add; /* length of off processor stash for
					SetValues */
   int	    current_num_elmts_add; /* current no. of elements stored in stash */
   int	    off_proc_i_indx_add; /* pointer to first empty space in 
				off_proc_i_add */
   int     *off_proc_i_add; /* length 2*num_off_procs_elmts, contains info pairs
			(row no., no. of elmts) */
   int     *off_proc_j_add; /* contains column indices */
   double  *off_proc_data_add; /* contains corresponding data */
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

#define hypre_AuxParCSRMatrixMaxOffProcElmtsSet(matrix)  ((matrix) -> max_off_proc_elmts_set)
#define hypre_AuxParCSRMatrixCurrentNumElmtsSet(matrix)  ((matrix) -> current_num_elmts_set)
#define hypre_AuxParCSRMatrixOffProcIIndxSet(matrix)  ((matrix) -> off_proc_i_indx_set)
#define hypre_AuxParCSRMatrixOffProcISet(matrix)  ((matrix) -> off_proc_i_set)
#define hypre_AuxParCSRMatrixOffProcJSet(matrix)  ((matrix) -> off_proc_j_set)
#define hypre_AuxParCSRMatrixOffProcDataSet(matrix)  ((matrix) -> off_proc_data_set)
#define hypre_AuxParCSRMatrixMaxOffProcElmtsAdd(matrix)  ((matrix) -> max_off_proc_elmts_add)
#define hypre_AuxParCSRMatrixCurrentNumElmtsAdd(matrix)  ((matrix) -> current_num_elmts_add)
#define hypre_AuxParCSRMatrixOffProcIIndxAdd(matrix)  ((matrix) -> off_proc_i_indx_add)
#define hypre_AuxParCSRMatrixOffProcIAdd(matrix)  ((matrix) -> off_proc_i_add)
#define hypre_AuxParCSRMatrixOffProcJAdd(matrix)  ((matrix) -> off_proc_j_add)
#define hypre_AuxParCSRMatrixOffProcDataAdd(matrix)  ((matrix) -> off_proc_data_add)

#endif
