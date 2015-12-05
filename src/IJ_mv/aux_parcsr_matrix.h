/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/





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
   HYPRE_Int      local_num_rows;   /* defines number of rows on this processors */
   HYPRE_Int      local_num_cols;   /* defines number of cols of diag */

   HYPRE_Int      need_aux; /* if need_aux = 1, aux_j, aux_data are used to
			generate the parcsr matrix (default),
			for need_aux = 0, data is put directly into
			parcsr structure (requires the knowledge of
			offd_i and diag_i ) */

   HYPRE_Int     *row_length; /* row_length_diag[i] contains number of stored
				elements in i-th row */
   HYPRE_Int     *row_space; /* row_space_diag[i] contains space allocated to
				i-th row */
   HYPRE_Int    **aux_j;	/* contains collected column indices */
   double **aux_data; /* contains collected data */

   HYPRE_Int     *indx_diag; /* indx_diag[i] points to first empty space of portion
			 in diag_j , diag_data assigned to row i */  
   HYPRE_Int     *indx_offd; /* indx_offd[i] points to first empty space of portion
			 in offd_j , offd_data assigned to row i */  
   HYPRE_Int	    max_off_proc_elmts; /* length of off processor stash set for
					SetValues and AddTOValues */
   HYPRE_Int	    current_num_elmts; /* current no. of elements stored in stash */
   HYPRE_Int	    off_proc_i_indx; /* pointer to first empty space in 
				set_off_proc_i_set */
   HYPRE_Int     *off_proc_i; /* length 2*num_off_procs_elmts, contains info pairs
			(code, no. of elmts) where code contains global
			row no. if  SetValues, and (-global row no. -1)
			if  AddToValues*/
   HYPRE_Int     *off_proc_j; /* contains column indices */
   double  *off_proc_data; /* contains corresponding data */
   HYPRE_Int	    cancel_indx; /* number of elements that have to be deleted due
			   to setting values from another processor */
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
#define hypre_AuxParCSRMatrixCancelIndx(matrix)  ((matrix) -> cancel_indx)

#endif
