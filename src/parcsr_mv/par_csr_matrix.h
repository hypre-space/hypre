/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Header info for Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_PAR_CSR_MATRIX_HEADER
#define hypre_PAR_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm		comm;

   HYPRE_Int     		global_num_rows;
   HYPRE_Int     		global_num_cols;
   HYPRE_Int			first_row_index;
   HYPRE_Int			first_col_diag;
   /* need to know entire local range in case row_starts and col_starts 
      are null  (i.e., bgl) AHB 6/05*/
   HYPRE_Int                  last_row_index;
   HYPRE_Int                  last_col_diag;

   hypre_CSRMatrix	*diag;
   hypre_CSRMatrix	*offd;
   HYPRE_Int			*col_map_offd; 
	/* maps columns of offd to global columns */
   HYPRE_Int 			*row_starts; 
	/* array of length num_procs+1, row_starts[i] contains the 
	   global number of the first row on proc i,  
	   first_row_index = row_starts[my_id],
	   row_starts[num_procs] = global_num_rows */
   HYPRE_Int 			*col_starts;
	/* array of length num_procs+1, col_starts[i] contains the 
	   global number of the first column of diag on proc i,  
	   first_col_diag = col_starts[my_id],
	   col_starts[num_procs] = global_num_cols */

   hypre_ParCSRCommPkg	*comm_pkg;
   hypre_ParCSRCommPkg	*comm_pkgT;
   
   /* Does the ParCSRMatrix create/destroy `diag', `offd', `col_map_offd'? */
   HYPRE_Int      owns_data;
   /* Does the ParCSRMatrix create/destroy `row_starts', `col_starts'? */
   HYPRE_Int      owns_row_starts;
   HYPRE_Int      owns_col_starts;

   HYPRE_Int      num_nonzeros;
   double   d_num_nonzeros;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   HYPRE_Int     *rowindices;
   double  *rowvalues;
   HYPRE_Int      getrowactive;

   hypre_IJAssumedPart *assumed_partition; /* only populated if no_global_partition option
                                              is used (compile-time option)*/


} hypre_ParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRMatrixComm(matrix)		  ((matrix) -> comm)
#define hypre_ParCSRMatrixGlobalNumRows(matrix)   ((matrix) -> global_num_rows)
#define hypre_ParCSRMatrixGlobalNumCols(matrix)   ((matrix) -> global_num_cols)
#define hypre_ParCSRMatrixFirstRowIndex(matrix)   ((matrix) -> first_row_index)
#define hypre_ParCSRMatrixFirstColDiag(matrix)    ((matrix) -> first_col_diag)
#define hypre_ParCSRMatrixLastRowIndex(matrix)    ((matrix) -> last_row_index)
#define hypre_ParCSRMatrixLastColDiag(matrix)     ((matrix) -> last_col_diag)
#define hypre_ParCSRMatrixDiag(matrix)  	  ((matrix) -> diag)
#define hypre_ParCSRMatrixOffd(matrix)  	  ((matrix) -> offd)
#define hypre_ParCSRMatrixColMapOffd(matrix)  	  ((matrix) -> col_map_offd)
#define hypre_ParCSRMatrixRowStarts(matrix)       ((matrix) -> row_starts)
#define hypre_ParCSRMatrixColStarts(matrix)       ((matrix) -> col_starts)
#define hypre_ParCSRMatrixCommPkg(matrix)	  ((matrix) -> comm_pkg)
#define hypre_ParCSRMatrixCommPkgT(matrix)	  ((matrix) -> comm_pkgT)
#define hypre_ParCSRMatrixOwnsData(matrix)        ((matrix) -> owns_data)
#define hypre_ParCSRMatrixOwnsRowStarts(matrix)   ((matrix) -> owns_row_starts)
#define hypre_ParCSRMatrixOwnsColStarts(matrix)   ((matrix) -> owns_col_starts)
#define hypre_ParCSRMatrixNumRows(matrix) \
hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(matrix))
#define hypre_ParCSRMatrixNumCols(matrix) \
hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(matrix))
#define hypre_ParCSRMatrixNumNonzeros(matrix)     ((matrix) -> num_nonzeros)
#define hypre_ParCSRMatrixDNumNonzeros(matrix)    ((matrix) -> d_num_nonzeros)
#define hypre_ParCSRMatrixRowindices(matrix)      ((matrix) -> rowindices)
#define hypre_ParCSRMatrixRowvalues(matrix)       ((matrix) -> rowvalues)
#define hypre_ParCSRMatrixGetrowactive(matrix)    ((matrix) -> getrowactive)
#define hypre_ParCSRMatrixAssumedPartition(matrix) ((matrix) -> assumed_partition)



/*--------------------------------------------------------------------------
 * Parallel CSR Boolean Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;
   HYPRE_Int                   global_num_rows;
   HYPRE_Int                   global_num_cols;
   HYPRE_Int                   first_row_index;
   HYPRE_Int                   first_col_diag;
   HYPRE_Int                   last_row_index;
   HYPRE_Int                   last_col_diag;
   hypre_CSRBooleanMatrix *diag;
   hypre_CSRBooleanMatrix *offd;
   HYPRE_Int	                *col_map_offd; 
   HYPRE_Int 	                *row_starts; 
   HYPRE_Int 	                *col_starts;
   hypre_ParCSRCommPkg  *comm_pkg;
   hypre_ParCSRCommPkg  *comm_pkgT;
   HYPRE_Int                   owns_data;
   HYPRE_Int                   owns_row_starts;
   HYPRE_Int                   owns_col_starts;
   HYPRE_Int                   num_nonzeros;
   HYPRE_Int                  *rowindices;
   HYPRE_Int                   getrowactive;

} hypre_ParCSRBooleanMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Boolean Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRBooleanMatrix_Get_Comm(matrix)          ((matrix)->comm)
#define hypre_ParCSRBooleanMatrix_Get_GlobalNRows(matrix)   ((matrix)->global_num_rows)
#define hypre_ParCSRBooleanMatrix_Get_GlobalNCols(matrix)   ((matrix)->global_num_cols)
#define hypre_ParCSRBooleanMatrix_Get_StartRow(matrix)      ((matrix)->first_row_index)
#define hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(matrix) ((matrix)->first_row_index)
#define hypre_ParCSRBooleanMatrix_Get_FirstColDiag(matrix)  ((matrix)->first_col_diag)
#define hypre_ParCSRBooleanMatrix_Get_LastRowIndex(matrix)  ((matrix)->last_row_index)
#define hypre_ParCSRBooleanMatrix_Get_LastColDiag(matrix)   ((matrix)->last_col_diag)
#define hypre_ParCSRBooleanMatrix_Get_Diag(matrix)          ((matrix)->diag)
#define hypre_ParCSRBooleanMatrix_Get_Offd(matrix)          ((matrix)->offd)
#define hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix)    ((matrix)->col_map_offd)
#define hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix)     ((matrix)->row_starts)
#define hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix)     ((matrix)->col_starts)
#define hypre_ParCSRBooleanMatrix_Get_CommPkg(matrix)       ((matrix)->comm_pkg)
#define hypre_ParCSRBooleanMatrix_Get_CommPkgT(matrix)      ((matrix)->comm_pkgT)
#define hypre_ParCSRBooleanMatrix_Get_OwnsData(matrix)      ((matrix)->owns_data)
#define hypre_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) ((matrix)->owns_row_starts)
#define hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) ((matrix)->owns_col_starts)
#define hypre_ParCSRBooleanMatrix_Get_NRows(matrix)         ((matrix->diag->num_rows))
#define hypre_ParCSRBooleanMatrix_Get_NCols(matrix)         ((matrix->diag->num_cols))
#define hypre_ParCSRBooleanMatrix_Get_NNZ(matrix)           ((matrix)->num_nonzeros)
#define hypre_ParCSRBooleanMatrix_Get_Rowindices(matrix)    ((matrix)->rowindices)
#define hypre_ParCSRBooleanMatrix_Get_Getrowactive(matrix)  ((matrix)->getrowactive)

#endif
