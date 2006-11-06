/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Header info for Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

                                                                                                               
#ifndef hypre_PAR_CSR_BLOCK_MATRIX_HEADER
#define hypre_PAR_CSR_BLOCK_MATRIX_HEADER
                                                                                                               
#include "utilities.h"
#include "csr_block_matrix.h"
#include "parcsr_mv.h"
                                                                                                               
#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Parallel CSR Block Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm		comm;

   int     		global_num_rows;
   int     		global_num_cols;
   int			first_row_index;
   int			first_col_diag;

 /* need to know entire local range in case row_starts and col_starts 
    are null */ 
   int                  last_row_index;
   int                  last_col_diag;

   hypre_CSRBlockMatrix	*diag;
   hypre_CSRBlockMatrix	*offd;
   int			*col_map_offd; 
	/* maps columns of offd to global columns */
   int 			*row_starts; 
	/* array of length num_procs+1, row_starts[i] contains the 
	   global number of the first row on proc i,  
	   first_row_index = row_starts[my_id],
	   row_starts[num_procs] = global_num_rows */
   int 			*col_starts;
	/* array of length num_procs+1, col_starts[i] contains the 
	   global number of the first column of diag on proc i,  
	   first_col_diag = col_starts[my_id],
	   col_starts[num_procs] = global_num_cols */

   hypre_ParCSRCommPkg	*comm_pkg;
   hypre_ParCSRCommPkg	*comm_pkgT;
   
   /* Does the ParCSRBlockMatrix create/destroy `diag', `offd', `col_map_offd'? */
   int      owns_data;
   /* Does the ParCSRBlockMatrix create/destroy `row_starts', `col_starts'? */
   int      owns_row_starts;
   int      owns_col_starts;

   int      num_nonzeros;
   double   d_num_nonzeros;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   int     *rowindices;
   double  *rowvalues;
   int      getrowactive;

   hypre_IJAssumedPart *assumed_partition; /* only populated if no_global_partition option
                                              is used (compile-time option)*/

} hypre_ParCSRBlockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRBlockMatrixComm(matrix)	       ((matrix)->comm)
#define hypre_ParCSRBlockMatrixGlobalNumRows(matrix)   ((matrix)->global_num_rows)
#define hypre_ParCSRBlockMatrixGlobalNumCols(matrix)   ((matrix)->global_num_cols)
#define hypre_ParCSRBlockMatrixFirstRowIndex(matrix)   ((matrix)->first_row_index)
#define hypre_ParCSRBlockMatrixFirstColDiag(matrix)    ((matrix)->first_col_diag)
#define hypre_ParCSRBlockMatrixLastRowIndex(matrix)    ((matrix) -> last_row_index)
#define hypre_ParCSRBlockMatrixLastColDiag(matrix)     ((matrix) -> last_col_diag)
#define hypre_ParCSRBlockMatrixBlockSize(matrix)       ((matrix)->diag->block_size)
#define hypre_ParCSRBlockMatrixDiag(matrix)  	       ((matrix) -> diag)
#define hypre_ParCSRBlockMatrixOffd(matrix)  	       ((matrix) -> offd)
#define hypre_ParCSRBlockMatrixColMapOffd(matrix)      ((matrix) -> col_map_offd)
#define hypre_ParCSRBlockMatrixRowStarts(matrix)       ((matrix) -> row_starts)
#define hypre_ParCSRBlockMatrixColStarts(matrix)       ((matrix) -> col_starts)
#define hypre_ParCSRBlockMatrixCommPkg(matrix)	       ((matrix) -> comm_pkg)
#define hypre_ParCSRBlockMatrixCommPkgT(matrix)	       ((matrix) -> comm_pkgT)
#define hypre_ParCSRBlockMatrixOwnsData(matrix)        ((matrix) -> owns_data)
#define hypre_ParCSRBlockMatrixOwnsRowStarts(matrix)   ((matrix) -> owns_row_starts)
#define hypre_ParCSRBlockMatrixOwnsColStarts(matrix)   ((matrix) -> owns_col_starts)
#define hypre_ParCSRBlockMatrixNumRows(matrix) \
hypre_CSRBlockMatrixNumRows(hypre_ParCSRBlockMatrixDiag(matrix))
#define hypre_ParCSRBlockMatrixNumCols(matrix) \
hypre_CSRBlockMatrixNumCols(hypre_ParCSRBlockMatrixDiag(matrix))
#define hypre_ParCSRBlockMatrixNumNonzeros(matrix)     ((matrix) -> num_nonzeros)
#define hypre_ParCSRBlockMatrixDNumNonzeros(matrix)    ((matrix) -> d_num_nonzeros)
#define hypre_ParCSRBlockMatrixRowindices(matrix)      ((matrix) -> rowindices)
#define hypre_ParCSRBlockMatrixRowvalues(matrix)       ((matrix) -> rowvalues)
#define hypre_ParCSRBlockMatrixGetrowactive(matrix)    ((matrix) -> getrowactive)
#define hypre_ParCSRBlockMatrixAssumedPartition(matrix) ((matrix) -> assumed_partition)


hypre_CSRBlockMatrix *
hypre_ParCSRBlockMatrixExtractBExt(hypre_ParCSRBlockMatrix *B,
                                   hypre_ParCSRBlockMatrix *A, int data);

hypre_ParCSRBlockMatrix *
hypre_ParCSRBlockMatrixCreate(MPI_Comm comm, int block_size, int global_num_rows,
                              int global_num_cols, int *row_starts, int *col_starts,
                              int num_cols_offd, int num_nonzeros_diag,
                              int num_nonzeros_offd);

int 
hypre_ParCSRBlockMatrixDestroy( hypre_ParCSRBlockMatrix *matrix );
   


int
hypre_BoomerAMGBuildBlockInterp( hypre_ParCSRBlockMatrix   *A,
                                 int                  *CF_marker,
                                 hypre_ParCSRMatrix   *S,
                                 int                  *num_cpts_global,
                                 int                   num_functions,
                                 int                  *dof_func,
                                 int                   debug_flag,
                                 double                trunc_factor,
                                 int 		      *col_offd_S_to_A,
                                 hypre_ParCSRBlockMatrix  **P_ptr);
   
int
hypre_BoomerAMGBuildBlockInterpDiag( hypre_ParCSRBlockMatrix   *A,
                                     int                  *CF_marker,
                                     hypre_ParCSRMatrix   *S,
                                     int                  *num_cpts_global,
                                     int                   num_functions,
                                     int                  *dof_func,
                                     int                   debug_flag,
                                     double                trunc_factor,
                                     int 		      *col_offd_S_to_A,
                                     hypre_ParCSRBlockMatrix  **P_ptr);

int hypre_BoomerAMGBlockInterpTruncation( hypre_ParCSRBlockMatrix *P,
                                      double trunc_factor);
   


int  hypre_BoomerAMGBlockRelaxIF( hypre_ParCSRBlockMatrix *A,
                             hypre_ParVector    *f,
                             int                *cf_marker,
                             int                 relax_type,
                             int                 relax_order,
                             int                 cycle_type,
                             double              relax_weight,
                             double              omega,
                             hypre_ParVector    *u,
                             hypre_ParVector    *Vtemp );
   

int  hypre_BoomerAMGBlockRelax( hypre_ParCSRBlockMatrix *A,
                                hypre_ParVector    *f,
                                int                *cf_marker,
                                int                 relax_type,
                                int                 relax_points,
                                double              relax_weight,
                                double              omega,
                                hypre_ParVector    *u,
                                hypre_ParVector    *Vtemp );
   
int
hypre_GetCommPkgBlockRTFromCommPkgBlockA( hypre_ParCSRBlockMatrix *RT,
			       	hypre_ParCSRBlockMatrix *A,
                                          int *fine_to_coarse_offd);
   

hypre_ParCSRCommHandle *
hypre_ParCSRBlockCommHandleCreate(int job, int bnnz, hypre_ParCSRCommPkg *comm_pkg,
                                  void *send_data, void *recv_data );


int
hypre_ParCSRBlockCommHandleDestroy(hypre_ParCSRCommHandle *comm_handle);
   


int
hypre_BlockMatvecCommPkgCreate(hypre_ParCSRBlockMatrix *A);


int
hypre_ParCSRBlockMatrixCreateAssumedPartition( hypre_ParCSRBlockMatrix *matrix);
   
int 
hypre_ParCSRBlockMatrixDestroyAssumedPartition(hypre_ParCSRBlockMatrix *matrix );
   

  
hypre_ParCSRMatrix *
hypre_ParCSRBlockMatrixConvertToParCSRMatrix(hypre_ParCSRBlockMatrix *matrix);
   

hypre_ParCSRBlockMatrix *
hypre_ParCSRBlockMatrixConvertFromParCSRMatrix(hypre_ParCSRMatrix *matrix,
                                               int matrix_C_block_size );
   

int
hypre_ParCSRBlockMatrixRAP(hypre_ParCSRBlockMatrix  *RT,
                           hypre_ParCSRBlockMatrix  *A,
                           hypre_ParCSRBlockMatrix  *P,
                           hypre_ParCSRBlockMatrix **RAP_ptr );
   
int 
hypre_ParCSRBlockMatrixSetNumNonzeros( hypre_ParCSRBlockMatrix *matrix);
   
int 
hypre_ParCSRBlockMatrixSetDNumNonzeros( hypre_ParCSRBlockMatrix *matrix);
   
int
hypre_BoomerAMGBlockCreateNodalA(hypre_ParCSRBlockMatrix    *A,
                                 int  option,
                                 hypre_ParCSRMatrix   **AN_ptr);
   
hypre_ParVector *
hypre_ParVectorCreateFromBlock(MPI_Comm comm,
                               int p_global_size, 
                               int *p_partitioning, int block_size);
   
int
hypre_ParCSRBlockMatrixMatvec(double alpha, hypre_ParCSRBlockMatrix *A,
                              hypre_ParVector *x, double beta,
                              hypre_ParVector *y);
int   
hypre_ParCSRBlockMatrixMatvecT( double                  alpha,
                                hypre_ParCSRBlockMatrix *A,
                                hypre_ParVector         *x,
                                double                   beta,
                                hypre_ParVector          *y);
   

#ifdef __cplusplus
}
#endif
#endif
