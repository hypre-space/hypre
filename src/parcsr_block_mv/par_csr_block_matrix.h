/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_PAR_CSR_BLOCK_MATRIX_HEADER
#define hypre_PAR_CSR_BLOCK_MATRIX_HEADER

#include "_hypre_utilities.h"
#include "csr_block_matrix.h"
#include "_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Parallel CSR Block Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm             comm;

   HYPRE_BigInt         global_num_rows;
   HYPRE_BigInt         global_num_cols;
   HYPRE_BigInt         first_row_index;
   HYPRE_BigInt         first_col_diag;

 /* need to know entire local range in case row_starts and col_starts 
    are null */ 
   HYPRE_BigInt         last_row_index;
   HYPRE_BigInt         last_col_diag;

   hypre_CSRBlockMatrix *diag;
   hypre_CSRBlockMatrix *offd;
   HYPRE_BigInt         *col_map_offd; 
        /* maps columns of offd to global columns */
   HYPRE_BigInt         *row_starts; 
        /* array of length num_procs+1, row_starts[i] contains the 
           global number of the first row on proc i,  
           first_row_index = row_starts[my_id],
           row_starts[num_procs] = global_num_rows */
   HYPRE_BigInt         *col_starts;
        /* array of length num_procs+1, col_starts[i] contains the 
           global number of the first column of diag on proc i,  
           first_col_diag = col_starts[my_id],
           col_starts[num_procs] = global_num_cols */

   hypre_ParCSRCommPkg  *comm_pkg;
   hypre_ParCSRCommPkg  *comm_pkgT;
   
   /* Does the ParCSRBlockMatrix create/destroy `diag', `offd', `col_map_offd'? */
   HYPRE_Int      owns_data;
   /* Does the ParCSRBlockMatrix create/destroy `row_starts', `col_starts'? */
   HYPRE_Int      owns_row_starts;
   HYPRE_Int      owns_col_starts;

   HYPRE_BigInt   num_nonzeros;
   HYPRE_Real     d_num_nonzeros;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   HYPRE_Int     *rowindices;
   HYPRE_Complex *rowvalues;
   HYPRE_Int      getrowactive;

   hypre_IJAssumedPart *assumed_partition; /* only populated if no_global_partition option
                                              is used (compile-time option)*/

} hypre_ParCSRBlockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRBlockMatrixComm(matrix)            ((matrix)->comm)
#define hypre_ParCSRBlockMatrixGlobalNumRows(matrix)   ((matrix)->global_num_rows)
#define hypre_ParCSRBlockMatrixGlobalNumCols(matrix)   ((matrix)->global_num_cols)
#define hypre_ParCSRBlockMatrixFirstRowIndex(matrix)   ((matrix)->first_row_index)
#define hypre_ParCSRBlockMatrixFirstColDiag(matrix)    ((matrix)->first_col_diag)
#define hypre_ParCSRBlockMatrixLastRowIndex(matrix)    ((matrix) -> last_row_index)
#define hypre_ParCSRBlockMatrixLastColDiag(matrix)     ((matrix) -> last_col_diag)
#define hypre_ParCSRBlockMatrixBlockSize(matrix)       ((matrix)->diag->block_size)
#define hypre_ParCSRBlockMatrixDiag(matrix)            ((matrix) -> diag)
#define hypre_ParCSRBlockMatrixOffd(matrix)            ((matrix) -> offd)
#define hypre_ParCSRBlockMatrixColMapOffd(matrix)      ((matrix) -> col_map_offd)
#define hypre_ParCSRBlockMatrixRowStarts(matrix)       ((matrix) -> row_starts)
#define hypre_ParCSRBlockMatrixColStarts(matrix)       ((matrix) -> col_starts)
#define hypre_ParCSRBlockMatrixCommPkg(matrix)         ((matrix) -> comm_pkg)
#define hypre_ParCSRBlockMatrixCommPkgT(matrix)        ((matrix) -> comm_pkgT)
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
                                   hypre_ParCSRBlockMatrix *A, HYPRE_Int data);

hypre_ParCSRBlockMatrix *
hypre_ParCSRBlockMatrixCreate(MPI_Comm comm, HYPRE_Int block_size, HYPRE_BigInt global_num_rows,
                              HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts,
                              HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag,
                              HYPRE_Int num_nonzeros_offd);

HYPRE_Int 
hypre_ParCSRBlockMatrixDestroy( hypre_ParCSRBlockMatrix *matrix );
   


HYPRE_Int
hypre_BoomerAMGBuildBlockInterp( hypre_ParCSRBlockMatrix    *A,
                                 HYPRE_Int                  *CF_marker,
                                 hypre_ParCSRMatrix         *S,
                                 HYPRE_BigInt               *num_cpts_global,
                                 HYPRE_Int                   num_functions,
                                 HYPRE_Int                  *dof_func,
                                 HYPRE_Int                   debug_flag,
                                 HYPRE_Real                  trunc_factor,
                                 HYPRE_Int                   max_elmts,
                                 HYPRE_Int                   add_weak_to_diag,    
                                 HYPRE_Int                  *col_offd_S_to_A,
                                 hypre_ParCSRBlockMatrix   **P_ptr);
   


HYPRE_Int
hypre_BoomerAMGBuildBlockInterpRV( hypre_ParCSRBlockMatrix   *A,
                                   HYPRE_Int                 *CF_marker,
                                   hypre_ParCSRMatrix        *S,
                                   HYPRE_BigInt              *num_cpts_global,
                                   HYPRE_Int                  num_functions,
                                   HYPRE_Int                 *dof_func,
                                   HYPRE_Int                  debug_flag,
                                   HYPRE_Real                 trunc_factor,
                                   HYPRE_Int                  max_elmts,
                                   HYPRE_Int                 *col_offd_S_to_A,
                                   hypre_ParCSRBlockMatrix  **P_ptr);
   
HYPRE_Int
hypre_BoomerAMGBuildBlockInterpRV2( hypre_ParCSRBlockMatrix    *A,
                                    HYPRE_Int                  *CF_marker,
                                    hypre_ParCSRMatrix         *S,
                                    HYPRE_BigInt               *num_cpts_global,
                                    HYPRE_Int                   num_functions,
                                    HYPRE_Int                  *dof_func,
                                    HYPRE_Int                   debug_flag,
                                    HYPRE_Real                  trunc_factor,
                                    HYPRE_Int                   max_elmts,
                                    HYPRE_Int                  *col_offd_S_to_A,
                                    hypre_ParCSRBlockMatrix   **P_ptr);
HYPRE_Int
hypre_BoomerAMGBuildBlockInterpDiag( hypre_ParCSRBlockMatrix  *A,
                                     HYPRE_Int                *CF_marker,
                                     hypre_ParCSRMatrix       *S,
                                     HYPRE_BigInt             *num_cpts_global,
                                     HYPRE_Int                 num_functions,
                                     HYPRE_Int                *dof_func,
                                     HYPRE_Int                 debug_flag,
                                     HYPRE_Real                trunc_factor,
                                     HYPRE_Int                 max_elmts,
                                     HYPRE_Int                 add_weak_to_diag,
                                     HYPRE_Int                *col_offd_S_to_A,
                                     hypre_ParCSRBlockMatrix  **P_ptr);

HYPRE_Int hypre_BoomerAMGBlockInterpTruncation( hypre_ParCSRBlockMatrix *P,
                                                HYPRE_Real trunc_factor,
                                                HYPRE_Int max_elements);
   

HYPRE_Int
hypre_BoomerAMGBuildBlockDirInterp( hypre_ParCSRBlockMatrix   *A,
                                    HYPRE_Int                 *CF_marker,
                                    hypre_ParCSRMatrix        *S,
                                    HYPRE_BigInt              *num_cpts_global,
                                    HYPRE_Int                  num_functions,
                                    HYPRE_Int                 *dof_func,
                                    HYPRE_Int                  debug_flag,
                                    HYPRE_Real                 trunc_factor,
                                    HYPRE_Int                  max_elmts,
                                    HYPRE_Int                 *col_offd_S_to_A,
                                    hypre_ParCSRBlockMatrix  **P_ptr);
   

HYPRE_Int  hypre_BoomerAMGBlockRelaxIF( hypre_ParCSRBlockMatrix *A,
                                        hypre_ParVector    *f,
                                        HYPRE_Int          *cf_marker,
                                        HYPRE_Int           relax_type,
                                        HYPRE_Int           relax_order,
                                        HYPRE_Int           cycle_type,
                                        HYPRE_Real          relax_weight,
                                        HYPRE_Real          omega,
                                        hypre_ParVector    *u,
                                        hypre_ParVector    *Vtemp );
   

HYPRE_Int  hypre_BoomerAMGBlockRelax( hypre_ParCSRBlockMatrix *A,
                                      hypre_ParVector    *f,
                                      HYPRE_Int          *cf_marker,
                                      HYPRE_Int           relax_type,
                                      HYPRE_Int           relax_points,
                                      HYPRE_Real          relax_weight,
                                      HYPRE_Real          omega,
                                      hypre_ParVector    *u,
                                      hypre_ParVector    *Vtemp );
   
HYPRE_Int
hypre_GetCommPkgBlockRTFromCommPkgBlockA( hypre_ParCSRBlockMatrix *RT,
                                          hypre_ParCSRBlockMatrix *A,
                                          HYPRE_Int *tmp_map_offd,
                                          HYPRE_BigInt *fine_to_coarse_offd);
   

hypre_ParCSRCommHandle *
hypre_ParCSRBlockCommHandleCreate(HYPRE_Int job, HYPRE_Int bnnz, hypre_ParCSRCommPkg *comm_pkg,
                                  void *send_data, void *recv_data );


HYPRE_Int
hypre_ParCSRBlockCommHandleDestroy(hypre_ParCSRCommHandle *comm_handle);
   


HYPRE_Int
hypre_BlockMatvecCommPkgCreate(hypre_ParCSRBlockMatrix *A);


HYPRE_Int
hypre_ParCSRBlockMatrixCreateAssumedPartition( hypre_ParCSRBlockMatrix *matrix);
   
HYPRE_Int 
hypre_ParCSRBlockMatrixDestroyAssumedPartition(hypre_ParCSRBlockMatrix *matrix );
   

  
hypre_ParCSRMatrix *
hypre_ParCSRBlockMatrixConvertToParCSRMatrix(hypre_ParCSRBlockMatrix *matrix);
   

hypre_ParCSRBlockMatrix *
hypre_ParCSRBlockMatrixConvertFromParCSRMatrix(hypre_ParCSRMatrix *matrix,
                                               HYPRE_Int matrix_C_block_size );
   

HYPRE_Int
hypre_ParCSRBlockMatrixRAP(hypre_ParCSRBlockMatrix  *RT,
                           hypre_ParCSRBlockMatrix  *A,
                           hypre_ParCSRBlockMatrix  *P,
                           hypre_ParCSRBlockMatrix **RAP_ptr );
   
HYPRE_Int 
hypre_ParCSRBlockMatrixSetNumNonzeros( hypre_ParCSRBlockMatrix *matrix);
   
HYPRE_Int 
hypre_ParCSRBlockMatrixSetDNumNonzeros( hypre_ParCSRBlockMatrix *matrix);
   
HYPRE_Int
hypre_BoomerAMGBlockCreateNodalA(hypre_ParCSRBlockMatrix    *A,
                                 HYPRE_Int  option, HYPRE_Int diag_option,
                                 hypre_ParCSRMatrix   **AN_ptr);
   
hypre_ParVector *
hypre_ParVectorCreateFromBlock(MPI_Comm comm,
                               HYPRE_BigInt p_global_size, 
                               HYPRE_BigInt *p_partitioning, HYPRE_Int block_size);
   
HYPRE_Int
hypre_ParCSRBlockMatrixMatvec(HYPRE_Complex alpha, hypre_ParCSRBlockMatrix *A,
                              hypre_ParVector *x, HYPRE_Complex beta,
                              hypre_ParVector *y);
HYPRE_Int   
hypre_ParCSRBlockMatrixMatvecT( HYPRE_Complex           alpha,
                                hypre_ParCSRBlockMatrix *A,
                                hypre_ParVector         *x,
                                HYPRE_Complex            beta,
                                hypre_ParVector          *y);
   





void hypre_block_qsort( HYPRE_Int *v,
                        HYPRE_Complex *w,
                        HYPRE_Complex *blk_array,
                        HYPRE_Int block_size,
                        HYPRE_Int  left,
                        HYPRE_Int  right );
   

void hypre_swap_blk( HYPRE_Complex *v,
               HYPRE_Int block_size,
               HYPRE_Int  i,
               HYPRE_Int  j );
   

#ifdef __cplusplus
}
#endif
#endif
