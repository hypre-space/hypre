/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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

#ifndef hypre_PAR_CSR_MATRIX_HEADER
#define hypre_PAR_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_PAR_CSR_MATRIX_STRUCT
#define HYPRE_PAR_CSR_MATRIX_STRUCT
#endif

typedef struct hypre_ParCSRMatrix_struct
{
   MPI_Comm              comm;

   HYPRE_BigInt          global_num_rows;
   HYPRE_BigInt          global_num_cols;
   HYPRE_BigInt          global_num_rownnz;
   HYPRE_BigInt          num_nonzeros;
   HYPRE_Real            d_num_nonzeros;

   HYPRE_BigInt          first_row_index;
   HYPRE_BigInt          first_col_diag;
   /* need to know entire local range in case row_starts and col_starts
      are null  (i.e., bgl) AHB 6/05*/
   HYPRE_BigInt          last_row_index;
   HYPRE_BigInt          last_col_diag;

   hypre_CSRMatrix      *diag;
   hypre_CSRMatrix      *offd;
   hypre_CSRMatrix      *diagT, *offdT;
   /* JSP: transposed matrices are created lazily and optional */
   HYPRE_BigInt         *col_map_offd;
   HYPRE_BigInt         *device_col_map_offd;
   /* maps columns of offd to global columns */
   HYPRE_BigInt          row_starts[2];
   /* row_starts[0] is start of local rows
      row_starts[1] is start of next processor's rows */
   HYPRE_BigInt          col_starts[2];
   /* col_starts[0] is start of local columns
      col_starts[1] is start of next processor's columns */

   hypre_ParCSRCommPkg  *comm_pkg;
   hypre_ParCSRCommPkg  *comm_pkgT;

   /* Does the ParCSRMatrix create/destroy `diag', `offd', `col_map_offd'? */
   HYPRE_Int             owns_data;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   HYPRE_BigInt         *rowindices;
   HYPRE_Complex        *rowvalues;
   HYPRE_Int             getrowactive;

   hypre_IJAssumedPart  *assumed_partition;
   HYPRE_Int             owns_assumed_partition;
   /* Array to store ordering of local diagonal block to relax. In particular,
   used for triangulr matrices that are not ordered to be triangular. */
   HYPRE_Int            *proc_ordering;

   /* Save block diagonal inverse */
   HYPRE_Int             bdiag_size;
   HYPRE_Complex        *bdiaginv;
   hypre_ParCSRCommPkg  *bdiaginv_comm_pkg;

#if defined(HYPRE_USING_GPU)
   /* these two arrays are reserveed for SoC matrices on GPUs to help build interpolation */
   HYPRE_Int            *soc_diag_j;
   HYPRE_Int            *soc_offd_j;
#endif

} hypre_ParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRMatrixComm(matrix)                   ((matrix) -> comm)
#define hypre_ParCSRMatrixGlobalNumRows(matrix)          ((matrix) -> global_num_rows)
#define hypre_ParCSRMatrixGlobalNumCols(matrix)          ((matrix) -> global_num_cols)
#define hypre_ParCSRMatrixGlobalNumRownnz(matrix)        ((matrix) -> global_num_rownnz)
#define hypre_ParCSRMatrixNumNonzeros(matrix)            ((matrix) -> num_nonzeros)
#define hypre_ParCSRMatrixDNumNonzeros(matrix)           ((matrix) -> d_num_nonzeros)
#define hypre_ParCSRMatrixFirstRowIndex(matrix)          ((matrix) -> first_row_index)
#define hypre_ParCSRMatrixFirstColDiag(matrix)           ((matrix) -> first_col_diag)
#define hypre_ParCSRMatrixLastRowIndex(matrix)           ((matrix) -> last_row_index)
#define hypre_ParCSRMatrixLastColDiag(matrix)            ((matrix) -> last_col_diag)
#define hypre_ParCSRMatrixDiag(matrix)                   ((matrix) -> diag)
#define hypre_ParCSRMatrixOffd(matrix)                   ((matrix) -> offd)
#define hypre_ParCSRMatrixDiagT(matrix)                  ((matrix) -> diagT)
#define hypre_ParCSRMatrixOffdT(matrix)                  ((matrix) -> offdT)
#define hypre_ParCSRMatrixColMapOffd(matrix)             ((matrix) -> col_map_offd)
#define hypre_ParCSRMatrixDeviceColMapOffd(matrix)       ((matrix) -> device_col_map_offd)
#define hypre_ParCSRMatrixRowStarts(matrix)              ((matrix) -> row_starts)
#define hypre_ParCSRMatrixColStarts(matrix)              ((matrix) -> col_starts)
#define hypre_ParCSRMatrixCommPkg(matrix)                ((matrix) -> comm_pkg)
#define hypre_ParCSRMatrixCommPkgT(matrix)               ((matrix) -> comm_pkgT)
#define hypre_ParCSRMatrixOwnsData(matrix)               ((matrix) -> owns_data)
#define hypre_ParCSRMatrixRowindices(matrix)             ((matrix) -> rowindices)
#define hypre_ParCSRMatrixRowvalues(matrix)              ((matrix) -> rowvalues)
#define hypre_ParCSRMatrixGetrowactive(matrix)           ((matrix) -> getrowactive)
#define hypre_ParCSRMatrixAssumedPartition(matrix)       ((matrix) -> assumed_partition)
#define hypre_ParCSRMatrixOwnsAssumedPartition(matrix)   ((matrix) -> owns_assumed_partition)
#define hypre_ParCSRMatrixProcOrdering(matrix)           ((matrix) -> proc_ordering)
#if defined(HYPRE_USING_GPU)
#define hypre_ParCSRMatrixSocDiagJ(matrix)               ((matrix) -> soc_diag_j)
#define hypre_ParCSRMatrixSocOffdJ(matrix)               ((matrix) -> soc_offd_j)
#endif

#define hypre_ParCSRMatrixNumRows(matrix) hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(matrix))
#define hypre_ParCSRMatrixNumCols(matrix) hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(matrix))

static inline HYPRE_MemoryLocation
hypre_ParCSRMatrixMemoryLocation(hypre_ParCSRMatrix *matrix)
{
   if (!matrix) { return HYPRE_MEMORY_UNDEFINED; }

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(matrix);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(matrix);
   HYPRE_MemoryLocation memory_diag = diag ? hypre_CSRMatrixMemoryLocation(
                                         diag) : HYPRE_MEMORY_UNDEFINED;
   HYPRE_MemoryLocation memory_offd = offd ? hypre_CSRMatrixMemoryLocation(
                                         offd) : HYPRE_MEMORY_UNDEFINED;

   if (diag && offd)
   {
      if (memory_diag != memory_offd)
      {
         char err_msg[1024];
         hypre_sprintf(err_msg, "Error: ParCSRMatrix Memory Location Diag (%d) != Offd (%d)\n", memory_diag,
                       memory_offd);
         hypre_error_w_msg(HYPRE_ERROR_MEMORY, err_msg);
         hypre_assert(0);

         return HYPRE_MEMORY_UNDEFINED;
      }

      return memory_diag;
   }

   if (diag) { return memory_diag; }
   if (offd) { return memory_offd; }

   return HYPRE_MEMORY_UNDEFINED;
}

/*--------------------------------------------------------------------------
 * Parallel CSR Boolean Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;
   HYPRE_BigInt            global_num_rows;
   HYPRE_BigInt            global_num_cols;
   HYPRE_BigInt            first_row_index;
   HYPRE_BigInt            first_col_diag;
   HYPRE_BigInt            last_row_index;
   HYPRE_BigInt            last_col_diag;
   hypre_CSRBooleanMatrix *diag;
   hypre_CSRBooleanMatrix *offd;
   HYPRE_BigInt           *col_map_offd;
   HYPRE_BigInt           *row_starts;
   HYPRE_BigInt           *col_starts;
   hypre_ParCSRCommPkg    *comm_pkg;
   hypre_ParCSRCommPkg    *comm_pkgT;
   HYPRE_Int               owns_data;
   HYPRE_Int               owns_row_starts;
   HYPRE_Int               owns_col_starts;
   HYPRE_BigInt            num_nonzeros;
   HYPRE_BigInt           *rowindices;
   HYPRE_Int               getrowactive;

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
