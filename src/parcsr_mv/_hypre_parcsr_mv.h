/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/


#include <HYPRE_config.h>

#include "HYPRE_parcsr_mv.h"

#ifndef hypre_PARCSR_MV_HEADER
#define hypre_PARCSR_MV_HEADER

#include "_hypre_utilities.h"
#include "seq_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HYPRE_PAR_CSR_COMMUNICATION_HEADER
#define HYPRE_PAR_CSR_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_USING_PERSISTENT_COMM
typedef enum CommPkgJobType
{
   HYPRE_COMM_PKG_JOB_COMPLEX = 0,
   HYPRE_COMM_PKG_JOB_COMPLEX_TRANSPOSE,
   HYPRE_COMM_PKG_JOB_INT,
   HYPRE_COMM_PKG_JOB_INT_TRANSPOSE,
   HYPRE_COMM_PKG_JOB_BIGINT,
   HYPRE_COMM_PKG_JOB_BIGINT_TRANSPOSE,
   NUM_OF_COMM_PKG_JOB_TYPE,
} CommPkgJobType;
#endif

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommHandle, hypre_ParCSRPersistentCommHandle
 *--------------------------------------------------------------------------*/
struct _hypre_ParCSRCommPkg;

typedef struct
{
   struct _hypre_ParCSRCommPkg *comm_pkg;
   HYPRE_Int             send_memory_location;
   HYPRE_Int             recv_memory_location;
   HYPRE_Int             num_send_bytes;
   HYPRE_Int             num_recv_bytes;
   void                 *send_data;
   void                 *recv_data;
   void                 *send_data_buffer;
   void                 *recv_data_buffer;
   HYPRE_Int             num_requests;
   hypre_MPI_Request    *requests;
} hypre_ParCSRCommHandle;

typedef hypre_ParCSRCommHandle hypre_ParCSRPersistentCommHandle;

typedef struct _hypre_ParCSRCommPkg
{
   MPI_Comm                     comm;

   HYPRE_Int                    num_sends;
   HYPRE_Int                   *send_procs;
   HYPRE_Int                   *send_map_starts;
   HYPRE_Int                   *send_map_elmts;
   HYPRE_Int                   *device_send_map_elmts;

   HYPRE_Int                    num_recvs;
   HYPRE_Int                   *recv_procs;
   HYPRE_Int                   *recv_vec_starts;

   /* remote communication information */
   hypre_MPI_Datatype          *send_mpi_types;
   hypre_MPI_Datatype          *recv_mpi_types;

#ifdef HYPRE_USING_PERSISTENT_COMM
   hypre_ParCSRPersistentCommHandle *persistent_comm_handles[NUM_OF_COMM_PKG_JOB_TYPE];
#endif

   /* temporary memory for matvec. cudaMalloc is expensive. alloc once and reuse */
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Complex *tmp_data;
   HYPRE_Complex *buf_data;
#endif
} hypre_ParCSRCommPkg;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommPkg
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRCommPkgComm(comm_pkg)                (comm_pkg -> comm)
#define hypre_ParCSRCommPkgNumSends(comm_pkg)            (comm_pkg -> num_sends)
#define hypre_ParCSRCommPkgSendProcs(comm_pkg)           (comm_pkg -> send_procs)
#define hypre_ParCSRCommPkgSendProc(comm_pkg, i)         (comm_pkg -> send_procs[i])
#define hypre_ParCSRCommPkgSendMapStarts(comm_pkg)       (comm_pkg -> send_map_starts)
#define hypre_ParCSRCommPkgSendMapStart(comm_pkg,i)      (comm_pkg -> send_map_starts[i])
#define hypre_ParCSRCommPkgSendMapElmts(comm_pkg)        (comm_pkg -> send_map_elmts)
#define hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg)  (comm_pkg -> device_send_map_elmts)
#define hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)       (comm_pkg -> send_map_elmts[i])
#define hypre_ParCSRCommPkgDeviceSendMapElmt(comm_pkg,i) (comm_pkg -> device_send_map_elmts[i])
#define hypre_ParCSRCommPkgNumRecvs(comm_pkg)            (comm_pkg -> num_recvs)
#define hypre_ParCSRCommPkgRecvProcs(comm_pkg)           (comm_pkg -> recv_procs)
#define hypre_ParCSRCommPkgRecvProc(comm_pkg, i)         (comm_pkg -> recv_procs[i])
#define hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)       (comm_pkg -> recv_vec_starts)
#define hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i)      (comm_pkg -> recv_vec_starts[i])
#define hypre_ParCSRCommPkgSendMPITypes(comm_pkg)        (comm_pkg -> send_mpi_types)
#define hypre_ParCSRCommPkgSendMPIType(comm_pkg,i)       (comm_pkg -> send_mpi_types[i])
#define hypre_ParCSRCommPkgRecvMPITypes(comm_pkg)        (comm_pkg -> recv_mpi_types)
#define hypre_ParCSRCommPkgRecvMPIType(comm_pkg,i)       (comm_pkg -> recv_mpi_types[i])

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#define hypre_ParCSRCommPkgTmpData(comm_pkg)           ((comm_pkg) -> tmp_data)
#define hypre_ParCSRCommPkgBufData(comm_pkg)           ((comm_pkg) -> buf_data)
#endif

static inline void
hypre_ParCSRCommPkgCopySendMapElmtsToDevice(hypre_ParCSRCommPkg *comm_pkg)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   if (hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) == NULL)
   {
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) =
         hypre_TAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                      HYPRE_MEMORY_DEVICE);

      hypre_TMemcpy(hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                    hypre_ParCSRCommPkgSendMapElmts(comm_pkg),
                    HYPRE_Int,
                    hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }
#endif
}

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommHandle
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRCommHandleCommPkg(comm_handle)                (comm_handle -> comm_pkg)
#define hypre_ParCSRCommHandleSendMemoryLocation(comm_handle)     (comm_handle -> send_memory_location)
#define hypre_ParCSRCommHandleRecvMemoryLocation(comm_handle)     (comm_handle -> recv_memory_location)
#define hypre_ParCSRCommHandleNumSendBytes(comm_handle)           (comm_handle -> num_send_bytes)
#define hypre_ParCSRCommHandleNumRecvBytes(comm_handle)           (comm_handle -> num_recv_bytes)
#define hypre_ParCSRCommHandleSendData(comm_handle)               (comm_handle -> send_data)
#define hypre_ParCSRCommHandleRecvData(comm_handle)               (comm_handle -> recv_data)
#define hypre_ParCSRCommHandleSendDataBuffer(comm_handle)         (comm_handle -> send_data_buffer)
#define hypre_ParCSRCommHandleRecvDataBuffer(comm_handle)         (comm_handle -> recv_data_buffer)
#define hypre_ParCSRCommHandleNumRequests(comm_handle)            (comm_handle -> num_requests)
#define hypre_ParCSRCommHandleRequests(comm_handle)               (comm_handle -> requests)
#define hypre_ParCSRCommHandleRequest(comm_handle, i)             (comm_handle -> requests[i])

#endif /* HYPRE_PAR_CSR_COMMUNICATION_HEADER */
#ifndef hypre_PARCSR_ASSUMED_PART
#define  hypre_PARCSR_ASSUMED_PART

typedef struct
{
   HYPRE_Int                   length;
   HYPRE_BigInt                row_start;
   HYPRE_BigInt                row_end;
   HYPRE_Int                   storage_length;
   HYPRE_Int                  *proc_list;
   HYPRE_BigInt               *row_start_list;
   HYPRE_BigInt               *row_end_list;
   HYPRE_Int                  *sort_index;
} hypre_IJAssumedPart;




#endif /* hypre_PARCSR_ASSUMED_PART */

#ifndef hypre_NEW_COMMPKG
#define hypre_NEW_COMMPKG

typedef struct
{
   HYPRE_Int       length;
   HYPRE_Int       storage_length;
   HYPRE_Int      *id;
   HYPRE_Int      *vec_starts;
   HYPRE_Int       element_storage_length;
   HYPRE_BigInt   *elements;
   HYPRE_Real     *d_elements; /* Is this used anywhere? */
   void           *v_elements;

}  hypre_ProcListElements;

#endif /* hypre_NEW_COMMPKG */

/******************************************************************************
 *
 * Header info for Parallel Vector data structure
 *
 *****************************************************************************/

#ifndef hypre_PAR_VECTOR_HEADER
#define hypre_PAR_VECTOR_HEADER


/*--------------------------------------------------------------------------
 * hypre_ParVector
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_PAR_VECTOR_STRUCT
#define HYPRE_PAR_VECTOR_STRUCT
#endif

typedef struct hypre_ParVector_struct
{
   MPI_Comm              comm;

   HYPRE_BigInt          global_size;
   HYPRE_BigInt          first_index;
   HYPRE_BigInt          last_index;
   HYPRE_BigInt         *partitioning;
   /* stores actual length of data in local vector to allow memory
    * manipulations for temporary vectors*/
   HYPRE_Int             actual_local_size;
   hypre_Vector         *local_vector;

   /* Does the Vector create/destroy `data'? */
   HYPRE_Int             owns_data;
   HYPRE_Int             owns_partitioning;

   hypre_IJAssumedPart  *assumed_partition; /* only populated if no_global_partition option
                                              is used (compile-time option) AND this partition
                                              needed
                                              (for setting off-proc elements, for example)*/


} hypre_ParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_ParVectorComm(vector)             ((vector) -> comm)
#define hypre_ParVectorGlobalSize(vector)       ((vector) -> global_size)
#define hypre_ParVectorFirstIndex(vector)       ((vector) -> first_index)
#define hypre_ParVectorLastIndex(vector)        ((vector) -> last_index)
#define hypre_ParVectorPartitioning(vector)     ((vector) -> partitioning)
#define hypre_ParVectorActualLocalSize(vector)  ((vector) -> actual_local_size)
#define hypre_ParVectorLocalVector(vector)      ((vector) -> local_vector)
#define hypre_ParVectorOwnsData(vector)         ((vector) -> owns_data)
#define hypre_ParVectorOwnsPartitioning(vector) ((vector) -> owns_partitioning)
#define hypre_ParVectorNumVectors(vector)\
 (hypre_VectorNumVectors( hypre_ParVectorLocalVector(vector) ))

#define hypre_ParVectorAssumedPartition(vector) ((vector) -> assumed_partition)


#endif
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

   /* Does the ParCSRMatrix create/destroy `diag', `offd', `col_map_offd'? */
   HYPRE_Int             owns_data;
   /* Does the ParCSRMatrix create/destroy `row_starts', `col_starts'? */
   HYPRE_Int             owns_row_starts;
   HYPRE_Int             owns_col_starts;

   HYPRE_BigInt          num_nonzeros;
   HYPRE_Real            d_num_nonzeros;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   HYPRE_BigInt         *rowindices;
   HYPRE_Complex        *rowvalues;
   HYPRE_Int             getrowactive;

   hypre_IJAssumedPart  *assumed_partition; /* only populated if
                                              no_global_partition option is used
                                              (compile-time option)*/
   HYPRE_Int             owns_assumed_partition;
   /* Array to store ordering of local diagonal block to relax. In particular,
   used for triangulr matrices that are not ordered to be triangular. */
   HYPRE_Int            *proc_ordering;

   /* Save block diagonal inverse */
   HYPRE_Int             bdiag_size;
   HYPRE_Complex        *bdiaginv;
   hypre_ParCSRCommPkg  *bdiaginv_comm_pkg;

} hypre_ParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRMatrixComm(matrix)             ((matrix) -> comm)
#define hypre_ParCSRMatrixGlobalNumRows(matrix)    ((matrix) -> global_num_rows)
#define hypre_ParCSRMatrixGlobalNumCols(matrix)    ((matrix) -> global_num_cols)
#define hypre_ParCSRMatrixFirstRowIndex(matrix)    ((matrix) -> first_row_index)
#define hypre_ParCSRMatrixFirstColDiag(matrix)     ((matrix) -> first_col_diag)
#define hypre_ParCSRMatrixLastRowIndex(matrix)     ((matrix) -> last_row_index)
#define hypre_ParCSRMatrixLastColDiag(matrix)      ((matrix) -> last_col_diag)
#define hypre_ParCSRMatrixDiag(matrix)             ((matrix) -> diag)
#define hypre_ParCSRMatrixOffd(matrix)             ((matrix) -> offd)
#define hypre_ParCSRMatrixDiagT(matrix)            ((matrix) -> diagT)
#define hypre_ParCSRMatrixOffdT(matrix)            ((matrix) -> offdT)
#define hypre_ParCSRMatrixColMapOffd(matrix)       ((matrix) -> col_map_offd)
#define hypre_ParCSRMatrixDeviceColMapOffd(matrix) ((matrix) -> device_col_map_offd)
#define hypre_ParCSRMatrixRowStarts(matrix)        ((matrix) -> row_starts)
#define hypre_ParCSRMatrixColStarts(matrix)        ((matrix) -> col_starts)
#define hypre_ParCSRMatrixCommPkg(matrix)          ((matrix) -> comm_pkg)
#define hypre_ParCSRMatrixCommPkgT(matrix)         ((matrix) -> comm_pkgT)
#define hypre_ParCSRMatrixOwnsData(matrix)         ((matrix) -> owns_data)
#define hypre_ParCSRMatrixOwnsRowStarts(matrix)    ((matrix) -> owns_row_starts)
#define hypre_ParCSRMatrixOwnsColStarts(matrix)    ((matrix) -> owns_col_starts)
#define hypre_ParCSRMatrixNumNonzeros(matrix)      ((matrix) -> num_nonzeros)
#define hypre_ParCSRMatrixDNumNonzeros(matrix)     ((matrix) -> d_num_nonzeros)
#define hypre_ParCSRMatrixRowindices(matrix)       ((matrix) -> rowindices)
#define hypre_ParCSRMatrixRowvalues(matrix)        ((matrix) -> rowvalues)
#define hypre_ParCSRMatrixGetrowactive(matrix)     ((matrix) -> getrowactive)
#define hypre_ParCSRMatrixAssumedPartition(matrix) ((matrix) -> assumed_partition)
#define hypre_ParCSRMatrixOwnsAssumedPartition(matrix)   ((matrix) -> owns_assumed_partition)
#define hypre_ParCSRMatrixProcOrdering(matrix)    ((matrix) -> proc_ordering)

#define hypre_ParCSRMatrixNumRows(matrix) hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(matrix))
#define hypre_ParCSRMatrixNumCols(matrix) hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(matrix))

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
/******************************************************************************
 *
 * Tree structure for keeping track of numbers (e.g. column numbers) -
 * when you get them one at a time, in no particular order, possibly very
 * sparse.  In a scalable manner you want to be able to store them and find
 * out whether a number has been stored.
 * All decimal numbers will fit in a tree with 10 branches (digits)
 * off each node.  We also have a terminal "digit" to indicate that the entire
 * number has been seen.  E.g., 1234 would be entered in a tree as:
 * (numbering the digits off a node as 0 1 2 3 4 5 6 7 8 9 TERM )
 *                          root
 *                           |
 *                   - - - - 4 - - - - - -
 *                           |
 *                     - - - 3 - - - - - - -
 *                           |
 *                       - - 2 - - - - - - - -
 *                           |
 *                         - 1 - - - - - - - - -
 *                           |
 *       - - - - - - - - - - T
 *
 *
 * This tree represents a number through its decimal expansion, but if needed
 * base depends on how the numbers encountered are distributed.  Totally
 * The more clustered, the larger the base should be in my judgement.
 *
 *****************************************************************************/

#ifndef hypre_NUMBERS_HEADER
#define hypre_NUMBERS_HEADER

typedef struct hypre_NumbersNode{
   struct hypre_NumbersNode * digit[11];
} hypre_NumbersNode;

hypre_NumbersNode * hypre_NumbersNewNode(void);
void hypre_NumbersDeleteNode( hypre_NumbersNode * node );
HYPRE_Int hypre_NumbersEnter( hypre_NumbersNode * node, const HYPRE_Int n );
HYPRE_Int hypre_NumbersNEntered( hypre_NumbersNode * node );
HYPRE_Int hypre_NumbersQuery( hypre_NumbersNode * node, const HYPRE_Int n );
HYPRE_Int * hypre_NumbersArray( hypre_NumbersNode * node );


#endif
/******************************************************************************
 *
 * Header info for Parallel Chord Matrix data structures
 *
 *****************************************************************************/

#include <HYPRE_config.h>

#ifndef hypre_PAR_CHORD_MATRIX_HEADER
#define hypre_PAR_CHORD_MATRIX_HEADER

#include "_hypre_utilities.h"
#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * Parallel Chord Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm comm;

  /*  A structure: -------------------------------------------------------- */
  HYPRE_Int num_inprocessors;
  HYPRE_Int *inprocessor;

  /* receiving in idof from different (in)processors; ---------------------- */
  HYPRE_Int *num_idofs_inprocessor;
  HYPRE_Int **idof_inprocessor;

  /* symmetric information: ----------------------------------------------- */
  /* this can be replaces by CSR format: ---------------------------------- */
  HYPRE_Int     *num_inchords;
  HYPRE_Int     **inchord_idof;
  HYPRE_Int     **inchord_rdof;
  HYPRE_Complex **inchord_data;

  HYPRE_Int num_idofs;
  HYPRE_Int num_rdofs;

  HYPRE_Int *firstindex_idof; /* not owned by my_id; ---------------------- */
  HYPRE_Int *firstindex_rdof; /* not owned by my_id; ---------------------- */

  /* --------------------------- mirror information: ---------------------- */
  /* participation of rdof in different processors; ----------------------- */

  HYPRE_Int num_toprocessors;
  HYPRE_Int *toprocessor;

  /* rdofs to be sentto toprocessors; -------------------------------------
     ---------------------------------------------------------------------- */
  HYPRE_Int *num_rdofs_toprocessor;
  HYPRE_Int **rdof_toprocessor;

} hypre_ParChordMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParChordMatrixComm(matrix)                  ((matrix) -> comm)

/*  matrix structure: ----------------------------------------------------- */

#define hypre_ParChordMatrixNumInprocessors(matrix)  ((matrix) -> num_inprocessors)
#define hypre_ParChordMatrixInprocessor(matrix) ((matrix) -> inprocessor)
#define hypre_ParChordMatrixNumIdofsInprocessor(matrix) ((matrix) -> num_idofs_inprocessor)
#define hypre_ParChordMatrixIdofInprocessor(matrix) ((matrix) -> idof_inprocessor)

#define hypre_ParChordMatrixNumInchords(matrix) ((matrix) -> num_inchords)

#define hypre_ParChordMatrixInchordIdof(matrix) ((matrix) -> inchord_idof)
#define hypre_ParChordMatrixInchordRdof(matrix) ((matrix) -> inchord_rdof)
#define hypre_ParChordMatrixInchordData(matrix) ((matrix) -> inchord_data)
#define hypre_ParChordMatrixNumIdofs(matrix)    ((matrix) -> num_idofs)
#define hypre_ParChordMatrixNumRdofs(matrix)    ((matrix) -> num_rdofs)

#define hypre_ParChordMatrixFirstindexIdof(matrix) ((matrix) -> firstindex_idof)
#define hypre_ParChordMatrixFirstindexRdof(matrix) ((matrix) -> firstindex_rdof)

/* participation of rdof in different processors; ---------- */

#define hypre_ParChordMatrixNumToprocessors(matrix) ((matrix) -> num_toprocessors)
#define hypre_ParChordMatrixToprocessor(matrix)  ((matrix) -> toprocessor)
#define hypre_ParChordMatrixNumRdofsToprocessor(matrix) ((matrix) -> num_rdofs_toprocessor)
#define hypre_ParChordMatrixRdofToprocessor(matrix) ((matrix) -> rdof_toprocessor)


#endif
#ifndef hypre_PAR_MAKE_SYSTEM
#define  hypre_PAR_MAKE_SYSTEM

typedef struct
{
   hypre_ParCSRMatrix * A;
   hypre_ParVector * x;
   hypre_ParVector * b;

} HYPRE_ParCSR_System_Problem;

#endif /* hypre_PAR_MAKE_SYSTEM */

/* communicationT.c */
void hypre_RowsWithColumn_original ( HYPRE_Int *rowmin , HYPRE_Int *rowmax , HYPRE_BigInt column , hypre_ParCSRMatrix *A );
void hypre_RowsWithColumn ( HYPRE_Int *rowmin , HYPRE_Int *rowmax , HYPRE_BigInt column , HYPRE_Int num_rows_diag , HYPRE_BigInt firstColDiag , HYPRE_BigInt *colMapOffd , HYPRE_Int *mat_i_diag , HYPRE_Int *mat_j_diag , HYPRE_Int *mat_i_offd , HYPRE_Int *mat_j_offd );
void hypre_MatTCommPkgCreate_core ( MPI_Comm comm , HYPRE_BigInt *col_map_offd , HYPRE_BigInt first_col_diag , HYPRE_BigInt *col_starts , HYPRE_Int num_rows_diag , HYPRE_Int num_cols_diag , HYPRE_Int num_cols_offd , HYPRE_BigInt *row_starts , HYPRE_BigInt firstColDiag , HYPRE_BigInt *colMapOffd , HYPRE_Int *mat_i_diag , HYPRE_Int *mat_j_diag , HYPRE_Int *mat_i_offd , HYPRE_Int *mat_j_offd , HYPRE_Int data , HYPRE_Int *p_num_recvs , HYPRE_Int **p_recv_procs , HYPRE_Int **p_recv_vec_starts , HYPRE_Int *p_num_sends , HYPRE_Int **p_send_procs , HYPRE_Int **p_send_map_starts , HYPRE_Int **p_send_map_elmts );
HYPRE_Int hypre_MatTCommPkgCreate ( hypre_ParCSRMatrix *A );

/* driver_aat.c */

/* driver_boolaat.c */

/* driver_boolmatmul.c */

/* driver.c */

/* driver_matmul.c */

/* driver_mat_multivec.c */

/* driver_matvec.c */

/* driver_multivec.c */

/* HYPRE_parcsr_matrix.c */
HYPRE_Int HYPRE_ParCSRMatrixCreate ( MPI_Comm comm , HYPRE_BigInt global_num_rows , HYPRE_BigInt global_num_cols , HYPRE_BigInt *row_starts , HYPRE_BigInt *col_starts , HYPRE_Int num_cols_offd , HYPRE_Int num_nonzeros_diag , HYPRE_Int num_nonzeros_offd , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixDestroy ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixInitialize ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixBigInitialize ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixRead ( MPI_Comm comm , const char *file_name , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixPrint ( HYPRE_ParCSRMatrix matrix , const char *file_name );
HYPRE_Int HYPRE_ParCSRMatrixGetComm ( HYPRE_ParCSRMatrix matrix , MPI_Comm *comm );
HYPRE_Int HYPRE_ParCSRMatrixGetDims ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt *M , HYPRE_BigInt *N );
HYPRE_Int HYPRE_ParCSRMatrixGetRowPartitioning ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetColPartitioning ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt **col_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetLocalRange ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt *row_start , HYPRE_BigInt *row_end , HYPRE_BigInt *col_start , HYPRE_BigInt *col_end );
HYPRE_Int HYPRE_ParCSRMatrixGetRow ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Complex **values );
HYPRE_Int HYPRE_ParCSRMatrixRestoreRow ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Complex **values );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix ( MPI_Comm comm , HYPRE_CSRMatrix A_CSR , HYPRE_BigInt *row_partitioning , HYPRE_BigInt *col_partitioning , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning ( MPI_Comm comm , HYPRE_CSRMatrix A_CSR , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixMatvec ( HYPRE_Complex alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , HYPRE_Complex beta , HYPRE_ParVector y );
HYPRE_Int HYPRE_ParCSRMatrixMatvecT ( HYPRE_Complex alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , HYPRE_Complex beta , HYPRE_ParVector y );

/* HYPRE_parcsr_vector.c */
HYPRE_Int HYPRE_ParVectorCreate ( MPI_Comm comm , HYPRE_BigInt global_size , HYPRE_BigInt *partitioning , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParMultiVectorCreate ( MPI_Comm comm , HYPRE_BigInt global_size , HYPRE_BigInt *partitioning , HYPRE_Int number_vectors , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorDestroy ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorInitialize ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorRead ( MPI_Comm comm , const char *file_name , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorPrint ( HYPRE_ParVector vector , const char *file_name );
HYPRE_Int HYPRE_ParVectorSetConstantValues ( HYPRE_ParVector vector , HYPRE_Complex value );
HYPRE_Int HYPRE_ParVectorSetRandomValues ( HYPRE_ParVector vector , HYPRE_Int seed );
HYPRE_Int HYPRE_ParVectorCopy ( HYPRE_ParVector x , HYPRE_ParVector y );
HYPRE_ParVector HYPRE_ParVectorCloneShallow ( HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorScale ( HYPRE_Complex value , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorAxpy ( HYPRE_Complex alpha , HYPRE_ParVector x , HYPRE_ParVector y );
HYPRE_Int HYPRE_ParVectorInnerProd ( HYPRE_ParVector x , HYPRE_ParVector y , HYPRE_Real *prod );
HYPRE_Int HYPRE_VectorToParVector ( MPI_Comm comm , HYPRE_Vector b , HYPRE_BigInt *partitioning , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorGetValues ( HYPRE_ParVector vector, HYPRE_Int num_values, HYPRE_BigInt *indices , HYPRE_Complex *values);

/* new_commpkg.c */
HYPRE_Int hypre_PrintCommpkg ( hypre_ParCSRMatrix *A , const char *file_name );
HYPRE_Int hypre_ParCSRCommPkgCreateApart_core ( MPI_Comm comm , HYPRE_BigInt *col_map_off_d , HYPRE_BigInt first_col_diag , HYPRE_Int num_cols_off_d , HYPRE_BigInt global_num_cols , HYPRE_Int *p_num_recvs , HYPRE_Int **p_recv_procs , HYPRE_Int **p_recv_vec_starts , HYPRE_Int *p_num_sends , HYPRE_Int **p_send_procs , HYPRE_Int **p_send_map_starts , HYPRE_Int **p_send_map_elements , hypre_IJAssumedPart *apart );
HYPRE_Int hypre_ParCSRCommPkgCreateApart ( MPI_Comm  comm, HYPRE_BigInt *col_map_off_d, HYPRE_BigInt  first_col_diag, HYPRE_Int  num_cols_off_d, HYPRE_BigInt  global_num_cols, hypre_IJAssumedPart *apart, hypre_ParCSRCommPkg *comm_pkg );
HYPRE_Int hypre_NewCommPkgDestroy ( hypre_ParCSRMatrix *parcsr_A );
HYPRE_Int hypre_RangeFillResponseIJDetermineRecvProcs ( void *p_recv_contact_buf , HYPRE_Int contact_size , HYPRE_Int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseIJDetermineSendProcs ( void *p_recv_contact_buf , HYPRE_Int contact_size , HYPRE_Int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , HYPRE_Int *response_message_size );

/* numbers.c */
hypre_NumbersNode *hypre_NumbersNewNode ( void );
void hypre_NumbersDeleteNode ( hypre_NumbersNode *node );
HYPRE_Int hypre_NumbersEnter ( hypre_NumbersNode *node , const HYPRE_Int n );
HYPRE_Int hypre_NumbersNEntered ( hypre_NumbersNode *node );
HYPRE_Int hypre_NumbersQuery ( hypre_NumbersNode *node , const HYPRE_Int n );
HYPRE_Int *hypre_NumbersArray ( hypre_NumbersNode *node );

/* parchord_to_parcsr.c */
void hypre_ParChordMatrix_RowStarts ( hypre_ParChordMatrix *Ac , MPI_Comm comm , HYPRE_BigInt **row_starts , HYPRE_BigInt *global_num_cols );
HYPRE_Int hypre_ParChordMatrixToParCSRMatrix ( hypre_ParChordMatrix *Ac , MPI_Comm comm , hypre_ParCSRMatrix **pAp );
HYPRE_Int hypre_ParCSRMatrixToParChordMatrix ( hypre_ParCSRMatrix *Ap , MPI_Comm comm , hypre_ParChordMatrix **pAc );

/* par_csr_aat.c */
void hypre_ParAat_RowSizes ( HYPRE_Int **C_diag_i , HYPRE_Int **C_offd_i , HYPRE_Int *B_marker , HYPRE_Int *A_diag_i , HYPRE_Int *A_diag_j , HYPRE_Int *A_offd_i , HYPRE_Int *A_offd_j , HYPRE_BigInt *A_col_map_offd , HYPRE_Int *A_ext_i , HYPRE_BigInt *A_ext_j , HYPRE_BigInt *A_ext_row_map , HYPRE_Int *C_diag_size , HYPRE_Int *C_offd_size , HYPRE_Int num_rows_diag_A , HYPRE_Int num_cols_offd_A , HYPRE_Int num_rows_A_ext , HYPRE_BigInt first_col_diag_A , HYPRE_BigInt first_row_index_A );
hypre_ParCSRMatrix *hypre_ParCSRAAt ( hypre_ParCSRMatrix *A );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractAExt ( hypre_ParCSRMatrix *A , HYPRE_Int data , HYPRE_BigInt **pA_ext_row_map );

/* par_csr_assumed_part.c */
HYPRE_Int hypre_LocateAssummedPartition ( MPI_Comm comm , HYPRE_BigInt row_start , HYPRE_BigInt row_end , HYPRE_BigInt global_first_row , HYPRE_BigInt global_num_rows , hypre_IJAssumedPart *part , HYPRE_Int myid );
HYPRE_Int hypre_ParCSRMatrixCreateAssumedPartition ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_AssumedPartitionDestroy ( hypre_IJAssumedPart *apart );
HYPRE_Int hypre_GetAssumedPartitionProcFromRow ( MPI_Comm comm , HYPRE_BigInt row , HYPRE_BigInt global_first_row , HYPRE_BigInt global_num_rows , HYPRE_Int *proc_id );
HYPRE_Int hypre_GetAssumedPartitionRowRange ( MPI_Comm comm , HYPRE_Int proc_id , HYPRE_BigInt global_first_row , HYPRE_BigInt global_num_rows , HYPRE_BigInt *row_start , HYPRE_BigInt *row_end );
HYPRE_Int hypre_ParVectorCreateAssumedPartition ( hypre_ParVector *vector );

/* par_csr_bool_matop.c */
hypre_ParCSRBooleanMatrix *hypre_ParBooleanMatmul ( hypre_ParCSRBooleanMatrix *A , hypre_ParCSRBooleanMatrix *B );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractBExt ( hypre_ParCSRBooleanMatrix *B , hypre_ParCSRBooleanMatrix *A );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractAExt ( hypre_ParCSRBooleanMatrix *A , HYPRE_BigInt **pA_ext_row_map );
hypre_ParCSRBooleanMatrix *hypre_ParBooleanAAt ( hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanMatTCommPkgCreate ( hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanMatvecCommPkgCreate ( hypre_ParCSRBooleanMatrix *A );

/* par_csr_bool_matrix.c */
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixCreate ( HYPRE_Int num_rows , HYPRE_Int num_cols , HYPRE_Int num_nonzeros );
HYPRE_Int hypre_CSRBooleanMatrixDestroy ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixInitialize ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixBigInitialize ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixSetDataOwner ( hypre_CSRBooleanMatrix *matrix , HYPRE_Int owns_data );
HYPRE_Int hypre_CSRBooleanMatrixSetBigDataOwner ( hypre_CSRBooleanMatrix *matrix , HYPRE_Int owns_data );
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixRead ( const char *file_name );
HYPRE_Int hypre_CSRBooleanMatrixPrint ( hypre_CSRBooleanMatrix *matrix , const char *file_name );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixCreate ( MPI_Comm comm , HYPRE_BigInt global_num_rows , HYPRE_BigInt global_num_cols , HYPRE_BigInt *row_starts , HYPRE_BigInt *col_starts , HYPRE_Int num_cols_offd , HYPRE_Int num_nonzeros_diag , HYPRE_Int num_nonzeros_offd );
HYPRE_Int hypre_ParCSRBooleanMatrixDestroy ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixInitialize ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixSetNNZ ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixSetDataOwner ( hypre_ParCSRBooleanMatrix *matrix , HYPRE_Int owns_data );
HYPRE_Int hypre_ParCSRBooleanMatrixSetRowStartsOwner ( hypre_ParCSRBooleanMatrix *matrix , HYPRE_Int owns_row_starts );
HYPRE_Int hypre_ParCSRBooleanMatrixSetColStartsOwner ( hypre_ParCSRBooleanMatrix *matrix , HYPRE_Int owns_col_starts );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixRead ( MPI_Comm comm , const char *file_name );
HYPRE_Int hypre_ParCSRBooleanMatrixPrint ( hypre_ParCSRBooleanMatrix *matrix , const char *file_name );
HYPRE_Int hypre_ParCSRBooleanMatrixPrintIJ ( hypre_ParCSRBooleanMatrix *matrix , const char *filename );
HYPRE_Int hypre_ParCSRBooleanMatrixGetLocalRange ( hypre_ParCSRBooleanMatrix *matrix , HYPRE_BigInt *row_start , HYPRE_BigInt *row_end , HYPRE_BigInt *col_start , HYPRE_BigInt *col_end );
HYPRE_Int hypre_ParCSRBooleanMatrixGetRow ( hypre_ParCSRBooleanMatrix *mat , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind );
HYPRE_Int hypre_ParCSRBooleanMatrixRestoreRow ( hypre_ParCSRBooleanMatrix *matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind );
HYPRE_Int hypre_BuildCSRBooleanMatrixMPIDataType ( HYPRE_Int num_nonzeros , HYPRE_Int num_rows , HYPRE_Int *a_i , HYPRE_Int *a_j , hypre_MPI_Datatype *csr_matrix_datatype );
hypre_ParCSRBooleanMatrix *hypre_CSRBooleanMatrixToParCSRBooleanMatrix ( MPI_Comm comm , hypre_CSRBooleanMatrix *A , HYPRE_BigInt *row_starts , HYPRE_BigInt *col_starts );
HYPRE_Int hypre_BooleanGenerateDiagAndOffd ( hypre_CSRBooleanMatrix *A , hypre_ParCSRBooleanMatrix *matrix , HYPRE_BigInt first_col_diag , HYPRE_BigInt last_col_diag );

/* par_csr_communication.c */
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate ( HYPRE_Int job , hypre_ParCSRCommPkg *comm_pkg , void *send_data , void *recv_data );
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate_v2 ( HYPRE_Int job, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int send_memory_location, void *send_data_in, HYPRE_Int recv_memory_location, void *recv_data_in );
HYPRE_Int hypre_ParCSRCommHandleDestroy ( hypre_ParCSRCommHandle *comm_handle );
void hypre_ParCSRCommPkgCreate_core ( MPI_Comm comm , HYPRE_BigInt *col_map_offd , HYPRE_BigInt first_col_diag , HYPRE_BigInt *col_starts , HYPRE_Int num_cols_diag , HYPRE_Int num_cols_offd , HYPRE_Int *p_num_recvs , HYPRE_Int **p_recv_procs , HYPRE_Int **p_recv_vec_starts , HYPRE_Int *p_num_sends , HYPRE_Int **p_send_procs , HYPRE_Int **p_send_map_starts , HYPRE_Int **p_send_map_elmts );
HYPRE_Int
hypre_ParCSRCommPkgCreate(MPI_Comm comm, HYPRE_BigInt *col_map_offd, HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts, HYPRE_Int num_cols_diag, HYPRE_Int num_cols_offd, hypre_ParCSRCommPkg *comm_pkg);
HYPRE_Int hypre_MatvecCommPkgCreate ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_MatvecCommPkgDestroy ( hypre_ParCSRCommPkg *comm_pkg );
HYPRE_Int hypre_BuildCSRMatrixMPIDataType ( HYPRE_Int num_nonzeros , HYPRE_Int num_rows , HYPRE_Complex *a_data , HYPRE_Int *a_i , HYPRE_Int *a_j , hypre_MPI_Datatype *csr_matrix_datatype );
HYPRE_Int hypre_BuildCSRJDataType ( HYPRE_Int num_nonzeros , HYPRE_Complex *a_data , HYPRE_Int *a_j , hypre_MPI_Datatype *csr_jdata_datatype );
HYPRE_Int hypre_ParCSRFindExtendCommPkg(MPI_Comm comm, HYPRE_BigInt global_num_cols, HYPRE_BigInt first_col_diag, HYPRE_Int num_cols_diag, HYPRE_BigInt *col_starts, hypre_IJAssumedPart *apart, HYPRE_Int indices_len, HYPRE_BigInt *indices, hypre_ParCSRCommPkg **extend_comm_pkg);
/* par_csr_matop.c */
void hypre_ParMatmul_RowSizes ( HYPRE_Int **C_diag_i , HYPRE_Int **C_offd_i , HYPRE_Int *A_diag_i , HYPRE_Int *A_diag_j , HYPRE_Int *A_offd_i , HYPRE_Int *A_offd_j , HYPRE_Int *B_diag_i , HYPRE_Int *B_diag_j , HYPRE_Int *B_offd_i , HYPRE_Int *B_offd_j , HYPRE_Int *B_ext_diag_i , HYPRE_Int *B_ext_diag_j , HYPRE_Int *B_ext_offd_i , HYPRE_Int *B_ext_offd_j , HYPRE_Int *map_B_to_C , HYPRE_Int *C_diag_size , HYPRE_Int *C_offd_size , HYPRE_Int num_rows_diag_A , HYPRE_Int num_cols_offd_A , HYPRE_Int allsquare , HYPRE_Int num_cols_diag_B , HYPRE_Int num_cols_offd_B , HYPRE_Int num_cols_offd_C );
hypre_ParCSRMatrix *hypre_ParMatmul ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B );
void hypre_ParCSRMatrixExtractBExt_Arrays ( HYPRE_Int **pB_ext_i , HYPRE_BigInt **pB_ext_j , HYPRE_Complex **pB_ext_data , HYPRE_BigInt **pB_ext_row_map , HYPRE_Int *num_nonzeros , HYPRE_Int data , HYPRE_Int find_row_map , MPI_Comm comm , hypre_ParCSRCommPkg *comm_pkg , HYPRE_Int num_cols_B , HYPRE_Int num_recvs , HYPRE_Int num_sends , HYPRE_BigInt first_col_diag , HYPRE_BigInt *row_starts , HYPRE_Int *recv_vec_starts , HYPRE_Int *send_map_starts , HYPRE_Int *send_map_elmts , HYPRE_Int *diag_i , HYPRE_Int *diag_j , HYPRE_Int *offd_i , HYPRE_Int *offd_j , HYPRE_BigInt *col_map_offd , HYPRE_Real *diag_data , HYPRE_Real *offd_data );
void hypre_ParCSRMatrixExtractBExt_Arrays_Overlap ( HYPRE_Int **pB_ext_i , HYPRE_BigInt **pB_ext_j , HYPRE_Complex **pB_ext_data , HYPRE_BigInt **pB_ext_row_map , HYPRE_Int *num_nonzeros , HYPRE_Int data , HYPRE_Int find_row_map , MPI_Comm comm , hypre_ParCSRCommPkg *comm_pkg , HYPRE_Int num_cols_B , HYPRE_Int num_recvs , HYPRE_Int num_sends , HYPRE_BigInt first_col_diag , HYPRE_BigInt *row_starts , HYPRE_Int *recv_vec_starts , HYPRE_Int *send_map_starts , HYPRE_Int *send_map_elmts , HYPRE_Int *diag_i , HYPRE_Int *diag_j , HYPRE_Int *offd_i , HYPRE_Int *offd_j , HYPRE_BigInt *col_map_offd , HYPRE_Real *diag_data , HYPRE_Real *offd_data, hypre_ParCSRCommHandle **comm_handle_idx, hypre_ParCSRCommHandle **comm_handle_data, HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd, HYPRE_Int skip_fine, HYPRE_Int skip_same_sign );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt ( hypre_ParCSRMatrix *B , hypre_ParCSRMatrix *A , HYPRE_Int data );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt_Overlap ( hypre_ParCSRMatrix *B , hypre_ParCSRMatrix *A , HYPRE_Int data, hypre_ParCSRCommHandle **comm_handle_idx, hypre_ParCSRCommHandle **comm_handle_data, HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd, HYPRE_Int skip_fine, HYPRE_Int skip_same_sign );
HYPRE_Int hypre_ParCSRMatrixExtractBExtDeviceInit( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A, HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ParCSRMatrixExtractBExtDeviceWait(void *request);
hypre_CSRMatrix* hypre_ParCSRMatrixExtractBExtDevice( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A, HYPRE_Int want_data );
HYPRE_Int hypre_ParCSRMatrixTranspose ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **AT_ptr , HYPRE_Int data );
void hypre_ParCSRMatrixGenSpanningTree ( hypre_ParCSRMatrix *G_csr , HYPRE_Int **indices , HYPRE_Int G_type );
void hypre_ParCSRMatrixExtractSubmatrices ( hypre_ParCSRMatrix *A_csr , HYPRE_Int *indices2 , hypre_ParCSRMatrix ***submatrices );
void hypre_ParCSRMatrixExtractRowSubmatrices ( hypre_ParCSRMatrix *A_csr , HYPRE_Int *indices2 , hypre_ParCSRMatrix ***submatrices );
HYPRE_Complex hypre_ParCSRMatrixLocalSumElts ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixAminvDB ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B , HYPRE_Complex *d , hypre_ParCSRMatrix **C_ptr );
hypre_ParCSRMatrix *hypre_ParTMatmul ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B );
HYPRE_Real hypre_ParCSRMatrixFnorm( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ExchangeExternalRowsInit( hypre_CSRMatrix *B_ext, hypre_ParCSRCommPkg *comm_pkg_A, void **request_ptr);
hypre_CSRMatrix* hypre_ExchangeExternalRowsWait(void *vequest);
HYPRE_Int hypre_ExchangeExternalRowsDeviceInit( hypre_CSRMatrix *B_ext, hypre_ParCSRCommPkg *comm_pkg_A, void **request_ptr);
hypre_CSRMatrix* hypre_ExchangeExternalRowsDeviceWait(void *vrequest);

#ifdef HYPRE_USING_PERSISTENT_COMM
hypre_ParCSRPersistentCommHandle* hypre_ParCSRPersistentCommHandleCreate(HYPRE_Int job, hypre_ParCSRCommPkg *comm_pkg);
hypre_ParCSRPersistentCommHandle* hypre_ParCSRCommPkgGetPersistentCommHandle(HYPRE_Int job, hypre_ParCSRCommPkg *comm_pkg);
void hypre_ParCSRPersistentCommHandleDestroy(hypre_ParCSRPersistentCommHandle *comm_handle);
void hypre_ParCSRPersistentCommHandleStart(hypre_ParCSRPersistentCommHandle *comm_handle, HYPRE_Int send_memory_location, void *send_data);
void hypre_ParCSRPersistentCommHandleWait(hypre_ParCSRPersistentCommHandle *comm_handle, HYPRE_Int recv_memory_location, void *recv_data);
#endif

HYPRE_Int hypre_ParcsrGetExternalRowsInit( hypre_ParCSRMatrix *A, HYPRE_Int indices_len, HYPRE_BigInt *indices, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ParcsrGetExternalRowsWait(void *vrequest);
HYPRE_Int hypre_ParcsrGetExternalRowsDeviceInit( hypre_ParCSRMatrix *A, HYPRE_Int indices_len, HYPRE_BigInt *indices, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ParcsrGetExternalRowsDeviceWait(void *vrequest);

HYPRE_Int hypre_ParvecBdiagInvScal( hypre_ParVector *b, HYPRE_Int blockSize, hypre_ParVector **bs, hypre_ParCSRMatrix *A);

HYPRE_Int hypre_ParcsrBdiagInvScal( hypre_ParCSRMatrix *A, HYPRE_Int blockSize, hypre_ParCSRMatrix **As);

HYPRE_Int hypre_ParCSRMatrixExtractSubmatrixFC( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker, HYPRE_BigInt *cpts_starts, const char *job, hypre_ParCSRMatrix **B_ptr, HYPRE_Real strength_thresh);

HYPRE_Int hypre_ParcsrAdd( HYPRE_Complex alpha, hypre_ParCSRMatrix *A, HYPRE_Complex beta, hypre_ParCSRMatrix *B, hypre_ParCSRMatrix **Cout);

/* par_csr_matop_marked.c */
void hypre_ParMatmul_RowSizes_Marked ( HYPRE_Int **C_diag_i , HYPRE_Int **C_offd_i , HYPRE_Int **B_marker , HYPRE_Int *A_diag_i , HYPRE_Int *A_diag_j , HYPRE_Int *A_offd_i , HYPRE_Int *A_offd_j , HYPRE_Int *B_diag_i , HYPRE_Int *B_diag_j , HYPRE_Int *B_offd_i , HYPRE_Int *B_offd_j , HYPRE_Int *B_ext_diag_i , HYPRE_Int *B_ext_diag_j , HYPRE_Int *B_ext_offd_i , HYPRE_Int *B_ext_offd_j , HYPRE_Int *map_B_to_C , HYPRE_Int *C_diag_size , HYPRE_Int *C_offd_size , HYPRE_Int num_rows_diag_A , HYPRE_Int num_cols_offd_A , HYPRE_Int allsquare , HYPRE_Int num_cols_diag_B , HYPRE_Int num_cols_offd_B , HYPRE_Int num_cols_offd_C , HYPRE_Int *CF_marker , HYPRE_Int *dof_func , HYPRE_Int *dof_func_offd );
hypre_ParCSRMatrix *hypre_ParMatmul_FC ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , HYPRE_Int *CF_marker , HYPRE_Int *dof_func , HYPRE_Int *dof_func_offd );
void hypre_ParMatScaleDiagInv_F ( hypre_ParCSRMatrix *C , hypre_ParCSRMatrix *A , HYPRE_Complex weight , HYPRE_Int *CF_marker );
hypre_ParCSRMatrix *hypre_ParMatMinus_F ( hypre_ParCSRMatrix *P , hypre_ParCSRMatrix *C , HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixZero_F ( hypre_ParCSRMatrix *P , HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixCopy_C ( hypre_ParCSRMatrix *P , hypre_ParCSRMatrix *C , HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixDropEntries ( hypre_ParCSRMatrix *C , hypre_ParCSRMatrix *P , HYPRE_Int *CF_marker );

/* par_csr_matrix.c */
hypre_ParCSRMatrix *hypre_ParCSRMatrixCreate ( MPI_Comm comm , HYPRE_BigInt global_num_rows , HYPRE_BigInt global_num_cols , HYPRE_BigInt *row_starts , HYPRE_BigInt *col_starts , HYPRE_Int num_cols_offd , HYPRE_Int num_nonzeros_diag , HYPRE_Int num_nonzeros_offd );
HYPRE_Int hypre_ParCSRMatrixDestroy ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixInitialize_v2( hypre_ParCSRMatrix *matrix, HYPRE_Int memory_location );
HYPRE_Int hypre_ParCSRMatrixInitialize ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetNumNonzeros ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetDNumNonzeros ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetDataOwner ( hypre_ParCSRMatrix *matrix , HYPRE_Int owns_data );
HYPRE_Int hypre_ParCSRMatrixSetRowStartsOwner ( hypre_ParCSRMatrix *matrix , HYPRE_Int owns_row_starts );
HYPRE_Int hypre_ParCSRMatrixSetColStartsOwner ( hypre_ParCSRMatrix *matrix , HYPRE_Int owns_col_starts );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRead ( MPI_Comm comm , const char *file_name );
HYPRE_Int hypre_ParCSRMatrixPrint ( hypre_ParCSRMatrix *matrix , const char *file_name );
HYPRE_Int hypre_ParCSRMatrixPrintIJ ( const hypre_ParCSRMatrix *matrix , const HYPRE_Int base_i , const HYPRE_Int base_j , const char *filename );
HYPRE_Int hypre_ParCSRMatrixReadIJ ( MPI_Comm comm , const char *filename , HYPRE_Int *base_i_ptr , HYPRE_Int *base_j_ptr , hypre_ParCSRMatrix **matrix_ptr );
HYPRE_Int hypre_ParCSRMatrixGetLocalRange ( hypre_ParCSRMatrix *matrix , HYPRE_BigInt *row_start , HYPRE_BigInt *row_end , HYPRE_BigInt *col_start , HYPRE_BigInt *col_end );
HYPRE_Int hypre_ParCSRMatrixGetRow ( hypre_ParCSRMatrix *mat , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Complex **values );
HYPRE_Int hypre_ParCSRMatrixRestoreRow ( hypre_ParCSRMatrix *matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Complex **values );
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix ( MPI_Comm comm , hypre_CSRMatrix *A , HYPRE_BigInt *row_starts , HYPRE_BigInt *col_starts );
HYPRE_Int GenerateDiagAndOffd ( hypre_CSRMatrix *A , hypre_ParCSRMatrix *matrix , HYPRE_BigInt first_col_diag , HYPRE_BigInt last_col_diag );
hypre_CSRMatrix *hypre_MergeDiagAndOffd ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_MergeDiagAndOffdDevice ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll ( hypre_ParCSRMatrix *par_matrix );
HYPRE_Int hypre_ParCSRMatrixCopy ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B , HYPRE_Int copy_data );
HYPRE_Int hypre_FillResponseParToCSRMatrix ( void *p_recv_contact_buf , HYPRE_Int contact_size , HYPRE_Int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , HYPRE_Int *response_message_size );
hypre_ParCSRMatrix *hypre_ParCSRMatrixUnion ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B );
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone ( hypre_ParCSRMatrix *A, HYPRE_Int copy_data );
#define hypre_ParCSRMatrixCompleteClone(A) hypre_ParCSRMatrixClone(A,0)
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone_v2 ( hypre_ParCSRMatrix *A, HYPRE_Int copy_data, HYPRE_Int memory_location );
#ifdef HYPRE_USING_CUDA
//hypre_int hypre_ParCSRMatrixIsManaged(hypre_ParCSRMatrix *a);
#endif
HYPRE_Int hypre_ParCSRMatrixDropSmallEntries( hypre_ParCSRMatrix *A, HYPRE_Real tol, HYPRE_Int type);

/* par_csr_matvec.c */
// y = alpha*A*x + beta*b
HYPRE_Int hypre_ParCSRMatrixMatvecOutOfPlace ( HYPRE_Complex alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , HYPRE_Complex beta , hypre_ParVector *b, hypre_ParVector *y );
// y = alpha*A*x + beta*y
HYPRE_Int hypre_ParCSRMatrixMatvec ( HYPRE_Complex alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , HYPRE_Complex beta , hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecT ( HYPRE_Complex alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , HYPRE_Complex beta , hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvec_FF ( HYPRE_Complex alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , HYPRE_Complex beta , hypre_ParVector *y , HYPRE_Int *CF_marker , HYPRE_Int fpt );

/* par_csr_triplemat.c */
hypre_ParCSRMatrix *hypre_ParCSRMatMat( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatMatHost( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatMatDevice( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );

hypre_ParCSRMatrix *hypre_ParCSRTMatMatKTHost( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B, HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKTDevice( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B, HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKT( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B, HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMat( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B);
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAPKT( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *P , HYPRE_Int keepTranspose );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAP( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *P );

/* par_make_system.c */
HYPRE_ParCSR_System_Problem *HYPRE_Generate2DSystem ( HYPRE_ParCSRMatrix H_L1 , HYPRE_ParCSRMatrix H_L2 , HYPRE_ParVector H_b1 , HYPRE_ParVector H_b2 , HYPRE_ParVector H_x1 , HYPRE_ParVector H_x2 , HYPRE_Complex *M_vals );
HYPRE_Int HYPRE_Destroy2DSystem ( HYPRE_ParCSR_System_Problem *sys_prob );

/* par_vector.c */
hypre_ParVector *hypre_ParVectorCreate ( MPI_Comm comm , HYPRE_BigInt global_size , HYPRE_BigInt *partitioning );
hypre_ParVector *hypre_ParMultiVectorCreate ( MPI_Comm comm , HYPRE_BigInt global_size , HYPRE_BigInt *partitioning , HYPRE_Int num_vectors );
HYPRE_Int hypre_ParVectorDestroy ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorInitialize ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorSetDataOwner ( hypre_ParVector *vector , HYPRE_Int owns_data );
HYPRE_Int hypre_ParVectorSetPartitioningOwner ( hypre_ParVector *vector , HYPRE_Int owns_partitioning );
HYPRE_Int hypre_ParVectorSetNumVectors ( hypre_ParVector *vector , HYPRE_Int num_vectors );
hypre_ParVector *hypre_ParVectorRead ( MPI_Comm comm , const char *file_name );
HYPRE_Int hypre_ParVectorPrint ( hypre_ParVector *vector , const char *file_name );
HYPRE_Int hypre_ParVectorSetConstantValues ( hypre_ParVector *v , HYPRE_Complex value );
HYPRE_Int hypre_ParVectorSetRandomValues ( hypre_ParVector *v , HYPRE_Int seed );
HYPRE_Int hypre_ParVectorCopy ( hypre_ParVector *x , hypre_ParVector *y );
hypre_ParVector *hypre_ParVectorCloneShallow ( hypre_ParVector *x );
HYPRE_Int hypre_ParVectorScale ( HYPRE_Complex alpha , hypre_ParVector *y );
HYPRE_Int hypre_ParVectorAxpy ( HYPRE_Complex alpha , hypre_ParVector *x , hypre_ParVector *y );
HYPRE_Int hypre_ParVectorMassAxpy ( HYPRE_Complex *alpha, hypre_ParVector **x, hypre_ParVector *y, HYPRE_Int k, HYPRE_Int unroll);
HYPRE_Real hypre_ParVectorInnerProd ( hypre_ParVector *x , hypre_ParVector *y );
HYPRE_Int hypre_ParVectorMassInnerProd ( hypre_ParVector *x , hypre_ParVector **y , HYPRE_Int k, HYPRE_Int unroll, HYPRE_Real *prod );
HYPRE_Int hypre_ParVectorMassDotpTwo ( hypre_ParVector *x , hypre_ParVector *y , hypre_ParVector **z, HYPRE_Int k, HYPRE_Int unroll, HYPRE_Real *prod_x , HYPRE_Real *prod_y );
hypre_ParVector *hypre_VectorToParVector ( MPI_Comm comm , hypre_Vector *v , HYPRE_BigInt *vec_starts );
hypre_Vector *hypre_ParVectorToVectorAll ( hypre_ParVector *par_v );
HYPRE_Int hypre_ParVectorPrintIJ ( hypre_ParVector *vector , HYPRE_Int base_j , const char *filename );
HYPRE_Int hypre_ParVectorReadIJ ( MPI_Comm comm , const char *filename , HYPRE_Int *base_j_ptr , hypre_ParVector **vector_ptr );
HYPRE_Int hypre_FillResponseParToVectorAll ( void *p_recv_contact_buf , HYPRE_Int contact_size , HYPRE_Int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , HYPRE_Int *response_message_size );
HYPRE_Complex hypre_ParVectorLocalSumElts ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorGetValues ( hypre_ParVector *vector, HYPRE_Int num_values, HYPRE_BigInt *indices , HYPRE_Complex *values);
#ifdef HYPRE_USING_CUDA
//hypre_int hypre_ParVectorIsManaged(hypre_ParVector *vector);
#endif

#ifdef __cplusplus
}
#endif

#endif

