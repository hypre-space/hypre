
#include "HYPRE_parcsr_mv.h"

#ifndef hypre_PARCSR_MV_HEADER
#define hypre_PARCSR_MV_HEADER

#include "utilities.h"
#include "seq_matrix_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * hypre_CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm               comm;

   int                    num_sends;
   int                   *send_procs;
   int			 *send_map_starts;
   int			 *send_map_elmts;

   int                    num_recvs;
   int                   *recv_procs;
   int                   *recv_vec_starts;

   /* remote communication information */
   MPI_Datatype          *send_mpi_types;
   MPI_Datatype          *recv_mpi_types;

} hypre_CommPkg;

/*--------------------------------------------------------------------------
 * hypre_CommHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_CommPkg  *comm_pkg;
   void 	  *send_data;
   void 	  *recv_data;

   int             num_requests;
   MPI_Request    *requests;

} hypre_CommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_VectorCommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_VectorCommPkgComm(comm_pkg)           (comm_pkg -> comm)
#define hypre_VectorCommPkgVecStarts(comm_pkg)      (comm_pkg -> vec_starts)
#define hypre_VectorCommPkgVecStart(comm_pkg,i)     (comm_pkg -> vec_starts[i])
#define hypre_VectorCommPkgVectorMPITypes(comm_pkg) (comm_pkg -> vector_mpi_types)
#define hypre_VectorCommPkgVectorMPIType(comm_pkg,i)(comm_pkg -> vector_mpi_types[i])
                                               
/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_CommPkgComm(comm_pkg)          (comm_pkg -> comm)
                                               
#define hypre_CommPkgNumSends(comm_pkg)      (comm_pkg -> num_sends)
#define hypre_CommPkgSendProcs(comm_pkg)     (comm_pkg -> send_procs)
#define hypre_CommPkgSendProc(comm_pkg, i)   (comm_pkg -> send_procs[i])
#define hypre_CommPkgSendMapStarts(comm_pkg) (comm_pkg -> send_map_starts)
#define hypre_CommPkgSendMapStart(comm_pkg,i)(comm_pkg -> send_map_starts[i])
#define hypre_CommPkgSendMapElmts(comm_pkg)  (comm_pkg -> send_map_elmts)
#define hypre_CommPkgSendMapElmt(comm_pkg,i) (comm_pkg -> send_map_elmts[i])

#define hypre_CommPkgNumRecvs(comm_pkg)      (comm_pkg -> num_recvs)
#define hypre_CommPkgRecvProcs(comm_pkg)     (comm_pkg -> recv_procs)
#define hypre_CommPkgRecvProc(comm_pkg, i)   (comm_pkg -> recv_procs[i])
#define hypre_CommPkgRecvVecStarts(comm_pkg) (comm_pkg -> recv_vec_starts)
#define hypre_CommPkgRecvVecStart(comm_pkg,i)(comm_pkg -> recv_vec_starts[i])

#define hypre_CommPkgSendMPITypes(comm_pkg)  (comm_pkg -> send_mpi_types)
#define hypre_CommPkgSendMPIType(comm_pkg,i) (comm_pkg -> send_mpi_types[i])

#define hypre_CommPkgRecvMPITypes(comm_pkg)  (comm_pkg -> recv_mpi_types)
#define hypre_CommPkgRecvMPIType(comm_pkg,i) (comm_pkg -> recv_mpi_types[i])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_CommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define hypre_CommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define hypre_CommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define hypre_CommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_CommHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_CommHandleRequest(comm_handle, i)  (comm_handle -> requests[i])

/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

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

typedef struct
{
   MPI_Comm	 comm;

   int      	 global_size;
   int      	 first_index;
   int      	*partitioning;
   hypre_Vector	*local_vector; 

   /* Does the Vector create/destroy `data'? */
   int      	 owns_data;
   int      	 owns_partitioning;

} hypre_ParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_ParVectorComm(vector)  	        ((vector) -> comm)
#define hypre_ParVectorGlobalSize(vector)       ((vector) -> global_size)
#define hypre_ParVectorFirstIndex(vector)       ((vector) -> first_index)
#define hypre_ParVectorPartitioning(vector)     ((vector) -> partitioning)
#define hypre_ParVectorLocalVector(vector)      ((vector) -> local_vector)
#define hypre_ParVectorOwnsData(vector)         ((vector) -> owns_data)
#define hypre_ParVectorOwnsPartitioning(vector) ((vector) -> owns_partitioning)

#endif
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

   int     		global_num_rows;
   int     		global_num_cols;
   int			first_row_index;
   int			first_col_diag;
   hypre_CSRMatrix	*diag;
   hypre_CSRMatrix	*offd;
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

   hypre_CommPkg	*comm_pkg;
   
   /* Does the ParCSRMatrix create/destroy `diag', `offd', `col_map_offd'? */
   int      owns_data;
   /* Does the ParCSRMatrix create/destroy `row_starts', `col_starts'? */
   int      owns_row_starts;
   int      owns_col_starts;

   int      num_nonzeros;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   int     *rowindices;
   double  *rowvalues;
   int      getrowactive;

} hypre_ParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRMatrixComm(matrix)		  ((matrix) -> comm)
#define hypre_ParCSRMatrixGlobalNumRows(matrix)   ((matrix) -> global_num_rows)
#define hypre_ParCSRMatrixGlobalNumCols(matrix)   ((matrix) -> global_num_cols)
#define hypre_ParCSRMatrixFirstRowIndex(matrix)   ((matrix) -> first_row_index)
#define hypre_ParCSRMatrixFirstColDiag(matrix)    ((matrix) -> first_col_diag)
#define hypre_ParCSRMatrixDiag(matrix)  	  ((matrix) -> diag)
#define hypre_ParCSRMatrixOffd(matrix)  	  ((matrix) -> offd)
#define hypre_ParCSRMatrixColMapOffd(matrix)  	  ((matrix) -> col_map_offd)
#define hypre_ParCSRMatrixRowStarts(matrix)       ((matrix) -> row_starts)
#define hypre_ParCSRMatrixColStarts(matrix)       ((matrix) -> col_starts)
#define hypre_ParCSRMatrixCommPkg(matrix)	  ((matrix) -> comm_pkg)
#define hypre_ParCSRMatrixOwnsData(matrix)        ((matrix) -> owns_data)
#define hypre_ParCSRMatrixOwnsRowStarts(matrix)   ((matrix) -> owns_row_starts)
#define hypre_ParCSRMatrixOwnsColStarts(matrix)   ((matrix) -> owns_col_starts)
#define hypre_ParCSRMatrixNumRows(matrix) \
hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(matrix))
#define hypre_ParCSRMatrixNumCols(matrix) \
hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(matrix))
#define hypre_ParCSRMatrixNumNonzeros(matrix)     ((matrix) -> num_nonzeros)
#define hypre_ParCSRMatrixRowindices(matrix)      ((matrix) -> rowindices)
#define hypre_ParCSRMatrixRowvalues(matrix)       ((matrix) -> rowvalues)
#define hypre_ParCSRMatrixGetrowactive(matrix)    ((matrix) -> getrowactive)

#endif
# define	P(s) s

/* F90_HYPRE_parcsr_matrix.c */
void hypre_F90_IFACE P((int hypre_newparcsrmatrix ));
void hypre_F90_IFACE P((int hypre_destroyparcsrmatrix ));
void hypre_F90_IFACE P((int hypre_initializeparcsrmatrix ));
void hypre_F90_IFACE P((int hypre_printparcsrmatrix ));
void hypre_F90_IFACE P((int hypre_getcommparcsr ));
void hypre_F90_IFACE P((int hypre_getdimsparcsr ));
void hypre_F90_IFACE P((int hypre_getlocalrangeparcsr ));
void hypre_F90_IFACE P((int hypre_getrowparcsrmatrix ));
void hypre_F90_IFACE P((int hypre_restorerowparcsrmatrix ));

/* F90_HYPRE_parcsr_vector.c */
void hypre_F90_IFACE P((int hypre_newparvector ));
void hypre_F90_IFACE P((int hypre_destroyparvector ));
void hypre_F90_IFACE P((int hypre_initializeparvector ));
void hypre_F90_IFACE P((int hypre_printparvector ));

/* F90_par_vector.c */
void hypre_F90_IFACE P((int hypre_createparvector ));
void hypre_F90_IFACE P((int hypre_setparvectordataowner ));
void hypre_F90_IFACE P((int hypre_setparvectorpartitioningo ));
void hypre_F90_IFACE P((int hypre_readparvector ));
void hypre_F90_IFACE P((int hypre_setparvectorconstantvalue ));
void hypre_F90_IFACE P((int hypre_setparvectorrandomvalues ));
void hypre_F90_IFACE P((int hypre_copyparvector ));
void hypre_F90_IFACE P((int hypre_scaleparvector ));
void hypre_F90_IFACE P((int hypre_paraxpy ));
void hypre_F90_IFACE P((int hypre_parinnerprod ));
void hypre_F90_IFACE P((int hypre_vectortoparvector ));
void hypre_F90_IFACE P((int hypre_parvectortovectorall ));

/* HYPRE_parcsr_matrix.c */
HYPRE_ParCSRMatrix HYPRE_CreateParCSRMatrix P((MPI_Comm comm , int global_num_rows , int global_num_cols , int *row_starts , int *col_starts , int num_cols_offd , int num_nonzeros_diag , int num_nonzeros_offd ));
int HYPRE_DestroyParCSRMatrix P((HYPRE_ParCSRMatrix matrix ));
int HYPRE_InitializeParCSRMatrix P((HYPRE_ParCSRMatrix matrix ));
void HYPRE_PrintParCSRMatrix P((HYPRE_ParCSRMatrix matrix , char *file_name ));
int HYPRE_GetCommParCSR P((HYPRE_ParCSRMatrix matrix , MPI_Comm *comm ));
int HYPRE_GetDimsParCSR P((HYPRE_ParCSRMatrix matrix , int *M , int *N ));
int HYPRE_GetLocalRangeParcsr P((HYPRE_ParCSRMatrix matrix , int *start , int *end ));
int HYPRE_GetRowParCSRMatrix P((HYPRE_ParCSRMatrix matrix , int row , int *size , int **col_ind , double **values ));
int HYPRE_RestoreRowParCSRMatrix P((HYPRE_ParCSRMatrix matrix , int row , int *size , int **col_ind , double **values ));

/* HYPRE_parcsr_vector.c */
HYPRE_ParVector HYPRE_CreateParVector P((MPI_Comm comm , int global_size , int *partitioning ));
int HYPRE_DestroyParVector P((HYPRE_ParVector vector ));
int HYPRE_InitializeParVector P((HYPRE_ParVector vector ));
int HYPRE_PrintParVector P((HYPRE_ParVector vector , char *file_name ));

/* communication.c */
hypre_CommHandle *hypre_InitializeCommunication P((int job , hypre_CommPkg *comm_pkg , void *send_data , void *recv_data ));
int hypre_FinalizeCommunication P((hypre_CommHandle *comm_handle ));
int hypre_GenerateMatvecCommunicationInfo P((hypre_ParCSRMatrix *A ));
int hypre_DestroyMatvecCommPkg P((hypre_CommPkg *comm_pkg ));
int hypre_BuildCSRMatrixMPIDataType P((int num_nonzeros , int num_rows , double *a_data , int *a_i , int *a_j , MPI_Datatype *csr_matrix_datatype ));
int hypre_BuildCSRJDataType P((int num_nonzeros , double *a_data , int *a_j , MPI_Datatype *csr_jdata_datatype ));

/* driver.c */
int main P((int argc , char *argv []));

/* driver_matmul.c */
int main P((int argc , char *argv []));

/* driver_matvec.c */
int main P((int argc , char *argv []));

/* par_csr_matop.c */
hypre_ParCSRMatrix *hypre_ParMatmul P((hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B ));
hypre_CSRMatrix *hypre_ExtractBExt P((hypre_ParCSRMatrix *B , hypre_ParCSRMatrix *A , int data ));

/* par_csr_matrix.c */
hypre_ParCSRMatrix *hypre_CreateParCSRMatrix P((MPI_Comm comm , int global_num_rows , int global_num_cols , int *row_starts , int *col_starts , int num_cols_offd , int num_nonzeros_diag , int num_nonzeros_offd ));
int hypre_DestroyParCSRMatrix P((hypre_ParCSRMatrix *matrix ));
int hypre_InitializeParCSRMatrix P((hypre_ParCSRMatrix *matrix ));
int hypre_SetParCSRMatrixNumNonzeros P((hypre_ParCSRMatrix *matrix ));
int hypre_SetParCSRMatrixDataOwner P((hypre_ParCSRMatrix *matrix , int owns_data ));
int hypre_SetParCSRMatrixRowStartsOwner P((hypre_ParCSRMatrix *matrix , int owns_row_starts ));
int hypre_SetParCSRMatrixColStartsOwner P((hypre_ParCSRMatrix *matrix , int owns_col_starts ));
hypre_ParCSRMatrix *hypre_ReadParCSRMatrix P((MPI_Comm comm , char *file_name ));
int hypre_PrintParCSRMatrix P((hypre_ParCSRMatrix *matrix , char *file_name ));
int hypre_GetLocalRangeParCSRMatrix P((hypre_ParCSRMatrix *matrix , int *start , int *end ));
int hypre_GetRowParCSRMatrix P((hypre_ParCSRMatrix *mat , int row , int *size , int **col_ind , double **values ));
int hypre_RestoreRowParCSRMatrix P((hypre_ParCSRMatrix *matrix , int row , int *size , int **col_ind , double **values ));
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix P((MPI_Comm comm , hypre_CSRMatrix *A , int *row_starts , int *col_starts ));
int GenerateDiagAndOffd P((hypre_CSRMatrix *A , hypre_ParCSRMatrix *matrix , int first_col_diag , int last_col_diag ));
hypre_CSRMatrix *hypre_MergeDiagAndOffd P((hypre_ParCSRMatrix *par_matrix ));
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll P((hypre_ParCSRMatrix *par_matrix ));
int hypre_CopyParCSRMatrix P((hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B , int copy_data ));

/* par_csr_matvec.c */
int hypre_ParMatvec P((double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y ));
int hypre_ParMatvecT P((double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y ));

/* par_vector.c */
hypre_ParVector *hypre_CreateParVector P((MPI_Comm comm , int global_size , int *partitioning ));
int hypre_DestroyParVector P((hypre_ParVector *vector ));
int hypre_InitializeParVector P((hypre_ParVector *vector ));
int hypre_SetParVectorDataOwner P((hypre_ParVector *vector , int owns_data ));
int hypre_SetParVectorPartitioningOwner P((hypre_ParVector *vector , int owns_partitioning ));
hypre_ParVector *hypre_ReadParVector P((MPI_Comm comm , char *file_name ));
int hypre_PrintParVector P((hypre_ParVector *vector , char *file_name ));
int hypre_SetParVectorConstantValues P((hypre_ParVector *v , double value ));
int hypre_SetParVectorRandomValues P((hypre_ParVector *v , int seed ));
int hypre_CopyParVector P((hypre_ParVector *x , hypre_ParVector *y ));
int hypre_ScaleParVector P((double alpha , hypre_ParVector *y ));
int hypre_ParAxpy P((double alpha , hypre_ParVector *x , hypre_ParVector *y ));
double hypre_ParInnerProd P((hypre_ParVector *x , hypre_ParVector *y ));
hypre_ParVector *hypre_VectorToParVector P((MPI_Comm comm , hypre_Vector *v , int *vec_starts ));
hypre_Vector *hypre_ParVectorToVectorAll P((hypre_ParVector *par_v ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

