
#include <HYPRE_config.h>

#include "HYPRE_parcsr_mv.h"

#ifndef hypre_PARCSR_MV_HEADER
#define hypre_PARCSR_MV_HEADER

#include "utilities.h"
#include "seq_matrix_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommPkg:
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

} hypre_ParCSRCommPkg;

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_ParCSRCommPkg  *comm_pkg;
   void 	  *send_data;
   void 	  *recv_data;

   int             num_requests;
   MPI_Request    *requests;

} hypre_ParCSRCommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_VectorCommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_VectorCommPkgComm(comm_pkg)           (comm_pkg -> comm)
#define hypre_VectorCommPkgVecStarts(comm_pkg)      (comm_pkg -> vec_starts)
#define hypre_VectorCommPkgVecStart(comm_pkg,i)     (comm_pkg -> vec_starts[i])
#define hypre_VectorCommPkgVectorMPITypes(comm_pkg) (comm_pkg -> vector_mpi_types)
#define hypre_VectorCommPkgVectorMPIType(comm_pkg,i)(comm_pkg -> vector_mpi_types[i])
                                               
/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_ParCSRCommPkgComm(comm_pkg)          (comm_pkg -> comm)
                                               
#define hypre_ParCSRCommPkgNumSends(comm_pkg)      (comm_pkg -> num_sends)
#define hypre_ParCSRCommPkgSendProcs(comm_pkg)     (comm_pkg -> send_procs)
#define hypre_ParCSRCommPkgSendProc(comm_pkg, i)   (comm_pkg -> send_procs[i])
#define hypre_ParCSRCommPkgSendMapStarts(comm_pkg) (comm_pkg -> send_map_starts)
#define hypre_ParCSRCommPkgSendMapStart(comm_pkg,i)(comm_pkg -> send_map_starts[i])
#define hypre_ParCSRCommPkgSendMapElmts(comm_pkg)  (comm_pkg -> send_map_elmts)
#define hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i) (comm_pkg -> send_map_elmts[i])

#define hypre_ParCSRCommPkgNumRecvs(comm_pkg)      (comm_pkg -> num_recvs)
#define hypre_ParCSRCommPkgRecvProcs(comm_pkg)     (comm_pkg -> recv_procs)
#define hypre_ParCSRCommPkgRecvProc(comm_pkg, i)   (comm_pkg -> recv_procs[i])
#define hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) (comm_pkg -> recv_vec_starts)
#define hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i)(comm_pkg -> recv_vec_starts[i])

#define hypre_ParCSRCommPkgSendMPITypes(comm_pkg)  (comm_pkg -> send_mpi_types)
#define hypre_ParCSRCommPkgSendMPIType(comm_pkg,i) (comm_pkg -> send_mpi_types[i])

#define hypre_ParCSRCommPkgRecvMPITypes(comm_pkg)  (comm_pkg -> recv_mpi_types)
#define hypre_ParCSRCommPkgRecvMPIType(comm_pkg,i) (comm_pkg -> recv_mpi_types[i])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_ParCSRCommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define hypre_ParCSRCommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define hypre_ParCSRCommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define hypre_ParCSRCommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_ParCSRCommHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_ParCSRCommHandleRequest(comm_handle, i)  (comm_handle -> requests[i])

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
 * Fortran <-> C interface macros
 *
 *****************************************************************************/

#ifndef HYPRE_FORTRAN_HEADER
#define HYPRE_FORTRAN_HEADER

#if defined(IRIX) || defined(DEC)
#define hypre_NAME_C_FOR_FORTRAN(name) name##_
#define hypre_NAME_FORTRAN_FOR_C(name) name##_
#else
#define hypre_NAME_C_FOR_FORTRAN(name) name##__
#define hypre_NAME_FORTRAN_FOR_C(name) name##_
#endif

#define hypre_F90_IFACE(iface_name) hypre_NAME_FORTRAN_FOR_C(iface_name)

#endif
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

   hypre_ParCSRCommPkg	*comm_pkg;
   
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
void hypre_F90_IFACE P((int hypre_parcsrmatrixcreate ));
void hypre_F90_IFACE P((int hypre_parcsrmatrixdestroy ));
void hypre_F90_IFACE P((int hypre_parcsrmatrixinitialize ));
void hypre_F90_IFACE P((int hypre_parcsrmatrixread ));
void hypre_F90_IFACE P((int hypre_parcsrmatrixprint ));
void hypre_F90_IFACE P((int hypre_parcsrmatrixgetcomm ));
void hypre_F90_IFACE P((int hypre_parcsrmatrixgetdims ));
void hypre_F90_IFACE P((int hypre_parcsrmatrixgetlocalrange ));
void hypre_F90_IFACE P((int hypre_parcsrmatrixgetrow ));
void hypre_F90_IFACE P((int hypre_parcsrmatrixrestorerow ));

/* F90_HYPRE_parcsr_vector.c */
void hypre_F90_IFACE P((int hypre_parvectorcreate ));
void hypre_F90_IFACE P((int hypre_parvectordestroy ));
void hypre_F90_IFACE P((int hypre_parvectorinitialize ));
void hypre_F90_IFACE P((int hypre_parvectorread ));
void hypre_F90_IFACE P((int hypre_parvectorprint ));

/* F90_par_vector.c */
void hypre_F90_IFACE P((int hypre_createparvector ));
void hypre_F90_IFACE P((int hypre_setparvectordataowner ));
void hypre_F90_IFACE P((int hypre_setparvectorpartitioningo ));
void hypre_F90_IFACE P((int hypre_setparvectorconstantvalue ));
void hypre_F90_IFACE P((int hypre_setparvectorrandomvalues ));
void hypre_F90_IFACE P((int hypre_copyparvector ));
void hypre_F90_IFACE P((int hypre_scaleparvector ));
void hypre_F90_IFACE P((int hypre_paraxpy ));
void hypre_F90_IFACE P((int hypre_parinnerprod ));
void hypre_F90_IFACE P((int hypre_vectortoparvector ));
void hypre_F90_IFACE P((int hypre_parvectortovectorall ));

/* F90_parcsr_matrix.c */
void hypre_F90_IFACE P((int hypre_parcsrmatrixglobalnumrows ));
void hypre_F90_IFACE P((int hypre_parcsrmatrixrowstarts ));

/* HYPRE_parcsr_matrix.c */
HYPRE_ParCSRMatrix HYPRE_ParCSRMatrixCreate P((MPI_Comm comm , int global_num_rows , int global_num_cols , int *row_starts , int *col_starts , int num_cols_offd , int num_nonzeros_diag , int num_nonzeros_offd ));
int HYPRE_ParCSRMatrixDestroy P((HYPRE_ParCSRMatrix matrix ));
int HYPRE_ParCSRMatrixInitialize P((HYPRE_ParCSRMatrix matrix ));
HYPRE_ParCSRMatrix HYPRE_ParCSRMatrixRead P((MPI_Comm comm , char *file_name ));
void HYPRE_ParCSRMatrixPrint P((HYPRE_ParCSRMatrix matrix , char *file_name ));
int HYPRE_ParCSRMatrixGetComm P((HYPRE_ParCSRMatrix matrix , MPI_Comm *comm ));
int HYPRE_ParCSRMatrixGetDims P((HYPRE_ParCSRMatrix matrix , int *M , int *N ));
int HYPRE_ParCSRMatrixGetRowPartitioning P((HYPRE_ParCSRMatrix matrix , int **row_partitioning_ptr ));
int HYPRE_ParCSRMatrixGetColPartitioning P((HYPRE_ParCSRMatrix matrix , int **col_partitioning_ptr ));
int HYPRE_ParCSRMatrixGetLocalRange P((HYPRE_ParCSRMatrix matrix , int *row_start , int *row_end , int *col_start , int *col_end ));
int HYPRE_ParCSRMatrixGetRow P((HYPRE_ParCSRMatrix matrix , int row , int *size , int **col_ind , double **values ));
int HYPRE_ParCSRMatrixRestoreRow P((HYPRE_ParCSRMatrix matrix , int row , int *size , int **col_ind , double **values ));
HYPRE_ParCSRMatrix HYPRE_CSRMatrixToParCSRMatrix P((MPI_Comm comm , HYPRE_CSRMatrix A_CSR , int *row_partitioning , int *col_partitioning ));
int HYPRE_ParCSRMatrixMatvec P((double alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , double beta , HYPRE_ParVector y ));

/* HYPRE_parcsr_vector.c */
HYPRE_ParVector HYPRE_ParVectorCreate P((MPI_Comm comm , int global_size , int *partitioning ));
int HYPRE_ParVectorDestroy P((HYPRE_ParVector vector ));
int HYPRE_ParVectorInitialize P((HYPRE_ParVector vector ));
HYPRE_ParVector HYPRE_ParVectorRead P((MPI_Comm comm , char *file_name ));
int HYPRE_ParVectorPrint P((HYPRE_ParVector vector , char *file_name ));
int HYPRE_ParVectorSetConstantValues P((HYPRE_ParVector vector , double value ));
int HYPRE_ParVectorSetRandomValues P((HYPRE_ParVector vector , int seed ));
int HYPRE_ParVectorCopy P((HYPRE_ParVector x , HYPRE_ParVector y ));
int HYPRE_ParVectorScale P((double value , HYPRE_ParVector x ));
double HYPRE_ParVectorInnerProd P((HYPRE_ParVector x , HYPRE_ParVector y ));
HYPRE_ParVector HYPRE_VectorToParVector P((MPI_Comm comm , HYPRE_Vector b , int *partitioning ));

/* communication.c */
hypre_ParCSRCommHandle *hypre_InitializeCommunication P((int job , hypre_ParCSRCommPkg *comm_pkg , void *send_data , void *recv_data ));
int hypre_FinalizeCommunication P((hypre_ParCSRCommHandle *comm_handle ));
int hypre_MatvecCommPkgCreate P((hypre_ParCSRMatrix *A ));
int hypre_MatvecCommPkgDestroy P((hypre_ParCSRCommPkg *comm_pkg ));
int hypre_BuildCSRMatrixMPIDataType P((int num_nonzeros , int num_rows , double *a_data , int *a_i , int *a_j , MPI_Datatype *csr_matrix_datatype ));
int hypre_BuildCSRJDataType P((int num_nonzeros , double *a_data , int *a_j , MPI_Datatype *csr_jdata_datatype ));

/* par_csr_matop.c */
hypre_ParCSRMatrix *hypre_ParMatmul P((hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B ));
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt P((hypre_ParCSRMatrix *B , hypre_ParCSRMatrix *A , int data ));

/* par_csr_matrix.c */
hypre_ParCSRMatrix *hypre_ParCSRMatrixCreate P((MPI_Comm comm , int global_num_rows , int global_num_cols , int *row_starts , int *col_starts , int num_cols_offd , int num_nonzeros_diag , int num_nonzeros_offd ));
int hypre_ParCSRMatrixDestroy P((hypre_ParCSRMatrix *matrix ));
int hypre_ParCSRMatrixInitialize P((hypre_ParCSRMatrix *matrix ));
int hypre_ParCSRMatrixSetNumNonzeros P((hypre_ParCSRMatrix *matrix ));
int hypre_ParCSRMatrixSetDataOwner P((hypre_ParCSRMatrix *matrix , int owns_data ));
int hypre_ParCSRMatrixSetRowStartsOwner P((hypre_ParCSRMatrix *matrix , int owns_row_starts ));
int hypre_ParCSRMatrixSetColStartsOwner P((hypre_ParCSRMatrix *matrix , int owns_col_starts ));
hypre_ParCSRMatrix *hypre_ParCSRMatrixRead P((MPI_Comm comm , char *file_name ));
int hypre_ParCSRMatrixPrint P((hypre_ParCSRMatrix *matrix , char *file_name ));
int hypre_ParCSRMatrixGetLocalRange P((hypre_ParCSRMatrix *matrix , int *row_start , int *row_end , int *col_start , int *col_end ));
int hypre_ParCSRMatrixGetRow P((hypre_ParCSRMatrix *mat , int row , int *size , int **col_ind , double **values ));
int hypre_ParCSRMatrixRestoreRow P((hypre_ParCSRMatrix *matrix , int row , int *size , int **col_ind , double **values ));
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix P((MPI_Comm comm , hypre_CSRMatrix *A , int *row_starts , int *col_starts ));
int GenerateDiagAndOffd P((hypre_CSRMatrix *A , hypre_ParCSRMatrix *matrix , int first_col_diag , int last_col_diag ));
hypre_CSRMatrix *hypre_MergeDiagAndOffd P((hypre_ParCSRMatrix *par_matrix ));
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll P((hypre_ParCSRMatrix *par_matrix ));
int hypre_ParCSRMatrixCopy P((hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B , int copy_data ));

/* par_csr_matvec.c */
int hypre_ParCSRMatrixMatvec P((double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y ));
int hypre_ParCSRMatrixMatvecT P((double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y ));

/* par_vector.c */
hypre_ParVector *hypre_ParVectorCreate P((MPI_Comm comm , int global_size , int *partitioning ));
int hypre_ParVectorDestroy P((hypre_ParVector *vector ));
int hypre_ParVectorInitialize P((hypre_ParVector *vector ));
int hypre_ParVectorSetDataOwner P((hypre_ParVector *vector , int owns_data ));
int hypre_ParVectorSetPartitioningOwner P((hypre_ParVector *vector , int owns_partitioning ));
hypre_ParVector *hypre_ParVectorRead P((MPI_Comm comm , char *file_name ));
int hypre_ParVectorPrint P((hypre_ParVector *vector , char *file_name ));
int hypre_ParVectorSetConstantValues P((hypre_ParVector *v , double value ));
int hypre_ParVectorSetRandomValues P((hypre_ParVector *v , int seed ));
int hypre_ParVectorCopy P((hypre_ParVector *x , hypre_ParVector *y ));
int hypre_ParVectorScale P((double alpha , hypre_ParVector *y ));
int hypre_ParVectorAxpy P((double alpha , hypre_ParVector *x , hypre_ParVector *y ));
double hypre_ParVectorInnerProd P((hypre_ParVector *x , hypre_ParVector *y ));
hypre_ParVector *hypre_VectorToParVector P((MPI_Comm comm , hypre_Vector *v , int *vec_starts ));
hypre_Vector *hypre_ParVectorToVectorAll P((hypre_ParVector *par_v ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

