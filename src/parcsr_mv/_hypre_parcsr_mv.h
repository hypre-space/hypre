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

#ifndef hypre_PARCSR_MV_HEADER
#define hypre_PARCSR_MV_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "HYPRE_parcsr_mv.h"
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

typedef struct
{
   MPI_Comm               comm;

   HYPRE_Int                    num_sends;
   HYPRE_Int                   *send_procs;
   HYPRE_Int			 *send_map_starts;
   HYPRE_Int			 *send_map_elmts;

   HYPRE_Int                    num_recvs;
   HYPRE_Int                   *recv_procs;
   HYPRE_Int                   *recv_vec_starts;

   /* remote communication information */
   hypre_MPI_Datatype          *send_mpi_types;
   hypre_MPI_Datatype          *recv_mpi_types;

} hypre_ParCSRCommPkg;

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_ParCSRCommPkg  *comm_pkg;
   void 	  *send_data;
   void 	  *recv_data;

   HYPRE_Int             num_requests;
   hypre_MPI_Request    *requests;

} hypre_ParCSRCommHandle;

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

#endif /* HYPRE_PAR_CSR_COMMUNICATION_HEADER */

#ifndef hypre_PARCSR_ASSUMED_PART
#define  hypre_PARCSR_ASSUMED_PART

typedef struct
{
   HYPRE_Int                   length;
   HYPRE_Int                   row_start;
   HYPRE_Int                   row_end;
   HYPRE_Int                   storage_length;
   HYPRE_Int                   *proc_list;
   HYPRE_Int		         *row_start_list;
   HYPRE_Int                   *row_end_list;  
  HYPRE_Int                    *sort_index;
} hypre_IJAssumedPart;




#endif /* hypre_PARCSR_ASSUMED_PART */





#ifndef hypre_NEW_COMMPKG
#define hypre_NEW_COMMPKG


typedef struct
{
   HYPRE_Int                   length;
   HYPRE_Int                   storage_length; 
   HYPRE_Int                   *id;
   HYPRE_Int                   *vec_starts;
   HYPRE_Int                   element_storage_length; 
   HYPRE_Int                   *elements;
   double                *d_elements;
   void                  *v_elements;
   
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

typedef struct
{
   MPI_Comm	 comm;

   HYPRE_Int      	 global_size;
   HYPRE_Int      	 first_index;
   HYPRE_Int           last_index;
   HYPRE_Int      	*partitioning;
   hypre_Vector	*local_vector; 

   /* Does the Vector create/destroy `data'? */
   HYPRE_Int      	 owns_data;
   HYPRE_Int      	 owns_partitioning;

   hypre_IJAssumedPart *assumed_partition; /* only populated if no_global_partition option
                                              is used (compile-time option) AND this partition
                                              needed
                                              (for setting off-proc elements, for example)*/


} hypre_ParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_ParVectorComm(vector)  	        ((vector) -> comm)
#define hypre_ParVectorGlobalSize(vector)       ((vector) -> global_size)
#define hypre_ParVectorFirstIndex(vector)       ((vector) -> first_index)
#define hypre_ParVectorLastIndex(vector)        ((vector) -> last_index)
#define hypre_ParVectorPartitioning(vector)     ((vector) -> partitioning)
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

typedef struct {
   void * digit[11];
/* ... should be   hypre_NumbersNode * digit[11]; */
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
  HYPRE_Int *num_inchords;
  HYPRE_Int **inchord_idof;
  HYPRE_Int **inchord_rdof;
  double **inchord_data;

  HYPRE_Int num_idofs;
  HYPRE_Int num_rdofs;

  HYPRE_Int *firstindex_idof; /* not owned by my_id; ----------------------------- */
  HYPRE_Int *firstindex_rdof; /* not owned by my_id; ----------------------------- */

  /* --------------------------- mirror information: ---------------------- */
  /* participation of rdof in different processors; ------------------------ */

  HYPRE_Int num_toprocessors;
  HYPRE_Int *toprocessor;

  /* rdofs to be sentto toprocessors; --------------------------------------
     ----------------------------------------------------------------------- */
  HYPRE_Int *num_rdofs_toprocessor;
  HYPRE_Int **rdof_toprocessor;


} hypre_ParChordMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParChordMatrixComm(matrix)		  ((matrix) -> comm)

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
void RowsWithColumn_original ( HYPRE_Int *rowmin , HYPRE_Int *rowmax , HYPRE_Int column , hypre_ParCSRMatrix *A );
void RowsWithColumn ( HYPRE_Int *rowmin , HYPRE_Int *rowmax , HYPRE_Int column , HYPRE_Int num_rows_diag , HYPRE_Int firstColDiag , HYPRE_Int *colMapOffd , HYPRE_Int *mat_i_diag , HYPRE_Int *mat_j_diag , HYPRE_Int *mat_i_offd , HYPRE_Int *mat_j_offd );
void hypre_MatTCommPkgCreate_core ( MPI_Comm comm , HYPRE_Int *col_map_offd , HYPRE_Int first_col_diag , HYPRE_Int *col_starts , HYPRE_Int num_rows_diag , HYPRE_Int num_cols_diag , HYPRE_Int num_cols_offd , HYPRE_Int *row_starts , HYPRE_Int firstColDiag , HYPRE_Int *colMapOffd , HYPRE_Int *mat_i_diag , HYPRE_Int *mat_j_diag , HYPRE_Int *mat_i_offd , HYPRE_Int *mat_j_offd , HYPRE_Int data , HYPRE_Int *p_num_recvs , HYPRE_Int **p_recv_procs , HYPRE_Int **p_recv_vec_starts , HYPRE_Int *p_num_sends , HYPRE_Int **p_send_procs , HYPRE_Int **p_send_map_starts , HYPRE_Int **p_send_map_elmts );
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
HYPRE_Int HYPRE_ParCSRMatrixCreate ( MPI_Comm comm , HYPRE_Int global_num_rows , HYPRE_Int global_num_cols , HYPRE_Int *row_starts , HYPRE_Int *col_starts , HYPRE_Int num_cols_offd , HYPRE_Int num_nonzeros_diag , HYPRE_Int num_nonzeros_offd , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixDestroy ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixInitialize ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixRead ( MPI_Comm comm , const char *file_name , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixPrint ( HYPRE_ParCSRMatrix matrix , const char *file_name );
HYPRE_Int HYPRE_ParCSRMatrixGetComm ( HYPRE_ParCSRMatrix matrix , MPI_Comm *comm );
HYPRE_Int HYPRE_ParCSRMatrixGetDims ( HYPRE_ParCSRMatrix matrix , HYPRE_Int *M , HYPRE_Int *N );
HYPRE_Int HYPRE_ParCSRMatrixGetRowPartitioning ( HYPRE_ParCSRMatrix matrix , HYPRE_Int **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetColPartitioning ( HYPRE_ParCSRMatrix matrix , HYPRE_Int **col_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetLocalRange ( HYPRE_ParCSRMatrix matrix , HYPRE_Int *row_start , HYPRE_Int *row_end , HYPRE_Int *col_start , HYPRE_Int *col_end );
HYPRE_Int HYPRE_ParCSRMatrixGetRow ( HYPRE_ParCSRMatrix matrix , HYPRE_Int row , HYPRE_Int *size , HYPRE_Int **col_ind , double **values );
HYPRE_Int HYPRE_ParCSRMatrixRestoreRow ( HYPRE_ParCSRMatrix matrix , HYPRE_Int row , HYPRE_Int *size , HYPRE_Int **col_ind , double **values );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix ( MPI_Comm comm , HYPRE_CSRMatrix A_CSR , HYPRE_Int *row_partitioning , HYPRE_Int *col_partitioning , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning ( MPI_Comm comm , HYPRE_CSRMatrix A_CSR , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixMatvec ( double alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , double beta , HYPRE_ParVector y );
HYPRE_Int HYPRE_ParCSRMatrixMatvecT ( double alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , double beta , HYPRE_ParVector y );

/* HYPRE_parcsr_vector.c */
HYPRE_Int HYPRE_ParVectorCreate ( MPI_Comm comm , HYPRE_Int global_size , HYPRE_Int *partitioning , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParMultiVectorCreate ( MPI_Comm comm , HYPRE_Int global_size , HYPRE_Int *partitioning , HYPRE_Int number_vectors , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorDestroy ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorInitialize ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorRead ( MPI_Comm comm , const char *file_name , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorPrint ( HYPRE_ParVector vector , const char *file_name );
HYPRE_Int HYPRE_ParVectorSetConstantValues ( HYPRE_ParVector vector , double value );
HYPRE_Int HYPRE_ParVectorSetRandomValues ( HYPRE_ParVector vector , HYPRE_Int seed );
HYPRE_Int HYPRE_ParVectorCopy ( HYPRE_ParVector x , HYPRE_ParVector y );
HYPRE_ParVector HYPRE_ParVectorCloneShallow ( HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorScale ( double value , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorAxpy ( double alpha , HYPRE_ParVector x , HYPRE_ParVector y );
HYPRE_Int HYPRE_ParVectorInnerProd ( HYPRE_ParVector x , HYPRE_ParVector y , double *prod );
HYPRE_Int HYPRE_VectorToParVector ( MPI_Comm comm , HYPRE_Vector b , HYPRE_Int *partitioning , HYPRE_ParVector *vector );

/* new_commpkg.c */
HYPRE_Int PrintCommpkg ( hypre_ParCSRMatrix *A , const char *file_name );
HYPRE_Int hypre_NewCommPkgCreate_core ( MPI_Comm comm , HYPRE_Int *col_map_off_d , HYPRE_Int first_col_diag , HYPRE_Int col_start , HYPRE_Int col_end , HYPRE_Int num_cols_off_d , HYPRE_Int global_num_cols , HYPRE_Int *p_num_recvs , HYPRE_Int **p_recv_procs , HYPRE_Int **p_recv_vec_starts , HYPRE_Int *p_num_sends , HYPRE_Int **p_send_procs , HYPRE_Int **p_send_map_starts , HYPRE_Int **p_send_map_elements , hypre_IJAssumedPart *apart );
HYPRE_Int hypre_NewCommPkgCreate ( hypre_ParCSRMatrix *parcsr_A );
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
void hypre_ParChordMatrix_RowStarts ( hypre_ParChordMatrix *Ac , MPI_Comm comm , HYPRE_Int **row_starts , HYPRE_Int *global_num_cols );
HYPRE_Int hypre_ParChordMatrixToParCSRMatrix ( hypre_ParChordMatrix *Ac , MPI_Comm comm , hypre_ParCSRMatrix **pAp );
HYPRE_Int hypre_ParCSRMatrixToParChordMatrix ( hypre_ParCSRMatrix *Ap , MPI_Comm comm , hypre_ParChordMatrix **pAc );

/* par_csr_aat.c */
void hypre_ParAat_RowSizes ( HYPRE_Int **C_diag_i , HYPRE_Int **C_offd_i , HYPRE_Int *B_marker , HYPRE_Int *A_diag_i , HYPRE_Int *A_diag_j , HYPRE_Int *A_offd_i , HYPRE_Int *A_offd_j , HYPRE_Int *A_col_map_offd , HYPRE_Int *A_ext_i , HYPRE_Int *A_ext_j , HYPRE_Int *A_ext_row_map , HYPRE_Int *C_diag_size , HYPRE_Int *C_offd_size , HYPRE_Int num_rows_diag_A , HYPRE_Int num_cols_offd_A , HYPRE_Int num_rows_A_ext , HYPRE_Int first_col_diag_A , HYPRE_Int first_row_index_A );
hypre_ParCSRMatrix *hypre_ParCSRAAt ( hypre_ParCSRMatrix *A );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractAExt ( hypre_ParCSRMatrix *A , HYPRE_Int data , HYPRE_Int **pA_ext_row_map );

/* par_csr_assumed_part.c */
HYPRE_Int hypre_LocateAssummedPartition ( MPI_Comm comm , HYPRE_Int row_start , HYPRE_Int row_end , HYPRE_Int global_num_rows , hypre_IJAssumedPart *part , HYPRE_Int myid );
HYPRE_Int hypre_ParCSRMatrixCreateAssumedPartition ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixDestroyAssumedPartition ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_GetAssumedPartitionProcFromRow ( MPI_Comm comm , HYPRE_Int row , HYPRE_Int global_num_rows , HYPRE_Int *proc_id );
HYPRE_Int hypre_GetAssumedPartitionRowRange ( MPI_Comm comm , HYPRE_Int proc_id , HYPRE_Int global_num_rows , HYPRE_Int *row_start , HYPRE_Int *row_end );
HYPRE_Int hypre_ParVectorCreateAssumedPartition ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorDestroyAssumedPartition ( hypre_ParVector *vector );

/* par_csr_bool_matop.c */
hypre_ParCSRBooleanMatrix *hypre_ParBooleanMatmul ( hypre_ParCSRBooleanMatrix *A , hypre_ParCSRBooleanMatrix *B );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractBExt ( hypre_ParCSRBooleanMatrix *B , hypre_ParCSRBooleanMatrix *A );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractAExt ( hypre_ParCSRBooleanMatrix *A , HYPRE_Int **pA_ext_row_map );
hypre_ParCSRBooleanMatrix *hypre_ParBooleanAAt ( hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanMatTCommPkgCreate ( hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanMatvecCommPkgCreate ( hypre_ParCSRBooleanMatrix *A );

/* par_csr_bool_matrix.c */
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixCreate ( HYPRE_Int num_rows , HYPRE_Int num_cols , HYPRE_Int num_nonzeros );
HYPRE_Int hypre_CSRBooleanMatrixDestroy ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixInitialize ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixSetDataOwner ( hypre_CSRBooleanMatrix *matrix , HYPRE_Int owns_data );
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixRead ( const char *file_name );
HYPRE_Int hypre_CSRBooleanMatrixPrint ( hypre_CSRBooleanMatrix *matrix , const char *file_name );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixCreate ( MPI_Comm comm , HYPRE_Int global_num_rows , HYPRE_Int global_num_cols , HYPRE_Int *row_starts , HYPRE_Int *col_starts , HYPRE_Int num_cols_offd , HYPRE_Int num_nonzeros_diag , HYPRE_Int num_nonzeros_offd );
HYPRE_Int hypre_ParCSRBooleanMatrixDestroy ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixInitialize ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixSetNNZ ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixSetDataOwner ( hypre_ParCSRBooleanMatrix *matrix , HYPRE_Int owns_data );
HYPRE_Int hypre_ParCSRBooleanMatrixSetRowStartsOwner ( hypre_ParCSRBooleanMatrix *matrix , HYPRE_Int owns_row_starts );
HYPRE_Int hypre_ParCSRBooleanMatrixSetColStartsOwner ( hypre_ParCSRBooleanMatrix *matrix , HYPRE_Int owns_col_starts );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixRead ( MPI_Comm comm , const char *file_name );
HYPRE_Int hypre_ParCSRBooleanMatrixPrint ( hypre_ParCSRBooleanMatrix *matrix , const char *file_name );
HYPRE_Int hypre_ParCSRBooleanMatrixPrintIJ ( hypre_ParCSRBooleanMatrix *matrix , const char *filename );
HYPRE_Int hypre_ParCSRBooleanMatrixGetLocalRange ( hypre_ParCSRBooleanMatrix *matrix , HYPRE_Int *row_start , HYPRE_Int *row_end , HYPRE_Int *col_start , HYPRE_Int *col_end );
HYPRE_Int hypre_ParCSRBooleanMatrixGetRow ( hypre_ParCSRBooleanMatrix *mat , HYPRE_Int row , HYPRE_Int *size , HYPRE_Int **col_ind );
HYPRE_Int hypre_ParCSRBooleanMatrixRestoreRow ( hypre_ParCSRBooleanMatrix *matrix , HYPRE_Int row , HYPRE_Int *size , HYPRE_Int **col_ind );
HYPRE_Int hypre_BuildCSRBooleanMatrixMPIDataType ( HYPRE_Int num_nonzeros , HYPRE_Int num_rows , HYPRE_Int *a_i , HYPRE_Int *a_j , hypre_MPI_Datatype *csr_matrix_datatype );
hypre_ParCSRBooleanMatrix *hypre_CSRBooleanMatrixToParCSRBooleanMatrix ( MPI_Comm comm , hypre_CSRBooleanMatrix *A , HYPRE_Int *row_starts , HYPRE_Int *col_starts );
HYPRE_Int BooleanGenerateDiagAndOffd ( hypre_CSRBooleanMatrix *A , hypre_ParCSRBooleanMatrix *matrix , HYPRE_Int first_col_diag , HYPRE_Int last_col_diag );

/* par_csr_communication.c */
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate ( HYPRE_Int job , hypre_ParCSRCommPkg *comm_pkg , void *send_data , void *recv_data );
HYPRE_Int hypre_ParCSRCommHandleDestroy ( hypre_ParCSRCommHandle *comm_handle );
void hypre_MatvecCommPkgCreate_core ( MPI_Comm comm , HYPRE_Int *col_map_offd , HYPRE_Int first_col_diag , HYPRE_Int *col_starts , HYPRE_Int num_cols_diag , HYPRE_Int num_cols_offd , HYPRE_Int firstColDiag , HYPRE_Int *colMapOffd , HYPRE_Int data , HYPRE_Int *p_num_recvs , HYPRE_Int **p_recv_procs , HYPRE_Int **p_recv_vec_starts , HYPRE_Int *p_num_sends , HYPRE_Int **p_send_procs , HYPRE_Int **p_send_map_starts , HYPRE_Int **p_send_map_elmts );
HYPRE_Int hypre_MatvecCommPkgCreate ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_MatvecCommPkgDestroy ( hypre_ParCSRCommPkg *comm_pkg );
HYPRE_Int hypre_BuildCSRMatrixMPIDataType ( HYPRE_Int num_nonzeros , HYPRE_Int num_rows , double *a_data , HYPRE_Int *a_i , HYPRE_Int *a_j , hypre_MPI_Datatype *csr_matrix_datatype );
HYPRE_Int hypre_BuildCSRJDataType ( HYPRE_Int num_nonzeros , double *a_data , HYPRE_Int *a_j , hypre_MPI_Datatype *csr_jdata_datatype );

/* par_csr_matop.c */
void hypre_ParMatmul_RowSizes ( HYPRE_Int **C_diag_i , HYPRE_Int **C_offd_i , HYPRE_Int **B_marker , HYPRE_Int *A_diag_i , HYPRE_Int *A_diag_j , HYPRE_Int *A_offd_i , HYPRE_Int *A_offd_j , HYPRE_Int *B_diag_i , HYPRE_Int *B_diag_j , HYPRE_Int *B_offd_i , HYPRE_Int *B_offd_j , HYPRE_Int *B_ext_diag_i , HYPRE_Int *B_ext_diag_j , HYPRE_Int *B_ext_offd_i , HYPRE_Int *B_ext_offd_j , HYPRE_Int *map_B_to_C , HYPRE_Int *C_diag_size , HYPRE_Int *C_offd_size , HYPRE_Int num_rows_diag_A , HYPRE_Int num_cols_offd_A , HYPRE_Int allsquare , HYPRE_Int num_cols_diag_B , HYPRE_Int num_cols_offd_B , HYPRE_Int num_cols_offd_C );
hypre_ParCSRMatrix *hypre_ParMatmul ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B );
void hypre_ParCSRMatrixExtractBExt_Arrays ( HYPRE_Int **pB_ext_i , HYPRE_Int **pB_ext_j , double **pB_ext_data , HYPRE_Int **pB_ext_row_map , HYPRE_Int *num_nonzeros , HYPRE_Int data , HYPRE_Int find_row_map , MPI_Comm comm , hypre_ParCSRCommPkg *comm_pkg , HYPRE_Int num_cols_B , HYPRE_Int num_recvs , HYPRE_Int num_sends , HYPRE_Int first_col_diag , HYPRE_Int first_row_index , HYPRE_Int *recv_vec_starts , HYPRE_Int *send_map_starts , HYPRE_Int *send_map_elmts , HYPRE_Int *diag_i , HYPRE_Int *diag_j , HYPRE_Int *offd_i , HYPRE_Int *offd_j , HYPRE_Int *col_map_offd , double *diag_data , double *offd_data );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt ( hypre_ParCSRMatrix *B , hypre_ParCSRMatrix *A , HYPRE_Int data );
HYPRE_Int hypre_ParCSRMatrixTranspose ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **AT_ptr , HYPRE_Int data );
void hypre_ParCSRMatrixGenSpanningTree ( hypre_ParCSRMatrix *G_csr , HYPRE_Int **indices , HYPRE_Int G_type );
void hypre_ParCSRMatrixExtractSubmatrices ( hypre_ParCSRMatrix *A_csr , HYPRE_Int *indices2 , hypre_ParCSRMatrix ***submatrices );
void hypre_ParCSRMatrixExtractRowSubmatrices ( hypre_ParCSRMatrix *A_csr , HYPRE_Int *indices2 , hypre_ParCSRMatrix ***submatrices );
double hypre_ParCSRMatrixLocalSumElts ( hypre_ParCSRMatrix *A );

/* par_csr_matop_marked.c */
void hypre_ParMatmul_RowSizes_Marked ( HYPRE_Int **C_diag_i , HYPRE_Int **C_offd_i , HYPRE_Int **B_marker , HYPRE_Int *A_diag_i , HYPRE_Int *A_diag_j , HYPRE_Int *A_offd_i , HYPRE_Int *A_offd_j , HYPRE_Int *B_diag_i , HYPRE_Int *B_diag_j , HYPRE_Int *B_offd_i , HYPRE_Int *B_offd_j , HYPRE_Int *B_ext_diag_i , HYPRE_Int *B_ext_diag_j , HYPRE_Int *B_ext_offd_i , HYPRE_Int *B_ext_offd_j , HYPRE_Int *map_B_to_C , HYPRE_Int *C_diag_size , HYPRE_Int *C_offd_size , HYPRE_Int num_rows_diag_A , HYPRE_Int num_cols_offd_A , HYPRE_Int allsquare , HYPRE_Int num_cols_diag_B , HYPRE_Int num_cols_offd_B , HYPRE_Int num_cols_offd_C , HYPRE_Int *CF_marker , HYPRE_Int *dof_func , HYPRE_Int *dof_func_offd );
hypre_ParCSRMatrix *hypre_ParMatmul_FC ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , HYPRE_Int *CF_marker , HYPRE_Int *dof_func , HYPRE_Int *dof_func_offd );
void hypre_ParMatScaleDiagInv_F ( hypre_ParCSRMatrix *C , hypre_ParCSRMatrix *A , double weight , HYPRE_Int *CF_marker );
hypre_ParCSRMatrix *hypre_ParMatMinus_F ( hypre_ParCSRMatrix *P , hypre_ParCSRMatrix *C , HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixZero_F ( hypre_ParCSRMatrix *P , HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixCopy_C ( hypre_ParCSRMatrix *P , hypre_ParCSRMatrix *C , HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixDropEntries ( hypre_ParCSRMatrix *C , hypre_ParCSRMatrix *P , HYPRE_Int *CF_marker );

/* par_csr_matrix.c */
hypre_ParCSRMatrix *hypre_ParCSRMatrixCreate ( MPI_Comm comm , HYPRE_Int global_num_rows , HYPRE_Int global_num_cols , HYPRE_Int *row_starts , HYPRE_Int *col_starts , HYPRE_Int num_cols_offd , HYPRE_Int num_nonzeros_diag , HYPRE_Int num_nonzeros_offd );
HYPRE_Int hypre_ParCSRMatrixDestroy ( hypre_ParCSRMatrix *matrix );
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
HYPRE_Int hypre_ParCSRMatrixGetLocalRange ( hypre_ParCSRMatrix *matrix , HYPRE_Int *row_start , HYPRE_Int *row_end , HYPRE_Int *col_start , HYPRE_Int *col_end );
HYPRE_Int hypre_ParCSRMatrixGetRow ( hypre_ParCSRMatrix *mat , HYPRE_Int row , HYPRE_Int *size , HYPRE_Int **col_ind , double **values );
HYPRE_Int hypre_ParCSRMatrixRestoreRow ( hypre_ParCSRMatrix *matrix , HYPRE_Int row , HYPRE_Int *size , HYPRE_Int **col_ind , double **values );
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix ( MPI_Comm comm , hypre_CSRMatrix *A , HYPRE_Int *row_starts , HYPRE_Int *col_starts );
HYPRE_Int GenerateDiagAndOffd ( hypre_CSRMatrix *A , hypre_ParCSRMatrix *matrix , HYPRE_Int first_col_diag , HYPRE_Int last_col_diag );
hypre_CSRMatrix *hypre_MergeDiagAndOffd ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll ( hypre_ParCSRMatrix *par_matrix );
HYPRE_Int hypre_ParCSRMatrixCopy ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B , HYPRE_Int copy_data );
HYPRE_Int hypre_FillResponseParToCSRMatrix ( void *p_recv_contact_buf , HYPRE_Int contact_size , HYPRE_Int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , HYPRE_Int *response_message_size );
hypre_ParCSRMatrix *hypre_ParCSRMatrixCompleteClone ( hypre_ParCSRMatrix *A );
hypre_ParCSRMatrix *hypre_ParCSRMatrixUnion ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B );

/* par_csr_matvec.c */
HYPRE_Int hypre_ParCSRMatrixMatvec ( double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecT ( double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvec_FF ( double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y , HYPRE_Int *CF_marker , HYPRE_Int fpt );

/* par_make_system.c */
HYPRE_ParCSR_System_Problem *HYPRE_Generate2DSystem ( HYPRE_ParCSRMatrix H_L1 , HYPRE_ParCSRMatrix H_L2 , HYPRE_ParVector H_b1 , HYPRE_ParVector H_b2 , HYPRE_ParVector H_x1 , HYPRE_ParVector H_x2 , double *M_vals );
HYPRE_Int HYPRE_Destroy2DSystem ( HYPRE_ParCSR_System_Problem *sys_prob );

/* par_vector.c */
hypre_ParVector *hypre_ParVectorCreate ( MPI_Comm comm , HYPRE_Int global_size , HYPRE_Int *partitioning );
hypre_ParVector *hypre_ParMultiVectorCreate ( MPI_Comm comm , HYPRE_Int global_size , HYPRE_Int *partitioning , HYPRE_Int num_vectors );
HYPRE_Int hypre_ParVectorDestroy ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorInitialize ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorSetDataOwner ( hypre_ParVector *vector , HYPRE_Int owns_data );
HYPRE_Int hypre_ParVectorSetPartitioningOwner ( hypre_ParVector *vector , HYPRE_Int owns_partitioning );
HYPRE_Int hypre_ParVectorSetNumVectors ( hypre_ParVector *vector , HYPRE_Int num_vectors );
hypre_ParVector *hypre_ParVectorRead ( MPI_Comm comm , const char *file_name );
HYPRE_Int hypre_ParVectorPrint ( hypre_ParVector *vector , const char *file_name );
HYPRE_Int hypre_ParVectorSetConstantValues ( hypre_ParVector *v , double value );
HYPRE_Int hypre_ParVectorSetRandomValues ( hypre_ParVector *v , HYPRE_Int seed );
HYPRE_Int hypre_ParVectorCopy ( hypre_ParVector *x , hypre_ParVector *y );
hypre_ParVector *hypre_ParVectorCloneShallow ( hypre_ParVector *x );
HYPRE_Int hypre_ParVectorScale ( double alpha , hypre_ParVector *y );
HYPRE_Int hypre_ParVectorAxpy ( double alpha , hypre_ParVector *x , hypre_ParVector *y );
double hypre_ParVectorInnerProd ( hypre_ParVector *x , hypre_ParVector *y );
hypre_ParVector *hypre_VectorToParVector ( MPI_Comm comm , hypre_Vector *v , HYPRE_Int *vec_starts );
hypre_Vector *hypre_ParVectorToVectorAll ( hypre_ParVector *par_v );
HYPRE_Int hypre_ParVectorPrintIJ ( hypre_ParVector *vector , HYPRE_Int base_j , const char *filename );
HYPRE_Int hypre_ParVectorReadIJ ( MPI_Comm comm , const char *filename , HYPRE_Int *base_j_ptr , hypre_ParVector **vector_ptr );
HYPRE_Int hypre_FillResponseParToVectorAll ( void *p_recv_contact_buf , HYPRE_Int contact_size , HYPRE_Int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , HYPRE_Int *response_message_size );
double hypre_ParVectorLocalSumElts ( hypre_ParVector *vector );

#ifdef __cplusplus
}
#endif

#endif

