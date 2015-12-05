/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/

#ifndef hypre_IJ_HEADER
#define hypre_IJ_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <HYPRE_config.h>

#include "_hypre_utilities.h"
#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

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
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Header info for Auxiliary Parallel Vector data structures
 *
 * Note: this vector currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_AUX_PAR_VECTOR_HEADER
#define hypre_AUX_PAR_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int	    max_off_proc_elmts; /* length of off processor stash for
					SetValues and AddToValues*/
   HYPRE_Int	    current_num_elmts; /* current no. of elements stored in stash */
   HYPRE_Int     *off_proc_i; /* contains column indices */
   double  *off_proc_data; /* contains corresponding data */
   HYPRE_Int	    cancel_indx; /* number of elements that have to be deleted due
                           to setting values from another processor */
} hypre_AuxParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParVectorMaxOffProcElmts(matrix)  ((matrix) -> max_off_proc_elmts)
#define hypre_AuxParVectorCurrentNumElmts(matrix)  ((matrix) -> current_num_elmts)
#define hypre_AuxParVectorOffProcI(matrix)  ((matrix) -> off_proc_i)
#define hypre_AuxParVectorOffProcData(matrix)  ((matrix) -> off_proc_data)
#define hypre_AuxParVectorCancelIndx(matrix)  ((matrix) -> cancel_indx)

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Header info for the hypre_IJMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_IJ_MATRIX_HEADER
#define hypre_IJ_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * hypre_IJMatrix:
 *--------------------------------------------------------------------------*/

typedef struct hypre_IJMatrix_struct
{
   MPI_Comm    comm;

   HYPRE_Int        *row_partitioning;    /* distribution of rows across processors */
   HYPRE_Int        *col_partitioning;    /* distribution of columns */

   HYPRE_Int         object_type;         /* Indicates the type of "object" */
   void       *object;              /* Structure for storing local portion */
   void       *translator;          /* optional storage_type specfic structure
                                       for holding additional local info */
   HYPRE_Int         assemble_flag;       /* indicates whether matrix has been 
				       assembled */

   HYPRE_Int         global_first_row;    /* these for data items are necessary */
   HYPRE_Int         global_first_col;    /*   to be able to avoind using the global */
   HYPRE_Int         global_num_rows;     /*   global partition */ 
   HYPRE_Int         global_num_cols;
   HYPRE_Int         print_level;
   


} hypre_IJMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJMatrix
 *--------------------------------------------------------------------------*/

#define hypre_IJMatrixComm(matrix)              ((matrix) -> comm)

#define hypre_IJMatrixRowPartitioning(matrix)   ((matrix) -> row_partitioning)
#define hypre_IJMatrixColPartitioning(matrix)   ((matrix) -> col_partitioning)

#define hypre_IJMatrixObjectType(matrix)        ((matrix) -> object_type)
#define hypre_IJMatrixObject(matrix)            ((matrix) -> object)
#define hypre_IJMatrixTranslator(matrix)        ((matrix) -> translator)

#define hypre_IJMatrixAssembleFlag(matrix)      ((matrix) -> assemble_flag)


#define hypre_IJMatrixGlobalFirstRow(matrix)      ((matrix) -> global_first_row)
#define hypre_IJMatrixGlobalFirstCol(matrix)      ((matrix) -> global_first_col)
#define hypre_IJMatrixGlobalNumRows(matrix)       ((matrix) -> global_num_rows)
#define hypre_IJMatrixGlobalNumCols(matrix)       ((matrix) -> global_num_cols)
#define hypre_IJMatrixPrintLevel(matrix)       ((matrix) -> print_level)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/

#ifdef PETSC_AVAILABLE
/* IJMatrix_petsc.c */
HYPRE_Int
hypre_GetIJMatrixParCSRMatrix( HYPRE_IJMatrix IJmatrix, Mat *reference )
#endif
  
#ifdef ISIS_AVAILABLE
/* IJMatrix_isis.c */
HYPRE_Int
hypre_GetIJMatrixISISMatrix( HYPRE_IJMatrix IJmatrix, RowMatrix *reference )
#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Header info for the hypre_IJMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_IJ_VECTOR_HEADER
#define hypre_IJ_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_IJVector:
 *--------------------------------------------------------------------------*/

typedef struct hypre_IJVector_struct
{
   MPI_Comm      comm;

   HYPRE_Int 		*partitioning;      /* Indicates partitioning over tasks */

   HYPRE_Int           object_type;       /* Indicates the type of "local storage" */

   void         *object;            /* Structure for storing local portion */

   void         *translator;        /* Structure for storing off processor
				       information */

   HYPRE_Int         global_first_row;    /* these for data items are necessary */
   HYPRE_Int         global_num_rows;     /*   to be able to avoid using the global */
                                    /*    global partition */ 
   HYPRE_Int	       print_level; 
   


} hypre_IJVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJVector
 *--------------------------------------------------------------------------*/

#define hypre_IJVectorComm(vector)           ((vector) -> comm)

#define hypre_IJVectorPartitioning(vector)   ((vector) -> partitioning)

#define hypre_IJVectorObjectType(vector)     ((vector) -> object_type)

#define hypre_IJVectorObject(vector)         ((vector) -> object)

#define hypre_IJVectorTranslator(vector)     ((vector) -> translator)

#define hypre_IJVectorGlobalFirstRow(vector)  ((vector) -> global_first_row)

#define hypre_IJVectorGlobalNumRows(vector)  ((vector) -> global_num_rows)

#define hypre_IJVectorPrintLevel(vector)  ((vector) -> print_level)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/
/* #include "./internal_protos.h" */

#endif

/* aux_parcsr_matrix.c */
HYPRE_Int hypre_AuxParCSRMatrixCreate ( hypre_AuxParCSRMatrix **aux_matrix , HYPRE_Int local_num_rows , HYPRE_Int local_num_cols , HYPRE_Int *sizes );
HYPRE_Int hypre_AuxParCSRMatrixDestroy ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixInitialize ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixSetMaxOffPRocElmts ( hypre_AuxParCSRMatrix *matrix , HYPRE_Int max_off_proc_elmts );

/* aux_par_vector.c */
HYPRE_Int hypre_AuxParVectorCreate ( hypre_AuxParVector **aux_vector );
HYPRE_Int hypre_AuxParVectorDestroy ( hypre_AuxParVector *vector );
HYPRE_Int hypre_AuxParVectorInitialize ( hypre_AuxParVector *vector );
HYPRE_Int hypre_AuxParVectorSetMaxOffPRocElmts ( hypre_AuxParVector *vector , HYPRE_Int max_off_proc_elmts );

/* IJMatrix.c */
HYPRE_Int hypre_IJMatrixGetRowPartitioning ( HYPRE_IJMatrix matrix , HYPRE_Int **row_partitioning );
HYPRE_Int hypre_IJMatrixGetColPartitioning ( HYPRE_IJMatrix matrix , HYPRE_Int **col_partitioning );
HYPRE_Int hypre_IJMatrixSetObject ( HYPRE_IJMatrix matrix , void *object );

/* IJMatrix_isis.c */
HYPRE_Int hypre_IJMatrixSetLocalSizeISIS ( hypre_IJMatrix *matrix , HYPRE_Int local_m , HYPRE_Int local_n );
HYPRE_Int hypre_IJMatrixCreateISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetRowSizesISIS ( hypre_IJMatrix *matrix , HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetDiagRowSizesISIS ( hypre_IJMatrix *matrix , HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetOffDiagRowSizesISIS ( hypre_IJMatrix *matrix , HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixInitializeISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixInsertBlockISIS ( hypre_IJMatrix *matrix , HYPRE_Int m , HYPRE_Int n , HYPRE_Int *rows , HYPRE_Int *cols , double *coeffs );
HYPRE_Int hypre_IJMatrixAddToBlockISIS ( hypre_IJMatrix *matrix , HYPRE_Int m , HYPRE_Int n , HYPRE_Int *rows , HYPRE_Int *cols , double *coeffs );
HYPRE_Int hypre_IJMatrixInsertRowISIS ( hypre_IJMatrix *matrix , HYPRE_Int n , HYPRE_Int row , HYPRE_Int *indices , double *coeffs );
HYPRE_Int hypre_IJMatrixAddToRowISIS ( hypre_IJMatrix *matrix , HYPRE_Int n , HYPRE_Int row , HYPRE_Int *indices , double *coeffs );
HYPRE_Int hypre_IJMatrixAssembleISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixDistributeISIS ( hypre_IJMatrix *matrix , HYPRE_Int *row_starts , HYPRE_Int *col_starts );
HYPRE_Int hypre_IJMatrixApplyISIS ( hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b );
HYPRE_Int hypre_IJMatrixDestroyISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetTotalSizeISIS ( hypre_IJMatrix *matrix , HYPRE_Int size );

/* IJMatrix_parcsr.c */
HYPRE_Int hypre_IJMatrixCreateParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetRowSizesParCSR ( hypre_IJMatrix *matrix , const HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetDiagOffdSizesParCSR ( hypre_IJMatrix *matrix , const HYPRE_Int *diag_sizes , const HYPRE_Int *offdiag_sizes );
HYPRE_Int hypre_IJMatrixSetMaxOffProcElmtsParCSR ( hypre_IJMatrix *matrix , HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJMatrixInitializeParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixGetRowCountsParCSR ( hypre_IJMatrix *matrix , HYPRE_Int nrows , HYPRE_Int *rows , HYPRE_Int *ncols );
HYPRE_Int hypre_IJMatrixGetValuesParCSR ( hypre_IJMatrix *matrix , HYPRE_Int nrows , HYPRE_Int *ncols , HYPRE_Int *rows , HYPRE_Int *cols , double *values );
HYPRE_Int hypre_IJMatrixSetValuesParCSR ( hypre_IJMatrix *matrix , HYPRE_Int nrows , HYPRE_Int *ncols , const HYPRE_Int *rows , const HYPRE_Int *cols , const double *values );
HYPRE_Int hypre_IJMatrixAddToValuesParCSR ( hypre_IJMatrix *matrix , HYPRE_Int nrows , HYPRE_Int *ncols , const HYPRE_Int *rows , const HYPRE_Int *cols , const double *values );
HYPRE_Int hypre_IJMatrixAssembleParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixDestroyParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixAssembleOffProcValsParCSR ( hypre_IJMatrix *matrix , HYPRE_Int off_proc_i_indx , HYPRE_Int max_off_proc_elmts , HYPRE_Int current_num_elmts , HYPRE_Int *off_proc_i , HYPRE_Int *off_proc_j , double *off_proc_data );
HYPRE_Int hypre_FillResponseIJOffProcVals ( void *p_recv_contact_buf , HYPRE_Int contact_size , HYPRE_Int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , HYPRE_Int *response_message_size );
HYPRE_Int hypre_FindProc ( HYPRE_Int *list , HYPRE_Int value , HYPRE_Int list_length );

/* IJMatrix_petsc.c */
HYPRE_Int hypre_IJMatrixSetLocalSizePETSc ( hypre_IJMatrix *matrix , HYPRE_Int local_m , HYPRE_Int local_n );
HYPRE_Int hypre_IJMatrixCreatePETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetRowSizesPETSc ( hypre_IJMatrix *matrix , HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetDiagRowSizesPETSc ( hypre_IJMatrix *matrix , HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetOffDiagRowSizesPETSc ( hypre_IJMatrix *matrix , HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixInitializePETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixInsertBlockPETSc ( hypre_IJMatrix *matrix , HYPRE_Int m , HYPRE_Int n , HYPRE_Int *rows , HYPRE_Int *cols , double *coeffs );
HYPRE_Int hypre_IJMatrixAddToBlockPETSc ( hypre_IJMatrix *matrix , HYPRE_Int m , HYPRE_Int n , HYPRE_Int *rows , HYPRE_Int *cols , double *coeffs );
HYPRE_Int hypre_IJMatrixInsertRowPETSc ( hypre_IJMatrix *matrix , HYPRE_Int n , HYPRE_Int row , HYPRE_Int *indices , double *coeffs );
HYPRE_Int hypre_IJMatrixAddToRowPETSc ( hypre_IJMatrix *matrix , HYPRE_Int n , HYPRE_Int row , HYPRE_Int *indices , double *coeffs );
HYPRE_Int hypre_IJMatrixAssemblePETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixDistributePETSc ( hypre_IJMatrix *matrix , HYPRE_Int *row_starts , HYPRE_Int *col_starts );
HYPRE_Int hypre_IJMatrixApplyPETSc ( hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b );
HYPRE_Int hypre_IJMatrixDestroyPETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetTotalSizePETSc ( hypre_IJMatrix *matrix , HYPRE_Int size );

/* IJVector.c */
HYPRE_Int hypre_IJVectorDistribute ( HYPRE_IJVector vector , const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorZeroValues ( HYPRE_IJVector vector );

/* IJVector_parcsr.c */
HYPRE_Int hypre_IJVectorCreatePar ( hypre_IJVector *vector , HYPRE_Int *IJpartitioning );
HYPRE_Int hypre_IJVectorDestroyPar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorInitializePar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorSetMaxOffProcElmtsPar ( hypre_IJVector *vector , HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJVectorDistributePar ( hypre_IJVector *vector , const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorZeroValuesPar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorSetValuesPar ( hypre_IJVector *vector , HYPRE_Int num_values , const HYPRE_Int *indices , const double *values );
HYPRE_Int hypre_IJVectorAddToValuesPar ( hypre_IJVector *vector , HYPRE_Int num_values , const HYPRE_Int *indices , const double *values );
HYPRE_Int hypre_IJVectorAssemblePar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorGetValuesPar ( hypre_IJVector *vector , HYPRE_Int num_values , const HYPRE_Int *indices , double *values );
HYPRE_Int hypre_IJVectorAssembleOffProcValsPar ( hypre_IJVector *vector , HYPRE_Int max_off_proc_elmts , HYPRE_Int current_num_elmts , HYPRE_Int *off_proc_i , double *off_proc_data );

/* HYPRE_IJMatrix.c */
HYPRE_Int HYPRE_IJMatrixCreate ( MPI_Comm comm , HYPRE_Int ilower , HYPRE_Int iupper , HYPRE_Int jlower , HYPRE_Int jupper , HYPRE_IJMatrix *matrix );
HYPRE_Int HYPRE_IJMatrixDestroy ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixInitialize ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixSetPrintLevel ( HYPRE_IJMatrix matrix , HYPRE_Int print_level );
HYPRE_Int HYPRE_IJMatrixSetValues ( HYPRE_IJMatrix matrix , HYPRE_Int nrows , HYPRE_Int *ncols , const HYPRE_Int *rows , const HYPRE_Int *cols , const double *values );
HYPRE_Int HYPRE_IJMatrixAddToValues ( HYPRE_IJMatrix matrix , HYPRE_Int nrows , HYPRE_Int *ncols , const HYPRE_Int *rows , const HYPRE_Int *cols , const double *values );
HYPRE_Int HYPRE_IJMatrixAssemble ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixGetRowCounts ( HYPRE_IJMatrix matrix , HYPRE_Int nrows , HYPRE_Int *rows , HYPRE_Int *ncols );
HYPRE_Int HYPRE_IJMatrixGetValues ( HYPRE_IJMatrix matrix , HYPRE_Int nrows , HYPRE_Int *ncols , HYPRE_Int *rows , HYPRE_Int *cols , double *values );
HYPRE_Int HYPRE_IJMatrixSetObjectType ( HYPRE_IJMatrix matrix , HYPRE_Int type );
HYPRE_Int HYPRE_IJMatrixGetObjectType ( HYPRE_IJMatrix matrix , HYPRE_Int *type );
HYPRE_Int HYPRE_IJMatrixGetLocalRange ( HYPRE_IJMatrix matrix , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int *jlower , HYPRE_Int *jupper );
HYPRE_Int HYPRE_IJMatrixGetObject ( HYPRE_IJMatrix matrix , void **object );
HYPRE_Int HYPRE_IJMatrixSetRowSizes ( HYPRE_IJMatrix matrix , const HYPRE_Int *sizes );
HYPRE_Int HYPRE_IJMatrixSetDiagOffdSizes ( HYPRE_IJMatrix matrix , const HYPRE_Int *diag_sizes , const HYPRE_Int *offdiag_sizes );
HYPRE_Int HYPRE_IJMatrixSetMaxOffProcElmts ( HYPRE_IJMatrix matrix , HYPRE_Int max_off_proc_elmts );
HYPRE_Int HYPRE_IJMatrixRead ( const char *filename , MPI_Comm comm , HYPRE_Int type , HYPRE_IJMatrix *matrix_ptr );
HYPRE_Int HYPRE_IJMatrixPrint ( HYPRE_IJMatrix matrix , const char *filename );

/* HYPRE_IJVector.c */
HYPRE_Int HYPRE_IJVectorCreate ( MPI_Comm comm , HYPRE_Int jlower , HYPRE_Int jupper , HYPRE_IJVector *vector );
HYPRE_Int HYPRE_IJVectorDestroy ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorInitialize ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorSetPrintLevel ( HYPRE_IJVector vector , HYPRE_Int print_level );
HYPRE_Int HYPRE_IJVectorSetValues ( HYPRE_IJVector vector , HYPRE_Int nvalues , const HYPRE_Int *indices , const double *values );
HYPRE_Int HYPRE_IJVectorAddToValues ( HYPRE_IJVector vector , HYPRE_Int nvalues , const HYPRE_Int *indices , const double *values );
HYPRE_Int HYPRE_IJVectorAssemble ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorGetValues ( HYPRE_IJVector vector , HYPRE_Int nvalues , const HYPRE_Int *indices , double *values );
HYPRE_Int HYPRE_IJVectorSetMaxOffProcElmts ( HYPRE_IJVector vector , HYPRE_Int max_off_proc_elmts );
HYPRE_Int HYPRE_IJVectorSetObjectType ( HYPRE_IJVector vector , HYPRE_Int type );
HYPRE_Int HYPRE_IJVectorGetObjectType ( HYPRE_IJVector vector , HYPRE_Int *type );
HYPRE_Int HYPRE_IJVectorGetLocalRange ( HYPRE_IJVector vector , HYPRE_Int *jlower , HYPRE_Int *jupper );
HYPRE_Int HYPRE_IJVectorGetObject ( HYPRE_IJVector vector , void **object );
HYPRE_Int HYPRE_IJVectorRead ( const char *filename , MPI_Comm comm , HYPRE_Int type , HYPRE_IJVector *vector_ptr );
HYPRE_Int HYPRE_IJVectorPrint ( HYPRE_IJVector vector , const char *filename );

#ifdef __cplusplus
}
#endif

#endif

