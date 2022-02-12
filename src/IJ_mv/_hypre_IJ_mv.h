
/*** DO NOT EDIT THIS FILE DIRECTLY (use 'headers' to generate) ***/

#ifndef hypre_IJ_HEADER
#define hypre_IJ_HEADER

#include <HYPRE_config.h>
#include "_hypre_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
   HYPRE_Int            local_num_rows;    /* defines number of rows on this processor */
   HYPRE_Int            local_num_rownnz;  /* defines number of nonzero rows on this processor */
   HYPRE_Int            local_num_cols;    /* defines number of cols of diag */

   HYPRE_Int            need_aux;                /* if need_aux = 1, aux_j, aux_data are used to
                                                    generate the parcsr matrix (default),
                                                    for need_aux = 0, data is put directly into
                                                    parcsr structure (requires the knowledge of
                                                    offd_i and diag_i ) */

   HYPRE_Int           *rownnz;                  /* row_nnz[i] contains the i-th nonzero row id */
   HYPRE_Int           *row_length;              /* row_length[i] contains number of stored
                                                    elements in i-th row */
   HYPRE_Int           *row_space;               /* row_space[i] contains space allocated to
                                                    i-th row */

   HYPRE_Int           *diag_sizes;              /* user input row lengths of diag */
   HYPRE_Int           *offd_sizes;              /* user input row lengths of diag */

   HYPRE_BigInt       **aux_j;                   /* contains collected column indices */
   HYPRE_Complex      **aux_data;                /* contains collected data */

   HYPRE_Int           *indx_diag;               /* indx_diag[i] points to first empty space of portion
                                                    in diag_j , diag_data assigned to row i */
   HYPRE_Int           *indx_offd;               /* indx_offd[i] points to first empty space of portion
                                                    in offd_j , offd_data assigned to row i */

   HYPRE_Int            max_off_proc_elmts;      /* length of off processor stash set for
                                                    SetValues and AddTOValues */
   HYPRE_Int            current_off_proc_elmts;  /* current no. of elements stored in stash */
   HYPRE_Int            off_proc_i_indx;         /* pointer to first empty space in
                                                    set_off_proc_i_set */
   HYPRE_BigInt        *off_proc_i;              /* length 2*num_off_procs_elmts, contains info pairs
                                                    (code, no. of elmts) where code contains global
                                                    row no. if  SetValues, and (-global row no. -1)
                                                    if  AddToValues */
   HYPRE_BigInt        *off_proc_j;              /* contains column indices
                                                  * ( global col id.)    if SetValues,
                                                  * (-global col id. -1) if AddToValues */
   HYPRE_Complex       *off_proc_data;           /* contains corresponding data */

   HYPRE_MemoryLocation memory_location;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Int            max_stack_elmts;
   HYPRE_Int            current_stack_elmts;
   HYPRE_BigInt        *stack_i;
   HYPRE_BigInt        *stack_j;
   HYPRE_Complex       *stack_data;
   char                *stack_sora;              /* Set (1) or Add (0) */
   HYPRE_Int            usr_on_proc_elmts;       /* user given num elmt on-proc */
   HYPRE_Int            usr_off_proc_elmts;      /* user given num elmt off-proc */
   HYPRE_Real           init_alloc_factor;
   HYPRE_Real           grow_factor;
#endif
} hypre_AuxParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParCSRMatrixLocalNumRows(matrix)         ((matrix) -> local_num_rows)
#define hypre_AuxParCSRMatrixLocalNumRownnz(matrix)       ((matrix) -> local_num_rownnz)
#define hypre_AuxParCSRMatrixLocalNumCols(matrix)         ((matrix) -> local_num_cols)

#define hypre_AuxParCSRMatrixNeedAux(matrix)              ((matrix) -> need_aux)
#define hypre_AuxParCSRMatrixRownnz(matrix)               ((matrix) -> rownnz)
#define hypre_AuxParCSRMatrixRowLength(matrix)            ((matrix) -> row_length)
#define hypre_AuxParCSRMatrixRowSpace(matrix)             ((matrix) -> row_space)
#define hypre_AuxParCSRMatrixAuxJ(matrix)                 ((matrix) -> aux_j)
#define hypre_AuxParCSRMatrixAuxData(matrix)              ((matrix) -> aux_data)

#define hypre_AuxParCSRMatrixIndxDiag(matrix)             ((matrix) -> indx_diag)
#define hypre_AuxParCSRMatrixIndxOffd(matrix)             ((matrix) -> indx_offd)

#define hypre_AuxParCSRMatrixDiagSizes(matrix)            ((matrix) -> diag_sizes)
#define hypre_AuxParCSRMatrixOffdSizes(matrix)            ((matrix) -> offd_sizes)

#define hypre_AuxParCSRMatrixMaxOffProcElmts(matrix)      ((matrix) -> max_off_proc_elmts)
#define hypre_AuxParCSRMatrixCurrentOffProcElmts(matrix)  ((matrix) -> current_off_proc_elmts)
#define hypre_AuxParCSRMatrixOffProcIIndx(matrix)         ((matrix) -> off_proc_i_indx)
#define hypre_AuxParCSRMatrixOffProcI(matrix)             ((matrix) -> off_proc_i)
#define hypre_AuxParCSRMatrixOffProcJ(matrix)             ((matrix) -> off_proc_j)
#define hypre_AuxParCSRMatrixOffProcData(matrix)          ((matrix) -> off_proc_data)

#define hypre_AuxParCSRMatrixMemoryLocation(matrix)       ((matrix) -> memory_location)

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#define hypre_AuxParCSRMatrixMaxStackElmts(matrix)        ((matrix) -> max_stack_elmts)
#define hypre_AuxParCSRMatrixCurrentStackElmts(matrix)    ((matrix) -> current_stack_elmts)
#define hypre_AuxParCSRMatrixStackI(matrix)               ((matrix) -> stack_i)
#define hypre_AuxParCSRMatrixStackJ(matrix)               ((matrix) -> stack_j)
#define hypre_AuxParCSRMatrixStackData(matrix)            ((matrix) -> stack_data)
#define hypre_AuxParCSRMatrixStackSorA(matrix)            ((matrix) -> stack_sora)
#define hypre_AuxParCSRMatrixUsrOnProcElmts(matrix)       ((matrix) -> usr_on_proc_elmts)
#define hypre_AuxParCSRMatrixUsrOffProcElmts(matrix)      ((matrix) -> usr_off_proc_elmts)
#define hypre_AuxParCSRMatrixInitAllocFactor(matrix)      ((matrix) -> init_alloc_factor)
#define hypre_AuxParCSRMatrixGrowFactor(matrix)           ((matrix) -> grow_factor)
#endif

#endif /* #ifndef hypre_AUX_PARCSR_MATRIX_HEADER */
/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
   HYPRE_Int            max_off_proc_elmts;      /* length of off processor stash for
                                                    SetValues and AddToValues*/
   HYPRE_Int            current_off_proc_elmts;  /* current no. of elements stored in stash */
   HYPRE_BigInt        *off_proc_i;              /* contains column indices */
   HYPRE_Complex       *off_proc_data;           /* contains corresponding data */

   HYPRE_MemoryLocation memory_location;

#if defined(HYPRE_USING_GPU)
   HYPRE_Int            max_stack_elmts;      /* length of stash for SetValues and AddToValues*/
   HYPRE_Int            current_stack_elmts;  /* current no. of elements stored in stash */
   HYPRE_BigInt        *stack_i;              /* contains row indices */
   HYPRE_Complex       *stack_data;           /* contains corresponding data */
   char                *stack_sora;
   HYPRE_Int            usr_off_proc_elmts;   /* the num of off-proc elements usr guided */
   HYPRE_Real           init_alloc_factor;
   HYPRE_Real           grow_factor;
#endif
} hypre_AuxParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParVectorMaxOffProcElmts(vector)      ((vector) -> max_off_proc_elmts)
#define hypre_AuxParVectorCurrentOffProcElmts(vector)  ((vector) -> current_off_proc_elmts)
#define hypre_AuxParVectorOffProcI(vector)             ((vector) -> off_proc_i)
#define hypre_AuxParVectorOffProcData(vector)          ((vector) -> off_proc_data)

#define hypre_AuxParVectorMemoryLocation(vector)       ((vector) -> memory_location)

#if defined(HYPRE_USING_GPU)
#define hypre_AuxParVectorMaxStackElmts(vector)        ((vector) -> max_stack_elmts)
#define hypre_AuxParVectorCurrentStackElmts(vector)    ((vector) -> current_stack_elmts)
#define hypre_AuxParVectorStackI(vector)               ((vector) -> stack_i)
#define hypre_AuxParVectorStackData(vector)            ((vector) -> stack_data)
#define hypre_AuxParVectorStackSorA(vector)            ((vector) -> stack_sora)
#define hypre_AuxParVectorUsrOffProcElmts(vector)      ((vector) -> usr_off_proc_elmts)
#define hypre_AuxParVectorInitAllocFactor(vector)      ((vector) -> init_alloc_factor)
#define hypre_AuxParVectorGrowFactor(vector)           ((vector) -> grow_factor)
#endif

#endif /* #ifndef hypre_AUX_PAR_VECTOR_HEADER */
/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
   MPI_Comm      comm;

   HYPRE_BigInt  row_partitioning[2]; /* distribution of rows across processors */
   HYPRE_BigInt  col_partitioning[2]; /* distribution of columns */

   HYPRE_Int     object_type;         /* Indicates the type of "object" */
   void         *object;              /* Structure for storing local portion */
   void         *translator;          /* optional storage_type specific structure
                                         for holding additional local info */
   void         *assumed_part;        /* IJMatrix assumed partition */
   HYPRE_Int     assemble_flag;       /* indicates whether matrix has been
                                         assembled */

   HYPRE_BigInt  global_first_row;    /* these four data items are necessary */
   HYPRE_BigInt  global_first_col;    /* to be able to avoid using the global */
   HYPRE_BigInt  global_num_rows;     /* global partition */
   HYPRE_BigInt  global_num_cols;
   HYPRE_Int     omp_flag;
   HYPRE_Int     print_level;

} hypre_IJMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJMatrix
 *--------------------------------------------------------------------------*/

#define hypre_IJMatrixComm(matrix)             ((matrix) -> comm)
#define hypre_IJMatrixRowPartitioning(matrix)  ((matrix) -> row_partitioning)
#define hypre_IJMatrixColPartitioning(matrix)  ((matrix) -> col_partitioning)

#define hypre_IJMatrixObjectType(matrix)       ((matrix) -> object_type)
#define hypre_IJMatrixObject(matrix)           ((matrix) -> object)
#define hypre_IJMatrixTranslator(matrix)       ((matrix) -> translator)
#define hypre_IJMatrixAssumedPart(matrix)      ((matrix) -> assumed_part)

#define hypre_IJMatrixAssembleFlag(matrix)     ((matrix) -> assemble_flag)

#define hypre_IJMatrixGlobalFirstRow(matrix)   ((matrix) -> global_first_row)
#define hypre_IJMatrixGlobalFirstCol(matrix)   ((matrix) -> global_first_col)
#define hypre_IJMatrixGlobalNumRows(matrix)    ((matrix) -> global_num_rows)
#define hypre_IJMatrixGlobalNumCols(matrix)    ((matrix) -> global_num_cols)
#define hypre_IJMatrixOMPFlag(matrix)          ((matrix) -> omp_flag)
#define hypre_IJMatrixPrintLevel(matrix)       ((matrix) -> print_level)

static inline HYPRE_MemoryLocation
hypre_IJMatrixMemoryLocation(hypre_IJMatrix *matrix)
{
   if ( hypre_IJMatrixObject(matrix) && hypre_IJMatrixObjectType(matrix) == HYPRE_PARCSR)
   {
      return hypre_ParCSRMatrixMemoryLocation( (hypre_ParCSRMatrix *) hypre_IJMatrixObject(matrix) );
   }

   return HYPRE_MEMORY_UNDEFINED;
}

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

#endif /* #ifndef hypre_IJ_MATRIX_HEADER */
/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

   HYPRE_BigInt  partitioning[2];   /* Indicates partitioning over tasks */

   HYPRE_Int     object_type;       /* Indicates the type of "local storage" */

   void         *object;            /* Structure for storing local portion */

   void         *translator;        /* Structure for storing off processor
                                       information */

   void         *assumed_part;      /* IJ Vector assumed partition */

   HYPRE_BigInt  global_first_row;  /* these for data items are necessary */
   HYPRE_BigInt  global_num_rows;   /* to be able to avoid using the global */
   /* global partition */
   HYPRE_Int     print_level;

} hypre_IJVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJVector
 *--------------------------------------------------------------------------*/

#define hypre_IJVectorComm(vector)            ((vector) -> comm)
#define hypre_IJVectorPartitioning(vector)    ((vector) -> partitioning)
#define hypre_IJVectorObjectType(vector)      ((vector) -> object_type)
#define hypre_IJVectorObject(vector)          ((vector) -> object)
#define hypre_IJVectorTranslator(vector)      ((vector) -> translator)
#define hypre_IJVectorAssumedPart(vector)     ((vector) -> assumed_part)
#define hypre_IJVectorGlobalFirstRow(vector)  ((vector) -> global_first_row)
#define hypre_IJVectorGlobalNumRows(vector)   ((vector) -> global_num_rows)
#define hypre_IJVectorPrintLevel(vector)      ((vector) -> print_level)

static inline HYPRE_MemoryLocation
hypre_IJVectorMemoryLocation(hypre_IJVector *vector)
{
   if ( hypre_IJVectorObject(vector) && hypre_IJVectorObjectType(vector) == HYPRE_PARCSR)
   {
      return hypre_ParVectorMemoryLocation( (hypre_ParVector *) hypre_IJVectorObject(vector) );
   }

   return HYPRE_MEMORY_UNDEFINED;
}

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/
/* #include "./internal_protos.h" */

#endif /* #ifndef hypre_IJ_VECTOR_HEADER */
/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* aux_parcsr_matrix.c */
HYPRE_Int hypre_AuxParCSRMatrixCreate ( hypre_AuxParCSRMatrix **aux_matrix,
                                        HYPRE_Int local_num_rows, HYPRE_Int local_num_cols, HYPRE_Int *sizes );
HYPRE_Int hypre_AuxParCSRMatrixDestroy ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixSetRownnz ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixInitialize ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixInitialize_v2( hypre_AuxParCSRMatrix *matrix,
                                              HYPRE_MemoryLocation memory_location );

/* aux_par_vector.c */
HYPRE_Int hypre_AuxParVectorCreate ( hypre_AuxParVector **aux_vector );
HYPRE_Int hypre_AuxParVectorDestroy ( hypre_AuxParVector *vector );
HYPRE_Int hypre_AuxParVectorInitialize ( hypre_AuxParVector *vector );
HYPRE_Int hypre_AuxParVectorInitialize_v2( hypre_AuxParVector *vector,
                                           HYPRE_MemoryLocation memory_location );

/* IJ_assumed_part.c */
HYPRE_Int hypre_IJMatrixCreateAssumedPartition ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJVectorCreateAssumedPartition ( hypre_IJVector *vector );

/* IJMatrix.c */
HYPRE_Int hypre_IJMatrixGetRowPartitioning ( HYPRE_IJMatrix matrix,
                                             HYPRE_BigInt **row_partitioning );
HYPRE_Int hypre_IJMatrixGetColPartitioning ( HYPRE_IJMatrix matrix,
                                             HYPRE_BigInt **col_partitioning );
HYPRE_Int hypre_IJMatrixSetObject ( HYPRE_IJMatrix matrix, void *object );

/* IJMatrix_isis.c */
HYPRE_Int hypre_IJMatrixSetLocalSizeISIS ( hypre_IJMatrix *matrix, HYPRE_Int local_m,
                                           HYPRE_Int local_n );
HYPRE_Int hypre_IJMatrixCreateISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetRowSizesISIS ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetDiagRowSizesISIS ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetOffDiagRowSizesISIS ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixInitializeISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixInsertBlockISIS ( hypre_IJMatrix *matrix, HYPRE_Int m, HYPRE_Int n,
                                          HYPRE_Int *rows, HYPRE_Int *cols, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAddToBlockISIS ( hypre_IJMatrix *matrix, HYPRE_Int m, HYPRE_Int n,
                                         HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixInsertRowISIS ( hypre_IJMatrix *matrix, HYPRE_Int n, HYPRE_BigInt row,
                                        HYPRE_BigInt *indices, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAddToRowISIS ( hypre_IJMatrix *matrix, HYPRE_Int n, HYPRE_BigInt row,
                                       HYPRE_BigInt *indices, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAssembleISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixDistributeISIS ( hypre_IJMatrix *matrix, HYPRE_BigInt *row_starts,
                                         HYPRE_BigInt *col_starts );
HYPRE_Int hypre_IJMatrixApplyISIS ( hypre_IJMatrix *matrix, hypre_ParVector *x,
                                    hypre_ParVector *b );
HYPRE_Int hypre_IJMatrixDestroyISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetTotalSizeISIS ( hypre_IJMatrix *matrix, HYPRE_Int size );

/* IJMatrix_parcsr.c */
HYPRE_Int hypre_IJMatrixCreateParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetRowSizesParCSR ( hypre_IJMatrix *matrix, const HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetDiagOffdSizesParCSR ( hypre_IJMatrix *matrix,
                                                 const HYPRE_Int *diag_sizes, const HYPRE_Int *offdiag_sizes );
HYPRE_Int hypre_IJMatrixSetMaxOffProcElmtsParCSR ( hypre_IJMatrix *matrix,
                                                   HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJMatrixInitializeParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixGetRowCountsParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                             HYPRE_BigInt *rows, HYPRE_Int *ncols );
HYPRE_Int hypre_IJMatrixGetValuesParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                          HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Complex *values );
HYPRE_Int hypre_IJMatrixSetValuesParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                          const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                          const HYPRE_Complex *values );
HYPRE_Int hypre_IJMatrixSetAddValuesParCSRDevice ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                                   HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                                   const HYPRE_Complex *values, const char *action );
HYPRE_Int hypre_IJMatrixSetConstantValuesParCSR ( hypre_IJMatrix *matrix, HYPRE_Complex value );
HYPRE_Int hypre_IJMatrixAddToValuesParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                            HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                            const HYPRE_Complex *values );
HYPRE_Int hypre_IJMatrixDestroyParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixTransposeParCSR ( hypre_IJMatrix  *matrix_A, hypre_IJMatrix *matrix_AT );
HYPRE_Int hypre_IJMatrixNormParCSR ( hypre_IJMatrix *matrix, HYPRE_Real *norm );
HYPRE_Int hypre_IJMatrixAddParCSR ( HYPRE_Complex alpha, hypre_IJMatrix *matrix_A,
                                    HYPRE_Complex beta, hypre_IJMatrix *matrix_B, hypre_IJMatrix *matrix_C );
HYPRE_Int hypre_IJMatrixAssembleOffProcValsParCSR ( hypre_IJMatrix *matrix,
                                                    HYPRE_Int off_proc_i_indx, HYPRE_Int max_off_proc_elmts, HYPRE_Int current_num_elmts,
                                                    HYPRE_MemoryLocation memory_location, HYPRE_BigInt *off_proc_i, HYPRE_BigInt *off_proc_j,
                                                    HYPRE_Complex *off_proc_data );
HYPRE_Int hypre_FillResponseIJOffProcVals ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                            HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                            HYPRE_Int *response_message_size );
HYPRE_Int hypre_FindProc ( HYPRE_BigInt *list, HYPRE_BigInt value, HYPRE_Int list_length );
HYPRE_Int hypre_IJMatrixAssembleParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetValuesOMPParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                             HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                             const HYPRE_Complex *values );
HYPRE_Int hypre_IJMatrixAddToValuesOMPParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                               HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                               const HYPRE_Complex *values );
HYPRE_Int hypre_IJMatrixAssembleParCSRDevice(hypre_IJMatrix *matrix);
HYPRE_Int hypre_IJMatrixInitializeParCSR_v2(hypre_IJMatrix *matrix,
                                            HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_IJMatrixSetConstantValuesParCSRDevice( hypre_IJMatrix *matrix,
                                                       HYPRE_Complex value );

/* IJMatrix_petsc.c */
HYPRE_Int hypre_IJMatrixSetLocalSizePETSc ( hypre_IJMatrix *matrix, HYPRE_Int local_m,
                                            HYPRE_Int local_n );
HYPRE_Int hypre_IJMatrixCreatePETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetRowSizesPETSc ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetDiagRowSizesPETSc ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetOffDiagRowSizesPETSc ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixInitializePETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixInsertBlockPETSc ( hypre_IJMatrix *matrix, HYPRE_Int m, HYPRE_Int n,
                                           HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAddToBlockPETSc ( hypre_IJMatrix *matrix, HYPRE_Int m, HYPRE_Int n,
                                          HYPRE_Int *rows, HYPRE_Int *cols, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixInsertRowPETSc ( hypre_IJMatrix *matrix, HYPRE_Int n, HYPRE_BigInt row,
                                         HYPRE_BigInt *indices, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAddToRowPETSc ( hypre_IJMatrix *matrix, HYPRE_Int n, HYPRE_BigInt row,
                                        HYPRE_BigInt *indices, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAssemblePETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixDistributePETSc ( hypre_IJMatrix *matrix, HYPRE_BigInt *row_starts,
                                          HYPRE_BigInt *col_starts );
HYPRE_Int hypre_IJMatrixApplyPETSc ( hypre_IJMatrix *matrix, hypre_ParVector *x,
                                     hypre_ParVector *b );
HYPRE_Int hypre_IJMatrixDestroyPETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetTotalSizePETSc ( hypre_IJMatrix *matrix, HYPRE_Int size );

/* IJVector.c */
HYPRE_Int hypre_IJVectorDistribute ( HYPRE_IJVector vector, const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorZeroValues ( HYPRE_IJVector vector );

/* IJVector_parcsr.c */
HYPRE_Int hypre_IJVectorCreatePar ( hypre_IJVector *vector, HYPRE_BigInt *IJpartitioning );
HYPRE_Int hypre_IJVectorDestroyPar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorInitializePar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorInitializePar_v2(hypre_IJVector *vector,
                                         HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_IJVectorSetMaxOffProcElmtsPar ( hypre_IJVector *vector,
                                                HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJVectorDistributePar ( hypre_IJVector *vector, const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorZeroValuesPar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorSetValuesPar ( hypre_IJVector *vector, HYPRE_Int num_values,
                                       const HYPRE_BigInt *indices, const HYPRE_Complex *values );
HYPRE_Int hypre_IJVectorAddToValuesPar ( hypre_IJVector *vector, HYPRE_Int num_values,
                                         const HYPRE_BigInt *indices, const HYPRE_Complex *values );
HYPRE_Int hypre_IJVectorAssemblePar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorGetValuesPar ( hypre_IJVector *vector, HYPRE_Int num_values,
                                       const HYPRE_BigInt *indices, HYPRE_Complex *values );
HYPRE_Int hypre_IJVectorAssembleOffProcValsPar ( hypre_IJVector *vector,
                                                 HYPRE_Int max_off_proc_elmts, HYPRE_Int current_num_elmts, HYPRE_MemoryLocation memory_location,
                                                 HYPRE_BigInt *off_proc_i, HYPRE_Complex *off_proc_data );
HYPRE_Int hypre_IJVectorSetAddValuesParDevice(hypre_IJVector *vector, HYPRE_Int num_values,
                                              const HYPRE_BigInt *indices, const HYPRE_Complex *values, const char *action);
HYPRE_Int hypre_IJVectorAssembleParDevice(hypre_IJVector *vector);

/* HYPRE_IJMatrix.c */
HYPRE_Int HYPRE_IJMatrixCreate ( MPI_Comm comm, HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                                 HYPRE_BigInt jlower, HYPRE_BigInt jupper, HYPRE_IJMatrix *matrix );
HYPRE_Int HYPRE_IJMatrixDestroy ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixInitialize ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixSetPrintLevel ( HYPRE_IJMatrix matrix, HYPRE_Int print_level );
HYPRE_Int HYPRE_IJMatrixSetValues ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                    const HYPRE_BigInt *rows, const HYPRE_BigInt *cols, const HYPRE_Complex *values );
HYPRE_Int HYPRE_IJMatrixSetConstantValues ( HYPRE_IJMatrix matrix, HYPRE_Complex value );
HYPRE_Int HYPRE_IJMatrixAddToValues ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                      const HYPRE_BigInt *rows, const HYPRE_BigInt *cols, const HYPRE_Complex *values );
HYPRE_Int HYPRE_IJMatrixAssemble ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixGetRowCounts ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_BigInt *rows,
                                       HYPRE_Int *ncols );
HYPRE_Int HYPRE_IJMatrixGetValues ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                    HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Complex *values );
HYPRE_Int HYPRE_IJMatrixSetObjectType ( HYPRE_IJMatrix matrix, HYPRE_Int type );
HYPRE_Int HYPRE_IJMatrixGetObjectType ( HYPRE_IJMatrix matrix, HYPRE_Int *type );
HYPRE_Int HYPRE_IJMatrixGetLocalRange ( HYPRE_IJMatrix matrix, HYPRE_BigInt *ilower,
                                        HYPRE_BigInt *iupper, HYPRE_BigInt *jlower, HYPRE_BigInt *jupper );
HYPRE_Int HYPRE_IJMatrixGetObject ( HYPRE_IJMatrix matrix, void **object );
HYPRE_Int HYPRE_IJMatrixSetRowSizes ( HYPRE_IJMatrix matrix, const HYPRE_Int *sizes );
HYPRE_Int HYPRE_IJMatrixSetDiagOffdSizes ( HYPRE_IJMatrix matrix, const HYPRE_Int *diag_sizes,
                                           const HYPRE_Int *offdiag_sizes );
HYPRE_Int HYPRE_IJMatrixSetMaxOffProcElmts ( HYPRE_IJMatrix matrix, HYPRE_Int max_off_proc_elmts );
HYPRE_Int HYPRE_IJMatrixRead ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                               HYPRE_IJMatrix *matrix_ptr );
HYPRE_Int HYPRE_IJMatrixPrint ( HYPRE_IJMatrix matrix, const char *filename );
HYPRE_Int HYPRE_IJMatrixSetOMPFlag ( HYPRE_IJMatrix matrix, HYPRE_Int omp_flag );
HYPRE_Int HYPRE_IJMatrixTranspose ( HYPRE_IJMatrix  matrix_A, HYPRE_IJMatrix *matrix_AT );
HYPRE_Int HYPRE_IJMatrixNorm ( HYPRE_IJMatrix matrix, HYPRE_Real *norm );
HYPRE_Int HYPRE_IJMatrixAdd ( HYPRE_Complex alpha, HYPRE_IJMatrix matrix_A, HYPRE_Complex beta,
                              HYPRE_IJMatrix matrix_B, HYPRE_IJMatrix *matrix_C );

/* HYPRE_IJVector.c */
HYPRE_Int HYPRE_IJVectorCreate ( MPI_Comm comm, HYPRE_BigInt jlower, HYPRE_BigInt jupper,
                                 HYPRE_IJVector *vector );
HYPRE_Int HYPRE_IJVectorDestroy ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorInitialize ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorSetPrintLevel ( HYPRE_IJVector vector, HYPRE_Int print_level );
HYPRE_Int HYPRE_IJVectorSetValues ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                    const HYPRE_BigInt *indices, const HYPRE_Complex *values );
HYPRE_Int HYPRE_IJVectorAddToValues ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                      const HYPRE_BigInt *indices, const HYPRE_Complex *values );
HYPRE_Int HYPRE_IJVectorAssemble ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorGetValues ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                    const HYPRE_BigInt *indices, HYPRE_Complex *values );
HYPRE_Int HYPRE_IJVectorSetMaxOffProcElmts ( HYPRE_IJVector vector, HYPRE_Int max_off_proc_elmts );
HYPRE_Int HYPRE_IJVectorSetObjectType ( HYPRE_IJVector vector, HYPRE_Int type );
HYPRE_Int HYPRE_IJVectorGetObjectType ( HYPRE_IJVector vector, HYPRE_Int *type );
HYPRE_Int HYPRE_IJVectorGetLocalRange ( HYPRE_IJVector vector, HYPRE_BigInt *jlower,
                                        HYPRE_BigInt *jupper );
HYPRE_Int HYPRE_IJVectorGetObject ( HYPRE_IJVector vector, void **object );
HYPRE_Int HYPRE_IJVectorRead ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                               HYPRE_IJVector *vector_ptr );
HYPRE_Int HYPRE_IJVectorPrint ( HYPRE_IJVector vector, const char *filename );

#ifdef __cplusplus
}
#endif

#endif

