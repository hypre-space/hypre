/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for HYPRE_mv library
 *
 *****************************************************************************/

#ifndef HYPRE_SEQ_MV_HEADER
#define HYPRE_SEQ_MV_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

struct hypre_CSRMatrix_struct;
typedef struct hypre_CSRMatrix_struct *HYPRE_CSRMatrix;
struct hypre_MappedMatrix_struct;
typedef struct hypre_MappedMatrix_struct *HYPRE_MappedMatrix;
struct hypre_MultiblockMatrix_struct;
typedef struct hypre_MultiblockMatrix_struct *HYPRE_MultiblockMatrix;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* HYPRE_csr_matrix.c */
HYPRE_CSRMatrix HYPRE_CSRMatrixCreate( HYPRE_Int num_rows, HYPRE_Int num_cols,
                                       HYPRE_Int *row_sizes );
HYPRE_Int HYPRE_CSRMatrixDestroy( HYPRE_CSRMatrix matrix );
HYPRE_Int HYPRE_CSRMatrixInitialize( HYPRE_CSRMatrix matrix );
HYPRE_CSRMatrix HYPRE_CSRMatrixRead( char *file_name );
void HYPRE_CSRMatrixPrint( HYPRE_CSRMatrix matrix, char *file_name );
HYPRE_Int HYPRE_CSRMatrixGetNumRows( HYPRE_CSRMatrix matrix, HYPRE_Int *num_rows );

/* HYPRE_mapped_matrix.c */
HYPRE_MappedMatrix HYPRE_MappedMatrixCreate( void );
HYPRE_Int HYPRE_MappedMatrixDestroy( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixLimitedDestroy( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixInitialize( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixAssemble( HYPRE_MappedMatrix matrix );
void HYPRE_MappedMatrixPrint( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixGetColIndex( HYPRE_MappedMatrix matrix, HYPRE_Int j );
void *HYPRE_MappedMatrixGetMatrix( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixSetMatrix( HYPRE_MappedMatrix matrix, void *matrix_data );
HYPRE_Int HYPRE_MappedMatrixSetColMap( HYPRE_MappedMatrix matrix, HYPRE_Int (*ColMap )(HYPRE_Int,
                                                                                       void *));
HYPRE_Int HYPRE_MappedMatrixSetMapData( HYPRE_MappedMatrix matrix, void *MapData );

/* HYPRE_multiblock_matrix.c */
HYPRE_MultiblockMatrix HYPRE_MultiblockMatrixCreate( void );
HYPRE_Int HYPRE_MultiblockMatrixDestroy( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixLimitedDestroy( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixInitialize( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixAssemble( HYPRE_MultiblockMatrix matrix );
void HYPRE_MultiblockMatrixPrint( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixSetNumSubmatrices( HYPRE_MultiblockMatrix matrix, HYPRE_Int n );
HYPRE_Int HYPRE_MultiblockMatrixSetSubmatrixType( HYPRE_MultiblockMatrix matrix, HYPRE_Int j,
                                                  HYPRE_Int type );

/* HYPRE_vector.c */
HYPRE_Vector HYPRE_VectorCreate( HYPRE_Int size );
HYPRE_Int HYPRE_VectorDestroy( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorInitialize( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorPrint( HYPRE_Vector vector, char *file_name );
HYPRE_Vector HYPRE_VectorRead( char *file_name );

typedef enum HYPRE_TimerID
{
   // timers for solver phase
   HYPRE_TIMER_ID_MATVEC = 0,
   HYPRE_TIMER_ID_BLAS1,
   HYPRE_TIMER_ID_RELAX,
   HYPRE_TIMER_ID_GS_ELIM_SOLVE,

   // timers for solve MPI
   HYPRE_TIMER_ID_PACK_UNPACK, // copying data to/from send/recv buf
   HYPRE_TIMER_ID_HALO_EXCHANGE, // halo exchange in matvec and relax
   HYPRE_TIMER_ID_ALL_REDUCE,

   // timers for setup phase
   // coarsening
   HYPRE_TIMER_ID_CREATES,
   HYPRE_TIMER_ID_CREATE_2NDS,
   HYPRE_TIMER_ID_PMIS,

   // interpolation
   HYPRE_TIMER_ID_EXTENDED_I_INTERP,
   HYPRE_TIMER_ID_PARTIAL_INTERP,
   HYPRE_TIMER_ID_MULTIPASS_INTERP,
   HYPRE_TIMER_ID_INTERP_TRUNC,
   HYPRE_TIMER_ID_MATMUL, // matrix-matrix multiplication
   HYPRE_TIMER_ID_COARSE_PARAMS,

   // rap
   HYPRE_TIMER_ID_RAP,

   // timers for setup MPI
   HYPRE_TIMER_ID_RENUMBER_COLIDX,
   HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA,

   // setup etc
   HYPRE_TIMER_ID_GS_ELIM_SETUP,

   // temporaries
   HYPRE_TIMER_ID_BEXT_A,
   HYPRE_TIMER_ID_BEXT_S,
   HYPRE_TIMER_ID_RENUMBER_COLIDX_RAP,
   HYPRE_TIMER_ID_MERGE,

   // csr matop
   HYPRE_TIMER_ID_SPGEMM_ROWNNZ,
   HYPRE_TIMER_ID_SPGEMM_ATTEMPT1,
   HYPRE_TIMER_ID_SPGEMM_ATTEMPT2,
   HYPRE_TIMER_ID_SPGEMM_SYMBOLIC,
   HYPRE_TIMER_ID_SPGEMM_NUMERIC,
   HYPRE_TIMER_ID_SPGEMM,
   HYPRE_TIMER_ID_SPADD,
   HYPRE_TIMER_ID_SPTRANS,

   HYPRE_TIMER_ID_COUNT
} HYPRE_TimerID;

extern HYPRE_Real hypre_profile_times[HYPRE_TIMER_ID_COUNT];

#ifdef __cplusplus
}

#endif

#endif
