/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




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
#ifndef HYPRE_VECTOR_STRUCT
#define HYPRE_VECTOR_STRUCT
struct hypre_Vector_struct;
typedef struct hypre_Vector_struct *HYPRE_Vector;
#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* HYPRE_csr_matrix.c */
HYPRE_CSRMatrix HYPRE_CSRMatrixCreate( HYPRE_Int num_rows , HYPRE_Int num_cols , HYPRE_Int *row_sizes );
HYPRE_Int HYPRE_CSRMatrixDestroy( HYPRE_CSRMatrix matrix );
HYPRE_Int HYPRE_CSRMatrixInitialize( HYPRE_CSRMatrix matrix );
HYPRE_CSRMatrix HYPRE_CSRMatrixRead( char *file_name );
void HYPRE_CSRMatrixPrint( HYPRE_CSRMatrix matrix , char *file_name );
HYPRE_Int HYPRE_CSRMatrixGetNumRows( HYPRE_CSRMatrix matrix , HYPRE_Int *num_rows );

/* HYPRE_mapped_matrix.c */
HYPRE_MappedMatrix HYPRE_MappedMatrixCreate( void );
HYPRE_Int HYPRE_MappedMatrixDestroy( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixLimitedDestroy( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixInitialize( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixAssemble( HYPRE_MappedMatrix matrix );
void HYPRE_MappedMatrixPrint( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixGetColIndex( HYPRE_MappedMatrix matrix , HYPRE_Int j );
void *HYPRE_MappedMatrixGetMatrix( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixSetMatrix( HYPRE_MappedMatrix matrix , void *matrix_data );
HYPRE_Int HYPRE_MappedMatrixSetColMap( HYPRE_MappedMatrix matrix , HYPRE_Int (*ColMap )(HYPRE_Int ,void *));
HYPRE_Int HYPRE_MappedMatrixSetMapData( HYPRE_MappedMatrix matrix , void *MapData );

/* HYPRE_multiblock_matrix.c */
HYPRE_MultiblockMatrix HYPRE_MultiblockMatrixCreate( void );
HYPRE_Int HYPRE_MultiblockMatrixDestroy( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixLimitedDestroy( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixInitialize( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixAssemble( HYPRE_MultiblockMatrix matrix );
void HYPRE_MultiblockMatrixPrint( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixSetNumSubmatrices( HYPRE_MultiblockMatrix matrix , HYPRE_Int n );
HYPRE_Int HYPRE_MultiblockMatrixSetSubmatrixType( HYPRE_MultiblockMatrix matrix , HYPRE_Int j , HYPRE_Int type );

/* HYPRE_vector.c */
HYPRE_Vector HYPRE_VectorCreate( HYPRE_Int size );
HYPRE_Int HYPRE_VectorDestroy( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorInitialize( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorPrint( HYPRE_Vector vector , char *file_name );
HYPRE_Vector HYPRE_VectorRead( char *file_name );

#ifdef __cplusplus
}

#endif

#endif
