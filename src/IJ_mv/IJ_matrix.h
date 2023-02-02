/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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
