/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_DIRECT_SOLVER_DATA_HEADER
#define hypre_DIRECT_SOLVER_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_DirectSolverBackend
 *--------------------------------------------------------------------------*/

typedef enum hypre_DirectSolverBackend_enum
{
  HYPRE_DIRECT_SOLVER_VENDOR = 0,
  HYPRE_DIRECT_SOLVER_CUSTOM = 1
} hypre_DirectSolverBackend;

/*--------------------------------------------------------------------------
 * hypre_DirectSolverMethod
 *--------------------------------------------------------------------------*/

typedef enum hypre_DirectSolverMethod_enum
{
  HYPRE_DIRECT_SOLVER_LU    = 0,
  HYPRE_DIRECT_SOLVER_LUPIV = 1,
  HYPRE_DIRECT_SOLVER_CHOL  = 2
} hypre_DirectSolverMethod;

/*--------------------------------------------------------------------------
 * hypre_MatrixType
 *--------------------------------------------------------------------------*/

typedef enum hypre_MatrixType_enum
{
  HYPRE_MATRIX_TYPE_UBATCHED_DENSE  = 0,
  HYPRE_MATRIX_TYPE_VBATCHED_DENSE  = 1,
  HYPRE_MATRIX_TYPE_UBATCHED_SPARSE = 2,
  HYPRE_MATRIX_TYPE_VBATCHED_SPARSE = 3
} hypre_MatrixType;

/*--------------------------------------------------------------------------
 * hypre_DirectSolverData
 *--------------------------------------------------------------------------*/

typedef struct hypre_DirectSolverData_struct
{
   hypre_DirectSolverBackend   backend;
   hypre_DirectSolverMethod    method;
   hypre_MatrixType            mat_type;

   HYPRE_Int                   size;
   HYPRE_Int                  *pivots;
   HYPRE_Int                  *info;
   HYPRE_MemoryLocation        memory_location;
} hypre_DirectSolverData;

/*--------------------------------------------------------------------------
 *  Accessor functions for the hypre_DirectSolverData structure
 *--------------------------------------------------------------------------*/

#define hypre_DirectSolverDataBackend(data)         ((data) -> backend)
#define hypre_DirectSolverDataMethod(data)          ((data) -> method)
#define hypre_DirectSolverDataMatType(data)         ((data) -> mat_type)
#define hypre_DirectSolverDataPivots(data)          ((data) -> pivots)
#define hypre_DirectSolverDataSize(data)            ((data) -> size)
#define hypre_DirectSolverDataInfo(data)            ((data) -> info)
#define hypre_DirectSolverDataMemoryLocation(data)  ((data) -> memory_location)

#endif
