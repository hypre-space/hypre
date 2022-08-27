/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_DIRECT_SOLVER_DATA_HEADER
#define hypre_DIRECT_SOLVER_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_DirectSolverData
 *--------------------------------------------------------------------------*/

typedef struct hypre_DirectSolverData_struct
{
   HYPRE_Int            option;
   HYPRE_Int            info;

#if defined (HYPRE_USING_CUSOLVER)
   cusolverDnHandle_t   cusolver_handle;
#endif
} hypre_DirectSolverData;

/*--------------------------------------------------------------------------
 *  Accessor functions for the hypre_DirectSolverData structure
 *--------------------------------------------------------------------------*/

#define hypre_DirectSolverDataOption(data)          ((data) -> option)
#define hypre_DirectSolverDataInfo(data)            ((data) -> info)
#define hypre_DirectSolverDataCuSolverHandle(data)  ((data) -> cusolver_handle)

#endif
