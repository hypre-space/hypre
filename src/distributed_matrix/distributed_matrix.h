/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the hypre_DistributedMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_DISTRIBUTED_MATRIX_HEADER
#define hypre_DISTRIBUTED_MATRIX_HEADER


#include "_hypre_utilities.h"


/*--------------------------------------------------------------------------
 * hypre_DistributedMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   HYPRE_BigInt M, N;                               /* number of rows and cols in matrix */

   void         *auxiliary_data;           /* Placeholder for implmentation specific
                                              data */

   void         *local_storage;            /* Structure for storing local portion */
   HYPRE_Int   	 local_storage_type;       /* Indicates the type of "local storage" */
   void         *translator;               /* optional storage_type specfic structure
                                              for holding additional local info */
#ifdef HYPRE_TIMING
   HYPRE_Int     GetRow_timer;
#endif
} hypre_DistributedMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_DistributedMatrix
 *--------------------------------------------------------------------------*/

#define hypre_DistributedMatrixContext(matrix)      ((matrix) -> context)
#define hypre_DistributedMatrixM(matrix)      ((matrix) -> M)
#define hypre_DistributedMatrixN(matrix)      ((matrix) -> N)
#define hypre_DistributedMatrixAuxiliaryData(matrix)         ((matrix) -> auxiliary_data)

#define hypre_DistributedMatrixLocalStorageType(matrix)  ((matrix) -> local_storage_type)
#define hypre_DistributedMatrixTranslator(matrix)   ((matrix) -> translator)
#define hypre_DistributedMatrixLocalStorage(matrix)         ((matrix) -> local_storage)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/
#include "HYPRE_distributed_matrix_mv.h"
#include "internal_protos.h"

#endif
