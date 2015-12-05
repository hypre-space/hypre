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

   HYPRE_Int M, N;                               /* number of rows and cols in matrix */

   void         *auxiliary_data;           /* Placeholder for implmentation specific
                                              data */

   void         *local_storage;            /* Structure for storing local portion */
   HYPRE_Int      	 local_storage_type;       /* Indicates the type of "local storage" */
   void         *translator;               /* optional storage_type specfic structure
                                              for holding additional local info */
#ifdef HYPRE_TIMING
   HYPRE_Int           GetRow_timer;
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
