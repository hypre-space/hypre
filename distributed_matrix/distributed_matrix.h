/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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

   int M, N;                               /* number of rows and cols in matrix */

   void         *auxiliary_data;           /* Placeholder for implmentation specific
                                              data */

   void         *local_storage;            /* Structure for storing local portion */
   int      	 local_storage_type;       /* Indicates the type of "local storage" */
   void         *translator;               /* optional storage_type specfic structure
                                              for holding additional local info */
#ifdef HYPRE_TIMING
   int           GetRow_timer;
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
