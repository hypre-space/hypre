/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for the hypre_DistributedMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_DISTRIBUTED_MATRIX_HEADER
#define hypre_DISTRIBUTED_MATRIX_HEADER

#include <../HYPRE_config.h>

#include "../utilities/general.h"
#include "../utilities/utilities.h"

#include "../HYPRE.h"

#include "./HYPRE_distributed_matrix_types.h"

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
#include "./hypre_protos.h"
#include "./internal_protos.h"

#endif
