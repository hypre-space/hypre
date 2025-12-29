/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for overlapping domain decomposition
 *
 *****************************************************************************/

#ifndef hypre_PAR_CSR_OVERLAP_HEADER
#define hypre_PAR_CSR_OVERLAP_HEADER

/*--------------------------------------------------------------------------
 * hypre_OverlapData
 *
 * Data structure for overlapping domain decomposition.
 * Stores information about the extended subdomain for a processor.
 *--------------------------------------------------------------------------*/

typedef struct hypre_OverlapData_struct
{
   /* Overlap configuration */
   HYPRE_Int            overlap_order;         /* Overlap order (delta >= 0) */

   /* Original local partition info */
   HYPRE_Int            num_local_rows;        /* Original local rows */
   HYPRE_BigInt         first_row_index;       /* First row owned by this proc */
   HYPRE_BigInt         last_row_index;        /* Last row owned by this proc */

   /* Extended subdomain information */
   HYPRE_Int            num_extended_rows;     /* Total rows in extended domain */
   HYPRE_Int            num_overlap_rows;      /* External rows (from overlap) */
   HYPRE_BigInt        *extended_row_indices;  /* Global indices of all extended rows */

   /* Mapping arrays */
   HYPRE_Int           *global_to_extended;    /* Map: global row -> extended local index */
   HYPRE_Int           *extended_to_global;    /* Map: extended index -> global row offset */
   HYPRE_Int           *row_is_owned;          /* 1 if row is owned, 0 if external */

   /* Communication package for fetching overlap data */
   hypre_ParCSRCommPkg *overlap_comm_pkg;

   /* External rows CSR matrix (fetched from other procs) */
   hypre_CSRMatrix     *external_rows;         /* CSR matrix of external rows */
   HYPRE_BigInt        *external_row_map;      /* Global row indices for external rows */

} hypre_OverlapData;

/*--------------------------------------------------------------------------
 * Accessor macros for hypre_OverlapData
 *--------------------------------------------------------------------------*/

#define hypre_OverlapDataOverlapOrder(data)          ((data)->overlap_order)
#define hypre_OverlapDataNumLocalRows(data)          ((data)->num_local_rows)
#define hypre_OverlapDataFirstRowIndex(data)         ((data)->first_row_index)
#define hypre_OverlapDataLastRowIndex(data)          ((data)->last_row_index)
#define hypre_OverlapDataNumExtendedRows(data)       ((data)->num_extended_rows)
#define hypre_OverlapDataNumOverlapRows(data)        ((data)->num_overlap_rows)
#define hypre_OverlapDataExtendedRowIndices(data)    ((data)->extended_row_indices)
#define hypre_OverlapDataGlobalToExtended(data)      ((data)->global_to_extended)
#define hypre_OverlapDataExtendedToGlobal(data)      ((data)->extended_to_global)
#define hypre_OverlapDataRowIsOwned(data)            ((data)->row_is_owned)
#define hypre_OverlapDataOverlapCommPkg(data)        ((data)->overlap_comm_pkg)
#define hypre_OverlapDataExternalRows(data)          ((data)->external_rows)
#define hypre_OverlapDataExternalRowMap(data)        ((data)->external_row_map)

#endif /* hypre_PAR_CSR_OVERLAP_HEADER */

