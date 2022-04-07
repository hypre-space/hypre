/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#ifndef hypre_PARCSR_ASSUMED_PART
#define hypre_PARCSR_ASSUMED_PART

typedef struct
{
   HYPRE_Int                   length;
   HYPRE_BigInt                row_start;
   HYPRE_BigInt                row_end;
   HYPRE_Int                   storage_length;
   HYPRE_Int                  *proc_list;
   HYPRE_BigInt               *row_start_list;
   HYPRE_BigInt               *row_end_list;
   HYPRE_Int                  *sort_index;
} hypre_IJAssumedPart;

#endif /* hypre_PARCSR_ASSUMED_PART */

