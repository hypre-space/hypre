/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the struct assumed partition
 *
 *****************************************************************************/

#ifndef hypre_ASSUMED_PART_HEADER
#define hypre_ASSUMED_PART_HEADER

typedef struct
{
   /* the entries will be the same for all procs */
   HYPRE_Int           ndim;             /* number of dimensions */
   hypre_BoxArray     *regions;          /* areas of the grid with boxes */
   HYPRE_Int           num_regions;      /* how many regions */
   HYPRE_Int          *proc_partitions;  /* proc ids assigned to each region
                                            (this is size num_regions +1) */
   hypre_Index        *divisions;        /* number of proc divisions in each
                                            direction for each region */
   /* these entries are specific to each proc */
   hypre_BoxArray     *my_partition;        /* my portion of grid (at most 2) */
   hypre_BoxArray     *my_partition_boxes;  /* boxes in my portion */
   HYPRE_Int          *my_partition_proc_ids;
   HYPRE_Int          *my_partition_boxnums;
   HYPRE_Int           my_partition_ids_size;
   HYPRE_Int           my_partition_ids_alloc;
   HYPRE_Int           my_partition_num_distinct_procs;

} hypre_StructAssumedPart;


/*Accessor macros */

#define hypre_StructAssumedPartNDim(apart) ((apart)->ndim)
#define hypre_StructAssumedPartRegions(apart) ((apart)->regions)
#define hypre_StructAssumedPartNumRegions(apart) ((apart)->num_regions)
#define hypre_StructAssumedPartDivisions(apart) ((apart)->divisions)
#define hypre_StructAssumedPartDivision(apart, i) ((apart)->divisions[i])
#define hypre_StructAssumedPartProcPartitions(apart) ((apart)->proc_partitions)
#define hypre_StructAssumedPartProcPartition(apart, i) ((apart)->proc_partitions[i])
#define hypre_StructAssumedPartMyPartition(apart) ((apart)->my_partition)
#define hypre_StructAssumedPartMyPartitionBoxes(apart) ((apart)->my_partition_boxes)
#define hypre_StructAssumedPartMyPartitionProcIds(apart) ((apart)->my_partition_proc_ids)
#define hypre_StructAssumedPartMyPartitionIdsSize(apart) ((apart)->my_partition_ids_size)
#define hypre_StructAssumedPartMyPartitionIdsAlloc(apart) ((apart)->my_partition_ids_alloc)
#define hypre_StructAssumedPartMyPartitionNumDistinctProcs(apart) ((apart)->my_partition_num_distinct_procs)
#define hypre_StructAssumedPartMyPartitionBoxnums(apart) ((apart)->my_partition_boxnums)

#define hypre_StructAssumedPartMyPartitionProcId(apart, i) ((apart)->my_partition_proc_ids[i])
#define hypre_StructAssumedPartMyPartitionBoxnum(apart, i) ((apart)->my_partition_boxnums[i])
#endif
