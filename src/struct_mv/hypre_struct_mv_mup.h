
/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * Header file of multiprecision function prototypes.
 * This is needed for mixed-precision algorithm development.
 *****************************************************************************/

#ifndef HYPRE_STRUCT_MV_MUP_HEADER
#define HYPRE_STRUCT_MV_MUP_HEADER

#if defined (HYPRE_MIXED_PRECISION)

HYPRE_Int hypre_APFillResponseStructAssumedPart_flt  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                                  HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                  HYPRE_Int *response_message_size );
HYPRE_Int hypre_APFillResponseStructAssumedPart_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                                  HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                  HYPRE_Int *response_message_size );
HYPRE_Int hypre_APFillResponseStructAssumedPart_long_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                                  HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                  HYPRE_Int *response_message_size );
HYPRE_Int hypre_APFindMyBoxesInRegions_flt  ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                         HYPRE_Int **p_count_array, hypre_float **p_vol_array );
HYPRE_Int hypre_APFindMyBoxesInRegions_dbl  ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                         HYPRE_Int **p_count_array, hypre_double **p_vol_array );
HYPRE_Int hypre_APFindMyBoxesInRegions_long_dbl  ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                         HYPRE_Int **p_count_array, hypre_long_double **p_vol_array );
HYPRE_Int hypre_APGetAllBoxesInRegions_flt  ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                         HYPRE_Int **p_count_array, hypre_float **p_vol_array, MPI_Comm comm );
HYPRE_Int hypre_APGetAllBoxesInRegions_dbl  ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                         HYPRE_Int **p_count_array, hypre_double **p_vol_array, MPI_Comm comm );
HYPRE_Int hypre_APGetAllBoxesInRegions_long_dbl  ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                         HYPRE_Int **p_count_array, hypre_long_double **p_vol_array, MPI_Comm comm );
HYPRE_Int hypre_APPruneRegions_flt  ( hypre_BoxArray *region_array, HYPRE_Int **p_count_array,
                                 hypre_float **p_vol_array );
HYPRE_Int hypre_APPruneRegions_dbl  ( hypre_BoxArray *region_array, HYPRE_Int **p_count_array,
                                 hypre_double **p_vol_array );
HYPRE_Int hypre_APPruneRegions_long_dbl  ( hypre_BoxArray *region_array, HYPRE_Int **p_count_array,
                                 hypre_long_double **p_vol_array );
HYPRE_Int hypre_APRefineRegionsByVol_flt  ( hypre_BoxArray *region_array, hypre_float *vol_array,
                                       HYPRE_Int max_regions, hypre_float gamma, HYPRE_Int dim, HYPRE_Int *return_code, MPI_Comm comm );
HYPRE_Int hypre_APRefineRegionsByVol_dbl  ( hypre_BoxArray *region_array, hypre_double *vol_array,
                                       HYPRE_Int max_regions, hypre_double gamma, HYPRE_Int dim, HYPRE_Int *return_code, MPI_Comm comm );
HYPRE_Int hypre_APRefineRegionsByVol_long_dbl  ( hypre_BoxArray *region_array, hypre_long_double *vol_array,
                                       HYPRE_Int max_regions, hypre_long_double gamma, HYPRE_Int dim, HYPRE_Int *return_code, MPI_Comm comm );
HYPRE_Int hypre_APShrinkRegions_flt  ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                  MPI_Comm comm );
HYPRE_Int hypre_APShrinkRegions_dbl  ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                  MPI_Comm comm );
HYPRE_Int hypre_APShrinkRegions_long_dbl  ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                  MPI_Comm comm );
HYPRE_Int hypre_APSubdivideRegion_flt  ( hypre_Box *region, HYPRE_Int dim, HYPRE_Int level,
                                    hypre_BoxArray *box_array, HYPRE_Int *num_new_boxes );
HYPRE_Int hypre_APSubdivideRegion_dbl  ( hypre_Box *region, HYPRE_Int dim, HYPRE_Int level,
                                    hypre_BoxArray *box_array, HYPRE_Int *num_new_boxes );
HYPRE_Int hypre_APSubdivideRegion_long_dbl  ( hypre_Box *region, HYPRE_Int dim, HYPRE_Int level,
                                    hypre_BoxArray *box_array, HYPRE_Int *num_new_boxes );
HYPRE_Int hypre_StructAssumedPartitionCreate_flt  ( HYPRE_Int dim, hypre_Box *bounding_box,
                                               hypre_float global_boxes_size, HYPRE_Int global_num_boxes, hypre_BoxArray *local_boxes,
                                               HYPRE_Int *local_boxnums, HYPRE_Int max_regions, HYPRE_Int max_refinements, hypre_float gamma,
                                               MPI_Comm comm, hypre_StructAssumedPart **p_assumed_partition );
HYPRE_Int hypre_StructAssumedPartitionCreate_dbl  ( HYPRE_Int dim, hypre_Box *bounding_box,
                                               hypre_double global_boxes_size, HYPRE_Int global_num_boxes, hypre_BoxArray *local_boxes,
                                               HYPRE_Int *local_boxnums, HYPRE_Int max_regions, HYPRE_Int max_refinements, hypre_double gamma,
                                               MPI_Comm comm, hypre_StructAssumedPart **p_assumed_partition );
HYPRE_Int hypre_StructAssumedPartitionCreate_long_dbl  ( HYPRE_Int dim, hypre_Box *bounding_box,
                                               hypre_long_double global_boxes_size, HYPRE_Int global_num_boxes, hypre_BoxArray *local_boxes,
                                               HYPRE_Int *local_boxnums, HYPRE_Int max_regions, HYPRE_Int max_refinements, hypre_long_double gamma,
                                               MPI_Comm comm, hypre_StructAssumedPart **p_assumed_partition );
HYPRE_Int hypre_StructAssumedPartitionDestroy_flt  ( hypre_StructAssumedPart *assumed_part );
HYPRE_Int hypre_StructAssumedPartitionDestroy_dbl  ( hypre_StructAssumedPart *assumed_part );
HYPRE_Int hypre_StructAssumedPartitionDestroy_long_dbl  ( hypre_StructAssumedPart *assumed_part );
HYPRE_Int hypre_StructAssumedPartitionGetProcsFromBox_flt  ( hypre_StructAssumedPart *assumed_part,
                                                        hypre_Box *box, HYPRE_Int *num_proc_array, HYPRE_Int *size_alloc_proc_array,
                                                        HYPRE_Int **p_proc_array );
HYPRE_Int hypre_StructAssumedPartitionGetProcsFromBox_dbl  ( hypre_StructAssumedPart *assumed_part,
                                                        hypre_Box *box, HYPRE_Int *num_proc_array, HYPRE_Int *size_alloc_proc_array,
                                                        HYPRE_Int **p_proc_array );
HYPRE_Int hypre_StructAssumedPartitionGetProcsFromBox_long_dbl  ( hypre_StructAssumedPart *assumed_part,
                                                        hypre_Box *box, HYPRE_Int *num_proc_array, HYPRE_Int *size_alloc_proc_array,
                                                        HYPRE_Int **p_proc_array );
HYPRE_Int hypre_StructAssumedPartitionGetRegionsFromProc_flt  ( hypre_StructAssumedPart *assumed_part,
                                                           HYPRE_Int proc_id, hypre_BoxArray *assumed_regions );
HYPRE_Int hypre_StructAssumedPartitionGetRegionsFromProc_dbl  ( hypre_StructAssumedPart *assumed_part,
                                                           HYPRE_Int proc_id, hypre_BoxArray *assumed_regions );
HYPRE_Int hypre_StructAssumedPartitionGetRegionsFromProc_long_dbl  ( hypre_StructAssumedPart *assumed_part,
                                                           HYPRE_Int proc_id, hypre_BoxArray *assumed_regions );
HYPRE_Int hypre_IntersectBoxes_flt  ( hypre_Box *box1, hypre_Box *box2, hypre_Box *ibox );
HYPRE_Int hypre_IntersectBoxes_dbl  ( hypre_Box *box1, hypre_Box *box2, hypre_Box *ibox );
HYPRE_Int hypre_IntersectBoxes_long_dbl  ( hypre_Box *box1, hypre_Box *box2, hypre_Box *ibox );
HYPRE_Int hypre_MinUnionBoxes_flt  ( hypre_BoxArray *boxes );
HYPRE_Int hypre_MinUnionBoxes_dbl  ( hypre_BoxArray *boxes );
HYPRE_Int hypre_MinUnionBoxes_long_dbl  ( hypre_BoxArray *boxes );
HYPRE_Int hypre_SubtractBoxArrays_flt  ( hypre_BoxArray *box_array1, hypre_BoxArray *box_array2,
                                    hypre_BoxArray *tmp_box_array );
HYPRE_Int hypre_SubtractBoxArrays_dbl  ( hypre_BoxArray *box_array1, hypre_BoxArray *box_array2,
                                    hypre_BoxArray *tmp_box_array );
HYPRE_Int hypre_SubtractBoxArrays_long_dbl  ( hypre_BoxArray *box_array1, hypre_BoxArray *box_array2,
                                    hypre_BoxArray *tmp_box_array );
HYPRE_Int hypre_SubtractBoxes_flt  ( hypre_Box *box1, hypre_Box *box2, hypre_BoxArray *box_array );
HYPRE_Int hypre_SubtractBoxes_dbl  ( hypre_Box *box1, hypre_Box *box2, hypre_BoxArray *box_array );
HYPRE_Int hypre_SubtractBoxes_long_dbl  ( hypre_Box *box1, hypre_Box *box2, hypre_BoxArray *box_array );
HYPRE_Int hypre_UnionBoxes_flt  ( hypre_BoxArray *boxes );
HYPRE_Int hypre_UnionBoxes_dbl  ( hypre_BoxArray *boxes );
HYPRE_Int hypre_UnionBoxes_long_dbl  ( hypre_BoxArray *boxes );
HYPRE_Int hypre_BoxBoundaryDG_flt  ( hypre_Box *box, hypre_StructGrid *g, hypre_BoxArray *boundarym,
                                hypre_BoxArray *boundaryp, HYPRE_Int d );
HYPRE_Int hypre_BoxBoundaryDG_dbl  ( hypre_Box *box, hypre_StructGrid *g, hypre_BoxArray *boundarym,
                                hypre_BoxArray *boundaryp, HYPRE_Int d );
HYPRE_Int hypre_BoxBoundaryDG_long_dbl  ( hypre_Box *box, hypre_StructGrid *g, hypre_BoxArray *boundarym,
                                hypre_BoxArray *boundaryp, HYPRE_Int d );
HYPRE_Int hypre_BoxBoundaryG_flt  ( hypre_Box *box, hypre_StructGrid *g, hypre_BoxArray *boundary );
HYPRE_Int hypre_BoxBoundaryG_dbl  ( hypre_Box *box, hypre_StructGrid *g, hypre_BoxArray *boundary );
HYPRE_Int hypre_BoxBoundaryG_long_dbl  ( hypre_Box *box, hypre_StructGrid *g, hypre_BoxArray *boundary );
HYPRE_Int hypre_BoxBoundaryIntersect_flt  ( hypre_Box *box, hypre_StructGrid *grid, HYPRE_Int d,
                                       HYPRE_Int dir, hypre_BoxArray *boundary );
HYPRE_Int hypre_BoxBoundaryIntersect_dbl  ( hypre_Box *box, hypre_StructGrid *grid, HYPRE_Int d,
                                       HYPRE_Int dir, hypre_BoxArray *boundary );
HYPRE_Int hypre_BoxBoundaryIntersect_long_dbl  ( hypre_Box *box, hypre_StructGrid *grid, HYPRE_Int d,
                                       HYPRE_Int dir, hypre_BoxArray *boundary );
HYPRE_Int hypre_GeneralBoxBoundaryIntersect_flt ( hypre_Box *box, hypre_StructGrid *grid,
                                             hypre_Index stencil_element, hypre_BoxArray *boundary );
HYPRE_Int hypre_GeneralBoxBoundaryIntersect_dbl ( hypre_Box *box, hypre_StructGrid *grid,
                                             hypre_Index stencil_element, hypre_BoxArray *boundary );
HYPRE_Int hypre_GeneralBoxBoundaryIntersect_long_dbl ( hypre_Box *box, hypre_StructGrid *grid,
                                             hypre_Index stencil_element, hypre_BoxArray *boundary );
HYPRE_Int hypre_BoxManAddEntry_flt  ( hypre_BoxManager *manager, hypre_Index imin, hypre_Index imax,
                                 HYPRE_Int proc_id, HYPRE_Int box_id, void *info );
HYPRE_Int hypre_BoxManAddEntry_dbl  ( hypre_BoxManager *manager, hypre_Index imin, hypre_Index imax,
                                 HYPRE_Int proc_id, HYPRE_Int box_id, void *info );
HYPRE_Int hypre_BoxManAddEntry_long_dbl  ( hypre_BoxManager *manager, hypre_Index imin, hypre_Index imax,
                                 HYPRE_Int proc_id, HYPRE_Int box_id, void *info );
HYPRE_Int hypre_BoxManAssemble_flt  ( hypre_BoxManager *manager );
HYPRE_Int hypre_BoxManAssemble_dbl  ( hypre_BoxManager *manager );
HYPRE_Int hypre_BoxManAssemble_long_dbl  ( hypre_BoxManager *manager );
HYPRE_Int hypre_BoxManCreate_flt  ( HYPRE_Int max_nentries, HYPRE_Int info_size, HYPRE_Int dim,
                               hypre_Box *bounding_box, MPI_Comm comm, hypre_BoxManager **manager_ptr );
HYPRE_Int hypre_BoxManCreate_dbl  ( HYPRE_Int max_nentries, HYPRE_Int info_size, HYPRE_Int dim,
                               hypre_Box *bounding_box, MPI_Comm comm, hypre_BoxManager **manager_ptr );
HYPRE_Int hypre_BoxManCreate_long_dbl  ( HYPRE_Int max_nentries, HYPRE_Int info_size, HYPRE_Int dim,
                               hypre_Box *bounding_box, MPI_Comm comm, hypre_BoxManager **manager_ptr );
HYPRE_Int hypre_BoxManDeleteMultipleEntriesAndInfo_flt  ( hypre_BoxManager *manager, HYPRE_Int *indices,
                                                     HYPRE_Int num );
HYPRE_Int hypre_BoxManDeleteMultipleEntriesAndInfo_dbl  ( hypre_BoxManager *manager, HYPRE_Int *indices,
                                                     HYPRE_Int num );
HYPRE_Int hypre_BoxManDeleteMultipleEntriesAndInfo_long_dbl  ( hypre_BoxManager *manager, HYPRE_Int *indices,
                                                     HYPRE_Int num );
HYPRE_Int hypre_BoxManDestroy_flt  ( hypre_BoxManager *manager );
HYPRE_Int hypre_BoxManDestroy_dbl  ( hypre_BoxManager *manager );
HYPRE_Int hypre_BoxManDestroy_long_dbl  ( hypre_BoxManager *manager );
HYPRE_Int hypre_BoxManEntryCopy_flt  ( hypre_BoxManEntry *fromentry, hypre_BoxManEntry *toentry );
HYPRE_Int hypre_BoxManEntryCopy_dbl  ( hypre_BoxManEntry *fromentry, hypre_BoxManEntry *toentry );
HYPRE_Int hypre_BoxManEntryCopy_long_dbl  ( hypre_BoxManEntry *fromentry, hypre_BoxManEntry *toentry );
HYPRE_Int hypre_BoxManEntryGetExtents_flt  ( hypre_BoxManEntry *entry, hypre_Index imin,
                                        hypre_Index imax );
HYPRE_Int hypre_BoxManEntryGetExtents_dbl  ( hypre_BoxManEntry *entry, hypre_Index imin,
                                        hypre_Index imax );
HYPRE_Int hypre_BoxManEntryGetExtents_long_dbl  ( hypre_BoxManEntry *entry, hypre_Index imin,
                                        hypre_Index imax );
HYPRE_Int hypre_BoxManEntryGetInfo_flt  ( hypre_BoxManEntry *entry, void **info_ptr );
HYPRE_Int hypre_BoxManEntryGetInfo_dbl  ( hypre_BoxManEntry *entry, void **info_ptr );
HYPRE_Int hypre_BoxManEntryGetInfo_long_dbl  ( hypre_BoxManEntry *entry, void **info_ptr );
HYPRE_Int hypre_BoxManGatherEntries_flt  ( hypre_BoxManager *manager, hypre_Index imin,
                                      hypre_Index imax );
HYPRE_Int hypre_BoxManGatherEntries_dbl  ( hypre_BoxManager *manager, hypre_Index imin,
                                      hypre_Index imax );
HYPRE_Int hypre_BoxManGatherEntries_long_dbl  ( hypre_BoxManager *manager, hypre_Index imin,
                                      hypre_Index imax );
HYPRE_Int hypre_BoxManGetAllEntries_flt  ( hypre_BoxManager *manager, HYPRE_Int *num_entries,
                                      hypre_BoxManEntry **entries );
HYPRE_Int hypre_BoxManGetAllEntries_dbl  ( hypre_BoxManager *manager, HYPRE_Int *num_entries,
                                      hypre_BoxManEntry **entries );
HYPRE_Int hypre_BoxManGetAllEntries_long_dbl  ( hypre_BoxManager *manager, HYPRE_Int *num_entries,
                                      hypre_BoxManEntry **entries );
HYPRE_Int hypre_BoxManGetAllEntriesBoxes_flt  ( hypre_BoxManager *manager, hypre_BoxArray *boxes );
HYPRE_Int hypre_BoxManGetAllEntriesBoxes_dbl  ( hypre_BoxManager *manager, hypre_BoxArray *boxes );
HYPRE_Int hypre_BoxManGetAllEntriesBoxes_long_dbl  ( hypre_BoxManager *manager, hypre_BoxArray *boxes );
HYPRE_Int hypre_BoxManGetAllEntriesBoxesProc_flt  ( hypre_BoxManager *manager, hypre_BoxArray *boxes,
                                               HYPRE_Int **procs_ptr );
HYPRE_Int hypre_BoxManGetAllEntriesBoxesProc_dbl  ( hypre_BoxManager *manager, hypre_BoxArray *boxes,
                                               HYPRE_Int **procs_ptr );
HYPRE_Int hypre_BoxManGetAllEntriesBoxesProc_long_dbl  ( hypre_BoxManager *manager, hypre_BoxArray *boxes,
                                               HYPRE_Int **procs_ptr );
HYPRE_Int hypre_BoxManGetAllGlobalKnown_flt  ( hypre_BoxManager *manager, HYPRE_Int *known );
HYPRE_Int hypre_BoxManGetAllGlobalKnown_dbl  ( hypre_BoxManager *manager, HYPRE_Int *known );
HYPRE_Int hypre_BoxManGetAllGlobalKnown_long_dbl  ( hypre_BoxManager *manager, HYPRE_Int *known );
HYPRE_Int hypre_BoxManGetAssumedPartition_flt  ( hypre_BoxManager *manager,
                                            hypre_StructAssumedPart **assumed_partition );
HYPRE_Int hypre_BoxManGetAssumedPartition_dbl  ( hypre_BoxManager *manager,
                                            hypre_StructAssumedPart **assumed_partition );
HYPRE_Int hypre_BoxManGetAssumedPartition_long_dbl  ( hypre_BoxManager *manager,
                                            hypre_StructAssumedPart **assumed_partition );
HYPRE_Int hypre_BoxManGetEntry_flt  ( hypre_BoxManager *manager, HYPRE_Int proc, HYPRE_Int id,
                                 hypre_BoxManEntry **entry_ptr );
HYPRE_Int hypre_BoxManGetEntry_dbl  ( hypre_BoxManager *manager, HYPRE_Int proc, HYPRE_Int id,
                                 hypre_BoxManEntry **entry_ptr );
HYPRE_Int hypre_BoxManGetEntry_long_dbl  ( hypre_BoxManager *manager, HYPRE_Int proc, HYPRE_Int id,
                                 hypre_BoxManEntry **entry_ptr );
HYPRE_Int hypre_BoxManGetGlobalIsGatherCalled_flt  ( hypre_BoxManager *manager, MPI_Comm comm,
                                                HYPRE_Int *is_gather );
HYPRE_Int hypre_BoxManGetGlobalIsGatherCalled_dbl  ( hypre_BoxManager *manager, MPI_Comm comm,
                                                HYPRE_Int *is_gather );
HYPRE_Int hypre_BoxManGetGlobalIsGatherCalled_long_dbl  ( hypre_BoxManager *manager, MPI_Comm comm,
                                                HYPRE_Int *is_gather );
HYPRE_Int hypre_BoxManGetIsEntriesSort_flt  ( hypre_BoxManager *manager, HYPRE_Int *is_sort );
HYPRE_Int hypre_BoxManGetIsEntriesSort_dbl  ( hypre_BoxManager *manager, HYPRE_Int *is_sort );
HYPRE_Int hypre_BoxManGetIsEntriesSort_long_dbl  ( hypre_BoxManager *manager, HYPRE_Int *is_sort );
HYPRE_Int hypre_BoxManGetLocalEntriesBoxes_flt  ( hypre_BoxManager *manager, hypre_BoxArray *boxes );
HYPRE_Int hypre_BoxManGetLocalEntriesBoxes_dbl  ( hypre_BoxManager *manager, hypre_BoxArray *boxes );
HYPRE_Int hypre_BoxManGetLocalEntriesBoxes_long_dbl  ( hypre_BoxManager *manager, hypre_BoxArray *boxes );
HYPRE_Int hypre_BoxManIncSize_flt  ( hypre_BoxManager *manager, HYPRE_Int inc_size );
HYPRE_Int hypre_BoxManIncSize_dbl  ( hypre_BoxManager *manager, HYPRE_Int inc_size );
HYPRE_Int hypre_BoxManIncSize_long_dbl  ( hypre_BoxManager *manager, HYPRE_Int inc_size );
HYPRE_Int hypre_BoxManIntersect_flt  ( hypre_BoxManager *manager, hypre_Index ilower, hypre_Index iupper,
                                  hypre_BoxManEntry ***entries_ptr, HYPRE_Int *nentries_ptr );
HYPRE_Int hypre_BoxManIntersect_dbl  ( hypre_BoxManager *manager, hypre_Index ilower, hypre_Index iupper,
                                  hypre_BoxManEntry ***entries_ptr, HYPRE_Int *nentries_ptr );
HYPRE_Int hypre_BoxManIntersect_long_dbl  ( hypre_BoxManager *manager, hypre_Index ilower, hypre_Index iupper,
                                  hypre_BoxManEntry ***entries_ptr, HYPRE_Int *nentries_ptr );
HYPRE_Int hypre_BoxManSetAllGlobalKnown_flt  ( hypre_BoxManager *manager, HYPRE_Int known );
HYPRE_Int hypre_BoxManSetAllGlobalKnown_dbl  ( hypre_BoxManager *manager, HYPRE_Int known );
HYPRE_Int hypre_BoxManSetAllGlobalKnown_long_dbl  ( hypre_BoxManager *manager, HYPRE_Int known );
HYPRE_Int hypre_BoxManSetAssumedPartition_flt  ( hypre_BoxManager *manager,
                                            hypre_StructAssumedPart *assumed_partition );
HYPRE_Int hypre_BoxManSetAssumedPartition_dbl  ( hypre_BoxManager *manager,
                                            hypre_StructAssumedPart *assumed_partition );
HYPRE_Int hypre_BoxManSetAssumedPartition_long_dbl  ( hypre_BoxManager *manager,
                                            hypre_StructAssumedPart *assumed_partition );
HYPRE_Int hypre_BoxManSetBoundingBox_flt  ( hypre_BoxManager *manager, hypre_Box *bounding_box );
HYPRE_Int hypre_BoxManSetBoundingBox_dbl  ( hypre_BoxManager *manager, hypre_Box *bounding_box );
HYPRE_Int hypre_BoxManSetBoundingBox_long_dbl  ( hypre_BoxManager *manager, hypre_Box *bounding_box );
HYPRE_Int hypre_BoxManSetIsEntriesSort_flt  ( hypre_BoxManager *manager, HYPRE_Int is_sort );
HYPRE_Int hypre_BoxManSetIsEntriesSort_dbl  ( hypre_BoxManager *manager, HYPRE_Int is_sort );
HYPRE_Int hypre_BoxManSetIsEntriesSort_long_dbl  ( hypre_BoxManager *manager, HYPRE_Int is_sort );
HYPRE_Int hypre_BoxManSetNumGhost_flt  ( hypre_BoxManager *manager, HYPRE_Int *num_ghost );
HYPRE_Int hypre_BoxManSetNumGhost_dbl  ( hypre_BoxManager *manager, HYPRE_Int *num_ghost );
HYPRE_Int hypre_BoxManSetNumGhost_long_dbl  ( hypre_BoxManager *manager, HYPRE_Int *num_ghost );
HYPRE_Int hypre_FillResponseBoxManAssemble1_flt  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                              HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseBoxManAssemble1_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                              HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseBoxManAssemble1_long_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                              HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseBoxManAssemble2_flt  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                              HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseBoxManAssemble2_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                              HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseBoxManAssemble2_long_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                              HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              HYPRE_Int *response_message_size );
HYPRE_Int hypre_AddIndexes_flt  ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim,
                             hypre_Index result );
HYPRE_Int hypre_AddIndexes_dbl  ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim,
                             hypre_Index result );
HYPRE_Int hypre_AddIndexes_long_dbl  ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim,
                             hypre_Index result );
HYPRE_Int hypre_AppendBox_flt  ( hypre_Box *box, hypre_BoxArray *box_array );
HYPRE_Int hypre_AppendBox_dbl  ( hypre_Box *box, hypre_BoxArray *box_array );
HYPRE_Int hypre_AppendBox_long_dbl  ( hypre_Box *box, hypre_BoxArray *box_array );
HYPRE_Int hypre_AppendBoxArray_flt  ( hypre_BoxArray *box_array_0, hypre_BoxArray *box_array_1 );
HYPRE_Int hypre_AppendBoxArray_dbl  ( hypre_BoxArray *box_array_0, hypre_BoxArray *box_array_1 );
HYPRE_Int hypre_AppendBoxArray_long_dbl  ( hypre_BoxArray *box_array_0, hypre_BoxArray *box_array_1 );
hypre_BoxArrayArray *hypre_BoxArrayArrayCreate_flt  ( HYPRE_Int size, HYPRE_Int ndim );
hypre_BoxArrayArray *hypre_BoxArrayArrayCreate_dbl  ( HYPRE_Int size, HYPRE_Int ndim );
hypre_BoxArrayArray *hypre_BoxArrayArrayCreate_long_dbl  ( HYPRE_Int size, HYPRE_Int ndim );
HYPRE_Int hypre_BoxArrayArrayDestroy_flt  ( hypre_BoxArrayArray *box_array_array );
HYPRE_Int hypre_BoxArrayArrayDestroy_dbl  ( hypre_BoxArrayArray *box_array_array );
HYPRE_Int hypre_BoxArrayArrayDestroy_long_dbl  ( hypre_BoxArrayArray *box_array_array );
hypre_BoxArrayArray *hypre_BoxArrayArrayDuplicate_flt  ( hypre_BoxArrayArray *box_array_array );
hypre_BoxArrayArray *hypre_BoxArrayArrayDuplicate_dbl  ( hypre_BoxArrayArray *box_array_array );
hypre_BoxArrayArray *hypre_BoxArrayArrayDuplicate_long_dbl  ( hypre_BoxArrayArray *box_array_array );
hypre_BoxArray *hypre_BoxArrayCreate_flt  ( HYPRE_Int size, HYPRE_Int ndim );
hypre_BoxArray *hypre_BoxArrayCreate_dbl  ( HYPRE_Int size, HYPRE_Int ndim );
hypre_BoxArray *hypre_BoxArrayCreate_long_dbl  ( HYPRE_Int size, HYPRE_Int ndim );
HYPRE_Int hypre_BoxArrayDestroy_flt  ( hypre_BoxArray *box_array );
HYPRE_Int hypre_BoxArrayDestroy_dbl  ( hypre_BoxArray *box_array );
HYPRE_Int hypre_BoxArrayDestroy_long_dbl  ( hypre_BoxArray *box_array );
hypre_BoxArray *hypre_BoxArrayDuplicate_flt  ( hypre_BoxArray *box_array );
hypre_BoxArray *hypre_BoxArrayDuplicate_dbl  ( hypre_BoxArray *box_array );
hypre_BoxArray *hypre_BoxArrayDuplicate_long_dbl  ( hypre_BoxArray *box_array );
HYPRE_Int hypre_BoxArraySetSize_flt  ( hypre_BoxArray *box_array, HYPRE_Int size );
HYPRE_Int hypre_BoxArraySetSize_dbl  ( hypre_BoxArray *box_array, HYPRE_Int size );
HYPRE_Int hypre_BoxArraySetSize_long_dbl  ( hypre_BoxArray *box_array, HYPRE_Int size );
hypre_Box *hypre_BoxCreate_flt  ( HYPRE_Int ndim );
hypre_Box *hypre_BoxCreate_dbl  ( HYPRE_Int ndim );
hypre_Box *hypre_BoxCreate_long_dbl  ( HYPRE_Int ndim );
HYPRE_Int hypre_BoxDestroy_flt  ( hypre_Box *box );
HYPRE_Int hypre_BoxDestroy_dbl  ( hypre_Box *box );
HYPRE_Int hypre_BoxDestroy_long_dbl  ( hypre_Box *box );
hypre_Box *hypre_BoxDuplicate_flt  ( hypre_Box *box );
hypre_Box *hypre_BoxDuplicate_dbl  ( hypre_Box *box );
hypre_Box *hypre_BoxDuplicate_long_dbl  ( hypre_Box *box );
HYPRE_Int hypre_BoxGetSize_flt  ( hypre_Box *box, hypre_Index size );
HYPRE_Int hypre_BoxGetSize_dbl  ( hypre_Box *box, hypre_Index size );
HYPRE_Int hypre_BoxGetSize_long_dbl  ( hypre_Box *box, hypre_Index size );
HYPRE_Int hypre_BoxGetStrideSize_flt  ( hypre_Box *box, hypre_Index stride, hypre_Index size );
HYPRE_Int hypre_BoxGetStrideSize_dbl  ( hypre_Box *box, hypre_Index stride, hypre_Index size );
HYPRE_Int hypre_BoxGetStrideSize_long_dbl  ( hypre_Box *box, hypre_Index stride, hypre_Index size );
HYPRE_Int hypre_BoxGetStrideVolume_flt  ( hypre_Box *box, hypre_Index stride, HYPRE_Int *volume_ptr );
HYPRE_Int hypre_BoxGetStrideVolume_dbl  ( hypre_Box *box, hypre_Index stride, HYPRE_Int *volume_ptr );
HYPRE_Int hypre_BoxGetStrideVolume_long_dbl  ( hypre_Box *box, hypre_Index stride, HYPRE_Int *volume_ptr );
HYPRE_Int hypre_BoxGrowByArray_flt  ( hypre_Box *box, HYPRE_Int *array );
HYPRE_Int hypre_BoxGrowByArray_dbl  ( hypre_Box *box, HYPRE_Int *array );
HYPRE_Int hypre_BoxGrowByArray_long_dbl  ( hypre_Box *box, HYPRE_Int *array );
HYPRE_Int hypre_BoxGrowByIndex_flt ( hypre_Box *box, hypre_Index  index );
HYPRE_Int hypre_BoxGrowByIndex_dbl ( hypre_Box *box, hypre_Index  index );
HYPRE_Int hypre_BoxGrowByIndex_long_dbl ( hypre_Box *box, hypre_Index  index );
HYPRE_Int hypre_BoxGrowByValue_flt ( hypre_Box *box, HYPRE_Int val );
HYPRE_Int hypre_BoxGrowByValue_dbl ( hypre_Box *box, HYPRE_Int val );
HYPRE_Int hypre_BoxGrowByValue_long_dbl ( hypre_Box *box, HYPRE_Int val );
HYPRE_Int hypre_BoxIndexRank_flt ( hypre_Box *box, hypre_Index index );
HYPRE_Int hypre_BoxIndexRank_dbl ( hypre_Box *box, hypre_Index index );
HYPRE_Int hypre_BoxIndexRank_long_dbl ( hypre_Box *box, hypre_Index index );
HYPRE_Int hypre_BoxInit_flt ( hypre_Box *box, HYPRE_Int  ndim );
HYPRE_Int hypre_BoxInit_dbl ( hypre_Box *box, HYPRE_Int  ndim );
HYPRE_Int hypre_BoxInit_long_dbl ( hypre_Box *box, HYPRE_Int  ndim );
HYPRE_Int hypre_BoxOffsetDistance_flt ( hypre_Box *box, hypre_Index index );
HYPRE_Int hypre_BoxOffsetDistance_dbl ( hypre_Box *box, hypre_Index index );
HYPRE_Int hypre_BoxOffsetDistance_long_dbl ( hypre_Box *box, hypre_Index index );
HYPRE_Int hypre_BoxPrint_flt  ( FILE *file, hypre_Box *box );
HYPRE_Int hypre_BoxPrint_dbl  ( FILE *file, hypre_Box *box );
HYPRE_Int hypre_BoxPrint_long_dbl  ( FILE *file, hypre_Box *box );
HYPRE_Int hypre_BoxRankIndex_flt ( hypre_Box *box, HYPRE_Int rank, hypre_Index index );
HYPRE_Int hypre_BoxRankIndex_dbl ( hypre_Box *box, HYPRE_Int rank, hypre_Index index );
HYPRE_Int hypre_BoxRankIndex_long_dbl ( hypre_Box *box, HYPRE_Int rank, hypre_Index index );
HYPRE_Int hypre_BoxRead_flt  ( FILE *file, HYPRE_Int ndim, hypre_Box **box_ptr );
HYPRE_Int hypre_BoxRead_dbl  ( FILE *file, HYPRE_Int ndim, hypre_Box **box_ptr );
HYPRE_Int hypre_BoxRead_long_dbl  ( FILE *file, HYPRE_Int ndim, hypre_Box **box_ptr );
HYPRE_Int hypre_BoxSetExtents_flt  ( hypre_Box *box, hypre_Index imin, hypre_Index imax );
HYPRE_Int hypre_BoxSetExtents_dbl  ( hypre_Box *box, hypre_Index imin, hypre_Index imax );
HYPRE_Int hypre_BoxSetExtents_long_dbl  ( hypre_Box *box, hypre_Index imin, hypre_Index imax );
HYPRE_Int hypre_BoxShiftNeg_flt ( hypre_Box *box, hypre_Index shift );
HYPRE_Int hypre_BoxShiftNeg_dbl ( hypre_Box *box, hypre_Index shift );
HYPRE_Int hypre_BoxShiftNeg_long_dbl ( hypre_Box *box, hypre_Index shift );
HYPRE_Int hypre_BoxShiftPos_flt ( hypre_Box *box, hypre_Index shift );
HYPRE_Int hypre_BoxShiftPos_dbl ( hypre_Box *box, hypre_Index shift );
HYPRE_Int hypre_BoxShiftPos_long_dbl ( hypre_Box *box, hypre_Index shift );
HYPRE_Int hypre_BoxVolume_flt ( hypre_Box *box );
HYPRE_Int hypre_BoxVolume_dbl ( hypre_Box *box );
HYPRE_Int hypre_BoxVolume_long_dbl ( hypre_Box *box );
HYPRE_Int hypre_CopyBox_flt ( hypre_Box *box1, hypre_Box *box2 );
HYPRE_Int hypre_CopyBox_dbl ( hypre_Box *box1, hypre_Box *box2 );
HYPRE_Int hypre_CopyBox_long_dbl ( hypre_Box *box1, hypre_Box *box2 );
HYPRE_Int hypre_CopyIndex_flt ( hypre_Index in_index, hypre_Index out_index );
HYPRE_Int hypre_CopyIndex_dbl ( hypre_Index in_index, hypre_Index out_index );
HYPRE_Int hypre_CopyIndex_long_dbl ( hypre_Index in_index, hypre_Index out_index );
HYPRE_Int hypre_CopyToCleanIndex_flt ( hypre_Index in_index, HYPRE_Int ndim, hypre_Index out_index );
HYPRE_Int hypre_CopyToCleanIndex_dbl ( hypre_Index in_index, HYPRE_Int ndim, hypre_Index out_index );
HYPRE_Int hypre_CopyToCleanIndex_long_dbl ( hypre_Index in_index, HYPRE_Int ndim, hypre_Index out_index );
HYPRE_Int hypre_DeleteBox_flt  ( hypre_BoxArray *box_array, HYPRE_Int index );
HYPRE_Int hypre_DeleteBox_dbl  ( hypre_BoxArray *box_array, HYPRE_Int index );
HYPRE_Int hypre_DeleteBox_long_dbl  ( hypre_BoxArray *box_array, HYPRE_Int index );
HYPRE_Int hypre_DeleteMultipleBoxes_flt  ( hypre_BoxArray *box_array, HYPRE_Int *indices,
                                      HYPRE_Int num );
HYPRE_Int hypre_DeleteMultipleBoxes_dbl  ( hypre_BoxArray *box_array, HYPRE_Int *indices,
                                      HYPRE_Int num );
HYPRE_Int hypre_DeleteMultipleBoxes_long_dbl  ( hypre_BoxArray *box_array, HYPRE_Int *indices,
                                      HYPRE_Int num );
hypre_float hypre_doubleBoxVolume_flt ( hypre_Box *box );
hypre_double hypre_doubleBoxVolume_dbl ( hypre_Box *box );
hypre_long_double hypre_doubleBoxVolume_long_dbl ( hypre_Box *box );
HYPRE_Int hypre_IndexEqual_flt  ( hypre_Index index, HYPRE_Int val, HYPRE_Int ndim );
HYPRE_Int hypre_IndexEqual_dbl  ( hypre_Index index, HYPRE_Int val, HYPRE_Int ndim );
HYPRE_Int hypre_IndexEqual_long_dbl  ( hypre_Index index, HYPRE_Int val, HYPRE_Int ndim );
HYPRE_Int hypre_IndexesEqual_flt  ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim );
HYPRE_Int hypre_IndexesEqual_dbl  ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim );
HYPRE_Int hypre_IndexesEqual_long_dbl  ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim );
HYPRE_Int hypre_IndexInBox_flt  ( hypre_Index index, hypre_Box *box );
HYPRE_Int hypre_IndexInBox_dbl  ( hypre_Index index, hypre_Box *box );
HYPRE_Int hypre_IndexInBox_long_dbl  ( hypre_Index index, hypre_Box *box );
HYPRE_Int hypre_IndexMax_flt ( hypre_Index index, HYPRE_Int ndim );
HYPRE_Int hypre_IndexMax_dbl ( hypre_Index index, HYPRE_Int ndim );
HYPRE_Int hypre_IndexMax_long_dbl ( hypre_Index index, HYPRE_Int ndim );
HYPRE_Int hypre_IndexMin_flt ( hypre_Index index, HYPRE_Int ndim );
HYPRE_Int hypre_IndexMin_dbl ( hypre_Index index, HYPRE_Int ndim );
HYPRE_Int hypre_IndexMin_long_dbl ( hypre_Index index, HYPRE_Int ndim );
HYPRE_Int hypre_IndexPrint_flt  ( FILE *file, HYPRE_Int ndim, hypre_Index index );
HYPRE_Int hypre_IndexPrint_dbl  ( FILE *file, HYPRE_Int ndim, hypre_Index index );
HYPRE_Int hypre_IndexPrint_long_dbl  ( FILE *file, HYPRE_Int ndim, hypre_Index index );
HYPRE_Int hypre_IndexRead_flt  ( FILE *file, HYPRE_Int ndim, hypre_Index index );
HYPRE_Int hypre_IndexRead_dbl  ( FILE *file, HYPRE_Int ndim, hypre_Index index );
HYPRE_Int hypre_IndexRead_long_dbl  ( FILE *file, HYPRE_Int ndim, hypre_Index index );
HYPRE_Int hypre_SetIndex_flt  ( hypre_Index index, HYPRE_Int val );
HYPRE_Int hypre_SetIndex_dbl  ( hypre_Index index, HYPRE_Int val );
HYPRE_Int hypre_SetIndex_long_dbl  ( hypre_Index index, HYPRE_Int val );
HYPRE_Int hypre_SubtractIndexes_flt  ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim,
                                  hypre_Index result );
HYPRE_Int hypre_SubtractIndexes_dbl  ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim,
                                  hypre_Index result );
HYPRE_Int hypre_SubtractIndexes_long_dbl  ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim,
                                  hypre_Index result );
HYPRE_Int hypre_CommInfoCreate_flt  ( hypre_BoxArrayArray *send_boxes, hypre_BoxArrayArray *recv_boxes,
                                 HYPRE_Int **send_procs, HYPRE_Int **recv_procs, HYPRE_Int **send_rboxnums,
                                 HYPRE_Int **recv_rboxnums, hypre_BoxArrayArray *send_rboxes, hypre_BoxArrayArray *recv_rboxes,
                                 HYPRE_Int boxes_match, hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CommInfoCreate_dbl  ( hypre_BoxArrayArray *send_boxes, hypre_BoxArrayArray *recv_boxes,
                                 HYPRE_Int **send_procs, HYPRE_Int **recv_procs, HYPRE_Int **send_rboxnums,
                                 HYPRE_Int **recv_rboxnums, hypre_BoxArrayArray *send_rboxes, hypre_BoxArrayArray *recv_rboxes,
                                 HYPRE_Int boxes_match, hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CommInfoCreate_long_dbl  ( hypre_BoxArrayArray *send_boxes, hypre_BoxArrayArray *recv_boxes,
                                 HYPRE_Int **send_procs, HYPRE_Int **recv_procs, HYPRE_Int **send_rboxnums,
                                 HYPRE_Int **recv_rboxnums, hypre_BoxArrayArray *send_rboxes, hypre_BoxArrayArray *recv_rboxes,
                                 HYPRE_Int boxes_match, hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CommInfoDestroy_flt  ( hypre_CommInfo *comm_info );
HYPRE_Int hypre_CommInfoDestroy_dbl  ( hypre_CommInfo *comm_info );
HYPRE_Int hypre_CommInfoDestroy_long_dbl  ( hypre_CommInfo *comm_info );
HYPRE_Int hypre_CommInfoGetTransforms_flt  ( hypre_CommInfo *comm_info, HYPRE_Int *num_transforms,
                                        hypre_Index **coords, hypre_Index **dirs );
HYPRE_Int hypre_CommInfoGetTransforms_dbl  ( hypre_CommInfo *comm_info, HYPRE_Int *num_transforms,
                                        hypre_Index **coords, hypre_Index **dirs );
HYPRE_Int hypre_CommInfoGetTransforms_long_dbl  ( hypre_CommInfo *comm_info, HYPRE_Int *num_transforms,
                                        hypre_Index **coords, hypre_Index **dirs );
HYPRE_Int hypre_CommInfoProjectRecv_flt  ( hypre_CommInfo *comm_info, hypre_Index index,
                                      hypre_Index stride );
HYPRE_Int hypre_CommInfoProjectRecv_dbl  ( hypre_CommInfo *comm_info, hypre_Index index,
                                      hypre_Index stride );
HYPRE_Int hypre_CommInfoProjectRecv_long_dbl  ( hypre_CommInfo *comm_info, hypre_Index index,
                                      hypre_Index stride );
HYPRE_Int hypre_CommInfoProjectSend_flt  ( hypre_CommInfo *comm_info, hypre_Index index,
                                      hypre_Index stride );
HYPRE_Int hypre_CommInfoProjectSend_dbl  ( hypre_CommInfo *comm_info, hypre_Index index,
                                      hypre_Index stride );
HYPRE_Int hypre_CommInfoProjectSend_long_dbl  ( hypre_CommInfo *comm_info, hypre_Index index,
                                      hypre_Index stride );
HYPRE_Int hypre_CommInfoSetTransforms_flt  ( hypre_CommInfo *comm_info, HYPRE_Int num_transforms,
                                        hypre_Index *coords, hypre_Index *dirs, HYPRE_Int **send_transforms, HYPRE_Int **recv_transforms );
HYPRE_Int hypre_CommInfoSetTransforms_dbl  ( hypre_CommInfo *comm_info, HYPRE_Int num_transforms,
                                        hypre_Index *coords, hypre_Index *dirs, HYPRE_Int **send_transforms, HYPRE_Int **recv_transforms );
HYPRE_Int hypre_CommInfoSetTransforms_long_dbl  ( hypre_CommInfo *comm_info, HYPRE_Int num_transforms,
                                        hypre_Index *coords, hypre_Index *dirs, HYPRE_Int **send_transforms, HYPRE_Int **recv_transforms );
HYPRE_Int hypre_CreateCommInfoFromGrids_flt  ( hypre_StructGrid *from_grid, hypre_StructGrid *to_grid,
                                          hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromGrids_dbl  ( hypre_StructGrid *from_grid, hypre_StructGrid *to_grid,
                                          hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromGrids_long_dbl  ( hypre_StructGrid *from_grid, hypre_StructGrid *to_grid,
                                          hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromNumGhost_flt  ( hypre_StructGrid *grid, HYPRE_Int *num_ghost,
                                             hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromNumGhost_dbl  ( hypre_StructGrid *grid, HYPRE_Int *num_ghost,
                                             hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromNumGhost_long_dbl  ( hypre_StructGrid *grid, HYPRE_Int *num_ghost,
                                             hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromStencil_flt  ( hypre_StructGrid *grid, hypre_StructStencil *stencil,
                                            hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromStencil_dbl  ( hypre_StructGrid *grid, hypre_StructStencil *stencil,
                                            hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromStencil_long_dbl  ( hypre_StructGrid *grid, hypre_StructStencil *stencil,
                                            hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_ComputeInfoCreate_flt  ( hypre_CommInfo *comm_info, hypre_BoxArrayArray *indt_boxes,
                                    hypre_BoxArrayArray *dept_boxes, hypre_ComputeInfo **compute_info_ptr );
HYPRE_Int hypre_ComputeInfoCreate_dbl  ( hypre_CommInfo *comm_info, hypre_BoxArrayArray *indt_boxes,
                                    hypre_BoxArrayArray *dept_boxes, hypre_ComputeInfo **compute_info_ptr );
HYPRE_Int hypre_ComputeInfoCreate_long_dbl  ( hypre_CommInfo *comm_info, hypre_BoxArrayArray *indt_boxes,
                                    hypre_BoxArrayArray *dept_boxes, hypre_ComputeInfo **compute_info_ptr );
HYPRE_Int hypre_ComputeInfoDestroy_flt  ( hypre_ComputeInfo *compute_info );
HYPRE_Int hypre_ComputeInfoDestroy_dbl  ( hypre_ComputeInfo *compute_info );
HYPRE_Int hypre_ComputeInfoDestroy_long_dbl  ( hypre_ComputeInfo *compute_info );
HYPRE_Int hypre_ComputeInfoProjectComp_flt  ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectComp_dbl  ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectComp_long_dbl  ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectRecv_flt  ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectRecv_dbl  ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectRecv_long_dbl  ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectSend_flt  ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectSend_dbl  ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectSend_long_dbl  ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputePkgCreate_flt  ( hypre_ComputeInfo *compute_info, hypre_BoxArray *data_space,
                                   HYPRE_Int num_values, hypre_StructGrid *grid, hypre_ComputePkg **compute_pkg_ptr );
HYPRE_Int hypre_ComputePkgCreate_dbl  ( hypre_ComputeInfo *compute_info, hypre_BoxArray *data_space,
                                   HYPRE_Int num_values, hypre_StructGrid *grid, hypre_ComputePkg **compute_pkg_ptr );
HYPRE_Int hypre_ComputePkgCreate_long_dbl  ( hypre_ComputeInfo *compute_info, hypre_BoxArray *data_space,
                                   HYPRE_Int num_values, hypre_StructGrid *grid, hypre_ComputePkg **compute_pkg_ptr );
HYPRE_Int hypre_ComputePkgDestroy_flt  ( hypre_ComputePkg *compute_pkg );
HYPRE_Int hypre_ComputePkgDestroy_dbl  ( hypre_ComputePkg *compute_pkg );
HYPRE_Int hypre_ComputePkgDestroy_long_dbl  ( hypre_ComputePkg *compute_pkg );
HYPRE_Int hypre_CreateComputeInfo_flt  ( hypre_StructGrid *grid, hypre_StructStencil *stencil,
                                    hypre_ComputeInfo **compute_info_ptr );
HYPRE_Int hypre_CreateComputeInfo_dbl  ( hypre_StructGrid *grid, hypre_StructStencil *stencil,
                                    hypre_ComputeInfo **compute_info_ptr );
HYPRE_Int hypre_CreateComputeInfo_long_dbl  ( hypre_StructGrid *grid, hypre_StructStencil *stencil,
                                    hypre_ComputeInfo **compute_info_ptr );
HYPRE_Int hypre_FinalizeIndtComputations_flt  ( hypre_CommHandle *comm_handle );
HYPRE_Int hypre_FinalizeIndtComputations_dbl  ( hypre_CommHandle *comm_handle );
HYPRE_Int hypre_FinalizeIndtComputations_long_dbl  ( hypre_CommHandle *comm_handle );
HYPRE_Int hypre_InitializeIndtComputations_flt  ( hypre_ComputePkg *compute_pkg, hypre_float *data,
                                             hypre_CommHandle **comm_handle_ptr );
HYPRE_Int hypre_InitializeIndtComputations_dbl  ( hypre_ComputePkg *compute_pkg, hypre_double *data,
                                             hypre_CommHandle **comm_handle_ptr );
HYPRE_Int hypre_InitializeIndtComputations_long_dbl  ( hypre_ComputePkg *compute_pkg, hypre_long_double *data,
                                             hypre_CommHandle **comm_handle_ptr );
HYPRE_Int HYPRE_StructGridAssemble_flt  ( HYPRE_StructGrid grid );
HYPRE_Int HYPRE_StructGridAssemble_dbl  ( HYPRE_StructGrid grid );
HYPRE_Int HYPRE_StructGridAssemble_long_dbl  ( HYPRE_StructGrid grid );
HYPRE_Int HYPRE_StructGridCreate_flt  ( MPI_Comm comm, HYPRE_Int dim, HYPRE_StructGrid *grid );
HYPRE_Int HYPRE_StructGridCreate_dbl  ( MPI_Comm comm, HYPRE_Int dim, HYPRE_StructGrid *grid );
HYPRE_Int HYPRE_StructGridCreate_long_dbl  ( MPI_Comm comm, HYPRE_Int dim, HYPRE_StructGrid *grid );
HYPRE_Int HYPRE_StructGridDestroy_flt  ( HYPRE_StructGrid grid );
HYPRE_Int HYPRE_StructGridDestroy_dbl  ( HYPRE_StructGrid grid );
HYPRE_Int HYPRE_StructGridDestroy_long_dbl  ( HYPRE_StructGrid grid );
HYPRE_Int HYPRE_StructGridSetExtents_flt  ( HYPRE_StructGrid grid, HYPRE_Int *ilower,
                                       HYPRE_Int *iupper );
HYPRE_Int HYPRE_StructGridSetExtents_dbl  ( HYPRE_StructGrid grid, HYPRE_Int *ilower,
                                       HYPRE_Int *iupper );
HYPRE_Int HYPRE_StructGridSetExtents_long_dbl  ( HYPRE_StructGrid grid, HYPRE_Int *ilower,
                                       HYPRE_Int *iupper );
HYPRE_Int HYPRE_StructGridSetNumGhost_flt  ( HYPRE_StructGrid grid, HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructGridSetNumGhost_dbl  ( HYPRE_StructGrid grid, HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructGridSetNumGhost_long_dbl  ( HYPRE_StructGrid grid, HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructGridSetPeriodic_flt  ( HYPRE_StructGrid grid, HYPRE_Int *periodic );
HYPRE_Int HYPRE_StructGridSetPeriodic_dbl  ( HYPRE_StructGrid grid, HYPRE_Int *periodic );
HYPRE_Int HYPRE_StructGridSetPeriodic_long_dbl  ( HYPRE_StructGrid grid, HYPRE_Int *periodic );
HYPRE_Int HYPRE_StructMatrixAddToBoxValues_flt  ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                             HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                             hypre_float *values );
HYPRE_Int HYPRE_StructMatrixAddToBoxValues_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                             HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                             hypre_double *values );
HYPRE_Int HYPRE_StructMatrixAddToBoxValues_long_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                             HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                             hypre_long_double *values );
HYPRE_Int HYPRE_StructMatrixAddToConstantValues_flt  ( HYPRE_StructMatrix matrix,
                                                  HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_float *values );
HYPRE_Int HYPRE_StructMatrixAddToConstantValues_dbl  ( HYPRE_StructMatrix matrix,
                                                  HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_double *values );
HYPRE_Int HYPRE_StructMatrixAddToConstantValues_long_dbl  ( HYPRE_StructMatrix matrix,
                                                  HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_long_double *values );
HYPRE_Int HYPRE_StructMatrixAddToValues_flt  ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                          HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_float *values );
HYPRE_Int HYPRE_StructMatrixAddToValues_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                          HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_double *values );
HYPRE_Int HYPRE_StructMatrixAddToValues_long_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                          HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_long_double *values );
HYPRE_Int HYPRE_StructMatrixAssemble_flt  ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixAssemble_dbl  ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixAssemble_long_dbl  ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixClearBoundary_flt ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixClearBoundary_dbl ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixClearBoundary_long_dbl ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixCreate_flt  ( MPI_Comm comm, HYPRE_StructGrid grid,
                                     HYPRE_StructStencil stencil, HYPRE_StructMatrix *matrix );
HYPRE_Int HYPRE_StructMatrixCreate_dbl  ( MPI_Comm comm, HYPRE_StructGrid grid,
                                     HYPRE_StructStencil stencil, HYPRE_StructMatrix *matrix );
HYPRE_Int HYPRE_StructMatrixCreate_long_dbl  ( MPI_Comm comm, HYPRE_StructGrid grid,
                                     HYPRE_StructStencil stencil, HYPRE_StructMatrix *matrix );
HYPRE_Int HYPRE_StructMatrixDestroy_flt  ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixDestroy_dbl  ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixDestroy_long_dbl  ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixGetBoxValues_flt  ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           hypre_float *values );
HYPRE_Int HYPRE_StructMatrixGetBoxValues_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           hypre_double *values );
HYPRE_Int HYPRE_StructMatrixGetBoxValues_long_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           hypre_long_double *values );
HYPRE_Int HYPRE_StructMatrixGetGrid_flt  ( HYPRE_StructMatrix matrix, HYPRE_StructGrid *grid );
HYPRE_Int HYPRE_StructMatrixGetGrid_dbl  ( HYPRE_StructMatrix matrix, HYPRE_StructGrid *grid );
HYPRE_Int HYPRE_StructMatrixGetGrid_long_dbl  ( HYPRE_StructMatrix matrix, HYPRE_StructGrid *grid );
HYPRE_Int HYPRE_StructMatrixGetValues_flt  ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_float *values );
HYPRE_Int HYPRE_StructMatrixGetValues_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_double *values );
HYPRE_Int HYPRE_StructMatrixGetValues_long_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_long_double *values );
HYPRE_Int HYPRE_StructMatrixInitialize_flt  ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixInitialize_dbl  ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixInitialize_long_dbl  ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixMatvec_flt  ( hypre_float alpha, HYPRE_StructMatrix A,
                                     HYPRE_StructVector x, hypre_float beta, HYPRE_StructVector y );
HYPRE_Int HYPRE_StructMatrixMatvec_dbl  ( hypre_double alpha, HYPRE_StructMatrix A,
                                     HYPRE_StructVector x, hypre_double beta, HYPRE_StructVector y );
HYPRE_Int HYPRE_StructMatrixMatvec_long_dbl  ( hypre_long_double alpha, HYPRE_StructMatrix A,
                                     HYPRE_StructVector x, hypre_long_double beta, HYPRE_StructVector y );
HYPRE_Int HYPRE_StructMatrixPrint_flt  ( const char *filename, HYPRE_StructMatrix matrix,
                                    HYPRE_Int all );
HYPRE_Int HYPRE_StructMatrixPrint_dbl  ( const char *filename, HYPRE_StructMatrix matrix,
                                    HYPRE_Int all );
HYPRE_Int HYPRE_StructMatrixPrint_long_dbl  ( const char *filename, HYPRE_StructMatrix matrix,
                                    HYPRE_Int all );
HYPRE_Int HYPRE_StructMatrixSetBoxValues_flt  ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           hypre_float *values );
HYPRE_Int HYPRE_StructMatrixSetBoxValues_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           hypre_double *values );
HYPRE_Int HYPRE_StructMatrixSetBoxValues_long_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           hypre_long_double *values );
HYPRE_Int HYPRE_StructMatrixSetConstantEntries_flt  ( HYPRE_StructMatrix matrix, HYPRE_Int nentries,
                                                 HYPRE_Int *entries );
HYPRE_Int HYPRE_StructMatrixSetConstantEntries_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int nentries,
                                                 HYPRE_Int *entries );
HYPRE_Int HYPRE_StructMatrixSetConstantEntries_long_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int nentries,
                                                 HYPRE_Int *entries );
HYPRE_Int HYPRE_StructMatrixSetConstantValues_flt  ( HYPRE_StructMatrix matrix,
                                                HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_float *values );
HYPRE_Int HYPRE_StructMatrixSetConstantValues_dbl  ( HYPRE_StructMatrix matrix,
                                                HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_double *values );
HYPRE_Int HYPRE_StructMatrixSetConstantValues_long_dbl  ( HYPRE_StructMatrix matrix,
                                                HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_long_double *values );
HYPRE_Int HYPRE_StructMatrixSetNumGhost_flt  ( HYPRE_StructMatrix matrix, HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructMatrixSetNumGhost_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructMatrixSetNumGhost_long_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructMatrixSetSymmetric_flt  ( HYPRE_StructMatrix matrix, HYPRE_Int symmetric );
HYPRE_Int HYPRE_StructMatrixSetSymmetric_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int symmetric );
HYPRE_Int HYPRE_StructMatrixSetSymmetric_long_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int symmetric );
HYPRE_Int HYPRE_StructMatrixSetValues_flt  ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_float *values );
HYPRE_Int HYPRE_StructMatrixSetValues_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_double *values );
HYPRE_Int HYPRE_StructMatrixSetValues_long_dbl  ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_long_double *values );
HYPRE_Int HYPRE_StructStencilCreate_flt  ( HYPRE_Int dim, HYPRE_Int size, HYPRE_StructStencil *stencil );
HYPRE_Int HYPRE_StructStencilCreate_dbl  ( HYPRE_Int dim, HYPRE_Int size, HYPRE_StructStencil *stencil );
HYPRE_Int HYPRE_StructStencilCreate_long_dbl  ( HYPRE_Int dim, HYPRE_Int size, HYPRE_StructStencil *stencil );
HYPRE_Int HYPRE_StructStencilDestroy_flt  ( HYPRE_StructStencil stencil );
HYPRE_Int HYPRE_StructStencilDestroy_dbl  ( HYPRE_StructStencil stencil );
HYPRE_Int HYPRE_StructStencilDestroy_long_dbl  ( HYPRE_StructStencil stencil );
HYPRE_Int HYPRE_StructStencilSetElement_flt  ( HYPRE_StructStencil stencil, HYPRE_Int element_index,
                                          HYPRE_Int *offset );
HYPRE_Int HYPRE_StructStencilSetElement_dbl  ( HYPRE_StructStencil stencil, HYPRE_Int element_index,
                                          HYPRE_Int *offset );
HYPRE_Int HYPRE_StructStencilSetElement_long_dbl  ( HYPRE_StructStencil stencil, HYPRE_Int element_index,
                                          HYPRE_Int *offset );
HYPRE_Int HYPRE_CommPkgDestroy_flt  ( HYPRE_CommPkg comm_pkg );
HYPRE_Int HYPRE_CommPkgDestroy_dbl  ( HYPRE_CommPkg comm_pkg );
HYPRE_Int HYPRE_CommPkgDestroy_long_dbl  ( HYPRE_CommPkg comm_pkg );
HYPRE_Int HYPRE_StructVectorAddToBoxValues_flt  ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                             HYPRE_Int *iupper, hypre_float *values );
HYPRE_Int HYPRE_StructVectorAddToBoxValues_dbl  ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                             HYPRE_Int *iupper, hypre_double *values );
HYPRE_Int HYPRE_StructVectorAddToBoxValues_long_dbl  ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                             HYPRE_Int *iupper, hypre_long_double *values );
HYPRE_Int HYPRE_StructVectorAddToValues_flt  ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                          hypre_float values );
HYPRE_Int HYPRE_StructVectorAddToValues_dbl  ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                          hypre_double values );
HYPRE_Int HYPRE_StructVectorAddToValues_long_dbl  ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                          hypre_long_double values );
HYPRE_Int HYPRE_StructVectorAssemble_flt  ( HYPRE_StructVector vector );
HYPRE_Int HYPRE_StructVectorAssemble_dbl  ( HYPRE_StructVector vector );
HYPRE_Int HYPRE_StructVectorAssemble_long_dbl  ( HYPRE_StructVector vector );
HYPRE_Int HYPRE_StructVectorCopy_flt  ( HYPRE_StructVector x, HYPRE_StructVector y );
HYPRE_Int HYPRE_StructVectorCopy_dbl  ( HYPRE_StructVector x, HYPRE_StructVector y );
HYPRE_Int HYPRE_StructVectorCopy_long_dbl  ( HYPRE_StructVector x, HYPRE_StructVector y );
HYPRE_Int HYPRE_StructVectorCreate_flt  ( MPI_Comm comm, HYPRE_StructGrid grid,
                                     HYPRE_StructVector *vector );
HYPRE_Int HYPRE_StructVectorCreate_dbl  ( MPI_Comm comm, HYPRE_StructGrid grid,
                                     HYPRE_StructVector *vector );
HYPRE_Int HYPRE_StructVectorCreate_long_dbl  ( MPI_Comm comm, HYPRE_StructGrid grid,
                                     HYPRE_StructVector *vector );
HYPRE_Int HYPRE_StructVectorDestroy_flt  ( HYPRE_StructVector struct_vector );
HYPRE_Int HYPRE_StructVectorDestroy_dbl  ( HYPRE_StructVector struct_vector );
HYPRE_Int HYPRE_StructVectorDestroy_long_dbl  ( HYPRE_StructVector struct_vector );
HYPRE_Int HYPRE_StructVectorGetBoxValues_flt  ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, hypre_float *values );
HYPRE_Int HYPRE_StructVectorGetBoxValues_dbl  ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, hypre_double *values );
HYPRE_Int HYPRE_StructVectorGetBoxValues_long_dbl  ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, hypre_long_double *values );
HYPRE_Int HYPRE_StructVectorGetMigrateCommPkg_flt  ( HYPRE_StructVector from_vector,
                                                HYPRE_StructVector to_vector, HYPRE_CommPkg *comm_pkg );
HYPRE_Int HYPRE_StructVectorGetMigrateCommPkg_dbl  ( HYPRE_StructVector from_vector,
                                                HYPRE_StructVector to_vector, HYPRE_CommPkg *comm_pkg );
HYPRE_Int HYPRE_StructVectorGetMigrateCommPkg_long_dbl  ( HYPRE_StructVector from_vector,
                                                HYPRE_StructVector to_vector, HYPRE_CommPkg *comm_pkg );
HYPRE_Int HYPRE_StructVectorGetValues_flt  ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                        hypre_float *values );
HYPRE_Int HYPRE_StructVectorGetValues_dbl  ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                        hypre_double *values );
HYPRE_Int HYPRE_StructVectorGetValues_long_dbl  ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                        hypre_long_double *values );
HYPRE_Int HYPRE_StructVectorInitialize_flt  ( HYPRE_StructVector vector );
HYPRE_Int HYPRE_StructVectorInitialize_dbl  ( HYPRE_StructVector vector );
HYPRE_Int HYPRE_StructVectorInitialize_long_dbl  ( HYPRE_StructVector vector );
HYPRE_Int HYPRE_StructVectorMigrate_flt  ( HYPRE_CommPkg comm_pkg, HYPRE_StructVector from_vector,
                                      HYPRE_StructVector to_vector );
HYPRE_Int HYPRE_StructVectorMigrate_dbl  ( HYPRE_CommPkg comm_pkg, HYPRE_StructVector from_vector,
                                      HYPRE_StructVector to_vector );
HYPRE_Int HYPRE_StructVectorMigrate_long_dbl  ( HYPRE_CommPkg comm_pkg, HYPRE_StructVector from_vector,
                                      HYPRE_StructVector to_vector );
HYPRE_Int HYPRE_StructVectorPrint_flt  ( const char *filename, HYPRE_StructVector vector,
                                    HYPRE_Int all );
HYPRE_Int HYPRE_StructVectorPrint_dbl  ( const char *filename, HYPRE_StructVector vector,
                                    HYPRE_Int all );
HYPRE_Int HYPRE_StructVectorPrint_long_dbl  ( const char *filename, HYPRE_StructVector vector,
                                    HYPRE_Int all );
HYPRE_Int HYPRE_StructVectorRead_flt  ( MPI_Comm comm, const char *filename,
                                   HYPRE_Int *num_ghost, HYPRE_StructVector *vector );
HYPRE_Int HYPRE_StructVectorRead_dbl  ( MPI_Comm comm, const char *filename,
                                   HYPRE_Int *num_ghost, HYPRE_StructVector *vector );
HYPRE_Int HYPRE_StructVectorRead_long_dbl  ( MPI_Comm comm, const char *filename,
                                   HYPRE_Int *num_ghost, HYPRE_StructVector *vector );
HYPRE_Int HYPRE_StructVectorScaleValues_flt  ( HYPRE_StructVector vector, hypre_float factor );
HYPRE_Int HYPRE_StructVectorScaleValues_dbl  ( HYPRE_StructVector vector, hypre_double factor );
HYPRE_Int HYPRE_StructVectorScaleValues_long_dbl  ( HYPRE_StructVector vector, hypre_long_double factor );
HYPRE_Int HYPRE_StructVectorSetBoxValues_flt  ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, hypre_float *values );
HYPRE_Int HYPRE_StructVectorSetBoxValues_dbl  ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, hypre_double *values );
HYPRE_Int HYPRE_StructVectorSetBoxValues_long_dbl  ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, hypre_long_double *values );
HYPRE_Int HYPRE_StructVectorSetConstantValues_flt  ( HYPRE_StructVector vector, hypre_float values );
HYPRE_Int HYPRE_StructVectorSetConstantValues_dbl  ( HYPRE_StructVector vector, hypre_double values );
HYPRE_Int HYPRE_StructVectorSetConstantValues_long_dbl  ( HYPRE_StructVector vector, hypre_long_double values );
HYPRE_Int HYPRE_StructVectorSetNumGhost_flt  ( HYPRE_StructVector vector, HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructVectorSetNumGhost_dbl  ( HYPRE_StructVector vector, HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructVectorSetNumGhost_long_dbl  ( HYPRE_StructVector vector, HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructVectorSetValues_flt  ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                        hypre_float values );
HYPRE_Int HYPRE_StructVectorSetValues_dbl  ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                        hypre_double values );
HYPRE_Int HYPRE_StructVectorSetValues_long_dbl  ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                        hypre_long_double values );
HYPRE_Int hypre_ProjectBox_flt  ( hypre_Box *box, hypre_Index index, hypre_Index stride );
HYPRE_Int hypre_ProjectBox_dbl  ( hypre_Box *box, hypre_Index index, hypre_Index stride );
HYPRE_Int hypre_ProjectBox_long_dbl  ( hypre_Box *box, hypre_Index index, hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArray_flt  ( hypre_BoxArray *box_array, hypre_Index index,
                                  hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArray_dbl  ( hypre_BoxArray *box_array, hypre_Index index,
                                  hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArray_long_dbl  ( hypre_BoxArray *box_array, hypre_Index index,
                                  hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArrayArray_flt  ( hypre_BoxArrayArray *box_array_array, hypre_Index index,
                                       hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArrayArray_dbl  ( hypre_BoxArrayArray *box_array_array, hypre_Index index,
                                       hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArrayArray_long_dbl  ( hypre_BoxArrayArray *box_array_array, hypre_Index index,
                                       hypre_Index stride );
HYPRE_Int hypre_StructAxpy_flt  ( hypre_float alpha, hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_StructAxpy_dbl  ( hypre_double alpha, hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_StructAxpy_long_dbl  ( hypre_long_double alpha, hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_CommPkgCreate_flt  ( hypre_CommInfo *comm_info, hypre_BoxArray *send_data_space,
                                hypre_BoxArray *recv_data_space, HYPRE_Int num_values, HYPRE_Int **orders, HYPRE_Int reverse,
                                MPI_Comm comm, hypre_CommPkg **comm_pkg_ptr );
HYPRE_Int hypre_CommPkgCreate_dbl  ( hypre_CommInfo *comm_info, hypre_BoxArray *send_data_space,
                                hypre_BoxArray *recv_data_space, HYPRE_Int num_values, HYPRE_Int **orders, HYPRE_Int reverse,
                                MPI_Comm comm, hypre_CommPkg **comm_pkg_ptr );
HYPRE_Int hypre_CommPkgCreate_long_dbl  ( hypre_CommInfo *comm_info, hypre_BoxArray *send_data_space,
                                hypre_BoxArray *recv_data_space, HYPRE_Int num_values, HYPRE_Int **orders, HYPRE_Int reverse,
                                MPI_Comm comm, hypre_CommPkg **comm_pkg_ptr );
HYPRE_Int hypre_CommPkgDestroy_flt  ( hypre_CommPkg *comm_pkg );
HYPRE_Int hypre_CommPkgDestroy_dbl  ( hypre_CommPkg *comm_pkg );
HYPRE_Int hypre_CommPkgDestroy_long_dbl  ( hypre_CommPkg *comm_pkg );
HYPRE_Int hypre_CommTypeSetEntries_flt  ( hypre_CommType *comm_type, HYPRE_Int *boxnums,
                                     hypre_Box *boxes, hypre_Index stride, hypre_Index coord, hypre_Index dir, HYPRE_Int *order,
                                     hypre_BoxArray *data_space, HYPRE_Int *data_offsets );
HYPRE_Int hypre_CommTypeSetEntries_dbl  ( hypre_CommType *comm_type, HYPRE_Int *boxnums,
                                     hypre_Box *boxes, hypre_Index stride, hypre_Index coord, hypre_Index dir, HYPRE_Int *order,
                                     hypre_BoxArray *data_space, HYPRE_Int *data_offsets );
HYPRE_Int hypre_CommTypeSetEntries_long_dbl  ( hypre_CommType *comm_type, HYPRE_Int *boxnums,
                                     hypre_Box *boxes, hypre_Index stride, hypre_Index coord, hypre_Index dir, HYPRE_Int *order,
                                     hypre_BoxArray *data_space, HYPRE_Int *data_offsets );
HYPRE_Int hypre_CommTypeSetEntry_flt  ( hypre_Box *box, hypre_Index stride, hypre_Index coord,
                                   hypre_Index dir, HYPRE_Int *order, hypre_Box *data_box, HYPRE_Int data_box_offset,
                                   hypre_CommEntryType *comm_entry );
HYPRE_Int hypre_CommTypeSetEntry_dbl  ( hypre_Box *box, hypre_Index stride, hypre_Index coord,
                                   hypre_Index dir, HYPRE_Int *order, hypre_Box *data_box, HYPRE_Int data_box_offset,
                                   hypre_CommEntryType *comm_entry );
HYPRE_Int hypre_CommTypeSetEntry_long_dbl  ( hypre_Box *box, hypre_Index stride, hypre_Index coord,
                                   hypre_Index dir, HYPRE_Int *order, hypre_Box *data_box, HYPRE_Int data_box_offset,
                                   hypre_CommEntryType *comm_entry );
HYPRE_Int hypre_ExchangeLocalData_flt  ( hypre_CommPkg *comm_pkg, hypre_float *send_data,
                                    hypre_float *recv_data, HYPRE_Int action );
HYPRE_Int hypre_ExchangeLocalData_dbl  ( hypre_CommPkg *comm_pkg, hypre_double *send_data,
                                    hypre_double *recv_data, HYPRE_Int action );
HYPRE_Int hypre_ExchangeLocalData_long_dbl  ( hypre_CommPkg *comm_pkg, hypre_long_double *send_data,
                                    hypre_long_double *recv_data, HYPRE_Int action );
HYPRE_Int hypre_FinalizeCommunication_flt  ( hypre_CommHandle *comm_handle );
HYPRE_Int hypre_FinalizeCommunication_dbl  ( hypre_CommHandle *comm_handle );
HYPRE_Int hypre_FinalizeCommunication_long_dbl  ( hypre_CommHandle *comm_handle );
HYPRE_Int hypre_InitializeCommunication_flt  ( hypre_CommPkg *comm_pkg, hypre_float *send_data,
                                          hypre_float *recv_data, HYPRE_Int action, HYPRE_Int tag, hypre_CommHandle **comm_handle_ptr );
HYPRE_Int hypre_InitializeCommunication_dbl  ( hypre_CommPkg *comm_pkg, hypre_double *send_data,
                                          hypre_double *recv_data, HYPRE_Int action, HYPRE_Int tag, hypre_CommHandle **comm_handle_ptr );
HYPRE_Int hypre_InitializeCommunication_long_dbl  ( hypre_CommPkg *comm_pkg, hypre_long_double *send_data,
                                          hypre_long_double *recv_data, HYPRE_Int action, HYPRE_Int tag, hypre_CommHandle **comm_handle_ptr );
HYPRE_Int hypre_StructCopy_flt  ( hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_StructCopy_dbl  ( hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_StructCopy_long_dbl  ( hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_StructPartialCopy_flt  ( hypre_StructVector *x, hypre_StructVector *y,
                                    hypre_BoxArrayArray *array_boxes );
HYPRE_Int hypre_StructPartialCopy_dbl  ( hypre_StructVector *x, hypre_StructVector *y,
                                    hypre_BoxArrayArray *array_boxes );
HYPRE_Int hypre_StructPartialCopy_long_dbl  ( hypre_StructVector *x, hypre_StructVector *y,
                                    hypre_BoxArrayArray *array_boxes );
HYPRE_Int hypre_ComputeBoxnums_flt  ( hypre_BoxArray *boxes, HYPRE_Int *procs, HYPRE_Int **boxnums_ptr );
HYPRE_Int hypre_ComputeBoxnums_dbl  ( hypre_BoxArray *boxes, HYPRE_Int *procs, HYPRE_Int **boxnums_ptr );
HYPRE_Int hypre_ComputeBoxnums_long_dbl  ( hypre_BoxArray *boxes, HYPRE_Int *procs, HYPRE_Int **boxnums_ptr );
HYPRE_Int hypre_GatherAllBoxes_flt  ( MPI_Comm comm, hypre_BoxArray *boxes, HYPRE_Int dim,
                                 hypre_BoxArray **all_boxes_ptr, HYPRE_Int **all_procs_ptr, HYPRE_Int *first_local_ptr );
HYPRE_Int hypre_GatherAllBoxes_dbl  ( MPI_Comm comm, hypre_BoxArray *boxes, HYPRE_Int dim,
                                 hypre_BoxArray **all_boxes_ptr, HYPRE_Int **all_procs_ptr, HYPRE_Int *first_local_ptr );
HYPRE_Int hypre_GatherAllBoxes_long_dbl  ( MPI_Comm comm, hypre_BoxArray *boxes, HYPRE_Int dim,
                                 hypre_BoxArray **all_boxes_ptr, HYPRE_Int **all_procs_ptr, HYPRE_Int *first_local_ptr );
HYPRE_Int hypre_StructGridAssemble_flt  ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridAssemble_dbl  ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridAssemble_long_dbl  ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridCreate_flt  ( MPI_Comm comm, HYPRE_Int dim, hypre_StructGrid **grid_ptr );
HYPRE_Int hypre_StructGridCreate_dbl  ( MPI_Comm comm, HYPRE_Int dim, hypre_StructGrid **grid_ptr );
HYPRE_Int hypre_StructGridCreate_long_dbl  ( MPI_Comm comm, HYPRE_Int dim, hypre_StructGrid **grid_ptr );
HYPRE_Int hypre_StructGridDestroy_flt  ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridDestroy_dbl  ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridDestroy_long_dbl  ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridGetMaxBoxSize_flt  ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridGetMaxBoxSize_dbl  ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridGetMaxBoxSize_long_dbl  ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridPrint_flt  ( FILE *file, hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridPrint_dbl  ( FILE *file, hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridPrint_long_dbl  ( FILE *file, hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridRead_flt  ( MPI_Comm comm, FILE *file, hypre_StructGrid **grid_ptr );
HYPRE_Int hypre_StructGridRead_dbl  ( MPI_Comm comm, FILE *file, hypre_StructGrid **grid_ptr );
HYPRE_Int hypre_StructGridRead_long_dbl  ( MPI_Comm comm, FILE *file, hypre_StructGrid **grid_ptr );
HYPRE_Int hypre_StructGridRef_flt  ( hypre_StructGrid *grid, hypre_StructGrid **grid_ref );
HYPRE_Int hypre_StructGridRef_dbl  ( hypre_StructGrid *grid, hypre_StructGrid **grid_ref );
HYPRE_Int hypre_StructGridRef_long_dbl  ( hypre_StructGrid *grid, hypre_StructGrid **grid_ref );
HYPRE_Int hypre_StructGridSetBoundingBox_flt  ( hypre_StructGrid *grid, hypre_Box *new_bb );
HYPRE_Int hypre_StructGridSetBoundingBox_dbl  ( hypre_StructGrid *grid, hypre_Box *new_bb );
HYPRE_Int hypre_StructGridSetBoundingBox_long_dbl  ( hypre_StructGrid *grid, hypre_Box *new_bb );
HYPRE_Int hypre_StructGridSetBoxes_flt  ( hypre_StructGrid *grid, hypre_BoxArray *boxes );
HYPRE_Int hypre_StructGridSetBoxes_dbl  ( hypre_StructGrid *grid, hypre_BoxArray *boxes );
HYPRE_Int hypre_StructGridSetBoxes_long_dbl  ( hypre_StructGrid *grid, hypre_BoxArray *boxes );
HYPRE_Int hypre_StructGridSetBoxManager_flt  ( hypre_StructGrid *grid, hypre_BoxManager *boxman );
HYPRE_Int hypre_StructGridSetBoxManager_dbl  ( hypre_StructGrid *grid, hypre_BoxManager *boxman );
HYPRE_Int hypre_StructGridSetBoxManager_long_dbl  ( hypre_StructGrid *grid, hypre_BoxManager *boxman );
HYPRE_Int hypre_StructGridSetExtents_flt  ( hypre_StructGrid *grid, hypre_Index ilower,
                                       hypre_Index iupper );
HYPRE_Int hypre_StructGridSetExtents_dbl  ( hypre_StructGrid *grid, hypre_Index ilower,
                                       hypre_Index iupper );
HYPRE_Int hypre_StructGridSetExtents_long_dbl  ( hypre_StructGrid *grid, hypre_Index ilower,
                                       hypre_Index iupper );
HYPRE_Int hypre_StructGridSetIDs_flt  ( hypre_StructGrid *grid, HYPRE_Int *ids );
HYPRE_Int hypre_StructGridSetIDs_dbl  ( hypre_StructGrid *grid, HYPRE_Int *ids );
HYPRE_Int hypre_StructGridSetIDs_long_dbl  ( hypre_StructGrid *grid, HYPRE_Int *ids );
HYPRE_Int hypre_StructGridSetMaxDistance_flt  ( hypre_StructGrid *grid, hypre_Index dist );
HYPRE_Int hypre_StructGridSetMaxDistance_dbl  ( hypre_StructGrid *grid, hypre_Index dist );
HYPRE_Int hypre_StructGridSetMaxDistance_long_dbl  ( hypre_StructGrid *grid, hypre_Index dist );
HYPRE_Int hypre_StructGridSetNumGhost_flt  ( hypre_StructGrid *grid, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructGridSetNumGhost_dbl  ( hypre_StructGrid *grid, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructGridSetNumGhost_long_dbl  ( hypre_StructGrid *grid, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructGridSetPeriodic_flt  ( hypre_StructGrid *grid, hypre_Index periodic );
HYPRE_Int hypre_StructGridSetPeriodic_dbl  ( hypre_StructGrid *grid, hypre_Index periodic );
HYPRE_Int hypre_StructGridSetPeriodic_long_dbl  ( hypre_StructGrid *grid, hypre_Index periodic );
hypre_float hypre_StructInnerProd_flt  ( hypre_StructVector *x, hypre_StructVector *y );
hypre_double hypre_StructInnerProd_dbl  ( hypre_StructVector *x, hypre_StructVector *y );
hypre_long_double hypre_StructInnerProd_long_dbl  ( hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_PrintBoxArrayData_flt  ( FILE *file, hypre_BoxArray *box_array,
                                    hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int dim, hypre_float *data );
HYPRE_Int hypre_PrintBoxArrayData_dbl  ( FILE *file, hypre_BoxArray *box_array,
                                    hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int dim, hypre_double *data );
HYPRE_Int hypre_PrintBoxArrayData_long_dbl  ( FILE *file, hypre_BoxArray *box_array,
                                    hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int dim, hypre_long_double *data );
HYPRE_Int hypre_PrintCCBoxArrayData_flt  ( FILE *file, hypre_BoxArray *box_array,
                                      hypre_BoxArray *data_space, HYPRE_Int num_values, hypre_float *data );
HYPRE_Int hypre_PrintCCBoxArrayData_dbl  ( FILE *file, hypre_BoxArray *box_array,
                                      hypre_BoxArray *data_space, HYPRE_Int num_values, hypre_double *data );
HYPRE_Int hypre_PrintCCBoxArrayData_long_dbl  ( FILE *file, hypre_BoxArray *box_array,
                                      hypre_BoxArray *data_space, HYPRE_Int num_values, hypre_long_double *data );
HYPRE_Int hypre_PrintCCVDBoxArrayData_flt  ( FILE *file, hypre_BoxArray *box_array,
                                        hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int center_rank, HYPRE_Int stencil_size,
                                        HYPRE_Int *symm_elements, HYPRE_Int dim, hypre_float *data );
HYPRE_Int hypre_PrintCCVDBoxArrayData_dbl  ( FILE *file, hypre_BoxArray *box_array,
                                        hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int center_rank, HYPRE_Int stencil_size,
                                        HYPRE_Int *symm_elements, HYPRE_Int dim, hypre_double *data );
HYPRE_Int hypre_PrintCCVDBoxArrayData_long_dbl  ( FILE *file, hypre_BoxArray *box_array,
                                        hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int center_rank, HYPRE_Int stencil_size,
                                        HYPRE_Int *symm_elements, HYPRE_Int dim, hypre_long_double *data );
HYPRE_Int hypre_ReadBoxArrayData_flt  ( FILE *file, hypre_BoxArray *box_array,
                                   hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int dim, hypre_float *data );
HYPRE_Int hypre_ReadBoxArrayData_dbl  ( FILE *file, hypre_BoxArray *box_array,
                                   hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int dim, hypre_double *data );
HYPRE_Int hypre_ReadBoxArrayData_long_dbl  ( FILE *file, hypre_BoxArray *box_array,
                                   hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int dim, hypre_long_double *data );
HYPRE_Int hypre_ReadBoxArrayData_CC_flt  ( FILE *file, hypre_BoxArray *box_array,
                                      hypre_BoxArray *data_space, HYPRE_Int stencil_size, HYPRE_Int real_stencil_size,
                                      HYPRE_Int constant_coefficient, HYPRE_Int dim, hypre_float *data );
HYPRE_Int hypre_ReadBoxArrayData_CC_dbl  ( FILE *file, hypre_BoxArray *box_array,
                                      hypre_BoxArray *data_space, HYPRE_Int stencil_size, HYPRE_Int real_stencil_size,
                                      HYPRE_Int constant_coefficient, HYPRE_Int dim, hypre_double *data );
HYPRE_Int hypre_ReadBoxArrayData_CC_long_dbl  ( FILE *file, hypre_BoxArray *box_array,
                                      hypre_BoxArray *data_space, HYPRE_Int stencil_size, HYPRE_Int real_stencil_size,
                                      HYPRE_Int constant_coefficient, HYPRE_Int dim, hypre_long_double *data );
hypre_StructMatrix *hypre_StructMatrixCreateMask_flt  ( hypre_StructMatrix *matrix,
                                                   HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices );
hypre_StructMatrix *hypre_StructMatrixCreateMask_dbl  ( hypre_StructMatrix *matrix,
                                                   HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices );
hypre_StructMatrix *hypre_StructMatrixCreateMask_long_dbl  ( hypre_StructMatrix *matrix,
                                                   HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices );
HYPRE_Int hypre_StructMatrixAssemble_flt  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixAssemble_dbl  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixAssemble_long_dbl  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixClearBoundary_flt ( hypre_StructMatrix *matrix);
HYPRE_Int hypre_StructMatrixClearBoundary_dbl ( hypre_StructMatrix *matrix);
HYPRE_Int hypre_StructMatrixClearBoundary_long_dbl ( hypre_StructMatrix *matrix);
HYPRE_Int hypre_StructMatrixClearBoxValues_flt  ( hypre_StructMatrix *matrix, hypre_Box *clear_box,
                                             HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixClearBoxValues_dbl  ( hypre_StructMatrix *matrix, hypre_Box *clear_box,
                                             HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixClearBoxValues_long_dbl  ( hypre_StructMatrix *matrix, hypre_Box *clear_box,
                                             HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixClearGhostValues_flt  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixClearGhostValues_dbl  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixClearGhostValues_long_dbl  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixClearValues_flt  ( hypre_StructMatrix *matrix, hypre_Index grid_index,
                                          HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixClearValues_dbl  ( hypre_StructMatrix *matrix, hypre_Index grid_index,
                                          HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixClearValues_long_dbl  ( hypre_StructMatrix *matrix, hypre_Index grid_index,
                                          HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Int boxnum, HYPRE_Int outside );
hypre_StructMatrix *hypre_StructMatrixCreate_flt  ( MPI_Comm comm, hypre_StructGrid *grid,
                                               hypre_StructStencil *user_stencil );
hypre_StructMatrix *hypre_StructMatrixCreate_dbl  ( MPI_Comm comm, hypre_StructGrid *grid,
                                               hypre_StructStencil *user_stencil );
hypre_StructMatrix *hypre_StructMatrixCreate_long_dbl  ( MPI_Comm comm, hypre_StructGrid *grid,
                                               hypre_StructStencil *user_stencil );
HYPRE_Int hypre_StructMatrixDestroy_flt  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixDestroy_dbl  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixDestroy_long_dbl  ( hypre_StructMatrix *matrix );
hypre_float *hypre_StructMatrixExtractPointerByIndex_flt  ( hypre_StructMatrix *matrix, HYPRE_Int b,
                                                         hypre_Index index );
hypre_double *hypre_StructMatrixExtractPointerByIndex_dbl  ( hypre_StructMatrix *matrix, HYPRE_Int b,
                                                         hypre_Index index );
hypre_long_double *hypre_StructMatrixExtractPointerByIndex_long_dbl  ( hypre_StructMatrix *matrix, HYPRE_Int b,
                                                         hypre_Index index );
HYPRE_Int hypre_StructMatrixInitialize_flt  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixInitialize_dbl  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixInitialize_long_dbl  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixInitializeData_flt  ( hypre_StructMatrix *matrix, hypre_float *data,
                                             hypre_float *data_const);
HYPRE_Int hypre_StructMatrixInitializeData_dbl  ( hypre_StructMatrix *matrix, hypre_double *data,
                                             hypre_double *data_const);
HYPRE_Int hypre_StructMatrixInitializeData_long_dbl  ( hypre_StructMatrix *matrix, hypre_long_double *data,
                                             hypre_long_double *data_const);
HYPRE_Int hypre_StructMatrixInitializeShell_flt  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixInitializeShell_dbl  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixInitializeShell_long_dbl  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixMigrate_flt  ( hypre_StructMatrix *from_matrix,
                                      hypre_StructMatrix *to_matrix );
HYPRE_Int hypre_StructMatrixMigrate_dbl  ( hypre_StructMatrix *from_matrix,
                                      hypre_StructMatrix *to_matrix );
HYPRE_Int hypre_StructMatrixMigrate_long_dbl  ( hypre_StructMatrix *from_matrix,
                                      hypre_StructMatrix *to_matrix );
HYPRE_Int hypre_StructMatrixPrint_flt  ( const char *filename, hypre_StructMatrix *matrix,
                                    HYPRE_Int all );
HYPRE_Int hypre_StructMatrixPrint_dbl  ( const char *filename, hypre_StructMatrix *matrix,
                                    HYPRE_Int all );
HYPRE_Int hypre_StructMatrixPrint_long_dbl  ( const char *filename, hypre_StructMatrix *matrix,
                                    HYPRE_Int all );
HYPRE_Int hypre_StructMatrixPrintData_flt  ( FILE *file, hypre_StructMatrix *matrix, HYPRE_Int all );
HYPRE_Int hypre_StructMatrixPrintData_dbl  ( FILE *file, hypre_StructMatrix *matrix, HYPRE_Int all );
HYPRE_Int hypre_StructMatrixPrintData_long_dbl  ( FILE *file, hypre_StructMatrix *matrix, HYPRE_Int all );
hypre_StructMatrix *hypre_StructMatrixRead_flt  ( MPI_Comm comm, const char *filename,
                                             HYPRE_Int *num_ghost );
hypre_StructMatrix *hypre_StructMatrixRead_dbl  ( MPI_Comm comm, const char *filename,
                                             HYPRE_Int *num_ghost );
hypre_StructMatrix *hypre_StructMatrixRead_long_dbl  ( MPI_Comm comm, const char *filename,
                                             HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructMatrixReadData_flt  ( FILE *file, hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixReadData_dbl  ( FILE *file, hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixReadData_long_dbl  ( FILE *file, hypre_StructMatrix *matrix );
hypre_StructMatrix *hypre_StructMatrixRef_flt  ( hypre_StructMatrix *matrix );
hypre_StructMatrix *hypre_StructMatrixRef_dbl  ( hypre_StructMatrix *matrix );
hypre_StructMatrix *hypre_StructMatrixRef_long_dbl  ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixSetBoxValues_flt  ( hypre_StructMatrix *matrix, hypre_Box *set_box,
                                           hypre_Box *value_box, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           hypre_float *values, HYPRE_Int action, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixSetBoxValues_dbl  ( hypre_StructMatrix *matrix, hypre_Box *set_box,
                                           hypre_Box *value_box, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           hypre_double *values, HYPRE_Int action, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixSetBoxValues_long_dbl  ( hypre_StructMatrix *matrix, hypre_Box *set_box,
                                           hypre_Box *value_box, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           hypre_long_double *values, HYPRE_Int action, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixSetConstantCoefficient_flt  ( hypre_StructMatrix *matrix,
                                                     HYPRE_Int constant_coefficient );
HYPRE_Int hypre_StructMatrixSetConstantCoefficient_dbl  ( hypre_StructMatrix *matrix,
                                                     HYPRE_Int constant_coefficient );
HYPRE_Int hypre_StructMatrixSetConstantCoefficient_long_dbl  ( hypre_StructMatrix *matrix,
                                                     HYPRE_Int constant_coefficient );
HYPRE_Int hypre_StructMatrixSetConstantEntries_flt  ( hypre_StructMatrix *matrix, HYPRE_Int nentries,
                                                 HYPRE_Int *entries );
HYPRE_Int hypre_StructMatrixSetConstantEntries_dbl  ( hypre_StructMatrix *matrix, HYPRE_Int nentries,
                                                 HYPRE_Int *entries );
HYPRE_Int hypre_StructMatrixSetConstantEntries_long_dbl  ( hypre_StructMatrix *matrix, HYPRE_Int nentries,
                                                 HYPRE_Int *entries );
HYPRE_Int hypre_StructMatrixSetConstantValues_flt  ( hypre_StructMatrix *matrix,
                                                HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_float *values,
                                                HYPRE_Int action );
HYPRE_Int hypre_StructMatrixSetConstantValues_dbl  ( hypre_StructMatrix *matrix,
                                                HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_double *values,
                                                HYPRE_Int action );
HYPRE_Int hypre_StructMatrixSetConstantValues_long_dbl  ( hypre_StructMatrix *matrix,
                                                HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_long_double *values,
                                                HYPRE_Int action );
HYPRE_Int hypre_StructMatrixSetNumGhost_flt  ( hypre_StructMatrix *matrix, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructMatrixSetNumGhost_dbl  ( hypre_StructMatrix *matrix, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructMatrixSetNumGhost_long_dbl  ( hypre_StructMatrix *matrix, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructMatrixSetValues_flt  ( hypre_StructMatrix *matrix, hypre_Index grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_float *values, HYPRE_Int action,
                                        HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixSetValues_dbl  ( hypre_StructMatrix *matrix, hypre_Index grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_double *values, HYPRE_Int action,
                                        HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixSetValues_long_dbl  ( hypre_StructMatrix *matrix, hypre_Index grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, hypre_long_double *values, HYPRE_Int action,
                                        HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatvec_flt  ( hypre_float alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                               hypre_float beta, hypre_StructVector *y );
HYPRE_Int hypre_StructMatvec_dbl  ( hypre_double alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                               hypre_double beta, hypre_StructVector *y );
HYPRE_Int hypre_StructMatvec_long_dbl  ( hypre_long_double alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                               hypre_long_double beta, hypre_StructVector *y );
HYPRE_Int hypre_StructMatvecCC0_flt  ( hypre_float alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC0_dbl  ( hypre_double alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC0_long_dbl  ( hypre_long_double alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC1_flt  ( hypre_float alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC1_dbl  ( hypre_double alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC1_long_dbl  ( hypre_long_double alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC2_flt  ( hypre_float alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC2_dbl  ( hypre_double alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC2_long_dbl  ( hypre_long_double alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCompute_flt  ( void *matvec_vdata, hypre_float alpha,
                                      hypre_StructMatrix *A, hypre_StructVector *x, hypre_float beta, hypre_StructVector *y );
HYPRE_Int hypre_StructMatvecCompute_dbl  ( void *matvec_vdata, hypre_double alpha,
                                      hypre_StructMatrix *A, hypre_StructVector *x, hypre_double beta, hypre_StructVector *y );
HYPRE_Int hypre_StructMatvecCompute_long_dbl  ( void *matvec_vdata, hypre_long_double alpha,
                                      hypre_StructMatrix *A, hypre_StructVector *x, hypre_long_double beta, hypre_StructVector *y );
void *hypre_StructMatvecCreate_flt  ( void );
void *hypre_StructMatvecCreate_dbl  ( void );
void *hypre_StructMatvecCreate_long_dbl  ( void );
HYPRE_Int hypre_StructMatvecDestroy_flt  ( void *matvec_vdata );
HYPRE_Int hypre_StructMatvecDestroy_dbl  ( void *matvec_vdata );
HYPRE_Int hypre_StructMatvecDestroy_long_dbl  ( void *matvec_vdata );
HYPRE_Int hypre_StructMatvecSetup_flt  ( void *matvec_vdata, hypre_StructMatrix *A,
                                    hypre_StructVector *x );
HYPRE_Int hypre_StructMatvecSetup_dbl  ( void *matvec_vdata, hypre_StructMatrix *A,
                                    hypre_StructVector *x );
HYPRE_Int hypre_StructMatvecSetup_long_dbl  ( void *matvec_vdata, hypre_StructMatrix *A,
                                    hypre_StructVector *x );
HYPRE_Int hypre_StructScale_flt  ( hypre_float alpha, hypre_StructVector *y );
HYPRE_Int hypre_StructScale_dbl  ( hypre_double alpha, hypre_StructVector *y );
HYPRE_Int hypre_StructScale_long_dbl  ( hypre_long_double alpha, hypre_StructVector *y );
hypre_StructStencil *hypre_StructStencilCreate_flt  ( HYPRE_Int dim, HYPRE_Int size,
                                                 hypre_Index *shape );
hypre_StructStencil *hypre_StructStencilCreate_dbl  ( HYPRE_Int dim, HYPRE_Int size,
                                                 hypre_Index *shape );
hypre_StructStencil *hypre_StructStencilCreate_long_dbl  ( HYPRE_Int dim, HYPRE_Int size,
                                                 hypre_Index *shape );
HYPRE_Int hypre_StructStencilDestroy_flt  ( hypre_StructStencil *stencil );
HYPRE_Int hypre_StructStencilDestroy_dbl  ( hypre_StructStencil *stencil );
HYPRE_Int hypre_StructStencilDestroy_long_dbl  ( hypre_StructStencil *stencil );
HYPRE_Int hypre_StructStencilElementRank_flt  ( hypre_StructStencil *stencil,
                                           hypre_Index stencil_element );
HYPRE_Int hypre_StructStencilElementRank_dbl  ( hypre_StructStencil *stencil,
                                           hypre_Index stencil_element );
HYPRE_Int hypre_StructStencilElementRank_long_dbl  ( hypre_StructStencil *stencil,
                                           hypre_Index stencil_element );
hypre_StructStencil *hypre_StructStencilRef_flt  ( hypre_StructStencil *stencil );
hypre_StructStencil *hypre_StructStencilRef_dbl  ( hypre_StructStencil *stencil );
hypre_StructStencil *hypre_StructStencilRef_long_dbl  ( hypre_StructStencil *stencil );
HYPRE_Int hypre_StructStencilSymmetrize_flt  ( hypre_StructStencil *stencil,
                                          hypre_StructStencil **symm_stencil_ptr, HYPRE_Int **symm_elements_ptr );
HYPRE_Int hypre_StructStencilSymmetrize_dbl  ( hypre_StructStencil *stencil,
                                          hypre_StructStencil **symm_stencil_ptr, HYPRE_Int **symm_elements_ptr );
HYPRE_Int hypre_StructStencilSymmetrize_long_dbl  ( hypre_StructStencil *stencil,
                                          hypre_StructStencil **symm_stencil_ptr, HYPRE_Int **symm_elements_ptr );
HYPRE_Int hypre_StructVectorAssemble_flt  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorAssemble_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorAssemble_long_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorClearAllValues_flt  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorClearAllValues_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorClearAllValues_long_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorClearBoundGhostValues_flt  ( hypre_StructVector *vector, HYPRE_Int force );
HYPRE_Int hypre_StructVectorClearBoundGhostValues_dbl  ( hypre_StructVector *vector, HYPRE_Int force );
HYPRE_Int hypre_StructVectorClearBoundGhostValues_long_dbl  ( hypre_StructVector *vector, HYPRE_Int force );
HYPRE_Int hypre_StructVectorClearBoxValues_flt  ( hypre_StructVector *vector, hypre_Box *clear_box,
                                             HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructVectorClearBoxValues_dbl  ( hypre_StructVector *vector, hypre_Box *clear_box,
                                             HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructVectorClearBoxValues_long_dbl  ( hypre_StructVector *vector, hypre_Box *clear_box,
                                             HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructVectorClearGhostValues_flt  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorClearGhostValues_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorClearGhostValues_long_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorClearValues_flt  ( hypre_StructVector *vector, hypre_Index grid_index,
                                          HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructVectorClearValues_dbl  ( hypre_StructVector *vector, hypre_Index grid_index,
                                          HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructVectorClearValues_long_dbl  ( hypre_StructVector *vector, hypre_Index grid_index,
                                          HYPRE_Int boxnum, HYPRE_Int outside );
hypre_StructVector *hypre_StructVectorClone_flt  ( hypre_StructVector *vector );
hypre_StructVector *hypre_StructVectorClone_dbl  ( hypre_StructVector *vector );
hypre_StructVector *hypre_StructVectorClone_long_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorCopy_flt  ( hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_StructVectorCopy_dbl  ( hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_StructVectorCopy_long_dbl  ( hypre_StructVector *x, hypre_StructVector *y );
hypre_StructVector *hypre_StructVectorCreate_flt  ( MPI_Comm comm, hypre_StructGrid *grid );
hypre_StructVector *hypre_StructVectorCreate_dbl  ( MPI_Comm comm, hypre_StructGrid *grid );
hypre_StructVector *hypre_StructVectorCreate_long_dbl  ( MPI_Comm comm, hypre_StructGrid *grid );
HYPRE_Int hypre_StructVectorDestroy_flt  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorDestroy_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorDestroy_long_dbl  ( hypre_StructVector *vector );
hypre_CommPkg *hypre_StructVectorGetMigrateCommPkg_flt  ( hypre_StructVector *from_vector,
                                                     hypre_StructVector *to_vector );
hypre_CommPkg *hypre_StructVectorGetMigrateCommPkg_dbl  ( hypre_StructVector *from_vector,
                                                     hypre_StructVector *to_vector );
hypre_CommPkg *hypre_StructVectorGetMigrateCommPkg_long_dbl  ( hypre_StructVector *from_vector,
                                                     hypre_StructVector *to_vector );
HYPRE_Int hypre_StructVectorInitialize_flt  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorInitialize_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorInitialize_long_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorInitializeData_flt  ( hypre_StructVector *vector, hypre_float *data);
HYPRE_Int hypre_StructVectorInitializeData_dbl  ( hypre_StructVector *vector, hypre_double *data);
HYPRE_Int hypre_StructVectorInitializeData_long_dbl  ( hypre_StructVector *vector, hypre_long_double *data);
HYPRE_Int hypre_StructVectorInitializeShell_flt  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorInitializeShell_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorInitializeShell_long_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorMigrate_flt  ( hypre_CommPkg *comm_pkg, hypre_StructVector *from_vector,
                                      hypre_StructVector *to_vector );
HYPRE_Int hypre_StructVectorMigrate_dbl  ( hypre_CommPkg *comm_pkg, hypre_StructVector *from_vector,
                                      hypre_StructVector *to_vector );
HYPRE_Int hypre_StructVectorMigrate_long_dbl  ( hypre_CommPkg *comm_pkg, hypre_StructVector *from_vector,
                                      hypre_StructVector *to_vector );
HYPRE_Int hypre_StructVectorPrint_flt  ( const char *filename, hypre_StructVector *vector,
                                    HYPRE_Int all );
HYPRE_Int hypre_StructVectorPrint_dbl  ( const char *filename, hypre_StructVector *vector,
                                    HYPRE_Int all );
HYPRE_Int hypre_StructVectorPrint_long_dbl  ( const char *filename, hypre_StructVector *vector,
                                    HYPRE_Int all );
HYPRE_Int hypre_StructVectorPrintData_flt  ( FILE *file, hypre_StructVector *vector, HYPRE_Int all );
HYPRE_Int hypre_StructVectorPrintData_dbl  ( FILE *file, hypre_StructVector *vector, HYPRE_Int all );
HYPRE_Int hypre_StructVectorPrintData_long_dbl  ( FILE *file, hypre_StructVector *vector, HYPRE_Int all );
hypre_StructVector *hypre_StructVectorRead_flt  ( MPI_Comm comm, const char *filename,
                                             HYPRE_Int *num_ghost );
hypre_StructVector *hypre_StructVectorRead_dbl  ( MPI_Comm comm, const char *filename,
                                             HYPRE_Int *num_ghost );
hypre_StructVector *hypre_StructVectorRead_long_dbl  ( MPI_Comm comm, const char *filename,
                                             HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructVectorReadData_flt  ( FILE *file, hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorReadData_dbl  ( FILE *file, hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorReadData_long_dbl  ( FILE *file, hypre_StructVector *vector );
hypre_StructVector *hypre_StructVectorRef_flt  ( hypre_StructVector *vector );
hypre_StructVector *hypre_StructVectorRef_dbl  ( hypre_StructVector *vector );
hypre_StructVector *hypre_StructVectorRef_long_dbl  ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorScaleValues_flt  ( hypre_StructVector *vector, hypre_float factor );
HYPRE_Int hypre_StructVectorScaleValues_dbl  ( hypre_StructVector *vector, hypre_double factor );
HYPRE_Int hypre_StructVectorScaleValues_long_dbl  ( hypre_StructVector *vector, hypre_long_double factor );
HYPRE_Int hypre_StructVectorSetBoxValues_flt  ( hypre_StructVector *vector, hypre_Box *set_box,
                                           hypre_Box *value_box, hypre_float *values, HYPRE_Int action, HYPRE_Int boxnum,
                                           HYPRE_Int outside );
HYPRE_Int hypre_StructVectorSetBoxValues_dbl  ( hypre_StructVector *vector, hypre_Box *set_box,
                                           hypre_Box *value_box, hypre_double *values, HYPRE_Int action, HYPRE_Int boxnum,
                                           HYPRE_Int outside );
HYPRE_Int hypre_StructVectorSetBoxValues_long_dbl  ( hypre_StructVector *vector, hypre_Box *set_box,
                                           hypre_Box *value_box, hypre_long_double *values, HYPRE_Int action, HYPRE_Int boxnum,
                                           HYPRE_Int outside );
HYPRE_Int hypre_StructVectorSetConstantValues_flt  ( hypre_StructVector *vector, hypre_float values );
HYPRE_Int hypre_StructVectorSetConstantValues_dbl  ( hypre_StructVector *vector, hypre_double values );
HYPRE_Int hypre_StructVectorSetConstantValues_long_dbl  ( hypre_StructVector *vector, hypre_long_double values );
HYPRE_Int hypre_StructVectorSetDataSize_flt (hypre_StructVector *vector, HYPRE_Int *data_size,
                                        HYPRE_Int *data_host_size);
HYPRE_Int hypre_StructVectorSetDataSize_dbl (hypre_StructVector *vector, HYPRE_Int *data_size,
                                        HYPRE_Int *data_host_size);
HYPRE_Int hypre_StructVectorSetDataSize_long_dbl (hypre_StructVector *vector, HYPRE_Int *data_size,
                                        HYPRE_Int *data_host_size);
HYPRE_Int hypre_StructVectorSetFunctionValues_flt  ( hypre_StructVector *vector,
                                                hypre_float (*fcn )( HYPRE_Int, HYPRE_Int, HYPRE_Int ));
HYPRE_Int hypre_StructVectorSetFunctionValues_dbl  ( hypre_StructVector *vector,
                                                hypre_double (*fcn )( HYPRE_Int, HYPRE_Int, HYPRE_Int ));
HYPRE_Int hypre_StructVectorSetFunctionValues_long_dbl  ( hypre_StructVector *vector,
                                                hypre_long_double (*fcn )( HYPRE_Int, HYPRE_Int, HYPRE_Int ));
HYPRE_Int hypre_StructVectorSetNumGhost_flt  ( hypre_StructVector *vector, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructVectorSetNumGhost_dbl  ( hypre_StructVector *vector, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructVectorSetNumGhost_long_dbl  ( hypre_StructVector *vector, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructVectorSetValues_flt  ( hypre_StructVector *vector, hypre_Index grid_index,
                                        hypre_float *values, HYPRE_Int action, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructVectorSetValues_dbl  ( hypre_StructVector *vector, hypre_Index grid_index,
                                        hypre_double *values, HYPRE_Int action, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructVectorSetValues_long_dbl  ( hypre_StructVector *vector, hypre_Index grid_index,
                                        hypre_long_double *values, HYPRE_Int action, HYPRE_Int boxnum, HYPRE_Int outside );

#endif

#endif
