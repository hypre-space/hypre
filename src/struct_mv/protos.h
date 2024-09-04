/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* assumed_part.c */
HYPRE_Int hypre_APSubdivideRegion ( hypre_Box *region, HYPRE_Int dim, HYPRE_Int level,
                                    hypre_BoxArray *box_array, HYPRE_Int *num_new_boxes );
HYPRE_Int hypre_APFindMyBoxesInRegions ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                         HYPRE_Int **p_count_array, HYPRE_Real **p_vol_array );
HYPRE_Int hypre_APGetAllBoxesInRegions ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                         HYPRE_Int **p_count_array, HYPRE_Real **p_vol_array, MPI_Comm comm );
HYPRE_Int hypre_APShrinkRegions ( hypre_BoxArray *region_array, hypre_BoxArray *my_box_array,
                                  MPI_Comm comm );
HYPRE_Int hypre_APPruneRegions ( hypre_BoxArray *region_array, HYPRE_Int **p_count_array,
                                 HYPRE_Real **p_vol_array );
HYPRE_Int hypre_APRefineRegionsByVol ( hypre_BoxArray *region_array, HYPRE_Real *vol_array,
                                       HYPRE_Int max_regions,
                                       HYPRE_Real gamma, HYPRE_Int dim, HYPRE_Int *return_code, MPI_Comm comm );
HYPRE_Int hypre_StructAssumedPartitionCreate ( HYPRE_Int dim, hypre_Box *bounding_box,
                                               HYPRE_Real global_boxes_size,
                                               HYPRE_Int global_num_boxes, hypre_BoxArray *local_boxes,
                                               HYPRE_Int max_regions, HYPRE_Int max_refinements, HYPRE_Real gamma,
                                               MPI_Comm comm, hypre_StructAssumedPart **p_assumed_partition );
HYPRE_Int hypre_StructAssumedPartitionDestroy ( hypre_StructAssumedPart *assumed_part );
HYPRE_Int hypre_APFillResponseStructAssumedPart ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                                  HYPRE_Int contact_proc,
                                                  void *ro, MPI_Comm comm, void **p_send_response_buf, HYPRE_Int *response_message_size );
HYPRE_Int hypre_StructAssumedPartitionGetRegionsFromProc ( hypre_StructAssumedPart *assumed_part,
                                                           HYPRE_Int proc_id,
                                                           hypre_BoxArray *assumed_regions );
HYPRE_Int hypre_StructAssumedPartitionGetProcsFromBox ( hypre_StructAssumedPart *assumed_part,
                                                        hypre_Box *box,
                                                        HYPRE_Int *num_proc_array, HYPRE_Int *size_alloc_proc_array,
                                                        HYPRE_Int **p_proc_array );
HYPRE_Int hypre_StructAssumedPartitionPrint ( const char *filename, hypre_StructAssumedPart *ap );
HYPRE_Int hypre_StructCoarsenAP ( hypre_StructAssumedPart *ap, hypre_Index origin,
                                  hypre_Index stride,
                                  hypre_StructAssumedPart **new_ap_ptr );

/* box_algebra.c */
HYPRE_Int hypre_BoxSplit ( hypre_Box *box, hypre_Index index, hypre_Box **lbox_ptr,
                           hypre_Box **rbox_ptr );
HYPRE_Int hypre_IntersectBoxes ( hypre_Box *box1, hypre_Box *box2, hypre_Box *ibox );
HYPRE_Int hypre_SubtractBoxes ( hypre_Box *box1, hypre_Box *box2, hypre_BoxArray *box_array );
HYPRE_Int hypre_SubtractBoxArrays ( hypre_BoxArray *box_array1, hypre_BoxArray *box_array2,
                                    hypre_BoxArray *tmp_box_array );
HYPRE_Int hypre_UnionBoxes ( hypre_BoxArray *boxes );
HYPRE_Int hypre_MinUnionBoxes ( hypre_BoxArray *boxes );

/* box_boundary.c */
HYPRE_Int hypre_BoxBoundaryIntersect ( hypre_Box *box, hypre_StructGrid *grid, HYPRE_Int d,
                                       HYPRE_Int dir, hypre_BoxArray *boundary );
HYPRE_Int hypre_BoxBoundaryG ( hypre_Box *box, hypre_StructGrid *g, hypre_BoxArray *boundary );
HYPRE_Int hypre_BoxBoundaryDG ( hypre_Box *box, hypre_StructGrid *g, hypre_BoxArray *boundarym,
                                hypre_BoxArray *boundaryp, HYPRE_Int d );
HYPRE_Int hypre_GeneralBoxBoundaryIntersect ( hypre_Box *box, hypre_StructGrid *grid,
                                              hypre_Index stencil_offset, hypre_BoxArray *boundary );

/* box.c */
HYPRE_Int hypre_SetIndex ( hypre_Index index, HYPRE_Int val );
HYPRE_Int hypre_CopyIndex ( hypre_Index in_index, hypre_Index out_index );
HYPRE_Int hypre_CopyToIndex ( hypre_Index in_index, HYPRE_Int ndim, hypre_Index out_index );
HYPRE_Int hypre_CopyToCleanIndex ( hypre_Index in_index, HYPRE_Int ndim, hypre_Index out_index );
HYPRE_Int hypre_IndexEqual ( hypre_Index index, HYPRE_Int val, HYPRE_Int ndim );
HYPRE_Int hypre_IndexMin ( hypre_Index index, HYPRE_Int ndim );
HYPRE_Int hypre_IndexMax ( hypre_Index index, HYPRE_Int ndim );
HYPRE_Int hypre_AddIndexes ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim,
                             hypre_Index result );
HYPRE_Int hypre_SubtractIndexes ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim,
                                  hypre_Index result );
HYPRE_Int hypre_IndexesEqual ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim );
HYPRE_Int hypre_IndexPrint ( FILE *file, HYPRE_Int ndim, hypre_Index index );
HYPRE_Int hypre_IndexRead ( FILE *file, HYPRE_Int ndim, hypre_Index index );
hypre_Box *hypre_BoxCreate ( HYPRE_Int ndim );
HYPRE_Int hypre_BoxDestroy ( hypre_Box *box );
HYPRE_Int hypre_BoxInit ( hypre_Box *box, HYPRE_Int  ndim );
HYPRE_Int hypre_BoxSetExtents ( hypre_Box *box, hypre_Index imin, hypre_Index imax );
HYPRE_Int hypre_CopyBox ( hypre_Box *box1, hypre_Box *box2 );
hypre_Box *hypre_BoxClone ( hypre_Box *box );
HYPRE_Int hypre_BoxVolume( hypre_Box *box );
HYPRE_Real hypre_doubleBoxVolume ( hypre_Box *box );
HYPRE_Int hypre_BoxStrideVolume ( hypre_Box *box, hypre_Index stride );
HYPRE_Int hypre_BoxPartialVolume ( hypre_Box *box, hypre_Index partial_volume);
HYPRE_Int hypre_BoxNnodes ( hypre_Box *box );
HYPRE_Int hypre_IndexInBox ( hypre_Index index, hypre_Box *box );
HYPRE_Int hypre_BoxMaxSize ( hypre_Box *box );
HYPRE_Int hypre_BoxGetSize ( hypre_Box *box, hypre_Index size );
HYPRE_Int hypre_BoxGetStrideSize ( hypre_Box *box, hypre_Index stride, hypre_Index size );
HYPRE_Int hypre_BoxGetStrideVolume ( hypre_Box *box, hypre_Index stride, HYPRE_Int *volume_ptr );
HYPRE_Int hypre_BoxIndexRank ( hypre_Box *box, hypre_Index index );
HYPRE_Int hypre_BoxRankIndex ( hypre_Box *box, HYPRE_Int rank, hypre_Index index );
HYPRE_Int hypre_BoxOffsetDistance ( hypre_Box *box, hypre_Index index );
HYPRE_Int hypre_BoxShiftPos ( hypre_Box *box, hypre_Index shift );
HYPRE_Int hypre_BoxShiftNeg ( hypre_Box *box, hypre_Index shift );
HYPRE_Int hypre_BoxGrowByIndex ( hypre_Box *box, hypre_Index  index );
HYPRE_Int hypre_BoxGrowByValue ( hypre_Box *box, HYPRE_Int val );
HYPRE_Int hypre_BoxGrowByBox ( hypre_Box *box, hypre_Box *gbox );
HYPRE_Int hypre_BoxGrowByArray ( hypre_Box *box, HYPRE_Int *array );
HYPRE_Int hypre_BoxPrint ( FILE *file, hypre_Box *box );
HYPRE_Int hypre_BoxRead ( FILE *file, HYPRE_Int ndim, hypre_Box **box_ptr );
hypre_BoxArray *hypre_BoxArrayCreate ( HYPRE_Int size, HYPRE_Int ndim );
HYPRE_Int hypre_BoxArrayCreateFromIndices ( HYPRE_Int ndim, HYPRE_Int num_indices_in,
                                            HYPRE_Int **indices_in, HYPRE_Real threshold, hypre_BoxArray **box_array_ptr );
HYPRE_Int hypre_BoxArrayDestroy ( hypre_BoxArray *box_array );
HYPRE_Int hypre_BoxArrayPrintToFile ( FILE *file, hypre_BoxArray *box_array );
HYPRE_Int hypre_BoxArrayReadFromFile( FILE *file, hypre_BoxArray **box_array_ptr );
HYPRE_Int hypre_BoxArrayPrint ( MPI_Comm comm, const char *filename, hypre_BoxArray *box_array );
HYPRE_Int hypre_BoxArraySetSize ( hypre_BoxArray *box_array, HYPRE_Int size );
hypre_BoxArray *hypre_BoxArrayClone ( hypre_BoxArray *box_array );
HYPRE_Int hypre_AppendBox ( hypre_Box *box, hypre_BoxArray *box_array );
HYPRE_Int hypre_DeleteBox ( hypre_BoxArray *box_array, HYPRE_Int index );
HYPRE_Int hypre_DeleteMultipleBoxes ( hypre_BoxArray *box_array, HYPRE_Int *indices,
                                      HYPRE_Int num );
HYPRE_Int hypre_AppendBoxArray ( hypre_BoxArray *box_array_0, hypre_BoxArray *box_array_1 );
HYPRE_Int hypre_BoxArrayVolume( hypre_BoxArray *box_array );
hypre_BoxArrayArray *hypre_BoxArrayArrayCreate ( HYPRE_Int size, HYPRE_Int ndim );
HYPRE_Int hypre_BoxArrayArrayDestroy ( hypre_BoxArrayArray *box_array_array );
hypre_BoxArrayArray *hypre_BoxArrayArrayClone ( hypre_BoxArrayArray *box_array_array );
HYPRE_Int hypre_BoxArrayArrayPrintToFile ( FILE *file, hypre_BoxArrayArray *box_array_array );
HYPRE_Int hypre_BoxArrayArrayPrint ( MPI_Comm comm, const char *filename,
                                     hypre_BoxArrayArray *box_array_array );

/* box_ds.c */
HYPRE_Int hypre_BoxBTNodeCreate ( HYPRE_Int ndim, hypre_BoxBTNode **btnode_ptr );
HYPRE_Int hypre_BoxBTNodeSetIndices ( hypre_BoxBTNode *btnode, HYPRE_Int num_indices,
                                      HYPRE_Int **indices );
HYPRE_Int hypre_BoxBTNodeInitialize ( hypre_BoxBTNode *btnode, HYPRE_Int num_indices,
                                      HYPRE_Int **indices, hypre_Box *box );
HYPRE_Int hypre_BoxBTNodeDestroy ( hypre_BoxBTNode *btnode );
HYPRE_Int hypre_BoxBinTreeCreate ( HYPRE_Int ndim, hypre_BoxBinTree **boxbt_ptr );
HYPRE_Int hypre_BoxBinTreeInitialize ( hypre_BoxBinTree  *boxbt, HYPRE_Int num_indices,
                                       HYPRE_Int **indices, hypre_Box *box );
HYPRE_Int hypre_BoxBinTreeDestroy ( hypre_BoxBinTree *boxbt );
HYPRE_Int hypre_BoxBTStackCreate ( hypre_BoxBTStack  **btstack_ptr );
HYPRE_Int hypre_BoxBTStackInitialize ( HYPRE_Int capacity, hypre_BoxBTStack *btstack );
HYPRE_Int hypre_BoxBTStackDestroy ( hypre_BoxBTStack *btstack );
HYPRE_Int hypre_BoxBTStackInsert ( hypre_BoxBTNode *btnode, hypre_BoxBTStack *btstack );
HYPRE_Int hypre_BoxBTStackDelete ( hypre_BoxBTStack *btstack, hypre_BoxBTNode **btnode_ptr );
HYPRE_Int hypre_BoxBTQueueCreate ( hypre_BoxBTQueue  **btqueue_ptr );
HYPRE_Int hypre_BoxBTQueueInitialize ( HYPRE_Int capacity, hypre_BoxBTQueue *btqueue );
HYPRE_Int hypre_BoxBTQueueDestroy ( hypre_BoxBTQueue *btqueue );
HYPRE_Int hypre_BoxBTQueueInsert ( hypre_BoxBTNode *btnode, hypre_BoxBTQueue *btqueue );
HYPRE_Int hypre_BoxBTQueueDelete ( hypre_BoxBTQueue *btqueue, hypre_BoxBTNode **btnode_ptr );

/* box_manager.c */
HYPRE_Int hypre_BoxManEntryGetInfo ( hypre_BoxManEntry *entry, void **info_ptr );
HYPRE_Int hypre_BoxManEntryGetExtents ( hypre_BoxManEntry *entry, hypre_Index imin,
                                        hypre_Index imax );
HYPRE_Int hypre_BoxManEntryGetStride ( hypre_BoxManEntry *entry, hypre_Index stride );
HYPRE_Int hypre_BoxManEntryCopy ( hypre_BoxManEntry *fromentry, hypre_BoxManEntry *toentry );
HYPRE_Int hypre_BoxManSetAllGlobalKnown ( hypre_BoxManager *manager, HYPRE_Int known );
HYPRE_Int hypre_BoxManGetAllGlobalKnown ( hypre_BoxManager *manager, HYPRE_Int *known );
HYPRE_Int hypre_BoxManSetIsEntriesSort ( hypre_BoxManager *manager, HYPRE_Int is_sort );
HYPRE_Int hypre_BoxManGetIsEntriesSort ( hypre_BoxManager *manager, HYPRE_Int *is_sort );
HYPRE_Int hypre_BoxManGetGlobalIsGatherCalled ( hypre_BoxManager *manager, MPI_Comm comm,
                                                HYPRE_Int *is_gather );
HYPRE_Int hypre_BoxManGetAssumedPartition ( hypre_BoxManager *manager,
                                            hypre_StructAssumedPart **assumed_partition );
HYPRE_Int hypre_BoxManSetAssumedPartition ( hypre_BoxManager *manager,
                                            hypre_StructAssumedPart *assumed_partition );
HYPRE_Int hypre_BoxManSetBoundingBox ( hypre_BoxManager *manager, hypre_Box *bounding_box );
HYPRE_Int hypre_BoxManSetNumGhost ( hypre_BoxManager *manager, HYPRE_Int *num_ghost );
HYPRE_Int hypre_BoxManDeleteMultipleEntriesAndInfo ( hypre_BoxManager *manager, HYPRE_Int *indices,
                                                     HYPRE_Int num );
HYPRE_Int hypre_BoxManCreate ( HYPRE_Int max_nentries, HYPRE_Int info_size, HYPRE_Int dim,
                               hypre_Box *bounding_box, MPI_Comm comm, hypre_BoxManager **manager_ptr );
HYPRE_Int hypre_BoxManIncSize ( hypre_BoxManager *manager, HYPRE_Int inc_size );
HYPRE_Int hypre_BoxManDestroy ( hypre_BoxManager *manager );
HYPRE_Int hypre_BoxManAddEntry ( hypre_BoxManager *manager, hypre_Index imin, hypre_Index imax,
                                 HYPRE_Int proc_id, HYPRE_Int box_id, void *info );
HYPRE_Int hypre_BoxManGetEntry ( hypre_BoxManager *manager, HYPRE_Int proc, HYPRE_Int id,
                                 hypre_BoxManEntry **entry_ptr );
HYPRE_Int hypre_BoxManGetAllEntries ( hypre_BoxManager *manager, HYPRE_Int *num_entries,
                                      hypre_BoxManEntry **entries );
HYPRE_Int hypre_BoxManGetAllEntriesBoxes ( hypre_BoxManager *manager, hypre_BoxArray *boxes );
HYPRE_Int hypre_BoxManGetLocalEntriesBoxes ( hypre_BoxManager *manager, hypre_BoxArray *boxes );
HYPRE_Int hypre_BoxManGetAllEntriesBoxesProc ( hypre_BoxManager *manager, hypre_BoxArray *boxes,
                                               HYPRE_Int **procs_ptr );
HYPRE_Int hypre_BoxManGatherEntries ( hypre_BoxManager *manager, hypre_Index imin,
                                      hypre_Index imax );
HYPRE_Int hypre_BoxManAssemble ( hypre_BoxManager *manager );
HYPRE_Int hypre_BoxManIntersect ( hypre_BoxManager *manager, hypre_Index ilower, hypre_Index iupper,
                                  hypre_BoxManEntry ***entries_ptr, HYPRE_Int *nentries_ptr );
HYPRE_Int hypre_FillResponseBoxManAssemble1 ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                              HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseBoxManAssemble2 ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                              HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                              HYPRE_Int *response_message_size );

/* coarsen.c */
HYPRE_Int hypre_MapToCoarseIndex ( hypre_Index index, hypre_IndexRef origin, hypre_Index stride,
                                   HYPRE_Int ndim );
HYPRE_Int hypre_MapToFineIndex ( hypre_Index index, hypre_IndexRef origin, hypre_Index stride,
                                 HYPRE_Int ndim );
HYPRE_Int hypre_StructMapFineToCoarse ( hypre_Index findex, hypre_Index origin, hypre_Index stride,
                                        hypre_Index cindex );
HYPRE_Int hypre_StructMapCoarseToFine ( hypre_Index cindex, hypre_Index origin, hypre_Index stride,
                                        hypre_Index findex );
HYPRE_Int
hypre_ComputeCoarseOriginStride ( hypre_Index coarse_origin, hypre_Index coarse_stride,
                                  hypre_IndexRef origin, hypre_Index stride, HYPRE_Int ndim );
HYPRE_Int hypre_CoarsenBox ( hypre_Box *box, hypre_IndexRef origin, hypre_Index stride );
HYPRE_Int hypre_CoarsenBoxNeg ( hypre_Box *box, hypre_Box *refbox, hypre_IndexRef origin,
                                hypre_Index stride );
HYPRE_Int hypre_RefineBox ( hypre_Box *box, hypre_IndexRef origin, hypre_Index stride );
HYPRE_Int hypre_CoarsenBoxArray ( hypre_BoxArray *box_array, hypre_IndexRef origin,
                                  hypre_Index stride );
HYPRE_Int hypre_CoarsenBoxArrayArray ( hypre_BoxArrayArray *box_array_array, hypre_IndexRef origin,
                                       hypre_Index stride );
HYPRE_Int hypre_CoarsenBoxArrayArrayNeg ( hypre_BoxArrayArray *boxaa, hypre_BoxArray *refboxa,
                                          hypre_IndexRef origin, hypre_Index stride, hypre_BoxArrayArray **new_boxaa_ptr );
HYPRE_Int hypre_StructCoarsen ( hypre_StructGrid *fgrid, hypre_IndexRef origin, hypre_Index stride,
                                HYPRE_Int prune, hypre_StructGrid **cgrid_ptr );

/* communication_info.c */
hypre_CommStencil *hypre_CommStencilCreate ( HYPRE_Int  ndim );
HYPRE_Int hypre_CommStencilSetEntry ( hypre_CommStencil *comm_stencil, hypre_Index offset );
HYPRE_Int hypre_CommStencilDestroy ( hypre_CommStencil *comm_stencil );
HYPRE_Int hypre_StructStencilPrint( FILE *file, hypre_StructStencil *stencil );
HYPRE_Int hypre_StructStencilRead( FILE *file, HYPRE_Int ndim, hypre_StructStencil **stencil_ptr );
HYPRE_Int hypre_CommStencilCreateNumGhost ( hypre_CommStencil *comm_stencil,
                                            HYPRE_Int **num_ghost_ptr );
HYPRE_Int hypre_CommInfoCreate ( hypre_BoxArrayArray *send_boxes, hypre_BoxArrayArray *recv_boxes,
                                 HYPRE_Int **send_procs, HYPRE_Int **recv_procs, HYPRE_Int **send_rboxnums,
                                 HYPRE_Int **recv_rboxnums, hypre_BoxArrayArray *send_rboxes, hypre_BoxArrayArray *recv_rboxes,
                                 HYPRE_Int boxes_match, hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CommInfoSetTransforms ( hypre_CommInfo *comm_info, HYPRE_Int num_transforms,
                                        hypre_Index *coords, hypre_Index *dirs, HYPRE_Int **send_transforms, HYPRE_Int **recv_transforms );
HYPRE_Int hypre_CommInfoGetTransforms ( hypre_CommInfo *comm_info, HYPRE_Int *num_transforms,
                                        hypre_Index **coords, hypre_Index **dirs );
HYPRE_Int hypre_CommInfoProjectSend ( hypre_CommInfo *comm_info, hypre_Index index,
                                      hypre_Index stride );
HYPRE_Int hypre_CommInfoProjectRecv ( hypre_CommInfo *comm_info, hypre_Index index,
                                      hypre_Index stride );
HYPRE_Int hypre_CommInfoDestroy ( hypre_CommInfo *comm_info );
HYPRE_Int hypre_CommInfoClone( hypre_CommInfo *comm_info, hypre_CommInfo **clone_ptr );
HYPRE_Int hypre_CreateCommInfo ( hypre_StructGrid *grid, hypre_CommStencil *comm_stencil,
                                 hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromStencil ( hypre_StructGrid *grid, hypre_StructStencil *stencil,
                                            hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromNumGhost ( hypre_StructGrid *grid, HYPRE_Int *num_ghost,
                                             hypre_CommInfo **comm_info_ptr );
HYPRE_Int hypre_CreateCommInfoFromGrids ( hypre_StructGrid *from_grid, hypre_StructGrid *to_grid,
                                          hypre_CommInfo **comm_info_ptr );

/* computation.c */
HYPRE_Int hypre_ComputeInfoCreate ( hypre_CommInfo *comm_info, hypre_BoxArrayArray *indt_boxes,
                                    hypre_BoxArrayArray *dept_boxes, hypre_ComputeInfo **compute_info_ptr );
HYPRE_Int hypre_ComputeInfoProjectSend ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectRecv ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputeInfoProjectComp ( hypre_ComputeInfo *compute_info, hypre_Index index,
                                         hypre_Index stride );
HYPRE_Int hypre_ComputeInfoDestroy ( hypre_ComputeInfo *compute_info );
HYPRE_Int hypre_CreateComputeInfo ( hypre_StructGrid *grid, hypre_StructStencil *stencil,
                                    hypre_ComputeInfo **compute_info_ptr );
HYPRE_Int hypre_ComputePkgCreate ( hypre_ComputeInfo *compute_info, hypre_BoxArray *data_space,
                                   HYPRE_Int num_values, hypre_StructGrid *grid, hypre_ComputePkg **compute_pkg_ptr );
HYPRE_Int hypre_ComputePkgDestroy ( hypre_ComputePkg *compute_pkg );
HYPRE_Int hypre_InitializeIndtComputations ( hypre_ComputePkg *compute_pkg, HYPRE_Complex *data,
                                             hypre_CommHandle **comm_handle_ptr );
HYPRE_Int hypre_FinalizeIndtComputations ( hypre_CommHandle *comm_handle );

/* project.c */
HYPRE_Int hypre_SnapIndexPos ( hypre_Index index, hypre_IndexRef origin, hypre_Index stride,
                               HYPRE_Int ndim );
HYPRE_Int hypre_SnapIndexNeg ( hypre_Index index, hypre_IndexRef origin, hypre_Index stride,
                               HYPRE_Int ndim );
HYPRE_Int hypre_ConvertToCanonicalIndex ( hypre_Index index, hypre_Index stride, HYPRE_Int ndim );
HYPRE_Int hypre_ProjectBox ( hypre_Box *box, hypre_IndexRef origin, hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArray ( hypre_BoxArray *box_array, hypre_IndexRef origin,
                                  hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArrayArray ( hypre_BoxArrayArray *box_array_array, hypre_IndexRef origin,
                                       hypre_Index stride );

/* struct_axpy.c */
HYPRE_Int hypre_StructAxpy ( HYPRE_Complex alpha, hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_StructVectorElmdivpy ( HYPRE_Complex alpha, hypre_StructVector *x,
                                       hypre_StructVector *z, HYPRE_Complex beta, hypre_StructVector *y );

/* struct_communication.c */
HYPRE_Int hypre_CommPkgCreate ( hypre_CommInfo *comm_info, hypre_BoxArray *send_data_space,
                                hypre_BoxArray *recv_data_space, HYPRE_Int num_values, HYPRE_Int **orders, HYPRE_Int reverse,
                                MPI_Comm comm, hypre_CommPkg **comm_pkg_ptr );
HYPRE_Int hypre_CommBlockSetEntries ( hypre_CommBlock *comm_block, HYPRE_Int *boxnums,
                                      hypre_Box *boxes, HYPRE_Int *orders, hypre_Index stride, hypre_BoxArray *data_space,
                                      HYPRE_Int *data_offsets );
HYPRE_Int hypre_CommBlockSetEntry ( hypre_CommBlock *comm_block, HYPRE_Int comm_num, hypre_Box *box,
                                    hypre_Index stride, hypre_Index coord, hypre_Index dir, HYPRE_Int *order, HYPRE_Int *rem_order,
                                    hypre_Box *data_box, HYPRE_Int data_box_offset );
HYPRE_Int hypre_CommPkgSetPrefixSizes ( hypre_CommPkg  *comm_pkg );
HYPRE_Int hypre_CommPkgAgglomerate ( HYPRE_Int num_comm_pkgs, hypre_CommPkg **comm_pkgs,
                                     hypre_CommPkg **agg_comm_pkg_ptr );
HYPRE_Int hypre_InitializeCommunication ( hypre_CommPkg *comm_pkg, HYPRE_Complex **send_data,
                                          HYPRE_Complex **recv_data, HYPRE_Int action, HYPRE_Int tag, hypre_CommHandle **comm_handle_ptr );
HYPRE_Int hypre_FinalizeCommunication ( hypre_CommHandle *comm_handle );
HYPRE_Int hypre_ExchangeLocalData ( hypre_CommPkg *comm_pkg, HYPRE_Complex **send_data,
                                    HYPRE_Complex **recv_data, HYPRE_Int action );
HYPRE_Int hypre_CommPkgDestroy ( hypre_CommPkg *comm_pkg );

/* struct_copy.c */
HYPRE_Int hypre_StructCopy ( hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_StructPartialCopy ( hypre_StructVector *x, hypre_StructVector *y,
                                    hypre_BoxArrayArray *array_boxes );

/* struct_data.c */
HYPRE_Int hypre_StructDataCopy ( HYPRE_Complex *fr_data, hypre_BoxArray *fr_data_space,
                                 HYPRE_Int *fr_ids, HYPRE_Complex *to_data, hypre_BoxArray *to_data_space, HYPRE_Int *to_ids,
                                 HYPRE_Int ndim, HYPRE_Int nval );
HYPRE_Int hypre_StructNumGhostFromStencil ( hypre_StructStencil *stencil,
                                            HYPRE_Int **num_ghost_ptr );

/* struct_grid.c */
HYPRE_Int hypre_StructGridCreate ( MPI_Comm comm, HYPRE_Int dim, hypre_StructGrid **grid_ptr );
HYPRE_Int hypre_StructGridRef ( hypre_StructGrid *grid, hypre_StructGrid **grid_ref );
HYPRE_Int hypre_StructGridDestroy ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridSetPeriodic ( hypre_StructGrid *grid, hypre_Index periodic );
HYPRE_Int hypre_StructGridSetExtents ( hypre_StructGrid *grid, hypre_Index ilower,
                                       hypre_Index iupper );
HYPRE_Int hypre_StructGridSetBoxes ( hypre_StructGrid *grid, hypre_BoxArray *boxes );
HYPRE_Int hypre_StructGridSetBoundingBox ( hypre_StructGrid *grid, hypre_Box *new_bb );
HYPRE_Int hypre_StructGridSetIDs ( hypre_StructGrid *grid, HYPRE_Int *ids );
HYPRE_Int hypre_StructGridSetBoxManager ( hypre_StructGrid *grid, hypre_BoxManager *boxman );
HYPRE_Int hypre_StructGridSetMaxDistance ( hypre_StructGrid *grid, hypre_Index dist );
HYPRE_Int hypre_StructGridComputeGlobalSize ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridAssemble ( hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridComputeBoxnums ( hypre_StructGrid *grid, HYPRE_Int nboxes,
                                           HYPRE_Int *boxnums, hypre_Index stride, HYPRE_Int *new_nboxes_ptr, HYPRE_Int **new_boxnums_ptr );
HYPRE_Int hypre_GatherAllBoxes ( MPI_Comm comm, hypre_BoxArray *boxes, HYPRE_Int dim,
                                 hypre_BoxArray **all_boxes_ptr, HYPRE_Int **all_procs_ptr, HYPRE_Int *first_local_ptr );
HYPRE_Int hypre_ComputeBoxnums ( hypre_BoxArray *boxes, HYPRE_Int *procs, HYPRE_Int **boxnums_ptr );
HYPRE_Int hypre_StructGridPrintVTK ( const char *filename, hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridPrint ( FILE *file, hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridRead ( MPI_Comm comm, FILE *file, hypre_StructGrid **grid_ptr );
HYPRE_Int hypre_StructGridSetNumGhost ( hypre_StructGrid *grid, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructGridGetMaxBoxSize ( hypre_StructGrid *grid );
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
HYPRE_Int hypre_StructGridSetDataLocation( HYPRE_StructGrid grid,
                                           HYPRE_MemoryLocation data_location );
#endif
/* struct_innerprod.c */
HYPRE_Real hypre_StructInnerProdLocal ( hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Real hypre_StructInnerProd ( hypre_StructVector *x, hypre_StructVector *y );

/* struct_io.c */
HYPRE_Int hypre_PrintBoxArrayData ( FILE *file, HYPRE_Int dim, hypre_BoxArray *box_array, hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int *value_ids, HYPRE_Complex *data );
HYPRE_Int
hypre_ReadBoxArrayData( FILE *file, HYPRE_Int ndim, hypre_BoxArray *box_array, HYPRE_Int *num_values_ptr, HYPRE_Int **value_ids_ptr, HYPRE_Complex **values_ptr );

/* struct_matmult.c */
HYPRE_Int hypre_StructMatmultCreate ( HYPRE_Int nmatrices_in, hypre_StructMatrix **matrices_in,
                                      HYPRE_Int nterms, HYPRE_Int *terms_in, HYPRE_Int *transposes_in,
                                      hypre_StructMatmultData **mmdata_ptr );
HYPRE_Int hypre_StructMatmultDestroy ( hypre_StructMatmultData *mmdata );
HYPRE_Int hypre_StructMatmultSetup ( hypre_StructMatmultData  *mmdata, hypre_StructMatrix **M_ptr );
HYPRE_Int hypre_StructMatmultCommunicate ( hypre_StructMatmultData *mmdata, hypre_StructMatrix *M );
HYPRE_Int hypre_StructMatmultCompute ( hypre_StructMatmultData *mmdata, hypre_StructMatrix *M );
HYPRE_Int hypre_StructMatmultCompute_core_double ( hypre_StructMatmultHelper *a, HYPRE_Int na,
                                                   HYPRE_Int ndim, hypre_Index loop_size, HYPRE_Int stencil_size, hypre_Box *fdbox,
                                                   hypre_Index fdstart, hypre_Index fdstride, hypre_Box *cdbox, hypre_Index cdstart,
                                                   hypre_Index cdstride, hypre_Box *Mdbox, hypre_Index Mdstart, hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_triple ( hypre_StructMatmultHelper *a, HYPRE_Int na,
                                                   HYPRE_Int ndim, hypre_Index loop_size, HYPRE_Int stencil_size, hypre_Box *fdbox,
                                                   hypre_Index fdstart, hypre_Index fdstride, hypre_Box *cdbox, hypre_Index cdstart,
                                                   hypre_Index cdstride, hypre_Box *Mdbox, hypre_Index Mdstart, hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_generic ( hypre_StructMatmultHelper *a, HYPRE_Int na,
                                                    HYPRE_Int nterms, HYPRE_Int ndim, hypre_Index loop_size, hypre_Box *fdbox, hypre_Index fdstart,
                                                    hypre_Index fdstride, hypre_Box *cdbox, hypre_Index cdstart, hypre_Index cdstride, hypre_Box *Mdbox,
                                                    hypre_Index Mdstart, hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_1d ( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                               HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr, HYPRE_Int ndim,
                                               hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                               hypre_Box *Mdbox, hypre_Index Mdstart, hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_1db ( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                                HYPRE_Int **order, HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr,
                                                HYPRE_Int ndim, hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                                hypre_Box *Mdbox, hypre_Index Mdstart, hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_1dbb( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                                HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr, HYPRE_Int ndim,
                                                hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                                hypre_Box *Mdbox, hypre_Index Mdstart, hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_2d( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                              HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr, HYPRE_Int ndim,
                                              hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                              hypre_Box *hdbox, hypre_Index hdstart, hypre_Index hdstride, hypre_Box *Mdbox, hypre_Index Mdstart,
                                              hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_2db ( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                                HYPRE_Int **order, HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr,
                                                HYPRE_Int ndim, hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                                hypre_Box *hdbox, hypre_Index hdstart, hypre_Index hdstride, hypre_Box *Mdbox, hypre_Index Mdstart,
                                                hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_1t ( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                               HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr, HYPRE_Int ndim,
                                               hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                               hypre_Box *Mdbox, hypre_Index Mdstart, hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_1tb ( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                                HYPRE_Int **order, HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr,
                                                HYPRE_Int ndim, hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                                hypre_Box *Mdbox, hypre_Index Mdstart, hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_1tbb( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                                HYPRE_Int **order, HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr,
                                                HYPRE_Int ndim, hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                                hypre_Box *Mdbox, hypre_Index Mdstart, hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_1tbbb( hypre_StructMatmultHelper *a,
                                                 HYPRE_Int ncomponents, HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr,
                                                 HYPRE_Int ndim, hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                                 hypre_Box *Mdbox, hypre_Index Mdstart, hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_2t( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                              HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr, HYPRE_Int ndim,
                                              hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                              hypre_Box *hdbox, hypre_Index hdstart, hypre_Index hdstride, hypre_Box *Mdbox, hypre_Index Mdstart,
                                              hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_2tb( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                               HYPRE_Int **order, HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr,
                                               HYPRE_Int ndim, hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                               hypre_Box *hdbox, hypre_Index hdstart, hypre_Index hdstride, hypre_Box *Mdbox, hypre_Index Mdstart,
                                               hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_2etb( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                                HYPRE_Int **order, HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr,
                                                HYPRE_Int ndim, hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                                hypre_Box *hdbox, hypre_Index hdstart, hypre_Index hdstride, hypre_Box *Mdbox, hypre_Index Mdstart,
                                                hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmultCompute_core_2tbb( hypre_StructMatmultHelper *a, HYPRE_Int ncomponents,
                                                HYPRE_Int **order, HYPRE_Complex *cprod, const HYPRE_Complex ***tptrs, HYPRE_Complex *mptr,
                                                HYPRE_Int ndim, hypre_Index loop_size, hypre_Box *gdbox, hypre_Index gdstart, hypre_Index gdstride,
                                                hypre_Box *hdbox, hypre_Index hdstart, hypre_Index hdstride, hypre_Box *Mdbox, hypre_Index Mdstart,
                                                hypre_Index Mdstride );
HYPRE_Int hypre_StructMatmult ( HYPRE_Int nmatrices, hypre_StructMatrix **matrices,
                                HYPRE_Int nterms, HYPRE_Int *terms, HYPRE_Int *trans, hypre_StructMatrix **M_ptr );
HYPRE_Int hypre_StructMatmat ( hypre_StructMatrix *A, hypre_StructMatrix *B,
                               hypre_StructMatrix **M_ptr );
HYPRE_Int hypre_StructMatrixPtAP ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                   hypre_StructMatrix **M_ptr );
HYPRE_Int hypre_StructMatrixRAP ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                  hypre_StructMatrix *P, hypre_StructMatrix **M_ptr );
HYPRE_Int hypre_StructMatrixRTtAP ( hypre_StructMatrix *RT, hypre_StructMatrix *A,
                                    hypre_StructMatrix *P, hypre_StructMatrix **M_ptr );

/* struct_matop.c */
HYPRE_Int hypre_StructMatrixComputeRowSum ( hypre_StructMatrix *A, HYPRE_Int type,
                                            hypre_StructVector *rowsum );
HYPRE_Int hypre_StructMatrixComputeRowSum_core_CC ( hypre_StructMatrix *A,
                                                    hypre_StructVector *rowsum, HYPRE_Int box_id, HYPRE_Int nentries, HYPRE_Int *entries,
                                                    hypre_Box *box, hypre_Box *Adbox, hypre_Box *rdbox, HYPRE_Int type );
HYPRE_Int hypre_StructMatrixComputeRowSum_core_VC ( hypre_StructMatrix *A,
                                                    hypre_StructVector *rowsum, HYPRE_Int box_id, HYPRE_Int nentries, HYPRE_Int *entries,
                                                    hypre_Box *box, hypre_Box *Adbox, hypre_Box *rdbox, HYPRE_Int type );

/* struct_matrix.c */
HYPRE_Int hypre_StructMatrixGetDataMapStride ( hypre_StructMatrix *matrix, hypre_IndexRef *stride );
HYPRE_Int hypre_StructMatrixMapDataIndex ( hypre_StructMatrix *matrix, hypre_Index dindex );
HYPRE_Int hypre_StructMatrixMapDataBox ( hypre_StructMatrix *matrix, hypre_Box *dbox );
HYPRE_Int hypre_StructMatrixMapDataStride ( hypre_StructMatrix *matrix, hypre_Index dstride );
HYPRE_Int hypre_StructMatrixUnMapDataIndex ( hypre_StructMatrix *matrix, hypre_Index dindex );
HYPRE_Int hypre_StructMatrixUnMapDataBox ( hypre_StructMatrix *matrix, hypre_Box *dbox );
HYPRE_Int hypre_StructMatrixUnMapDataStride ( hypre_StructMatrix *matrix, hypre_Index dstride );
HYPRE_Int hypre_StructMatrixPlaceStencil ( hypre_StructMatrix *matrix, HYPRE_Int entry,
                                           hypre_Index dindex, hypre_Index index );
HYPRE_Int hypre_StructMatrixGetStencilStride ( hypre_StructMatrix *matrix, hypre_Index stride );
HYPRE_Int hypre_StructMatrixGetStencilSpace ( hypre_StructMatrix *matrix, HYPRE_Int entry,
                                              HYPRE_Int transpose, hypre_Index origin, hypre_Index stride );
HYPRE_Int hypre_StructMatrixMapCommInfo ( hypre_StructMatrix *matrix, hypre_IndexRef origin,
                                          hypre_Index stride, hypre_CommInfo *comm_info );
HYPRE_Int hypre_StructMatrixCreateCommPkg ( hypre_StructMatrix *matrix, hypre_CommInfo *comm_info,
                                            hypre_CommPkg **comm_pkg_ptr, HYPRE_Complex ***comm_data_ptr);
HYPRE_Complex *hypre_StructMatrixExtractPointerByIndex ( hypre_StructMatrix *matrix, HYPRE_Int b,
                                                         hypre_Index index );
hypre_StructMatrix *hypre_StructMatrixCreate ( MPI_Comm comm, hypre_StructGrid *grid,
                                               hypre_StructStencil *user_stencil );
hypre_StructMatrix *hypre_StructMatrixRef ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixDestroy ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixSetRangeStride ( hypre_StructMatrix *matrix,
                                             hypre_IndexRef range_stride );
HYPRE_Int hypre_StructMatrixSetDomainStride ( hypre_StructMatrix *matrix,
                                              hypre_IndexRef domain_stride );
HYPRE_Int hypre_StructMatrixComputeDataSpace ( hypre_StructMatrix *matrix, HYPRE_Int *num_ghost,
                                               hypre_BoxArray **data_space_ptr );
HYPRE_Int hypre_StructMatrixResize ( hypre_StructMatrix *matrix, hypre_BoxArray *data_space );
HYPRE_Int hypre_StructMatrixRestore ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixForget ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixInitializeShell ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixInitializeData ( hypre_StructMatrix *matrix, HYPRE_Complex *data );
HYPRE_Int hypre_StructMatrixInitialize ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixSetValues ( hypre_StructMatrix *matrix, hypre_Index grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Complex *values, HYPRE_Int action,
                                        HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixSetBoxValues ( hypre_StructMatrix *matrix, hypre_Box *set_box,
                                           hypre_Box *value_box, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           HYPRE_Complex *values, HYPRE_Int action, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixSetConstantValues ( hypre_StructMatrix *matrix,
                                                HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Complex *values,
                                                HYPRE_Int action );
HYPRE_Int hypre_StructMatrixClearValues ( hypre_StructMatrix *matrix, hypre_Index grid_index,
                                          HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixClearBoxValues ( hypre_StructMatrix *matrix, hypre_Box *clear_box,
                                             HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructMatrixAssemble ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixSetConstantEntries ( hypre_StructMatrix *matrix, HYPRE_Int nentries,
                                                 HYPRE_Int *entries );
HYPRE_Int hypre_StructMatrixSetTranspose ( hypre_StructMatrix *matrix, HYPRE_Int transpose,
                                           HYPRE_Int *resize );
HYPRE_Int hypre_StructMatrixSetNumGhost ( hypre_StructMatrix *matrix, HYPRE_Int *num_ghost,
                                          HYPRE_Int *resize );
HYPRE_Int hypre_StructMatrixSetGhost ( hypre_StructMatrix *matrix, HYPRE_Int ghost,
                                       HYPRE_Int *resize );
HYPRE_Int hypre_StructMatrixClearGhostValues ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixPrintData ( FILE *file, hypre_StructMatrix *matrix, HYPRE_Int all );
HYPRE_Int hypre_StructMatrixReadData ( FILE *file, hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixPrint ( const char *filename, hypre_StructMatrix *matrix,
                                    HYPRE_Int all );
hypre_StructMatrix *hypre_StructMatrixRead ( MPI_Comm comm, const char *filename,
                                             HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructMatrixMigrate ( hypre_StructMatrix *from_matrix,
                                      hypre_StructMatrix *to_matrix );
HYPRE_Int hypre_StructMatrixClearBoundary ( hypre_StructMatrix *matrix);
HYPRE_Int hypre_StructMatrixGetDiagonal ( hypre_StructMatrix *matrix, hypre_StructVector *diag );

/* struct_matrix_mask.c */
hypre_StructMatrix *hypre_StructMatrixCreateMask ( hypre_StructMatrix *matrix,
                                                   HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices );

/* struct_matvec.c */
void *hypre_StructMatvecCreate ( void );
HYPRE_Int hypre_StructMatvecSetTranspose ( void *matvec_vdata, HYPRE_Int transpose );
HYPRE_Int hypre_StructMatvecSetActive ( void *matvec_vdata, HYPRE_Int active );
HYPRE_Int hypre_StructMatvecSetup ( void *matvec_vdata, hypre_StructMatrix *A,
                                    hypre_StructVector *x );
HYPRE_Int hypre_StructMatvecCompute ( void *matvec_vdata, HYPRE_Complex alpha,
                                      hypre_StructMatrix *A, hypre_StructVector *x, HYPRE_Complex beta, hypre_StructVector *y );
HYPRE_Int hypre_StructMatvecCompute_core_CC ( hypre_StructMatrix *A, hypre_StructVector *x,
                                              hypre_StructVector *y, HYPRE_Int box_id, HYPRE_Int nentries, HYPRE_Int *entries,
                                              hypre_Box *compute_box, hypre_Box *x_data_box, hypre_Box *y_data_box );
HYPRE_Int hypre_StructMatvecCompute_core_VC ( hypre_StructMatrix *A, hypre_StructVector *x,
                                              hypre_StructVector *y, HYPRE_Int box_id, HYPRE_Int nentries, HYPRE_Int *entries,
                                              hypre_Box *compute_box, hypre_Box *A_data_box, hypre_Box *x_data_box, hypre_Box *y_data_box );
HYPRE_Int hypre_StructMatvecDestroy ( void *matvec_vdata );
HYPRE_Int hypre_StructMatvec ( HYPRE_Complex alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                               HYPRE_Complex beta, hypre_StructVector *y );
HYPRE_Int hypre_StructMatvecT ( HYPRE_Complex alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                HYPRE_Complex beta, hypre_StructVector *y );

/* struct_scale.c */
HYPRE_Int hypre_StructScale ( HYPRE_Complex alpha, hypre_StructVector *y );

/* struct_stencil.c */
hypre_StructStencil *hypre_StructStencilCreate ( HYPRE_Int dim, HYPRE_Int size,
                                                 hypre_Index *shape );
hypre_StructStencil *hypre_StructStencilRef ( hypre_StructStencil *stencil );
HYPRE_Int hypre_StructStencilDestroy ( hypre_StructStencil *stencil );
HYPRE_Int hypre_StructStencilOffsetEntry ( hypre_StructStencil *stencil,
                                           hypre_Index stencil_offset );
HYPRE_Int hypre_StructStencilSymmetrize ( hypre_StructStencil *stencil,
                                          hypre_StructStencil **symm_stencil_ptr, HYPRE_Int **symm_offsets_ptr );

/* struct_vector.c */
HYPRE_Int hypre_StructVectorMapDataIndex ( hypre_StructVector *vector, hypre_Index dindex );
HYPRE_Int hypre_StructVectorMapDataBox ( hypre_StructVector *vector, hypre_Box *dbox );
HYPRE_Int hypre_StructVectorMapDataStride ( hypre_StructVector *vector, hypre_Index dstride );
HYPRE_Int hypre_StructVectorUnMapDataIndex ( hypre_StructVector *vector, hypre_Index dindex );
HYPRE_Int hypre_StructVectorUnMapDataBox ( hypre_StructVector *vector, hypre_Box *dbox );
HYPRE_Int hypre_StructVectorUnMapDataStride ( hypre_StructVector *vector, hypre_Index dstride );
HYPRE_Int hypre_StructVectorMapCommInfo ( hypre_StructVector *vector, hypre_CommInfo *comm_info );
hypre_StructVector *hypre_StructVectorCreate ( MPI_Comm comm, hypre_StructGrid *grid );
hypre_StructVector *hypre_StructVectorRef ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorDestroy ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorSetStride ( hypre_StructVector *vector, hypre_IndexRef stride );
HYPRE_Int hypre_StructVectorReindex ( hypre_StructVector *vector, hypre_StructGrid *grid,
                                      hypre_Index stride );
HYPRE_Int hypre_StructVectorComputeDataSpace ( hypre_StructVector *vector, HYPRE_Int *num_ghost,
                                               hypre_BoxArray **data_space_ptr );
HYPRE_Int hypre_StructVectorResize ( hypre_StructVector *vector, hypre_BoxArray     *data_space );
HYPRE_Int hypre_StructVectorRestore ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorForget ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorInitializeShell ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorInitializeData ( hypre_StructVector *vector, HYPRE_Complex *data );
HYPRE_Int hypre_StructVectorInitialize ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorSetValues ( hypre_StructVector *vector, hypre_Index grid_index,
                                        HYPRE_Complex *values, HYPRE_Int action, HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructVectorSetBoxValues ( hypre_StructVector *vector, hypre_Box *set_box,
                                           hypre_Box *value_box, HYPRE_Complex *values, HYPRE_Int action, HYPRE_Int boxnum,
                                           HYPRE_Int outside );
HYPRE_Int hypre_StructVectorClearValues ( hypre_StructVector *vector, hypre_Index grid_index,
                                          HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructVectorClearBoxValues ( hypre_StructVector *vector, hypre_Box *clear_box,
                                             HYPRE_Int boxnum, HYPRE_Int outside );
HYPRE_Int hypre_StructVectorClearAllValues ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorSetNumGhost ( hypre_StructVector *vector, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructVectorSetDataSize(hypre_StructVector *vector, HYPRE_Int *data_size,
                                        HYPRE_Int *data_host_size);
HYPRE_Int hypre_StructVectorAssemble ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorCopy ( hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_StructVectorSetConstantValues ( hypre_StructVector *vector, HYPRE_Complex value );
HYPRE_Int hypre_StructVectorSetRandomValues ( hypre_StructVector *vector, HYPRE_Int seed );
HYPRE_Int hypre_StructVectorSetFunctionValues ( hypre_StructVector *vector,
                                                HYPRE_Complex (*fcn )( HYPRE_Int, HYPRE_Int, HYPRE_Int ));
HYPRE_Int hypre_StructVectorClearGhostValues ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorClearBoundGhostValues ( hypre_StructVector *vector, HYPRE_Int force );
HYPRE_Int hypre_StructVectorScaleValues ( hypre_StructVector *vector, HYPRE_Complex factor );
hypre_CommPkg *hypre_StructVectorGetMigrateCommPkg ( hypre_StructVector *from_vector,
                                                     hypre_StructVector *to_vector );
HYPRE_Int hypre_StructVectorMigrate ( hypre_CommPkg *comm_pkg, hypre_StructVector *from_vector,
                                      hypre_StructVector *to_vector );
HYPRE_Int hypre_StructVectorPrintData ( FILE *file, hypre_StructVector *vector, HYPRE_Int all );
HYPRE_Int hypre_StructVectorReadData ( FILE *file, hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorPrint ( const char *filename, hypre_StructVector *vector,
                                    HYPRE_Int all );
hypre_StructVector *hypre_StructVectorRead ( MPI_Comm comm, const char *filename,
                                             HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructVectorMaxValue ( hypre_StructVector *vector, HYPRE_Real *max_value,
                                       HYPRE_Int *max_index, hypre_Index max_xyz_index );
hypre_StructVector *hypre_StructVectorClone ( hypre_StructVector *vector );
