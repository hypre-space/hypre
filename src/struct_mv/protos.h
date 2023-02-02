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
                                       HYPRE_Int max_regions, HYPRE_Real gamma, HYPRE_Int dim, HYPRE_Int *return_code, MPI_Comm comm );
HYPRE_Int hypre_StructAssumedPartitionCreate ( HYPRE_Int dim, hypre_Box *bounding_box,
                                               HYPRE_Real global_boxes_size, HYPRE_Int global_num_boxes, hypre_BoxArray *local_boxes,
                                               HYPRE_Int *local_boxnums, HYPRE_Int max_regions, HYPRE_Int max_refinements, HYPRE_Real gamma,
                                               MPI_Comm comm, hypre_StructAssumedPart **p_assumed_partition );
HYPRE_Int hypre_StructAssumedPartitionDestroy ( hypre_StructAssumedPart *assumed_part );
HYPRE_Int hypre_APFillResponseStructAssumedPart ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                                  HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                  HYPRE_Int *response_message_size );
HYPRE_Int hypre_StructAssumedPartitionGetRegionsFromProc ( hypre_StructAssumedPart *assumed_part,
                                                           HYPRE_Int proc_id, hypre_BoxArray *assumed_regions );
HYPRE_Int hypre_StructAssumedPartitionGetProcsFromBox ( hypre_StructAssumedPart *assumed_part,
                                                        hypre_Box *box, HYPRE_Int *num_proc_array, HYPRE_Int *size_alloc_proc_array,
                                                        HYPRE_Int **p_proc_array );

/* box_algebra.c */
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
HYPRE_Int hypre_GeneralBoxBoundaryIntersect( hypre_Box *box, hypre_StructGrid *grid,
                                             hypre_Index stencil_element, hypre_BoxArray *boundary );

/* box.c */
HYPRE_Int hypre_SetIndex ( hypre_Index index, HYPRE_Int val );
HYPRE_Int hypre_CopyIndex( hypre_Index in_index, hypre_Index out_index );
HYPRE_Int hypre_CopyToCleanIndex( hypre_Index in_index, HYPRE_Int ndim, hypre_Index out_index );
HYPRE_Int hypre_IndexEqual ( hypre_Index index, HYPRE_Int val, HYPRE_Int ndim );
HYPRE_Int hypre_IndexMin( hypre_Index index, HYPRE_Int ndim );
HYPRE_Int hypre_IndexMax( hypre_Index index, HYPRE_Int ndim );
HYPRE_Int hypre_AddIndexes ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim,
                             hypre_Index result );
HYPRE_Int hypre_SubtractIndexes ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim,
                                  hypre_Index result );
HYPRE_Int hypre_IndexesEqual ( hypre_Index index1, hypre_Index index2, HYPRE_Int ndim );
HYPRE_Int hypre_IndexPrint ( FILE *file, HYPRE_Int ndim, hypre_Index index );
HYPRE_Int hypre_IndexRead ( FILE *file, HYPRE_Int ndim, hypre_Index index );
hypre_Box *hypre_BoxCreate ( HYPRE_Int ndim );
HYPRE_Int hypre_BoxDestroy ( hypre_Box *box );
HYPRE_Int hypre_BoxInit( hypre_Box *box, HYPRE_Int  ndim );
HYPRE_Int hypre_BoxSetExtents ( hypre_Box *box, hypre_Index imin, hypre_Index imax );
HYPRE_Int hypre_CopyBox( hypre_Box *box1, hypre_Box *box2 );
hypre_Box *hypre_BoxDuplicate ( hypre_Box *box );
HYPRE_Int hypre_BoxVolume( hypre_Box *box );
HYPRE_Real hypre_doubleBoxVolume( hypre_Box *box );
HYPRE_Int hypre_IndexInBox ( hypre_Index index, hypre_Box *box );
HYPRE_Int hypre_BoxGetSize ( hypre_Box *box, hypre_Index size );
HYPRE_Int hypre_BoxGetStrideSize ( hypre_Box *box, hypre_Index stride, hypre_Index size );
HYPRE_Int hypre_BoxGetStrideVolume ( hypre_Box *box, hypre_Index stride, HYPRE_Int *volume_ptr );
HYPRE_Int hypre_BoxIndexRank( hypre_Box *box, hypre_Index index );
HYPRE_Int hypre_BoxRankIndex( hypre_Box *box, HYPRE_Int rank, hypre_Index index );
HYPRE_Int hypre_BoxOffsetDistance( hypre_Box *box, hypre_Index index );
HYPRE_Int hypre_BoxShiftPos( hypre_Box *box, hypre_Index shift );
HYPRE_Int hypre_BoxShiftNeg( hypre_Box *box, hypre_Index shift );
HYPRE_Int hypre_BoxGrowByIndex( hypre_Box *box, hypre_Index  index );
HYPRE_Int hypre_BoxGrowByValue( hypre_Box *box, HYPRE_Int val );
HYPRE_Int hypre_BoxGrowByArray ( hypre_Box *box, HYPRE_Int *array );
HYPRE_Int hypre_BoxPrint ( FILE *file, hypre_Box *box );
HYPRE_Int hypre_BoxRead ( FILE *file, HYPRE_Int ndim, hypre_Box **box_ptr );
hypre_BoxArray *hypre_BoxArrayCreate ( HYPRE_Int size, HYPRE_Int ndim );
HYPRE_Int hypre_BoxArrayDestroy ( hypre_BoxArray *box_array );
HYPRE_Int hypre_BoxArraySetSize ( hypre_BoxArray *box_array, HYPRE_Int size );
hypre_BoxArray *hypre_BoxArrayDuplicate ( hypre_BoxArray *box_array );
HYPRE_Int hypre_AppendBox ( hypre_Box *box, hypre_BoxArray *box_array );
HYPRE_Int hypre_DeleteBox ( hypre_BoxArray *box_array, HYPRE_Int index );
HYPRE_Int hypre_DeleteMultipleBoxes ( hypre_BoxArray *box_array, HYPRE_Int *indices,
                                      HYPRE_Int num );
HYPRE_Int hypre_AppendBoxArray ( hypre_BoxArray *box_array_0, hypre_BoxArray *box_array_1 );
hypre_BoxArrayArray *hypre_BoxArrayArrayCreate ( HYPRE_Int size, HYPRE_Int ndim );
HYPRE_Int hypre_BoxArrayArrayDestroy ( hypre_BoxArrayArray *box_array_array );
hypre_BoxArrayArray *hypre_BoxArrayArrayDuplicate ( hypre_BoxArrayArray *box_array_array );

/* box_manager.c */
HYPRE_Int hypre_BoxManEntryGetInfo ( hypre_BoxManEntry *entry, void **info_ptr );
HYPRE_Int hypre_BoxManEntryGetExtents ( hypre_BoxManEntry *entry, hypre_Index imin,
                                        hypre_Index imax );
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

/* communication_info.c */
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

/* HYPRE_struct_grid.c */
HYPRE_Int HYPRE_StructGridCreate ( MPI_Comm comm, HYPRE_Int dim, HYPRE_StructGrid *grid );
HYPRE_Int HYPRE_StructGridDestroy ( HYPRE_StructGrid grid );
HYPRE_Int HYPRE_StructGridSetExtents ( HYPRE_StructGrid grid, HYPRE_Int *ilower,
                                       HYPRE_Int *iupper );
HYPRE_Int HYPRE_StructGridSetPeriodic ( HYPRE_StructGrid grid, HYPRE_Int *periodic );
HYPRE_Int HYPRE_StructGridAssemble ( HYPRE_StructGrid grid );
HYPRE_Int HYPRE_StructGridSetNumGhost ( HYPRE_StructGrid grid, HYPRE_Int *num_ghost );

/* HYPRE_struct_matrix.c */
HYPRE_Int HYPRE_StructMatrixCreate ( MPI_Comm comm, HYPRE_StructGrid grid,
                                     HYPRE_StructStencil stencil, HYPRE_StructMatrix *matrix );
HYPRE_Int HYPRE_StructMatrixDestroy ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixInitialize ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixSetValues ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Complex *values );
HYPRE_Int HYPRE_StructMatrixGetValues ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                        HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Complex *values );
HYPRE_Int HYPRE_StructMatrixSetBoxValues ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           HYPRE_Complex *values );
HYPRE_Int HYPRE_StructMatrixGetBoxValues ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                           HYPRE_Complex *values );
HYPRE_Int HYPRE_StructMatrixSetConstantValues ( HYPRE_StructMatrix matrix,
                                                HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Complex *values );
HYPRE_Int HYPRE_StructMatrixAddToValues ( HYPRE_StructMatrix matrix, HYPRE_Int *grid_index,
                                          HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Complex *values );
HYPRE_Int HYPRE_StructMatrixAddToBoxValues ( HYPRE_StructMatrix matrix, HYPRE_Int *ilower,
                                             HYPRE_Int *iupper, HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices,
                                             HYPRE_Complex *values );
HYPRE_Int HYPRE_StructMatrixAddToConstantValues ( HYPRE_StructMatrix matrix,
                                                  HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices, HYPRE_Complex *values );
HYPRE_Int HYPRE_StructMatrixAssemble ( HYPRE_StructMatrix matrix );
HYPRE_Int HYPRE_StructMatrixSetNumGhost ( HYPRE_StructMatrix matrix, HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructMatrixGetGrid ( HYPRE_StructMatrix matrix, HYPRE_StructGrid *grid );
HYPRE_Int HYPRE_StructMatrixSetSymmetric ( HYPRE_StructMatrix matrix, HYPRE_Int symmetric );
HYPRE_Int HYPRE_StructMatrixSetConstantEntries ( HYPRE_StructMatrix matrix, HYPRE_Int nentries,
                                                 HYPRE_Int *entries );
HYPRE_Int HYPRE_StructMatrixPrint ( const char *filename, HYPRE_StructMatrix matrix,
                                    HYPRE_Int all );
HYPRE_Int HYPRE_StructMatrixMatvec ( HYPRE_Complex alpha, HYPRE_StructMatrix A,
                                     HYPRE_StructVector x, HYPRE_Complex beta, HYPRE_StructVector y );
HYPRE_Int HYPRE_StructMatrixClearBoundary( HYPRE_StructMatrix matrix );

/* HYPRE_struct_stencil.c */
HYPRE_Int HYPRE_StructStencilCreate ( HYPRE_Int dim, HYPRE_Int size, HYPRE_StructStencil *stencil );
HYPRE_Int HYPRE_StructStencilSetElement ( HYPRE_StructStencil stencil, HYPRE_Int element_index,
                                          HYPRE_Int *offset );
HYPRE_Int HYPRE_StructStencilDestroy ( HYPRE_StructStencil stencil );

/* HYPRE_struct_vector.c */
HYPRE_Int HYPRE_StructVectorCreate ( MPI_Comm comm, HYPRE_StructGrid grid,
                                     HYPRE_StructVector *vector );
HYPRE_Int HYPRE_StructVectorDestroy ( HYPRE_StructVector struct_vector );
HYPRE_Int HYPRE_StructVectorInitialize ( HYPRE_StructVector vector );
HYPRE_Int HYPRE_StructVectorSetValues ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                        HYPRE_Complex values );
HYPRE_Int HYPRE_StructVectorSetBoxValues ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, HYPRE_Complex *values );
HYPRE_Int HYPRE_StructVectorAddToValues ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                          HYPRE_Complex values );
HYPRE_Int HYPRE_StructVectorAddToBoxValues ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                             HYPRE_Int *iupper, HYPRE_Complex *values );
HYPRE_Int HYPRE_StructVectorScaleValues ( HYPRE_StructVector vector, HYPRE_Complex factor );
HYPRE_Int HYPRE_StructVectorGetValues ( HYPRE_StructVector vector, HYPRE_Int *grid_index,
                                        HYPRE_Complex *values );
HYPRE_Int HYPRE_StructVectorGetBoxValues ( HYPRE_StructVector vector, HYPRE_Int *ilower,
                                           HYPRE_Int *iupper, HYPRE_Complex *values );
HYPRE_Int HYPRE_StructVectorAssemble ( HYPRE_StructVector vector );
HYPRE_Int hypre_StructVectorPrintData ( FILE *file, hypre_StructVector *vector, HYPRE_Int all );
HYPRE_Int hypre_StructVectorReadData ( FILE *file, hypre_StructVector *vector );
HYPRE_Int HYPRE_StructVectorPrint ( const char *filename, HYPRE_StructVector vector,
                                    HYPRE_Int all );
HYPRE_Int HYPRE_StructVectorRead ( MPI_Comm comm, const char *filename,
                                   HYPRE_Int *num_ghost, HYPRE_StructVector *vector );
HYPRE_Int HYPRE_StructVectorSetNumGhost ( HYPRE_StructVector vector, HYPRE_Int *num_ghost );
HYPRE_Int HYPRE_StructVectorCopy ( HYPRE_StructVector x, HYPRE_StructVector y );
HYPRE_Int HYPRE_StructVectorSetConstantValues ( HYPRE_StructVector vector, HYPRE_Complex values );
HYPRE_Int HYPRE_StructVectorGetMigrateCommPkg ( HYPRE_StructVector from_vector,
                                                HYPRE_StructVector to_vector, HYPRE_CommPkg *comm_pkg );
HYPRE_Int HYPRE_StructVectorMigrate ( HYPRE_CommPkg comm_pkg, HYPRE_StructVector from_vector,
                                      HYPRE_StructVector to_vector );
HYPRE_Int HYPRE_CommPkgDestroy ( HYPRE_CommPkg comm_pkg );

/* project.c */
HYPRE_Int hypre_ProjectBox ( hypre_Box *box, hypre_Index index, hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArray ( hypre_BoxArray *box_array, hypre_Index index,
                                  hypre_Index stride );
HYPRE_Int hypre_ProjectBoxArrayArray ( hypre_BoxArrayArray *box_array_array, hypre_Index index,
                                       hypre_Index stride );

/* struct_axpy.c */
HYPRE_Int hypre_StructAxpy ( HYPRE_Complex alpha, hypre_StructVector *x, hypre_StructVector *y );

/* struct_communication.c */
HYPRE_Int hypre_CommPkgCreate ( hypre_CommInfo *comm_info, hypre_BoxArray *send_data_space,
                                hypre_BoxArray *recv_data_space, HYPRE_Int num_values, HYPRE_Int **orders, HYPRE_Int reverse,
                                MPI_Comm comm, hypre_CommPkg **comm_pkg_ptr );
HYPRE_Int hypre_CommTypeSetEntries ( hypre_CommType *comm_type, HYPRE_Int *boxnums,
                                     hypre_Box *boxes, hypre_Index stride, hypre_Index coord, hypre_Index dir, HYPRE_Int *order,
                                     hypre_BoxArray *data_space, HYPRE_Int *data_offsets );
HYPRE_Int hypre_CommTypeSetEntry ( hypre_Box *box, hypre_Index stride, hypre_Index coord,
                                   hypre_Index dir, HYPRE_Int *order, hypre_Box *data_box, HYPRE_Int data_box_offset,
                                   hypre_CommEntryType *comm_entry );
HYPRE_Int hypre_InitializeCommunication ( hypre_CommPkg *comm_pkg, HYPRE_Complex *send_data,
                                          HYPRE_Complex *recv_data, HYPRE_Int action, HYPRE_Int tag, hypre_CommHandle **comm_handle_ptr );
HYPRE_Int hypre_FinalizeCommunication ( hypre_CommHandle *comm_handle );
HYPRE_Int hypre_ExchangeLocalData ( hypre_CommPkg *comm_pkg, HYPRE_Complex *send_data,
                                    HYPRE_Complex *recv_data, HYPRE_Int action );
HYPRE_Int hypre_CommPkgDestroy ( hypre_CommPkg *comm_pkg );

/* struct_copy.c */
HYPRE_Int hypre_StructCopy ( hypre_StructVector *x, hypre_StructVector *y );
HYPRE_Int hypre_StructPartialCopy ( hypre_StructVector *x, hypre_StructVector *y,
                                    hypre_BoxArrayArray *array_boxes );

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
HYPRE_Int hypre_StructGridAssemble ( hypre_StructGrid *grid );
HYPRE_Int hypre_GatherAllBoxes ( MPI_Comm comm, hypre_BoxArray *boxes, HYPRE_Int dim,
                                 hypre_BoxArray **all_boxes_ptr, HYPRE_Int **all_procs_ptr, HYPRE_Int *first_local_ptr );
HYPRE_Int hypre_ComputeBoxnums ( hypre_BoxArray *boxes, HYPRE_Int *procs, HYPRE_Int **boxnums_ptr );
HYPRE_Int hypre_StructGridPrint ( FILE *file, hypre_StructGrid *grid );
HYPRE_Int hypre_StructGridRead ( MPI_Comm comm, FILE *file, hypre_StructGrid **grid_ptr );
HYPRE_Int hypre_StructGridSetNumGhost ( hypre_StructGrid *grid, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructGridGetMaxBoxSize ( hypre_StructGrid *grid );
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
HYPRE_Int hypre_StructGridSetDataLocation( HYPRE_StructGrid grid,
                                           HYPRE_MemoryLocation data_location );
#endif
/* struct_innerprod.c */
HYPRE_Real hypre_StructInnerProd ( hypre_StructVector *x, hypre_StructVector *y );

/* struct_io.c */
HYPRE_Int hypre_PrintBoxArrayData ( FILE *file, hypre_BoxArray *box_array,
                                    hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int dim, HYPRE_Complex *data );
HYPRE_Int hypre_PrintCCVDBoxArrayData ( FILE *file, hypre_BoxArray *box_array,
                                        hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int center_rank, HYPRE_Int stencil_size,
                                        HYPRE_Int *symm_elements, HYPRE_Int dim, HYPRE_Complex *data );
HYPRE_Int hypre_PrintCCBoxArrayData ( FILE *file, hypre_BoxArray *box_array,
                                      hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Complex *data );
HYPRE_Int hypre_ReadBoxArrayData ( FILE *file, hypre_BoxArray *box_array,
                                   hypre_BoxArray *data_space, HYPRE_Int num_values, HYPRE_Int dim, HYPRE_Complex *data );
HYPRE_Int hypre_ReadBoxArrayData_CC ( FILE *file, hypre_BoxArray *box_array,
                                      hypre_BoxArray *data_space, HYPRE_Int stencil_size, HYPRE_Int real_stencil_size,
                                      HYPRE_Int constant_coefficient, HYPRE_Int dim, HYPRE_Complex *data );

/* struct_matrix.c */
HYPRE_Complex *hypre_StructMatrixExtractPointerByIndex ( hypre_StructMatrix *matrix, HYPRE_Int b,
                                                         hypre_Index index );
hypre_StructMatrix *hypre_StructMatrixCreate ( MPI_Comm comm, hypre_StructGrid *grid,
                                               hypre_StructStencil *user_stencil );
hypre_StructMatrix *hypre_StructMatrixRef ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixDestroy ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixInitializeShell ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixInitializeData ( hypre_StructMatrix *matrix, HYPRE_Complex *data,
                                             HYPRE_Complex *data_const);
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
HYPRE_Int hypre_StructMatrixSetNumGhost ( hypre_StructMatrix *matrix, HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructMatrixSetConstantCoefficient ( hypre_StructMatrix *matrix,
                                                     HYPRE_Int constant_coefficient );
HYPRE_Int hypre_StructMatrixSetConstantEntries ( hypre_StructMatrix *matrix, HYPRE_Int nentries,
                                                 HYPRE_Int *entries );
HYPRE_Int hypre_StructMatrixClearGhostValues ( hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixPrintData ( FILE *file, hypre_StructMatrix *matrix, HYPRE_Int all );
HYPRE_Int hypre_StructMatrixReadData ( FILE *file, hypre_StructMatrix *matrix );
HYPRE_Int hypre_StructMatrixPrint ( const char *filename, hypre_StructMatrix *matrix,
                                    HYPRE_Int all );
hypre_StructMatrix *hypre_StructMatrixRead ( MPI_Comm comm, const char *filename,
                                             HYPRE_Int *num_ghost );
HYPRE_Int hypre_StructMatrixMigrate ( hypre_StructMatrix *from_matrix,
                                      hypre_StructMatrix *to_matrix );
HYPRE_Int hypre_StructMatrixClearBoundary( hypre_StructMatrix *matrix);

/* struct_matrix_mask.c */
hypre_StructMatrix *hypre_StructMatrixCreateMask ( hypre_StructMatrix *matrix,
                                                   HYPRE_Int num_stencil_indices, HYPRE_Int *stencil_indices );

/* struct_matvec.c */
void *hypre_StructMatvecCreate ( void );
HYPRE_Int hypre_StructMatvecSetup ( void *matvec_vdata, hypre_StructMatrix *A,
                                    hypre_StructVector *x );
HYPRE_Int hypre_StructMatvecCompute ( void *matvec_vdata, HYPRE_Complex alpha,
                                      hypre_StructMatrix *A, hypre_StructVector *x, HYPRE_Complex beta, hypre_StructVector *y );
HYPRE_Int hypre_StructMatvecCC0 ( HYPRE_Complex alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC1 ( HYPRE_Complex alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecCC2 ( HYPRE_Complex alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                                  hypre_StructVector *y, hypre_BoxArrayArray *compute_box_aa, hypre_IndexRef stride );
HYPRE_Int hypre_StructMatvecDestroy ( void *matvec_vdata );
HYPRE_Int hypre_StructMatvec ( HYPRE_Complex alpha, hypre_StructMatrix *A, hypre_StructVector *x,
                               HYPRE_Complex beta, hypre_StructVector *y );

/* struct_scale.c */
HYPRE_Int hypre_StructScale ( HYPRE_Complex alpha, hypre_StructVector *y );

/* struct_stencil.c */
hypre_StructStencil *hypre_StructStencilCreate ( HYPRE_Int dim, HYPRE_Int size,
                                                 hypre_Index *shape );
hypre_StructStencil *hypre_StructStencilRef ( hypre_StructStencil *stencil );
HYPRE_Int hypre_StructStencilDestroy ( hypre_StructStencil *stencil );
HYPRE_Int hypre_StructStencilElementRank ( hypre_StructStencil *stencil,
                                           hypre_Index stencil_element );
HYPRE_Int hypre_StructStencilSymmetrize ( hypre_StructStencil *stencil,
                                          hypre_StructStencil **symm_stencil_ptr, HYPRE_Int **symm_elements_ptr );

/* struct_vector.c */
hypre_StructVector *hypre_StructVectorCreate ( MPI_Comm comm, hypre_StructGrid *grid );
hypre_StructVector *hypre_StructVectorRef ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorDestroy ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorInitializeShell ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorInitializeData ( hypre_StructVector *vector, HYPRE_Complex *data);
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
HYPRE_Int hypre_StructVectorSetConstantValues ( hypre_StructVector *vector, HYPRE_Complex values );
HYPRE_Int hypre_StructVectorSetFunctionValues ( hypre_StructVector *vector,
                                                HYPRE_Complex (*fcn )( HYPRE_Int, HYPRE_Int, HYPRE_Int ));
HYPRE_Int hypre_StructVectorClearGhostValues ( hypre_StructVector *vector );
HYPRE_Int hypre_StructVectorClearBoundGhostValues ( hypre_StructVector *vector, HYPRE_Int force );
HYPRE_Int hypre_StructVectorScaleValues ( hypre_StructVector *vector, HYPRE_Complex factor );
hypre_CommPkg *hypre_StructVectorGetMigrateCommPkg ( hypre_StructVector *from_vector,
                                                     hypre_StructVector *to_vector );
HYPRE_Int hypre_StructVectorMigrate ( hypre_CommPkg *comm_pkg, hypre_StructVector *from_vector,
                                      hypre_StructVector *to_vector );
HYPRE_Int hypre_StructVectorPrint ( const char *filename, hypre_StructVector *vector,
                                    HYPRE_Int all );
hypre_StructVector *hypre_StructVectorRead ( MPI_Comm comm, const char *filename,
                                             HYPRE_Int *num_ghost );
hypre_StructVector *hypre_StructVectorClone ( hypre_StructVector *vector );
