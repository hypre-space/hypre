/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* HYPRE_sstruct_graph.c */
HYPRE_Int HYPRE_SStructGraphCreate ( MPI_Comm comm, HYPRE_SStructGrid grid,
                                     HYPRE_SStructGraph *graph_ptr );
HYPRE_Int HYPRE_SStructGraphDestroy ( HYPRE_SStructGraph graph );
HYPRE_Int HYPRE_SStructGraphSetDomainGrid ( HYPRE_SStructGraph graph,
                                            HYPRE_SStructGrid domain_grid );
HYPRE_Int HYPRE_SStructGraphSetStencil ( HYPRE_SStructGraph graph, HYPRE_Int part, HYPRE_Int var,
                                         HYPRE_SStructStencil stencil );
HYPRE_Int HYPRE_SStructGraphSetFEM ( HYPRE_SStructGraph graph, HYPRE_Int part );
HYPRE_Int HYPRE_SStructGraphSetFEMSparsity ( HYPRE_SStructGraph graph, HYPRE_Int part,
                                             HYPRE_Int nsparse, HYPRE_Int *sparsity );
HYPRE_Int HYPRE_SStructGraphAddEntries ( HYPRE_SStructGraph graph, HYPRE_Int part, HYPRE_Int *index,
                                         HYPRE_Int var, HYPRE_Int to_part, HYPRE_Int *to_index, HYPRE_Int to_var );
HYPRE_Int HYPRE_SStructGraphAssemble ( HYPRE_SStructGraph graph );
HYPRE_Int HYPRE_SStructGraphSetObjectType ( HYPRE_SStructGraph graph, HYPRE_Int type );
HYPRE_Int HYPRE_SStructGraphPrint ( FILE *file, HYPRE_SStructGraph graph );
HYPRE_Int HYPRE_SStructGraphRead ( FILE *file, HYPRE_SStructGrid grid,
                                   HYPRE_SStructStencil **stencils, HYPRE_SStructGraph *graph_ptr );

/* HYPRE_sstruct_grid.c */
HYPRE_Int HYPRE_SStructGridCreate ( MPI_Comm comm, HYPRE_Int ndim, HYPRE_Int nparts,
                                    HYPRE_SStructGrid *grid_ptr );
HYPRE_Int HYPRE_SStructGridDestroy ( HYPRE_SStructGrid grid );
HYPRE_Int HYPRE_SStructGridSetExtents ( HYPRE_SStructGrid grid, HYPRE_Int part, HYPRE_Int *ilower,
                                        HYPRE_Int *iupper );
HYPRE_Int HYPRE_SStructGridSetVariables ( HYPRE_SStructGrid grid, HYPRE_Int part, HYPRE_Int nvars,
                                          HYPRE_SStructVariable *vartypes );
HYPRE_Int HYPRE_SStructGridAddVariables ( HYPRE_SStructGrid grid, HYPRE_Int part, HYPRE_Int *index,
                                          HYPRE_Int nvars, HYPRE_SStructVariable *vartypes );
HYPRE_Int HYPRE_SStructGridSetFEMOrdering ( HYPRE_SStructGrid grid, HYPRE_Int part,
                                            HYPRE_Int *ordering );
HYPRE_Int HYPRE_SStructGridSetNeighborPart ( HYPRE_SStructGrid grid, HYPRE_Int part,
                                             HYPRE_Int *ilower, HYPRE_Int *iupper, HYPRE_Int nbor_part, HYPRE_Int *nbor_ilower,
                                             HYPRE_Int *nbor_iupper, HYPRE_Int *index_map, HYPRE_Int *index_dir );
HYPRE_Int HYPRE_SStructGridSetSharedPart ( HYPRE_SStructGrid grid, HYPRE_Int part,
                                           HYPRE_Int *ilower, HYPRE_Int *iupper, HYPRE_Int *offset, HYPRE_Int shared_part,
                                           HYPRE_Int *shared_ilower, HYPRE_Int *shared_iupper, HYPRE_Int *shared_offset, HYPRE_Int *index_map,
                                           HYPRE_Int *index_dir );
HYPRE_Int HYPRE_SStructGridAddUnstructuredPart ( HYPRE_SStructGrid grid, HYPRE_Int ilower,
                                                 HYPRE_Int iupper );
HYPRE_Int HYPRE_SStructGridAssemble ( HYPRE_SStructGrid grid );
HYPRE_Int HYPRE_SStructGridSetPeriodic ( HYPRE_SStructGrid grid, HYPRE_Int part,
                                         HYPRE_Int *periodic );
HYPRE_Int HYPRE_SStructGridSetNumGhost ( HYPRE_SStructGrid grid, HYPRE_Int *num_ghost );

/* HYPRE_sstruct_matrix.c */
HYPRE_Int HYPRE_SStructMatrixCreate ( MPI_Comm comm, HYPRE_SStructGraph graph,
                                      HYPRE_SStructMatrix *matrix_ptr );
HYPRE_Int HYPRE_SStructMatrixDestroy ( HYPRE_SStructMatrix matrix );
HYPRE_Int HYPRE_SStructMatrixInitialize ( HYPRE_SStructMatrix matrix );
HYPRE_Int HYPRE_SStructMatrixSetValues ( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                         HYPRE_Int *index, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructMatrixAddToValues ( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                           HYPRE_Int *index, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructMatrixAddFEMValues ( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                            HYPRE_Int *index, HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructMatrixGetValues ( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                         HYPRE_Int *index, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructMatrixGetFEMValues ( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                            HYPRE_Int *index, HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructMatrixSetBoxValues ( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                            HYPRE_Int *ilower, HYPRE_Int *iupper, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries,
                                            HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructMatrixAddToBoxValues ( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                              HYPRE_Int *ilower, HYPRE_Int *iupper, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries,
                                              HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructMatrixGetBoxValues ( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                            HYPRE_Int *ilower, HYPRE_Int *iupper, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries,
                                            HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructMatrixAssemble ( HYPRE_SStructMatrix matrix );
HYPRE_Int HYPRE_SStructMatrixSetSymmetric ( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                            HYPRE_Int var, HYPRE_Int to_var, HYPRE_Int symmetric );
HYPRE_Int HYPRE_SStructMatrixSetNSSymmetric ( HYPRE_SStructMatrix matrix, HYPRE_Int symmetric );
HYPRE_Int HYPRE_SStructMatrixSetObjectType ( HYPRE_SStructMatrix matrix, HYPRE_Int type );
HYPRE_Int HYPRE_SStructMatrixGetObject ( HYPRE_SStructMatrix matrix, void **object );
HYPRE_Int HYPRE_SStructMatrixPrint ( const char *filename, HYPRE_SStructMatrix matrix,
                                     HYPRE_Int all );
HYPRE_Int HYPRE_SStructMatrixRead ( MPI_Comm comm, const char *filename,
                                    HYPRE_SStructMatrix *matrix_ptr );
HYPRE_Int HYPRE_SStructMatrixMatvec ( HYPRE_Complex alpha, HYPRE_SStructMatrix A,
                                      HYPRE_SStructVector x, HYPRE_Complex beta, HYPRE_SStructVector y );

/* HYPRE_sstruct_stencil.c */
HYPRE_Int HYPRE_SStructStencilCreate ( HYPRE_Int ndim, HYPRE_Int size,
                                       HYPRE_SStructStencil *stencil_ptr );
HYPRE_Int HYPRE_SStructStencilDestroy ( HYPRE_SStructStencil stencil );
HYPRE_Int HYPRE_SStructStencilSetEntry ( HYPRE_SStructStencil stencil, HYPRE_Int entry,
                                         HYPRE_Int *offset, HYPRE_Int var );
HYPRE_Int HYPRE_SStructStencilPrint ( FILE *file, HYPRE_SStructStencil stencil );
HYPRE_Int HYPRE_SStructStencilRead ( FILE *file, HYPRE_SStructStencil *stencil_ptr );


/* HYPRE_sstruct_vector.c */
HYPRE_Int HYPRE_SStructVectorCreate ( MPI_Comm comm, HYPRE_SStructGrid grid,
                                      HYPRE_SStructVector *vector_ptr );
HYPRE_Int HYPRE_SStructVectorDestroy ( HYPRE_SStructVector vector );
HYPRE_Int HYPRE_SStructVectorInitialize ( HYPRE_SStructVector vector );
HYPRE_Int HYPRE_SStructVectorSetValues ( HYPRE_SStructVector vector, HYPRE_Int part,
                                         HYPRE_Int *index, HYPRE_Int var, HYPRE_Complex *value );
HYPRE_Int HYPRE_SStructVectorAddToValues ( HYPRE_SStructVector vector, HYPRE_Int part,
                                           HYPRE_Int *index, HYPRE_Int var, HYPRE_Complex *value );
HYPRE_Int HYPRE_SStructVectorAddFEMValues ( HYPRE_SStructVector vector, HYPRE_Int part,
                                            HYPRE_Int *index, HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructVectorGetValues ( HYPRE_SStructVector vector, HYPRE_Int part,
                                         HYPRE_Int *index, HYPRE_Int var, HYPRE_Complex *value );
HYPRE_Int HYPRE_SStructVectorGetFEMValues ( HYPRE_SStructVector vector, HYPRE_Int part,
                                            HYPRE_Int *index, HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructVectorSetBoxValues ( HYPRE_SStructVector vector, HYPRE_Int part,
                                            HYPRE_Int *ilower, HYPRE_Int *iupper, HYPRE_Int var, HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructVectorAddToBoxValues ( HYPRE_SStructVector vector, HYPRE_Int part,
                                              HYPRE_Int *ilower, HYPRE_Int *iupper, HYPRE_Int var, HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructVectorGetBoxValues ( HYPRE_SStructVector vector, HYPRE_Int part,
                                            HYPRE_Int *ilower, HYPRE_Int *iupper, HYPRE_Int var, HYPRE_Complex *values );
HYPRE_Int HYPRE_SStructVectorAssemble ( HYPRE_SStructVector vector );
HYPRE_Int HYPRE_SStructVectorGather ( HYPRE_SStructVector vector );
HYPRE_Int HYPRE_SStructVectorSetConstantValues ( HYPRE_SStructVector vector, HYPRE_Complex value );
HYPRE_Int HYPRE_SStructVectorSetObjectType ( HYPRE_SStructVector vector, HYPRE_Int type );
HYPRE_Int HYPRE_SStructVectorGetObject ( HYPRE_SStructVector vector, void **object );
HYPRE_Int HYPRE_SStructVectorPrint ( const char *filename, HYPRE_SStructVector vector,
                                     HYPRE_Int all );
HYPRE_Int HYPRE_SStructVectorRead ( MPI_Comm comm, const char *filename,
                                    HYPRE_SStructVector *vector );
HYPRE_Int HYPRE_SStructVectorCopy ( HYPRE_SStructVector x, HYPRE_SStructVector y );
HYPRE_Int HYPRE_SStructVectorScale ( HYPRE_Complex alpha, HYPRE_SStructVector y );
HYPRE_Int HYPRE_SStructInnerProd ( HYPRE_SStructVector x, HYPRE_SStructVector y,
                                   HYPRE_Real *result );
HYPRE_Int HYPRE_SStructAxpy ( HYPRE_Complex alpha, HYPRE_SStructVector x, HYPRE_SStructVector y );

/* sstruct_axpy.c */
HYPRE_Int hypre_SStructPAxpy ( HYPRE_Complex alpha, hypre_SStructPVector *px,
                               hypre_SStructPVector *py );
HYPRE_Int hypre_SStructAxpy ( HYPRE_Complex alpha, hypre_SStructVector *x, hypre_SStructVector *y );

/* sstruct_copy.c */
HYPRE_Int hypre_SStructPCopy ( hypre_SStructPVector *px, hypre_SStructPVector *py );
HYPRE_Int hypre_SStructPartialPCopy ( hypre_SStructPVector *px, hypre_SStructPVector *py,
                                      hypre_BoxArrayArray **array_boxes );
HYPRE_Int hypre_SStructCopy ( hypre_SStructVector *x, hypre_SStructVector *y );

/* sstruct_graph.c */
HYPRE_Int hypre_SStructGraphRef ( hypre_SStructGraph *graph, hypre_SStructGraph **graph_ref );
HYPRE_Int hypre_SStructGraphGetUVEntryRank( hypre_SStructGraph *graph, HYPRE_Int part,
                                            HYPRE_Int var, hypre_Index index, HYPRE_BigInt *rank );
HYPRE_Int hypre_SStructGraphFindBoxEndpt ( hypre_SStructGraph *graph, HYPRE_Int part, HYPRE_Int var,
                                           HYPRE_Int proc, HYPRE_Int endpt, HYPRE_Int boxi );
HYPRE_Int hypre_SStructGraphFindSGridEndpts ( hypre_SStructGraph *graph, HYPRE_Int part,
                                              HYPRE_Int var, HYPRE_Int proc, HYPRE_Int endpt, HYPRE_Int *endpts );

/* sstruct_grid.c */
HYPRE_Int hypre_SStructVariableGetOffset ( HYPRE_SStructVariable vartype, HYPRE_Int ndim,
                                           hypre_Index varoffset );
HYPRE_Int hypre_SStructPGridCreate ( MPI_Comm comm, HYPRE_Int ndim,
                                     hypre_SStructPGrid **pgrid_ptr );
HYPRE_Int hypre_SStructPGridDestroy ( hypre_SStructPGrid *pgrid );
HYPRE_Int hypre_SStructPGridSetExtents ( hypre_SStructPGrid *pgrid, hypre_Index ilower,
                                         hypre_Index iupper );
HYPRE_Int hypre_SStructPGridSetCellSGrid ( hypre_SStructPGrid *pgrid,
                                           hypre_StructGrid *cell_sgrid );
HYPRE_Int hypre_SStructPGridSetVariables ( hypre_SStructPGrid *pgrid, HYPRE_Int nvars,
                                           HYPRE_SStructVariable *vartypes );
HYPRE_Int hypre_SStructPGridSetPNeighbor ( hypre_SStructPGrid *pgrid, hypre_Box *pneighbor_box,
                                           hypre_Index pnbor_offset );
HYPRE_Int hypre_SStructPGridAssemble ( hypre_SStructPGrid *pgrid );
HYPRE_Int hypre_SStructPGridGetMaxBoxSize ( hypre_SStructPGrid *pgrid );
HYPRE_Int hypre_SStructGridRef ( hypre_SStructGrid *grid, hypre_SStructGrid **grid_ref );
HYPRE_Int hypre_SStructGridAssembleBoxManagers ( hypre_SStructGrid *grid );
HYPRE_Int hypre_SStructGridAssembleNborBoxManagers ( hypre_SStructGrid *grid );
HYPRE_Int hypre_SStructGridCreateCommInfo ( hypre_SStructGrid *grid );
HYPRE_Int hypre_SStructGridFindBoxManEntry ( hypre_SStructGrid *grid, HYPRE_Int part,
                                             hypre_Index index, HYPRE_Int var, hypre_BoxManEntry **entry_ptr );
HYPRE_Int hypre_SStructGridFindNborBoxManEntry ( hypre_SStructGrid *grid, HYPRE_Int part,
                                                 hypre_Index index, HYPRE_Int var, hypre_BoxManEntry **entry_ptr );
HYPRE_Int hypre_SStructGridBoxProcFindBoxManEntry ( hypre_SStructGrid *grid, HYPRE_Int part,
                                                    HYPRE_Int var, HYPRE_Int box, HYPRE_Int proc, hypre_BoxManEntry **entry_ptr );
HYPRE_Int hypre_SStructBoxManEntryGetCSRstrides ( hypre_BoxManEntry *entry, hypre_Index strides );
HYPRE_Int hypre_SStructBoxManEntryGetGhstrides ( hypre_BoxManEntry *entry, hypre_Index strides );
HYPRE_Int hypre_SStructBoxManEntryGetGlobalCSRank ( hypre_BoxManEntry *entry, hypre_Index index,
                                                    HYPRE_BigInt *rank_ptr );
HYPRE_Int hypre_SStructBoxManEntryGetGlobalGhrank ( hypre_BoxManEntry *entry, hypre_Index index,
                                                    HYPRE_BigInt *rank_ptr );
HYPRE_Int hypre_SStructBoxManEntryGetProcess ( hypre_BoxManEntry *entry, HYPRE_Int *proc_ptr );
HYPRE_Int hypre_SStructBoxManEntryGetBoxnum ( hypre_BoxManEntry *entry, HYPRE_Int *id_ptr );
HYPRE_Int hypre_SStructBoxManEntryGetPart ( hypre_BoxManEntry *entry, HYPRE_Int part,
                                            HYPRE_Int *part_ptr );
HYPRE_Int hypre_SStructIndexToNborIndex( hypre_Index index, hypre_Index root, hypre_Index nbor_root,
                                         hypre_Index coord, hypre_Index dir, HYPRE_Int ndim, hypre_Index nbor_index );
HYPRE_Int hypre_SStructBoxToNborBox ( hypre_Box *box, hypre_Index root, hypre_Index nbor_root,
                                      hypre_Index coord, hypre_Index dir );
HYPRE_Int hypre_SStructNborIndexToIndex( hypre_Index nbor_index, hypre_Index root,
                                         hypre_Index nbor_root, hypre_Index coord, hypre_Index dir, HYPRE_Int ndim, hypre_Index index );
HYPRE_Int hypre_SStructNborBoxToBox ( hypre_Box *nbor_box, hypre_Index root, hypre_Index nbor_root,
                                      hypre_Index coord, hypre_Index dir );
HYPRE_Int hypre_SStructVarToNborVar ( hypre_SStructGrid *grid, HYPRE_Int part, HYPRE_Int var,
                                      HYPRE_Int *coord, HYPRE_Int *nbor_var_ptr );
HYPRE_Int hypre_SStructGridSetNumGhost ( hypre_SStructGrid *grid, HYPRE_Int *num_ghost );
HYPRE_Int hypre_SStructBoxManEntryGetGlobalRank ( hypre_BoxManEntry *entry, hypre_Index index,
                                                  HYPRE_BigInt *rank_ptr, HYPRE_Int type );
HYPRE_Int hypre_SStructBoxManEntryGetStrides ( hypre_BoxManEntry *entry, hypre_Index strides,
                                               HYPRE_Int type );
HYPRE_Int hypre_SStructBoxNumMap ( hypre_SStructGrid *grid, HYPRE_Int part, HYPRE_Int boxnum,
                                   HYPRE_Int **num_varboxes_ptr, HYPRE_Int ***map_ptr );
HYPRE_Int hypre_SStructCellGridBoxNumMap ( hypre_SStructGrid *grid, HYPRE_Int part,
                                           HYPRE_Int ***num_varboxes_ptr, HYPRE_Int ****map_ptr );
HYPRE_Int hypre_SStructCellBoxToVarBox ( hypre_Box *box, hypre_Index offset, hypre_Index varoffset,
                                         HYPRE_Int *valid );
HYPRE_Int hypre_SStructGridIntersect ( hypre_SStructGrid *grid, HYPRE_Int part, HYPRE_Int var,
                                       hypre_Box *box, HYPRE_Int action, hypre_BoxManEntry ***entries_ptr, HYPRE_Int *nentries_ptr );
HYPRE_Int hypre_SStructGridGetMaxBoxSize ( hypre_SStructGrid *grid );
HYPRE_Int hypre_SStructGridPrint ( FILE *file, hypre_SStructGrid *grid );
HYPRE_Int hypre_SStructGridRead ( MPI_Comm comm, FILE *file, hypre_SStructGrid **grid_ptr );

/* sstruct_innerprod.c */
HYPRE_Int hypre_SStructPInnerProd ( hypre_SStructPVector *px, hypre_SStructPVector *py,
                                    HYPRE_Real *presult_ptr );
HYPRE_Int hypre_SStructInnerProd ( hypre_SStructVector *x, hypre_SStructVector *y,
                                   HYPRE_Real *result_ptr );

/* sstruct_matrix.c */
HYPRE_Int hypre_SStructPMatrixRef ( hypre_SStructPMatrix *matrix,
                                    hypre_SStructPMatrix **matrix_ref );
HYPRE_Int hypre_SStructPMatrixCreate ( MPI_Comm comm, hypre_SStructPGrid *pgrid,
                                       hypre_SStructStencil **stencils, hypre_SStructPMatrix **pmatrix_ptr );
HYPRE_Int hypre_SStructPMatrixDestroy ( hypre_SStructPMatrix *pmatrix );
HYPRE_Int hypre_SStructPMatrixInitialize ( hypre_SStructPMatrix *pmatrix );
HYPRE_Int hypre_SStructPMatrixSetValues ( hypre_SStructPMatrix *pmatrix, hypre_Index index,
                                          HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, HYPRE_Complex *values, HYPRE_Int action );
HYPRE_Int hypre_SStructPMatrixSetBoxValues( hypre_SStructPMatrix *pmatrix, hypre_Box *set_box,
                                            HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, hypre_Box *value_box, HYPRE_Complex *values,
                                            HYPRE_Int action );
HYPRE_Int hypre_SStructPMatrixAccumulate ( hypre_SStructPMatrix *pmatrix );
HYPRE_Int hypre_SStructPMatrixAssemble ( hypre_SStructPMatrix *pmatrix );
HYPRE_Int hypre_SStructPMatrixSetSymmetric ( hypre_SStructPMatrix *pmatrix, HYPRE_Int var,
                                             HYPRE_Int to_var, HYPRE_Int symmetric );
HYPRE_Int hypre_SStructPMatrixPrint ( const char *filename, hypre_SStructPMatrix *pmatrix,
                                      HYPRE_Int all );
HYPRE_Int hypre_SStructUMatrixInitialize ( hypre_SStructMatrix *matrix );
HYPRE_Int hypre_SStructUMatrixSetValues ( hypre_SStructMatrix *matrix, HYPRE_Int part,
                                          hypre_Index index, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, HYPRE_Complex *values,
                                          HYPRE_Int action );
HYPRE_Int hypre_SStructUMatrixSetBoxValues( hypre_SStructMatrix *matrix, HYPRE_Int part,
                                            hypre_Box *set_box, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, hypre_Box *value_box,
                                            HYPRE_Complex *values, HYPRE_Int action );
HYPRE_Int hypre_SStructUMatrixAssemble ( hypre_SStructMatrix *matrix );
HYPRE_Int hypre_SStructMatrixRef ( hypre_SStructMatrix *matrix, hypre_SStructMatrix **matrix_ref );
HYPRE_Int hypre_SStructMatrixSplitEntries ( hypre_SStructMatrix *matrix, HYPRE_Int part,
                                            HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, HYPRE_Int *nSentries_ptr,
                                            HYPRE_Int **Sentries_ptr, HYPRE_Int *nUentries_ptr, HYPRE_Int **Uentries_ptr );
HYPRE_Int hypre_SStructMatrixSetValues ( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                         HYPRE_Int *index, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, HYPRE_Complex *values,
                                         HYPRE_Int action );
HYPRE_Int hypre_SStructMatrixSetBoxValues( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                           hypre_Box *set_box, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, hypre_Box *value_box,
                                           HYPRE_Complex *values, HYPRE_Int action );
HYPRE_Int hypre_SStructMatrixSetInterPartValues( HYPRE_SStructMatrix matrix, HYPRE_Int part,
                                                 hypre_Box *set_box, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, hypre_Box *value_box,
                                                 HYPRE_Complex *values, HYPRE_Int action );
HYPRE_MemoryLocation hypre_SStructMatrixMemoryLocation(hypre_SStructMatrix *matrix);

/* sstruct_matvec.c */
HYPRE_Int hypre_SStructPMatvecCreate ( void **pmatvec_vdata_ptr );
HYPRE_Int hypre_SStructPMatvecSetup ( void *pmatvec_vdata, hypre_SStructPMatrix *pA,
                                      hypre_SStructPVector *px );
HYPRE_Int hypre_SStructPMatvecCompute ( void *pmatvec_vdata, HYPRE_Complex alpha,
                                        hypre_SStructPMatrix *pA, hypre_SStructPVector *px, HYPRE_Complex beta, hypre_SStructPVector *py );
HYPRE_Int hypre_SStructPMatvecDestroy ( void *pmatvec_vdata );
HYPRE_Int hypre_SStructPMatvec ( HYPRE_Complex alpha, hypre_SStructPMatrix *pA,
                                 hypre_SStructPVector *px, HYPRE_Complex beta, hypre_SStructPVector *py );
HYPRE_Int hypre_SStructMatvecCreate ( void **matvec_vdata_ptr );
HYPRE_Int hypre_SStructMatvecSetup ( void *matvec_vdata, hypre_SStructMatrix *A,
                                     hypre_SStructVector *x );
HYPRE_Int hypre_SStructMatvecCompute ( void *matvec_vdata, HYPRE_Complex alpha,
                                       hypre_SStructMatrix *A, hypre_SStructVector *x, HYPRE_Complex beta, hypre_SStructVector *y );
HYPRE_Int hypre_SStructMatvecDestroy ( void *matvec_vdata );
HYPRE_Int hypre_SStructMatvec ( HYPRE_Complex alpha, hypre_SStructMatrix *A, hypre_SStructVector *x,
                                HYPRE_Complex beta, hypre_SStructVector *y );

/* sstruct_scale.c */
HYPRE_Int hypre_SStructPScale ( HYPRE_Complex alpha, hypre_SStructPVector *py );
HYPRE_Int hypre_SStructScale ( HYPRE_Complex alpha, hypre_SStructVector *y );

/* sstruct_stencil.c */
HYPRE_Int hypre_SStructStencilRef ( hypre_SStructStencil *stencil,
                                    hypre_SStructStencil **stencil_ref );

/* sstruct_vector.c */
HYPRE_Int hypre_SStructPVectorRef ( hypre_SStructPVector *vector,
                                    hypre_SStructPVector **vector_ref );
HYPRE_Int hypre_SStructPVectorCreate ( MPI_Comm comm, hypre_SStructPGrid *pgrid,
                                       hypre_SStructPVector **pvector_ptr );
HYPRE_Int hypre_SStructPVectorDestroy ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructPVectorInitialize ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructPVectorSetValues ( hypre_SStructPVector *pvector, hypre_Index index,
                                          HYPRE_Int var, HYPRE_Complex *value, HYPRE_Int action );
HYPRE_Int hypre_SStructPVectorSetBoxValues( hypre_SStructPVector *pvector, hypre_Box *set_box,
                                            HYPRE_Int var, hypre_Box *value_box, HYPRE_Complex *values, HYPRE_Int action );
HYPRE_Int hypre_SStructPVectorAccumulate ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructPVectorAssemble ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructPVectorGather ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructPVectorGetValues ( hypre_SStructPVector *pvector, hypre_Index index,
                                          HYPRE_Int var, HYPRE_Complex *value );
HYPRE_Int hypre_SStructPVectorGetBoxValues( hypre_SStructPVector *pvector, hypre_Box *set_box,
                                            HYPRE_Int var, hypre_Box *value_box, HYPRE_Complex *values );
HYPRE_Int hypre_SStructPVectorSetConstantValues ( hypre_SStructPVector *pvector,
                                                  HYPRE_Complex value );
HYPRE_Int hypre_SStructPVectorPrint ( const char *filename, hypre_SStructPVector *pvector,
                                      HYPRE_Int all );
HYPRE_Int hypre_SStructVectorRef ( hypre_SStructVector *vector, hypre_SStructVector **vector_ref );
HYPRE_Int hypre_SStructVectorSetConstantValues ( hypre_SStructVector *vector, HYPRE_Complex value );
HYPRE_Int hypre_SStructVectorConvert ( hypre_SStructVector *vector,
                                       hypre_ParVector **parvector_ptr );
HYPRE_Int hypre_SStructVectorParConvert ( hypre_SStructVector *vector,
                                          hypre_ParVector **parvector_ptr );
HYPRE_Int hypre_SStructVectorRestore ( hypre_SStructVector *vector, hypre_ParVector *parvector );
HYPRE_Int hypre_SStructVectorParRestore ( hypre_SStructVector *vector, hypre_ParVector *parvector );
HYPRE_Int hypre_SStructPVectorInitializeShell ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructVectorInitializeShell ( hypre_SStructVector *vector );
HYPRE_Int hypre_SStructVectorClearGhostValues ( hypre_SStructVector *vector );
HYPRE_MemoryLocation hypre_SStructVectorMemoryLocation(hypre_SStructVector *vector);

