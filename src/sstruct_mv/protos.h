/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
HYPRE_Int hypre_SStructVectorPointwiseDivision ( hypre_SStructVector *x, hypre_SStructVector *y,
                                                 hypre_SStructVector **z_ptr );
HYPRE_Int hypre_SStructVectorPointwiseProduct ( hypre_SStructVector *x, hypre_SStructVector *y,
                                                hypre_SStructVector **z_ptr );
HYPRE_Int hypre_SStructVectorPointwiseInverse ( hypre_SStructVector *x,
                                                hypre_SStructVector **y_ptr );
