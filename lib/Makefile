#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../config/Makefile.config

 
BLASFILES =\
dasum.o\
daxpy.o\
dcopy.o\
ddot.o\
dgemm.o\
dgemv.o\
dger.o\
dnrm2.o\
drot.o\
dscal.o\
dswap.o\
dsymm.o\
dsymv.o\
dsyr2.o\
dsyr2k.o\
dsyrk.o\
dtrmm.o\
dtrmv.o\
dtrsm.o\
dtrsv.o\
idamax.o\
blas_utils.o


LAPACKFILES = \
dbdsqr.o\
dgebd2.o\
dgebrd.o\
dgelq2.o\
dgelqf.o\
dgels.o\
dgeqr2.o\
dgeqrf.o\
dgesvd.o\
dlabad.o\
dlabrd.o\
dlacpy.o\
dlae2.o\
dlaev2.o\
dlamch.o\
dlange.o\
dlanst.o\
dlansy.o\
dlapy2.o\
dlarf.o\
dlarfb.o\
dlarfg.o\
dlarft.o\
dlartg.o\
dlas2.o\
dlascl.o\
dlaset.o\
dlasq1.o\
dlasq2.o\
dlasq3.o\
dlasq4.o\
dlasq5.o\
dlasq6.o\
dlasr.o\
dlasrt.o\
dlassq.o\
dlasv2.o\
dlatrd.o\
dorg2l.o\
dorg2r.o\
dorgbr.o\
dorgl2.o\
dorglq.o\
dorgql.o\
dorgqr.o\
dorgtr.o\
dorm2r.o\
dormbr.o\
dorml2.o\
dormlq.o\
dormqr.o\
dpotf2.o\
dpotrf.o\
dpotrs.o\
dsteqr.o\
dsterf.o\
dsyev.o\
dsygs2.o\
dsygst.o\
dsygv.o\
dsytd2.o\
dsytrd.o\
ieeeck.o\
ilaenv.o\
lapack_utils.o\
lsame.o\
xerbla.o

SUPERLUFILES = \
 colamd.o\
 dcolumn_bmod.o\
 dcolumn_dfs.o\
 dcopy_to_ucol.o\
 dgscon.o\
 dgsequ.o\
 dgsrfs.o\
 dgssv.o\
 dgssvx.o\
 dgstrf.o\
 dgstrs.o\
 dlacon.o\
 dlangs.o\
 dlaqgs.o\
 dmemory.o\
 dpanel_bmod.o\
 dpanel_dfs.o\
 dpivotL.o\
 dpivotgrowth.o\
 dpruneL.o\
 dreadhb.o\
 dsnode_bmod.o\
 dsnode_dfs.o\
 dsp_blas2.o\
 dsp_blas3.o\
 dutil.o\
 get_perm_c.o\
 mmd.o\
 relax_snode.o\
 sp_coletree.o\
 sp_ienv.o\
 sp_preorder.o\
 superlu_memory.o\
 superlu_timer.o\
 superlu_util.o

FEIHYPREFILES = \
 bicgs.o\
 bicgstabl.o\
 FEI_HYPRE_Impl.o\
 HYPRE_LSC_aux.o\
 HYPRE_LSI_Dsuperlu.o\
 HYPRE_LSI_UZAWA.o\
 HYPRE_LSI_blkprec.o\
 HYPRE_LSI_ddict.o\
 HYPRE_LSI_ddilut.o\
 HYPRE_LSI_ml.o\
 HYPRE_LSI_mli.o\
 HYPRE_LSI_poly.o\
 HYPRE_LSI_schwarz.o\
 HYPRE_LinSysCore.o\
 HYPRE_SlideReduction.o\
 HYPRE_parcsr_TFQmr.o\
 HYPRE_parcsr_bicgs.o\
 HYPRE_parcsr_bicgstabl.o\
 HYPRE_parcsr_fgmres.o\
 HYPRE_parcsr_lsicg.o\
 HYPRE_parcsr_symqmr.o\
 SymQMR.o\
 TFQmr.o\
 cfei_hypre.o\
 fgmres.o\
 hypre_lsi_amge.o\
 hypre_lsi_ddamg.o\
 hypre_lsi_misc.o\
 hypre_schur_reduce.o\
 hypre_slide_reduce.o\
 lsicg.o\
 LLNL_FEI_Fei.o\
 LLNL_FEI_Impl.o\
 LLNL_FEI_LSCore.o\
 LLNL_FEI_Matrix.o\
 LLNL_FEI_Solver.o

FEMLIFILES = \
 mli_utils.o\
 mli_amgsa_calib.o\
 mli_amgsa_coarsen1.o\
 mli_amgsa_dd_fedata.o\
 mli_amgsa_dd_sfei.o\
 mli_method.o\
 mli_method_amgcr.o\
 mli_method_amgrs.o\
 mli_method_amgsa.o\
 mli.o\
 mli_oneLevel.o\
 cmli.o\
 mli_febase.o\
 mli_fedata.o\
 mli_fedata_utils.o\
 mli_sfei.o\
 mli_mapper.o\
 mli_matrix.o\
 mli_matrix_mult.o\
 mli_matrix_utils.o\
 mli_solver.o\
 mli_solver_arpacksuperlu.o\
 mli_solver_bjacobi.o\
 mli_solver_bsgs.o\
 mli_solver_cg.o\
 mli_solver_chebyshev.o\
 mli_solver_gmres.o\
 mli_solver_gs.o\
 mli_solver_hschwarz.o\
 mli_solver_hsgs.o\
 mli_solver_jacobi.o\
 mli_solver_kaczmarz.o\
 mli_solver_mls.o\
 mli_solver_mli.o\
 mli_solver_amg.o\
 mli_solver_parasails.o\
 mli_solver_seqsuperlu.o\
 mli_solver_sgs.o\
 mli_solver_superlu.o\
 mli_vector.o\
 mli_utils_fortran.o

IJMVFILES = \
 F90_HYPRE_IJMatrix.o\
 F90_HYPRE_IJVector.o\
 F90_IJMatrix.o\
 HYPRE_IJMatrix.o\
 HYPRE_IJVector.o\
 IJMatrix.o\
 IJMatrix_parcsr.o\
 IJVector.o\
 IJVector_parcsr.o\
 aux_parcsr_matrix.o\
 aux_par_vector.o

EUCLIDFILES = \
 Euclid_apply.o\
 Euclid_dh.o\
 ExternalRows_dh.o\
 Factor_dh.o\
 Hash_dh.o\
 Hash_i_dh.o\
 MatGenFD.o\
 Mat_dh.o\
 Mem_dh.o\
 Numbering_dh.o\
 Parser_dh.o\
 SortedList_dh.o\
 SortedSet_dh.o\
 SubdomainGraph_dh.o\
 TimeLog_dh.o\
 Timer_dh.o\
 Vec_dh.o\
 blas_dh.o\
 getRow_dh.o\
 globalObjects.o\
 ilu_mpi_bj.o\
 ilu_mpi_pilu.o\
 ilu_seq.o\
 io_dh.o\
 krylov_dh.o\
 mat_dh_private.o\
 shellSort_dh.o\
 sig_dh.o

PARASAILSFILES = \
 ConjGrad.o\
 DiagScale.o\
 FlexGmres.o\
 Hash.o\
 LoadBal.o\
 Matrix.o\
 Mem.o\
 Numbering.o\
 OrderStat.o\
 ParaSails.o\
 PrunedRows.o\
 RowPatt.o\
 StoredRows.o\
 hypre_ParaSails.o

PILUTFILES = \
 comm.o\
 HYPRE_DistributedMatrixPilutSolver.o\
 debug.o\
 distributed_qsort.o\
 distributed_qsort_si.o\
 ilut.o\
 parilut.o\
 parutil.o\
 pblas1.o\
 serilut.o\
 trifactor.o\
 util.o

DISTMATRIXFILES = \
 HYPRE_distributed_matrix.o\
 distributed_matrix.o\
 distributed_matrix_ISIS.o\
 distributed_matrix_PETSc.o\
 distributed_matrix_parcsr.o

KRYLOVFILES = \
 HYPRE_bicgstab.o\
 HYPRE_cgnr.o\
 HYPRE_gmres.o\
 HYPRE_pcg.o\
 bicgstab.o\
 cgnr.o\
 gmres.o\
 pcg.o

MATMATFILES = \
 HYPRE_BuildIJMatrixFromDistributedMatrix.o\
 HYPRE_ConvertPETScMatrixToDistributedMatrix.o\
 HYPRE_ConvertParCSRMatrixToDistributedMatrix.o

EIGENSOLVEFILES = \
 HYPRE_lobpcg.o\
 lobpcg.o\
 multivector.o\
 temp_multivector.o\
 HYPRE_parcsr_int.o\
 HYPRE_struct_int.o\
 HYPRE_sstruct_int.o\
 fortran_matrix.o

PARCSRLSFILES = \
 F90_HYPRE_parcsr_ParaSails.o\
 F90_HYPRE_parcsr_amg.o\
 F90_HYPRE_parcsr_bicgstab.o\
 F90_HYPRE_parcsr_cgnr.o\
 F90_HYPRE_parcsr_gmres.o\
 F90_HYPRE_parcsr_pcg.o\
 F90_HYPRE_parcsr_pilut.o\
 F90_hypre_laplace.o\
 HYPRE_parcsr_Euclid.o\
 HYPRE_parcsr_ParaSails.o\
 HYPRE_parcsr_amg.o\
 HYPRE_parcsr_bicgstab.o\
 HYPRE_parcsr_cgnr.o\
 HYPRE_parcsr_gmres.o\
 HYPRE_parcsr_hybrid.o\
 HYPRE_parcsr_pcg.o\
 HYPRE_parcsr_pilut.o\
 HYPRE_parcsr_schwarz.o\
 amg_hybrid.o\
 par_amg.o\
 par_amg_setup.o\
 par_amg_solve.o\
 par_amg_solveT.o\
 par_cg_relax_wt.o\
 par_coarse_parms.o\
 par_coarsen.o\
 par_cycle.o\
 par_difconv.o\
 par_gsmg.o\
 par_indepset.o\
 par_interp.o\
 par_laplace.o\
 par_laplace_27pt.o\
 par_laplace_9pt.o\
 par_nodal_systems.o\
 par_rap.o\
 par_rap_communication.o\
 par_relax.o\
 par_relax_interface.o\
 par_scaled_matnorm.o\
 par_schwarz.o\
 par_stats.o\
 par_strength.o\
 pcg_par.o\
 schwarz.o

PARCSRMVFILES = \
 F90_HYPRE_parcsr_matrix.o\
 F90_HYPRE_parcsr_vector.o\
 F90_par_vector.o\
 F90_parcsr_matrix.o\
 HYPRE_parcsr_matrix.o\
 HYPRE_parcsr_vector.o\
 communicationT.o\
 new_commpkg.o\
 numbers.o\
 par_csr_aat.o\
 par_csr_bool_matop.o\
 par_csr_bool_matrix.o\
 par_csr_communication.o\
 par_csr_matop.o\
 par_csr_matrix.o\
 par_csr_matvec.o\
 par_vector.o

SEQMVFILES = \
 HYPRE_csr_matrix.o\
 HYPRE_mapped_matrix.o\
 HYPRE_vector.o\
 HYPRE_multiblock_matrix.o\
 csr_matop.o\
 csr_matrix.o\
 csr_matvec.o\
 genpart.o\
 mapped_matrix.o\
 multiblock_matrix.o\
 vector.o

SSTRUCTLSFILES = \
 F90_HYPRE_sstruct_bicgstab.o\
 F90_HYPRE_sstruct_gmres.o\
 F90_HYPRE_sstruct_pcg.o\
 F90_HYPRE_sstruct_split.o\
 F90_HYPRE_sstruct_sys_pfmg.o\
 HYPRE_sstruct_bicgstab.o\
 HYPRE_sstruct_gmres.o\
 HYPRE_sstruct_InterFAC.o\
 HYPRE_sstruct_pcg.o\
 HYPRE_sstruct_split.o\
 HYPRE_sstruct_sys_pfmg.o\
 fac_amr_fcoarsen.o\
 fac_amr_rap.o\
 fac_amr_zero_data.o\
 fac_cf_coarsen.o\
 fac_CFInterfaceExtents.o\
 fac_cfstencil_box.o\
 fac_interp2.o\
 fac.o\
 fac_relax.o\
 fac_restrict2.o\
 fac_setup2.o\
 fac_solve3.o\
 fac_zero_cdata.o\
 fac_zero_stencilcoef.o\
 hypre_bsearch.o\
 krylov.o\
 krylov_sstruct.o\
 node_relax.o\
 sstruct_amr_intercommunication.o\
 sstruct_owninfo.o\
 sstruct_recvinfo.o\
 sstruct_sendinfo.o\
 sys_pfmg.o\
 sys_pfmg_relax.o\
 sys_pfmg_setup.o\
 sys_pfmg_setup_interp.o\
 sys_pfmg_setup_rap.o\
 sys_pfmg_solve.o\
 sys_semi_interp.o\
 sys_semi_restrict.o

SSTRUCTMVFILES = \
 F90_HYPRE_sstruct_graph.o\
 F90_HYPRE_sstruct_grid.o\
 F90_HYPRE_sstruct_matrix.o\
 F90_HYPRE_sstruct_stencil.o\
 F90_HYPRE_sstruct_vector.o\
 HYPRE_sstruct_graph.o\
 HYPRE_sstruct_grid.o\
 HYPRE_sstruct_matrix.o\
 HYPRE_sstruct_stencil.o\
 HYPRE_sstruct_vector.o\
 box_map.o\
 sstruct_axpy.o\
 sstruct_copy.o\
 sstruct_graph.o\
 sstruct_grid.o\
 sstruct_innerprod.o\
 sstruct_matrix.o\
 sstruct_matvec.o\
 sstruct_overlap_innerprod.o\
 sstruct_scale.o\
 sstruct_stencil.o\
 sstruct_vector.o

STRUCTLSFILES = \
 F90_HYPRE_struct_hybrid.o\
 F90_HYPRE_struct_pcg.o\
 F90_HYPRE_struct_pfmg.o\
 F90_HYPRE_struct_smg.o\
 HYPRE_struct_bicgstab.o\
 HYPRE_struct_gmres.o\
 HYPRE_struct_hybrid.o\
 HYPRE_struct_jacobi.o\
 HYPRE_struct_pcg.o\
 HYPRE_struct_pfmg.o\
 HYPRE_struct_smg.o\
 HYPRE_struct_sparse_msg.o\
 coarsen.o\
 cyclic_reduction.o\
 general.o\
 hybrid.o\
 jacobi.o\
 pcg_struct.o\
 pfmg.o\
 pfmg2_setup_rap.o\
 pfmg3_setup_rap.o\
 pfmg_relax.o\
 pfmg_setup.o\
 pfmg_setup_interp.o\
 pfmg_setup_rap.o\
 pfmg_setup_rap5.o\
 pfmg_setup_rap7.o\
 pfmg_solve.o\
 point_relax.o\
 red_black_gs.o\
 red_black_constantcoef_gs.o\
 semi_interp.o\
 semi_restrict.o\
 semi_setup_rap.o\
 smg.o\
 smg2_setup_rap.o\
 smg3_setup_rap.o\
 smg_axpy.o\
 smg_relax.o\
 smg_residual.o\
 smg_residual_unrolled.o\
 smg_setup.o\
 smg_setup_interp.o\
 smg_setup_rap.o\
 smg_setup_restrict.o\
 smg_solve.o\
 sparse_msg.o\
 sparse_msg2_setup_rap.o\
 sparse_msg3_setup_rap.o\
 sparse_msg_filter.o\
 sparse_msg_interp.o\
 sparse_msg_restrict.o\
 sparse_msg_setup.o\
 sparse_msg_setup_rap.o\
 sparse_msg_solve.o

STRUCTMVFILES = \
 F90_HYPRE_struct_grid.o\
 F90_HYPRE_struct_matrix.o\
 F90_HYPRE_struct_stencil.o\
 F90_HYPRE_struct_vector.o\
 HYPRE_struct_grid.o\
 HYPRE_struct_matrix.o\
 HYPRE_struct_stencil.o\
 HYPRE_struct_vector.o\
 box.o\
 box_algebra.o\
 box_alloc.o\
 box_boundary.o\
 box_neighbors.o\
 grow.o\
 communication_info.o\
 computation.o\
 project.o\
 struct_axpy.o\
 struct_communication.o\
 struct_copy.o\
 struct_grid.o\
 struct_io.o\
 struct_innerprod.o\
 struct_matrix.o\
 struct_matrix_mask.o\
 struct_matvec.o\
 struct_overlap_innerprod.o\
 struct_scale.o\
 struct_stencil.o\
 struct_vector.o

UTILFILES = \
 amg_linklist.o\
 binsearch.o\
 exchange_data.o\
 hypre_memory.o\
 hypre_qsort.o\
 memory_dmalloc.o\
 mpistubs.o\
 qsplit.o\
 random.o\
 thread_mpistubs.o\
 threading.o\
 timer.o\
 timing.o\
 umalloc_local.o

FILES = \
$(SUPERLUFILES)\
$(BLASFILES)\
$(LAPACKFILES)\
$(FEIHYPREFILES)\
$(FEMLIFILES)\
$(IJMVFILES)\
$(EUCLIDFILES)\
$(PARASAILSFILES)\
$(PILUTFILES)\
$(DISTMATRIXFILES)\
$(KRYLOVFILES)\
$(MATMATFILES)\
$(EIGENSOLVEFILES)\
$(PARCSRLSFILES)\
$(PARCSRMVFILES)\
$(SEQMVFILES)\
$(SSTRUCTLSFILES)\
$(SSTRUCTMVFILES)\
$(STRUCTLSFILES)\
$(STRUCTMVFILES)\
$(UTILFILES)

FILES_LSI = \
$(FEIHYPREFILES)\
$(FEMLIFILES)\
$(IJMVFILES)\
$(EUCLIDFILES)\
$(PARASAILSFILES)\
$(PILUTFILES)\
$(DISTMATRIXFILES)\
$(KRYLOVFILES)\
$(MATMATFILES)\
$(EIGENSOLVEFILES)\
$(PARCSRLSFILES)\
$(PARCSRMVFILES)\
$(SEQMVFILES)\
$(SSTRUCTLSFILES)\
$(SSTRUCTMVFILES)\
$(STRUCTLSFILES)\
$(STRUCTMVFILES)\
$(UTILFILES)

##################################################################
# Targets
##################################################################

all:
	@echo  "Building libHYPRE.a ... "
	${AR} libHYPRE.a ${FILES}
	${RANLIB} libHYPRE.a
	@echo  "Building libHYPRE_LSI.a ... "
	${AR} libHYPRE_LSI.a ${FILES_LSI}
	${RANLIB} libHYPRE_LSI.a
	cp -fp libHYPRE* ${HYPRE_BUILD_DIR}/lib

install: all
	cp -fp libHYPRE* ${HYPRE_LIB_INSTALL}

clean:
	rm -f *.o libHYPRE*

distclean: clean
