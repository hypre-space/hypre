## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE.la libHYPRE_LSI.la
libHYPRE_la_SOURCES = $(FILES)
libHYPRE_la_LDFLAGS = -avoid-version
libHYPRE_LSI_la_SOURCES = $(FILES)
libHYPRE_LSI_la_LDFLAGS = -avoid-version

if BLASNEEDED
BLASDIR = blas
BLASFILES = \
 $(top_srcdir)/blas/dasum.c\
 $(top_srcdir)/blas/daxpy.c\
 $(top_srcdir)/blas/dcopy.c\
 $(top_srcdir)/blas/ddot.c\
 $(top_srcdir)/blas/dgemm.c\
 $(top_srcdir)/blas/dgemv.c\
 $(top_srcdir)/blas/dger.c\
 $(top_srcdir)/blas/dnrm2.c\
 $(top_srcdir)/blas/drot.c\
 $(top_srcdir)/blas/dscal.c\
 $(top_srcdir)/blas/dsymv.c\
 $(top_srcdir)/blas/dsyr2.c\
 $(top_srcdir)/blas/dsyr2k.c\
 $(top_srcdir)/blas/dsyrk.c\
 $(top_srcdir)/blas/dtrmm.c\
 $(top_srcdir)/blas/dtrmv.c\
 $(top_srcdir)/blas/dtrsm.c\
 $(top_srcdir)/blas/dtrsv.c\
 $(top_srcdir)/blas/idamax.c\
 $(top_srcdir)/blas/blas_utils.c
else !BLASNEEDED
BLASDIR = 
BLASFILES = 
endif !BLASNEEDED

if LAPACKNEEDED
LAPACKDIR = lapack
LAPACKFILES = \
 $(top_srcdir)/dgels.c\
 $(top_srcdir)/dsyev.c\
 $(top_srcdir)/dsygv.c\
 $(top_srcdir)/lapack_utils.c
else !LAPACKNEEDED
LAPACKDIR = 
LAPACKFILES =
endif !LAPACKNEEDED

HYPRE_SRC_DIRS = \
 $(BLASDIR)\
 $(LAPACKDIR)\
 FEI_mv/SuperLU\
 FEI_mv/fei-hypre\
 FEI_mv/femli/lib\
 IJ_mv\
 distributed_ls/Euclid\
 distributed_ls/ParaSails\
 distributed_ls/pilut\
 distributed_matrix\
 krylov\
 matrix_matrix\
 parcsr_es/LOBPCG\
 parcsr_ls\
 parcsr_mv\
 seq_mv\
 sstruct_ls\
 sstruct_mv\
 struct_ls\
 struct_mv\
 utilities

FILES = \
 $(BLASFILES)\
 $(LAPACKFILES)\
 $(top_srcdir)/FEI_mv/SuperLU/colamd.c\
 $(top_srcdir)/FEI_mv/SuperLU/dcolumn_bmod.c\
 $(top_srcdir)/FEI_mv/SuperLU/dcolumn_dfs.c\
 $(top_srcdir)/FEI_mv/SuperLU/dcopy_to_ucol.c\
 $(top_srcdir)/FEI_mv/SuperLU/dgscon.c\
 $(top_srcdir)/FEI_mv/SuperLU/dgsequ.c\
 $(top_srcdir)/FEI_mv/SuperLU/dgsrfs.c\
 $(top_srcdir)/FEI_mv/SuperLU/dgssv.c\
 $(top_srcdir)/FEI_mv/SuperLU/dgssvx.c\
 $(top_srcdir)/FEI_mv/SuperLU/dgstrf.c\
 $(top_srcdir)/FEI_mv/SuperLU/dgstrs.c\
 $(top_srcdir)/FEI_mv/SuperLU/dlacon.c\
 $(top_srcdir)/FEI_mv/SuperLU/dlangs.c\
 $(top_srcdir)/FEI_mv/SuperLU/dlaqgs.c\
 $(top_srcdir)/FEI_mv/SuperLU/dmemory.c\
 $(top_srcdir)/FEI_mv/SuperLU/dpanel_bmod.c\
 $(top_srcdir)/FEI_mv/SuperLU/dpanel_dfs.c\
 $(top_srcdir)/FEI_mv/SuperLU/dpivotL.c\
 $(top_srcdir)/FEI_mv/SuperLU/dpivotgrowth.c\
 $(top_srcdir)/FEI_mv/SuperLU/dpruneL.c\
 $(top_srcdir)/FEI_mv/SuperLU/dreadhb.c\
 $(top_srcdir)/FEI_mv/SuperLU/dsnode_bmod.c\
 $(top_srcdir)/FEI_mv/SuperLU/dsnode_dfs.c\
 $(top_srcdir)/FEI_mv/SuperLU/dsp_blas2.c\
 $(top_srcdir)/FEI_mv/SuperLU/dsp_blas3.c\
 $(top_srcdir)/FEI_mv/SuperLU/dutil.c\
 $(top_srcdir)/FEI_mv/SuperLU/get_perm_c.c\
 $(top_srcdir)/FEI_mv/SuperLU/mmd.c\
 $(top_srcdir)/FEI_mv/SuperLU/relax_snode.c\
 $(top_srcdir)/FEI_mv/SuperLU/sp_coletree.c\
 $(top_srcdir)/FEI_mv/SuperLU/sp_ienv.c\
 $(top_srcdir)/FEI_mv/SuperLU/sp_preorder.c\
 $(top_srcdir)/FEI_mv/SuperLU/superlu_memory.c\
 $(top_srcdir)/FEI_mv/SuperLU/superlu_timer.c\
 $(top_srcdir)/FEI_mv/SuperLU/superlu_util.c\
 $(top_srcdir)/FEI_mv/fei-hypre/bicgs.c\
 $(top_srcdir)/FEI_mv/fei-hypre/bicgstabl.c\
 $(top_srcdir)/FEI_mv/fei-hypre/FEI_HYPRE_Impl.cxx\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_LSC_aux.cxx\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_LSI_Dsuperlu.c\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_LSI_UZAWA.cxx\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_LSI_blkprec.cxx\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_LSI_ddict.c\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_LSI_ddilut.c\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_LSI_ml.c\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_LSI_mli.cxx\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_LSI_poly.c\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_LSI_schwarz.c\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_LinSysCore.cxx\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_SlideReduction.cxx\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_parcsr_TFQmr.c\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_parcsr_bicgs.c\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_parcsr_bicgstabl.c\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_parcsr_fgmres.c\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_parcsr_lsicg.c\
 $(top_srcdir)/FEI_mv/fei-hypre/HYPRE_parcsr_symqmr.c\
 $(top_srcdir)/FEI_mv/fei-hypre/SymQMR.c\
 $(top_srcdir)/FEI_mv/fei-hypre/TFQmr.c\
 $(top_srcdir)/FEI_mv/fei-hypre/cfei_hypre.cxx\
 $(top_srcdir)/FEI_mv/fei-hypre/fgmres.c\
 $(top_srcdir)/FEI_mv/fei-hypre/hypre_lsi_amge.c\
 $(top_srcdir)/FEI_mv/fei-hypre/hypre_lsi_ddamg.c\
 $(top_srcdir)/FEI_mv/fei-hypre/hypre_lsi_misc.c\
 $(top_srcdir)/FEI_mv/fei-hypre/hypre_schur_reduce.cxx\
 $(top_srcdir)/FEI_mv/fei-hypre/hypre_slide_reduce.cxx\
 $(top_srcdir)/FEI_mv/fei-hypre/lsicg.c\
 $(top_srcdir)/FEI_mv/femli/util/mli_utils.c\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_amgsa_calib.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_amgsa_coarsen1.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_amgsa_dd_fedata.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_amgsa_dd_sfei.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method_amgcr.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method_amgrs.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method_amgsa.cxx\
 $(top_srcdir)/FEI_mv/femli/base/mli.cxx\
 $(top_srcdir)/FEI_mv/femli/base/mli_oneLevel.cxx\
 $(top_srcdir)/FEI_mv/femli/cintface/cmli.cxx\
 $(top_srcdir)/FEI_mv/femli/fedata/mli_fedata.cxx\
 $(top_srcdir)/FEI_mv/femli/fedata/mli_fedata_utils.cxx\
 $(top_srcdir)/FEI_mv/femli/fedata/mli_sfei.cxx\
 $(top_srcdir)/FEI_mv/femli/mapper/mli_mapper.cxx\
 $(top_srcdir)/FEI_mv/femli/matrix/mli_matrix.cxx\
 $(top_srcdir)/FEI_mv/femli/matrix/mli_matrix_mult.cxx\
 $(top_srcdir)/FEI_mv/femli/matrix/mli_matrix_utils.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_arpacksuperlu.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_bjacobi.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_bsgs.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_cg.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_chebyshev.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_gs.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_hschwarz.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_hsgs.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_jacobi.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_kaczmarz.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_mls.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_parasails.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_seqsuperlu.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_sgs.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_superlu.cxx\
 $(top_srcdir)/FEI_mv/femli/vector/mli_vector.cxx\
 $(top_srcdir)/FEI_mv/femli/util/mli_utils_fortran.f\
 $(top_srcdir)/IJ_mv/F90_HYPRE_IJMatrix.c\
 $(top_srcdir)/IJ_mv/F90_HYPRE_IJVector.c\
 $(top_srcdir)/IJ_mv/F90_IJMatrix.c\
 $(top_srcdir)/IJ_mv/HYPRE_IJMatrix.c\
 $(top_srcdir)/IJ_mv/HYPRE_IJVector.c\
 $(top_srcdir)/IJ_mv/IJMatrix.c\
 $(top_srcdir)/IJ_mv/IJMatrix_parcsr.c\
 $(top_srcdir)/IJ_mv/IJVector.c\
 $(top_srcdir)/IJ_mv/IJVector_parcsr.c\
 $(top_srcdir)/IJ_mv/aux_parcsr_matrix.c\
 $(top_srcdir)/distributed_ls/Euclid/Euclid_apply.c\
 $(top_srcdir)/distributed_ls/Euclid/Euclid_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/ExternalRows_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/Factor_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/Hash_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/Hash_i_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/MatGenFD.c\
 $(top_srcdir)/distributed_ls/Euclid/Mat_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/Mem_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/Numbering_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/Parser_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/SortedList_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/SortedSet_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/SubdomainGraph_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/TimeLog_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/Timer_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/Vec_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/blas_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/getRow_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/globalObjects.c\
 $(top_srcdir)/distributed_ls/Euclid/ilu_mpi_bj.c\
 $(top_srcdir)/distributed_ls/Euclid/ilu_mpi_pilu.c\
 $(top_srcdir)/distributed_ls/Euclid/ilu_seq.c\
 $(top_srcdir)/distributed_ls/Euclid/io_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/krylov_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/mat_dh_private.c\
 $(top_srcdir)/distributed_ls/Euclid/shellSort_dh.c\
 $(top_srcdir)/distributed_ls/Euclid/sig_dh.c\
 $(top_srcdir)/distributed_ls/ParaSails/ConjGrad.c\
 $(top_srcdir)/distributed_ls/ParaSails/DiagScale.c\
 $(top_srcdir)/distributed_ls/ParaSails/FlexGmres.c\
 $(top_srcdir)/distributed_ls/ParaSails/Hash.c\
 $(top_srcdir)/distributed_ls/ParaSails/LoadBal.c\
 $(top_srcdir)/distributed_ls/ParaSails/Matrix.c\
 $(top_srcdir)/distributed_ls/ParaSails/Mem.c\
 $(top_srcdir)/distributed_ls/ParaSails/Numbering.c\
 $(top_srcdir)/distributed_ls/ParaSails/OrderStat.c\
 $(top_srcdir)/distributed_ls/ParaSails/ParaSails.c\
 $(top_srcdir)/distributed_ls/ParaSails/PrunedRows.c\
 $(top_srcdir)/distributed_ls/ParaSails/RowPatt.c\
 $(top_srcdir)/distributed_ls/ParaSails/StoredRows.c\
 $(top_srcdir)/distributed_ls/ParaSails/hypre_ParaSails.c\
 $(top_srcdir)/distributed_ls/pilut/comm.c\
 $(top_srcdir)/distributed_ls/pilut/HYPRE_DistributedMatrixPilutSolver.c\
 $(top_srcdir)/distributed_ls/pilut/debug.c\
 $(top_srcdir)/distributed_ls/pilut/distributed_qsort.c\
 $(top_srcdir)/distributed_ls/pilut/distributed_qsort_si.c\
 $(top_srcdir)/distributed_ls/pilut/ilut.c\
 $(top_srcdir)/distributed_ls/pilut/parilut.c\
 $(top_srcdir)/distributed_ls/pilut/parutil.c\
 $(top_srcdir)/distributed_ls/pilut/pblas1.c\
 $(top_srcdir)/distributed_ls/pilut/serilut.c\
 $(top_srcdir)/distributed_ls/pilut/trifactor.c\
 $(top_srcdir)/distributed_ls/pilut/util.c\
 $(top_srcdir)/distributed_matrix/HYPRE_distributed_matrix.c\
 $(top_srcdir)/distributed_matrix/distributed_matrix.c\
 $(top_srcdir)/distributed_matrix/distributed_matrix_ISIS.c\
 $(top_srcdir)/distributed_matrix/distributed_matrix_PETSc.c\
 $(top_srcdir)/distributed_matrix/distributed_matrix_parcsr.c\
 $(top_srcdir)/krylov/HYPRE_bicgstab.c\
 $(top_srcdir)/krylov/HYPRE_cgnr.c\
 $(top_srcdir)/krylov/HYPRE_gmres.c\
 $(top_srcdir)/krylov/HYPRE_pcg.c\
 $(top_srcdir)/krylov/bicgstab.c\
 $(top_srcdir)/krylov/cgnr.c\
 $(top_srcdir)/krylov/gmres.c\
 $(top_srcdir)/krylov/pcg.c\
 $(top_srcdir)/matrix_matrix/HYPRE_ConvertPETScMatrixToDistributedMatrix.c\
 $(top_srcdir)/matrix_matrix/HYPRE_ConvertParCSRMatrixToDistributedMatrix.c\
 $(top_srcdir)/parcsr_es/LOBPCG/lobpcg.c\
 $(top_srcdir)/parcsr_es/LOBPCG/lobpcg_matrix.c\
 $(top_srcdir)/parcsr_es/LOBPCG/lobpcg_utilities.c\
 $(top_srcdir)/parcsr_ls/F90_HYPRE_parcsr_ParaSails.c\
 $(top_srcdir)/parcsr_ls/F90_HYPRE_parcsr_amg.c\
 $(top_srcdir)/parcsr_ls/F90_HYPRE_parcsr_bicgstab.c\
 $(top_srcdir)/parcsr_ls/F90_HYPRE_parcsr_cgnr.c\
 $(top_srcdir)/parcsr_ls/F90_HYPRE_parcsr_gmres.c\
 $(top_srcdir)/parcsr_ls/F90_HYPRE_parcsr_pcg.c\
 $(top_srcdir)/parcsr_ls/F90_HYPRE_parcsr_pilut.c\
 $(top_srcdir)/parcsr_ls/F90_hypre_laplace.c\
 $(top_srcdir)/parcsr_ls/HYPRE_parcsr_Euclid.c\
 $(top_srcdir)/parcsr_ls/HYPRE_parcsr_ParaSails.c\
 $(top_srcdir)/parcsr_ls/HYPRE_parcsr_amg.c\
 $(top_srcdir)/parcsr_ls/HYPRE_parcsr_bicgstab.c\
 $(top_srcdir)/parcsr_ls/HYPRE_parcsr_cgnr.c\
 $(top_srcdir)/parcsr_ls/HYPRE_parcsr_gmres.c\
 $(top_srcdir)/parcsr_ls/HYPRE_parcsr_hybrid.c\
 $(top_srcdir)/parcsr_ls/HYPRE_parcsr_pcg.c\
 $(top_srcdir)/parcsr_ls/HYPRE_parcsr_pilut.c\
 $(top_srcdir)/parcsr_ls/HYPRE_parcsr_schwarz.c\
 $(top_srcdir)/parcsr_ls/amg_hybrid.c\
 $(top_srcdir)/parcsr_ls/par_amg.c\
 $(top_srcdir)/parcsr_ls/par_amg_setup.c\
 $(top_srcdir)/parcsr_ls/par_amg_solve.c\
 $(top_srcdir)/parcsr_ls/par_amg_solveT.c\
 $(top_srcdir)/parcsr_ls/par_cg_relax_wt.c\
 $(top_srcdir)/parcsr_ls/par_coarse_parms.c\
 $(top_srcdir)/parcsr_ls/par_coarsen.c\
 $(top_srcdir)/parcsr_ls/par_cycle.c\
 $(top_srcdir)/parcsr_ls/par_difconv.c\
 $(top_srcdir)/parcsr_ls/par_gsmg.c\
 $(top_srcdir)/parcsr_ls/par_indepset.c\
 $(top_srcdir)/parcsr_ls/par_interp.c\
 $(top_srcdir)/parcsr_ls/par_laplace.c\
 $(top_srcdir)/parcsr_ls/par_laplace_27pt.c\
 $(top_srcdir)/parcsr_ls/par_laplace_9pt.c\
 $(top_srcdir)/parcsr_ls/par_rap.c\
 $(top_srcdir)/parcsr_ls/par_rap_communication.c\
 $(top_srcdir)/parcsr_ls/par_relax.c\
 $(top_srcdir)/parcsr_ls/par_relax_interface.c\
 $(top_srcdir)/parcsr_ls/par_scaled_matnorm.c\
 $(top_srcdir)/parcsr_ls/par_schwarz.c\
 $(top_srcdir)/parcsr_ls/par_stats.c\
 $(top_srcdir)/parcsr_ls/par_strength.c\
 $(top_srcdir)/parcsr_ls/pcg_par.c\
 $(top_srcdir)/parcsr_ls/schwarz.c\
 $(top_srcdir)/parcsr_mv/F90_HYPRE_parcsr_matrix.c\
 $(top_srcdir)/parcsr_mv/F90_HYPRE_parcsr_vector.c\
 $(top_srcdir)/parcsr_mv/F90_par_vector.c\
 $(top_srcdir)/parcsr_mv/F90_parcsr_matrix.c\
 $(top_srcdir)/parcsr_mv/HYPRE_parcsr_matrix.c\
 $(top_srcdir)/parcsr_mv/HYPRE_parcsr_vector.c\
 $(top_srcdir)/parcsr_mv/communicationT.c\
 $(top_srcdir)/parcsr_mv/numbers.c\
 $(top_srcdir)/parcsr_mv/par_csr_aat.c\
 $(top_srcdir)/parcsr_mv/par_csr_bool_matop.c\
 $(top_srcdir)/parcsr_mv/par_csr_bool_matrix.c\
 $(top_srcdir)/parcsr_mv/par_csr_communication.c\
 $(top_srcdir)/parcsr_mv/par_csr_matop.c\
 $(top_srcdir)/parcsr_mv/par_csr_matrix.c\
 $(top_srcdir)/parcsr_mv/par_csr_matvec.c\
 $(top_srcdir)/parcsr_mv/par_vector.c\
 $(top_srcdir)/seq_mv/HYPRE_csr_matrix.c\
 $(top_srcdir)/seq_mv/HYPRE_mapped_matrix.c\
 $(top_srcdir)/seq_mv/HYPRE_vector.c\
 $(top_srcdir)/seq_mv/HYPRE_multiblock_matrix.c\
 $(top_srcdir)/seq_mv/csr_matop.c\
 $(top_srcdir)/seq_mv/csr_matrix.c\
 $(top_srcdir)/seq_mv/csr_matvec.c\
 $(top_srcdir)/seq_mv/genpart.c\
 $(top_srcdir)/seq_mv/mapped_matrix.c\
 $(top_srcdir)/seq_mv/multiblock_matrix.c\
 $(top_srcdir)/seq_mv/vector.c\
 $(top_srcdir)/sstruct_ls/HYPRE_sstruct_bicgstab.c\
 $(top_srcdir)/sstruct_ls/HYPRE_sstruct_gmres.c\
 $(top_srcdir)/sstruct_ls/HYPRE_sstruct_pcg.c\
 $(top_srcdir)/sstruct_ls/HYPRE_sstruct_split.c\
 $(top_srcdir)/sstruct_ls/HYPRE_sstruct_sys_pfmg.c\
 $(top_srcdir)/sstruct_ls/krylov.c\
 $(top_srcdir)/sstruct_ls/krylov_sstruct.c\
 $(top_srcdir)/sstruct_ls/node_relax.c\
 $(top_srcdir)/sstruct_ls/sys_pfmg.c\
 $(top_srcdir)/sstruct_ls/sys_pfmg_relax.c\
 $(top_srcdir)/sstruct_ls/sys_pfmg_setup.c\
 $(top_srcdir)/sstruct_ls/sys_pfmg_setup_interp.c\
 $(top_srcdir)/sstruct_ls/sys_pfmg_setup_rap.c\
 $(top_srcdir)/sstruct_ls/sys_pfmg_solve.c\
 $(top_srcdir)/sstruct_ls/sys_semi_interp.c\
 $(top_srcdir)/sstruct_ls/sys_semi_restrict.c\
 $(top_srcdir)/sstruct_mv/HYPRE_sstruct_graph.c\
 $(top_srcdir)/sstruct_mv/HYPRE_sstruct_grid.c\
 $(top_srcdir)/sstruct_mv/HYPRE_sstruct_matrix.c\
 $(top_srcdir)/sstruct_mv/HYPRE_sstruct_stencil.c\
 $(top_srcdir)/sstruct_mv/HYPRE_sstruct_vector.c\
 $(top_srcdir)/sstruct_mv/box_map.c\
 $(top_srcdir)/sstruct_mv/sstruct_axpy.c\
 $(top_srcdir)/sstruct_mv/sstruct_copy.c\
 $(top_srcdir)/sstruct_mv/sstruct_graph.c\
 $(top_srcdir)/sstruct_mv/sstruct_grid.c\
 $(top_srcdir)/sstruct_mv/sstruct_innerprod.c\
 $(top_srcdir)/sstruct_mv/sstruct_matrix.c\
 $(top_srcdir)/sstruct_mv/sstruct_matvec.c\
 $(top_srcdir)/sstruct_mv/sstruct_scale.c\
 $(top_srcdir)/sstruct_mv/sstruct_stencil.c\
 $(top_srcdir)/sstruct_mv/sstruct_vector.c\
 $(top_srcdir)/struct_ls/F90_HYPRE_struct_hybrid.c\
 $(top_srcdir)/struct_ls/F90_HYPRE_struct_pcg.c\
 $(top_srcdir)/struct_ls/F90_HYPRE_struct_pfmg.c\
 $(top_srcdir)/struct_ls/F90_HYPRE_struct_smg.c\
 $(top_srcdir)/struct_ls/HYPRE_struct_bicgstab.c\
 $(top_srcdir)/struct_ls/HYPRE_struct_gmres.c\
 $(top_srcdir)/struct_ls/HYPRE_struct_hybrid.c\
 $(top_srcdir)/struct_ls/HYPRE_struct_jacobi.c\
 $(top_srcdir)/struct_ls/HYPRE_struct_pcg.c\
 $(top_srcdir)/struct_ls/HYPRE_struct_pfmg.c\
 $(top_srcdir)/struct_ls/HYPRE_struct_smg.c\
 $(top_srcdir)/struct_ls/HYPRE_struct_sparse_msg.c\
 $(top_srcdir)/struct_ls/coarsen.c\
 $(top_srcdir)/struct_ls/cyclic_reduction.c\
 $(top_srcdir)/struct_ls/general.c\
 $(top_srcdir)/struct_ls/hybrid.c\
 $(top_srcdir)/struct_ls/jacobi.c\
 $(top_srcdir)/struct_ls/pcg_struct.c\
 $(top_srcdir)/struct_ls/pfmg.c\
 $(top_srcdir)/struct_ls/pfmg2_setup_rap.c\
 $(top_srcdir)/struct_ls/pfmg3_setup_rap.c\
 $(top_srcdir)/struct_ls/pfmg_relax.c\
 $(top_srcdir)/struct_ls/pfmg_setup.c\
 $(top_srcdir)/struct_ls/pfmg_setup_interp.c\
 $(top_srcdir)/struct_ls/pfmg_setup_rap.c\
 $(top_srcdir)/struct_ls/pfmg_setup_rap5.c\
 $(top_srcdir)/struct_ls/pfmg_setup_rap7.c\
 $(top_srcdir)/struct_ls/pfmg_solve.c\
 $(top_srcdir)/struct_ls/point_relax.c\
 $(top_srcdir)/struct_ls/semi_interp.c\
 $(top_srcdir)/struct_ls/semi_restrict.c\
 $(top_srcdir)/struct_ls/semi_setup_rap.c\
 $(top_srcdir)/struct_ls/smg.c\
 $(top_srcdir)/struct_ls/smg2_setup_rap.c\
 $(top_srcdir)/struct_ls/smg3_setup_rap.c\
 $(top_srcdir)/struct_ls/smg_axpy.c\
 $(top_srcdir)/struct_ls/smg_relax.c\
 $(top_srcdir)/struct_ls/smg_residual.c\
 $(top_srcdir)/struct_ls/smg_setup.c\
 $(top_srcdir)/struct_ls/smg_setup_interp.c\
 $(top_srcdir)/struct_ls/smg_setup_rap.c\
 $(top_srcdir)/struct_ls/smg_setup_restrict.c\
 $(top_srcdir)/struct_ls/smg_solve.c\
 $(top_srcdir)/struct_ls/sparse_msg.c\
 $(top_srcdir)/struct_ls/sparse_msg2_setup_rap.c\
 $(top_srcdir)/struct_ls/sparse_msg3_setup_rap.c\
 $(top_srcdir)/struct_ls/sparse_msg_filter.c\
 $(top_srcdir)/struct_ls/sparse_msg_interp.c\
 $(top_srcdir)/struct_ls/sparse_msg_restrict.c\
 $(top_srcdir)/struct_ls/sparse_msg_setup.c\
 $(top_srcdir)/struct_ls/sparse_msg_setup_rap.c\
 $(top_srcdir)/struct_ls/sparse_msg_solve.c\
 $(top_srcdir)/struct_mv/F90_HYPRE_struct_grid.c\
 $(top_srcdir)/struct_mv/F90_HYPRE_struct_matrix.c\
 $(top_srcdir)/struct_mv/F90_HYPRE_struct_stencil.c\
 $(top_srcdir)/struct_mv/F90_HYPRE_struct_vector.c\
 $(top_srcdir)/struct_mv/HYPRE_struct_grid.c\
 $(top_srcdir)/struct_mv/HYPRE_struct_matrix.c\
 $(top_srcdir)/struct_mv/HYPRE_struct_stencil.c\
 $(top_srcdir)/struct_mv/HYPRE_struct_vector.c\
 $(top_srcdir)/struct_mv/box.c\
 $(top_srcdir)/struct_mv/box_algebra.c\
 $(top_srcdir)/struct_mv/box_alloc.c\
 $(top_srcdir)/struct_mv/box_neighbors.c\
 $(top_srcdir)/struct_mv/grow.c\
 $(top_srcdir)/struct_mv/communication_info.c\
 $(top_srcdir)/struct_mv/computation.c\
 $(top_srcdir)/struct_mv/project.c\
 $(top_srcdir)/struct_mv/struct_axpy.c\
 $(top_srcdir)/struct_mv/struct_communication.c\
 $(top_srcdir)/struct_mv/struct_copy.c\
 $(top_srcdir)/struct_mv/struct_grid.c\
 $(top_srcdir)/struct_mv/struct_io.c\
 $(top_srcdir)/struct_mv/struct_innerprod.c\
 $(top_srcdir)/struct_mv/struct_matrix.c\
 $(top_srcdir)/struct_mv/struct_matrix_mask.c\
 $(top_srcdir)/struct_mv/struct_matvec.c\
 $(top_srcdir)/struct_mv/struct_scale.c\
 $(top_srcdir)/struct_mv/struct_stencil.c\
 $(top_srcdir)/struct_mv/struct_vector.c\
 $(top_srcdir)/utilities/amg_linklist.c\
 $(top_srcdir)/utilities/binsearch.c\
 $(top_srcdir)/utilities/hypre_memory.c\
 $(top_srcdir)/utilities/hypre_qsort.c\
 $(top_srcdir)/utilities/memory_dmalloc.c\
 $(top_srcdir)/utilities/mpistubs.c\
 $(top_srcdir)/utilities/qsplit.c\
 $(top_srcdir)/utilities/random.c\
 $(top_srcdir)/utilities/thread_mpistubs.c\
 $(top_srcdir)/utilities/threading.c\
 $(top_srcdir)/utilities/timer.c\
 $(top_srcdir)/utilities/timing.c\
 $(top_srcdir)/utilities/umalloc_local.c

BUILT_SOURCES = hypre-stamp

clean-local:
	rm -f hypre-stamp

hypre-stamp:
	@ \
	touch hypre-stamp; \
	mkdir .libs; \
	for i in ${HYPRE_SRC_DIRS}; \
	do \
	  if [ -d ../$$i ]; \
	  then \
	    for j1 in ../$$i/*.lo; \
	    do \
	      test -f $$j1 && cp -f $$j1 .; \
	    done; \
	    for j2 in ../$$i/.libs/*.o; \
	    do \
	      test -f $$j2 && cp -f $$j2 .libs; \
	    done; \
	    for j3 in ../$$i/*.o; \
	    do \
	      test -f $$j3 && cp -f $$j3 .; \
	    done; \
	  fi; \
	done


