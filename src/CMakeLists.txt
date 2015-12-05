cmake_minimum_required (VERSION 2.8.8)
project (hypre)

# The version number.
set (HYPRE_VERSION 2.10.0b)
set (HYPRE_DATE    2015/01/22)
set (HYPRE_TIME    00:00:00)
set (HYPRE_BUGS    hypre-support@llnl.gov)
set (HYPRE_SRCDIR  "${PROJECT_SOURCE_DIR}")

if (${hypre_SOURCE_DIR} STREQUAL ${hypre_BINARY_DIR})
  message(FATAL_ERROR "In-place build not allowed! Please use a separate build directory. See the Users Manual or INSTALL file for details.")
endif ()

# Set default installation directory, but provide a means for users to change
set (HYPRE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/hypre" CACHE PATH
  "Installation directory for HYPRE")
set (CMAKE_INSTALL_PREFIX "${HYPRE_INSTALL_PREFIX}" CACHE INTERNAL "" FORCE)

# Set default compile optimization flag
set (HYPRE_BUILD_TYPE "Release" CACHE STRING
  "Optimization flags: set to Debug, Release, RelWithDebInfo, or MinSizeRel")
set (CMAKE_BUILD_TYPE "${HYPRE_BUILD_TYPE}" CACHE INTERNAL "" FORCE)

# Configuration options
option(HYPRE_SHARED               "Build a shared library" OFF)
option(HYPRE_BIGINT               "Use long long int for HYPRE_Int" OFF)
option(HYPRE_SEQUENTIAL           "Compile without MPI" OFF)
option(HYPRE_TIMING               "Use HYPRE timing routines" OFF)
option(HYPRE_USING_HYPRE_BLAS     "Use internal BLAS library" ON)
option(HYPRE_USING_HYPRE_LAPACK   "Use internal LAPACK library" ON)
option(HYPRE_NO_GLOBAL_PARTITION  "Use assumed partition" ON)
option(HYPRE_PRINT_ERRORS         "Print HYPRE errors" OFF)
option(HYPRE_USING_OPENMP         "Use OpenMP" OFF)
option(HYPRE_USING_FEI            "Use FEI" ON)

if (HYPRE_SHARED)
  # FEI doesn't currently compile with shared
  set (HYPRE_USING_FEI OFF CACHE BOOL "" FORCE)
endif ()

if (HYPRE_BIGINT)
  set (HYPRE_USING_FEI OFF CACHE BOOL "" FORCE)
  # In the future, also turn off babel
endif ()

if (HYPRE_SEQUENTIAL)
  set (HYPRE_NO_GLOBAL_PARTITION OFF CACHE BOOL "" FORCE)
endif ()

#   --with-examples         Build example codes.
#   --with-fei              Use internal FEI routines.
#   --with-superlu          Use internal SuperLU routines.
#   --with-mli              Use MLI

# Headers and sources
set (HYPRE_HEADERS "")
set (HYPRE_SOURCES "")

# Headers and sources: .
list (APPEND HYPRE_HEADERS
  "${PROJECT_BINARY_DIR}/HYPRE_config.h"
  HYPREf.h
  HYPRE.h
)

# Headers and sources: blas
if (HYPRE_USING_HYPRE_BLAS)
list (APPEND HYPRE_SOURCES
  blas/blas_utils.c
  blas/dasum.c
  blas/daxpy.c
  blas/dcopy.c
  blas/ddot.c
  blas/dgemm.c
  blas/dgemv.c
  blas/dger.c
  blas/dnrm2.c
  blas/drot.c
  blas/dscal.c
  blas/dswap.c
  blas/dsymm.c
  blas/dsymv.c
  blas/dsyr2.c
  blas/dsyr2k.c
  blas/dsyrk.c
  blas/dtrmm.c
  blas/dtrmv.c
  blas/dtrsm.c
  blas/dtrsv.c
  blas/idamax.c
)
endif (HYPRE_USING_HYPRE_BLAS)

# Headers and sources: lapack
if (HYPRE_USING_HYPRE_LAPACK)
list (APPEND HYPRE_SOURCES
  lapack/dbdsqr.c
  lapack/dgebd2.c
  lapack/dgebrd.c
  lapack/dgelq2.c
  lapack/dgelqf.c
  lapack/dgels.c
  lapack/dgeqr2.c
  lapack/dgeqrf.c
  lapack/dgesvd.c
  lapack/dgetrf.c
  lapack/dgetrs.c
  lapack/dgetf2.c
  lapack/dlabad.c
  lapack/dlabrd.c
  lapack/dlacpy.c
  lapack/dlae2.c
  lapack/dlaev2.c
  lapack/dlamch.c
  lapack/dlange.c
  lapack/dlanst.c
  lapack/dlansy.c
  lapack/dlapy2.c
  lapack/dlarfb.c
  lapack/dlarf.c
  lapack/dlarfg.c
  lapack/dlarft.c
  lapack/dlartg.c
  lapack/dlas2.c
  lapack/dlascl.c
  lapack/dlaset.c
  lapack/dlasq1.c
  lapack/dlasq2.c
  lapack/dlasq3.c
  lapack/dlasq4.c
  lapack/dlasq5.c
  lapack/dlasq6.c
  lapack/dlasr.c
  lapack/dlasrt.c
  lapack/dlassq.c
  lapack/dlaswp.c
  lapack/dlasv2.c
  lapack/dlatrd.c
  lapack/dorg2l.c
  lapack/dorg2r.c
  lapack/dorgbr.c
  lapack/dorgl2.c
  lapack/dorglq.c
  lapack/dorgql.c
  lapack/dorgqr.c
  lapack/dorgtr.c
  lapack/dorm2r.c
  lapack/dormbr.c
  lapack/dorml2.c
  lapack/dormlq.c
  lapack/dormqr.c
  lapack/dpotf2.c
  lapack/dpotrf.c
  lapack/dpotrs.c
  lapack/dsteqr.c
  lapack/dsterf.c
  lapack/dsyev.c
  lapack/dsygs2.c
  lapack/dsygst.c
  lapack/dsygv.c
  lapack/dsytd2.c
  lapack/dsytrd.c
  lapack/ieeeck.c
  lapack/ilaenv.c
  lapack/lapack_utils.c
  lapack/lsame.c
  lapack/xerbla.c
)
endif (HYPRE_USING_HYPRE_LAPACK)

# Headers and sources: utilities
list (APPEND HYPRE_HEADERS
  utilities/HYPRE_utilities.h
  utilities/_hypre_utilities.h
  utilities/HYPRE_error_f.h
  utilities/fortran.h
  utilities/fortran_matrix.h
)
list (APPEND HYPRE_SOURCES
  utilities/amg_linklist.c
  utilities/binsearch.c
  utilities/exchange_data.c
  utilities/F90_HYPRE_error.c
  utilities/fortran_matrix.c
  utilities/hypre_ap.c
  utilities/hypre_complex.c
  utilities/hypre_error.c
  utilities/hypre_memory.c
  utilities/hypre_mpi_comm_f2c.c
  utilities/hypre_printf.c
  utilities/hypre_qsort.c
  utilities/memory_dmalloc.c
  utilities/mpistubs.c
  utilities/qsplit.c
  utilities/random.c
  utilities/threading.c
  utilities/timer.c
  utilities/timing.c
  utilities/umalloc_local.c
)

# Headers and sources: multivector
list (APPEND HYPRE_HEADERS
  multivector/csr_matmultivec.h
  multivector/interpreter.h
  multivector/multivector.h
  multivector/par_csr_matmultivec.h
  multivector/par_csr_pmvcomm.h
  multivector/par_multivector.h
  multivector/seq_multivector.h
  multivector/temp_multivector.h
)
list (APPEND HYPRE_SOURCES
  multivector/multivector.c
  multivector/temp_multivector.c
)

# Headers and sources: krylov
list (APPEND HYPRE_HEADERS
  krylov/HYPRE_krylov.h
  krylov/HYPRE_lobpcg.h
  krylov/HYPRE_MatvecFunctions.h
  krylov/krylov.h
  krylov/lobpcg.h
)
list (APPEND HYPRE_SOURCES
  krylov/bicgstab.c
  krylov/cgnr.c
  krylov/gmres.c
  krylov/flexgmres.c
  krylov/lgmres.c
  krylov/HYPRE_bicgstab.c
  krylov/HYPRE_cgnr.c
  krylov/HYPRE_gmres.c
  krylov/HYPRE_lgmres.c
  krylov/HYPRE_flexgmres.c
  krylov/HYPRE_pcg.c
  krylov/pcg.c
  krylov/HYPRE_lobpcg.c
  krylov/lobpcg.c
)

# Headers and sources: seq_mv
list (APPEND HYPRE_HEADERS
  seq_mv/HYPRE_seq_mv.h
  seq_mv/seq_mv.h
)
list (APPEND HYPRE_SOURCES
  seq_mv/csr_matop.c
  seq_mv/csr_matrix.c
  seq_mv/csr_matvec.c
  seq_mv/genpart.c
  seq_mv/HYPRE_csr_matrix.c
  seq_mv/HYPRE_mapped_matrix.c
  seq_mv/HYPRE_multiblock_matrix.c
  seq_mv/HYPRE_vector.c
  seq_mv/mapped_matrix.c
  seq_mv/multiblock_matrix.c
  seq_mv/vector.c
)

# Headers and sources: parcsr_mv
list (APPEND HYPRE_HEADERS
  parcsr_mv/HYPRE_parcsr_mv.h
  parcsr_mv/_hypre_parcsr_mv.h
)
list (APPEND HYPRE_SOURCES
  parcsr_mv/communicationT.c
  parcsr_mv/F90_HYPRE_parcsr_matrix.c
  parcsr_mv/F90_HYPRE_parcsr_vector.c
  parcsr_mv/F90_parcsr_matrix.c
  parcsr_mv/F90_par_vector.c
  parcsr_mv/HYPRE_parcsr_matrix.c
  parcsr_mv/HYPRE_parcsr_vector.c
  parcsr_mv/new_commpkg.c
  parcsr_mv/numbers.c
  parcsr_mv/par_csr_aat.c
  parcsr_mv/par_csr_assumed_part.c
  parcsr_mv/par_csr_bool_matop.c
  parcsr_mv/par_csr_bool_matrix.c
  parcsr_mv/par_csr_communication.c
  parcsr_mv/par_csr_matop.c
  parcsr_mv/par_csr_matrix.c
  parcsr_mv/par_csr_matop_marked.c
  parcsr_mv/par_csr_matvec.c
  parcsr_mv/par_vector.c
  parcsr_mv/par_make_system.c
)

# Headers and sources: parcsr_block_mv
list (APPEND HYPRE_HEADERS
  parcsr_block_mv/par_csr_block_matrix.h
  parcsr_block_mv/csr_block_matrix.h
)
list (APPEND HYPRE_SOURCES
  parcsr_block_mv/csr_block_matrix.c
  parcsr_block_mv/csr_block_matvec.c
  parcsr_block_mv/par_csr_block_matrix.c
  parcsr_block_mv/par_csr_block_matvec.c
  parcsr_block_mv/par_csr_block_comm.c
  parcsr_block_mv/par_csr_block_rap.c
  parcsr_block_mv/par_csr_block_rap_communication.c
  parcsr_block_mv/par_csr_block_interp.c
  parcsr_block_mv/par_csr_block_relax.c
  parcsr_block_mv/par_block_nodal_systems.c
)

# Headers and sources: distributed_matrix
list (APPEND HYPRE_HEADERS
  distributed_matrix/distributed_matrix.h
)
list (APPEND HYPRE_SOURCES
  distributed_matrix/distributed_matrix.c
  distributed_matrix/HYPRE_distributed_matrix.c
  distributed_matrix/distributed_matrix_ISIS.c
  distributed_matrix/distributed_matrix_PETSc.c
  distributed_matrix/distributed_matrix_parcsr.c
)

# Headers and sources: IJ_mv
list (APPEND HYPRE_HEADERS
  IJ_mv/HYPRE_IJ_mv.h
  IJ_mv/_hypre_IJ_mv.h
)
list (APPEND HYPRE_SOURCES
  IJ_mv/aux_parcsr_matrix.c
  IJ_mv/aux_par_vector.c
  IJ_mv/F90_HYPRE_IJMatrix.c
  IJ_mv/F90_HYPRE_IJVector.c
  IJ_mv/F90_IJMatrix.c
  IJ_mv/HYPRE_IJMatrix.c
  IJ_mv/HYPRE_IJVector.c
  IJ_mv/IJMatrix.c
  IJ_mv/IJMatrix_parcsr.c
  IJ_mv/IJVector.c
  IJ_mv/IJVector_parcsr.c
)

# Headers and sources: matrix_matrix
list (APPEND HYPRE_HEADERS
  matrix_matrix/HYPRE_matrix_matrix_protos.h
)
list (APPEND HYPRE_SOURCES
  matrix_matrix/HYPRE_ConvertParCSRMatrixToDistributedMatrix.c
  matrix_matrix/HYPRE_ConvertPETScMatrixToDistributedMatrix.c
)

# Headers and sources: distributed_ls/pilut
list (APPEND HYPRE_HEADERS
  distributed_ls/pilut/HYPRE_DistributedMatrixPilutSolver_protos.h
  distributed_ls/pilut/HYPRE_DistributedMatrixPilutSolver_types.h
)
list (APPEND HYPRE_SOURCES
  distributed_ls/pilut/comm.c
  distributed_ls/pilut/debug.c
  distributed_ls/pilut/distributed_qsort.c
  distributed_ls/pilut/distributed_qsort_si.c
  distributed_ls/pilut/HYPRE_DistributedMatrixPilutSolver.c
  distributed_ls/pilut/ilut.c
  distributed_ls/pilut/parilut.c
  distributed_ls/pilut/parutil.c
  distributed_ls/pilut/pblas1.c
  distributed_ls/pilut/serilut.c
  distributed_ls/pilut/trifactor.c
  distributed_ls/pilut/util.c
)

# Headers and sources: distributed_ls/ParaSails
list (APPEND HYPRE_SOURCES
  distributed_ls/ParaSails/ConjGrad.c
  distributed_ls/ParaSails/DiagScale.c
  distributed_ls/ParaSails/FGmres.c
  distributed_ls/ParaSails/Hash.c
  distributed_ls/ParaSails/hypre_ParaSails.c
  distributed_ls/ParaSails/LoadBal.c
  distributed_ls/ParaSails/Matrix.c
  distributed_ls/ParaSails/Mem.c
  distributed_ls/ParaSails/Numbering.c
  distributed_ls/ParaSails/OrderStat.c
  distributed_ls/ParaSails/ParaSails.c
  distributed_ls/ParaSails/PrunedRows.c
  distributed_ls/ParaSails/RowPatt.c
  distributed_ls/ParaSails/StoredRows.c
)

# Headers and sources: distributed_ls/Euclid
list (APPEND HYPRE_SOURCES
  distributed_ls/Euclid/blas_dh.c
  distributed_ls/Euclid/Euclid_apply.c
  distributed_ls/Euclid/Euclid_dh.c
  distributed_ls/Euclid/ExternalRows_dh.c
  distributed_ls/Euclid/Factor_dh.c
  distributed_ls/Euclid/getRow_dh.c
  distributed_ls/Euclid/globalObjects.c
  distributed_ls/Euclid/Hash_dh.c
  distributed_ls/Euclid/Hash_i_dh.c
  distributed_ls/Euclid/ilu_mpi_bj.c
  distributed_ls/Euclid/ilu_mpi_pilu.c
  distributed_ls/Euclid/ilu_seq.c
  distributed_ls/Euclid/io_dh.c
  distributed_ls/Euclid/krylov_dh.c
  distributed_ls/Euclid/Mat_dh.c
  distributed_ls/Euclid/mat_dh_private.c
  distributed_ls/Euclid/MatGenFD.c
  distributed_ls/Euclid/Mem_dh.c
  distributed_ls/Euclid/Numbering_dh.c
  distributed_ls/Euclid/Parser_dh.c
  distributed_ls/Euclid/shellSort_dh.c
  distributed_ls/Euclid/sig_dh.c
  distributed_ls/Euclid/SortedList_dh.c
  distributed_ls/Euclid/SortedSet_dh.c
  distributed_ls/Euclid/SubdomainGraph_dh.c
  distributed_ls/Euclid/TimeLog_dh.c
  distributed_ls/Euclid/Timer_dh.c
  distributed_ls/Euclid/Vec_dh.c
)

# Headers and sources: parcsr_ls
list (APPEND HYPRE_HEADERS
  parcsr_ls/HYPRE_parcsr_ls.h
  parcsr_ls/_hypre_parcsr_ls.h
)
list (APPEND HYPRE_SOURCES
  parcsr_ls/amg_hybrid.c
  parcsr_ls/aux_interp.c
  parcsr_ls/F90_hypre_laplace.c
  parcsr_ls/F90_HYPRE_parcsr_amg.c
  parcsr_ls/F90_HYPRE_parcsr_bicgstab.c
  parcsr_ls/F90_HYPRE_parcsr_block.c
  parcsr_ls/F90_HYPRE_parcsr_cgnr.c
  parcsr_ls/F90_HYPRE_parcsr_Euclid.c
  parcsr_ls/F90_HYPRE_parcsr_gmres.c
  parcsr_ls/F90_HYPRE_parcsr_flexgmres.c
  parcsr_ls/F90_HYPRE_parcsr_lgmres.c
  parcsr_ls/F90_HYPRE_parcsr_hybrid.c
  parcsr_ls/F90_HYPRE_parcsr_int.c
  parcsr_ls/F90_HYPRE_parcsr_ParaSails.c
  parcsr_ls/F90_HYPRE_parcsr_pcg.c
  parcsr_ls/F90_HYPRE_parcsr_pilut.c
  parcsr_ls/F90_HYPRE_parcsr_schwarz.c
  parcsr_ls/F90_HYPRE_ams.c
  parcsr_ls/gen_redcs_mat.c
  parcsr_ls/HYPRE_parcsr_amg.c
  parcsr_ls/HYPRE_parcsr_bicgstab.c
  parcsr_ls/HYPRE_parcsr_block.c
  parcsr_ls/HYPRE_parcsr_cgnr.c
  parcsr_ls/HYPRE_parcsr_Euclid.c
  parcsr_ls/HYPRE_parcsr_gmres.c
  parcsr_ls/HYPRE_parcsr_flexgmres.c
  parcsr_ls/HYPRE_parcsr_lgmres.c
  parcsr_ls/HYPRE_parcsr_hybrid.c
  parcsr_ls/HYPRE_parcsr_int.c
  parcsr_ls/HYPRE_parcsr_ParaSails.c
  parcsr_ls/HYPRE_parcsr_pcg.c
  parcsr_ls/HYPRE_parcsr_pilut.c
  parcsr_ls/HYPRE_parcsr_schwarz.c
  parcsr_ls/HYPRE_ams.c
  parcsr_ls/HYPRE_ads.c
  parcsr_ls/HYPRE_ame.c
  parcsr_ls/par_amg.c
  parcsr_ls/par_amg_setup.c
  parcsr_ls/par_amg_solve.c
  parcsr_ls/par_amg_solveT.c
  parcsr_ls/par_cg_relax_wt.c
  parcsr_ls/par_coarsen.c
  parcsr_ls/par_cgc_coarsen.c
  parcsr_ls/par_coarse_parms.c
  parcsr_ls/par_coordinates.c
  parcsr_ls/par_cr.c
  parcsr_ls/par_cycle.c
  parcsr_ls/par_add_cycle.c
  parcsr_ls/par_difconv.c
  parcsr_ls/par_gsmg.c
  parcsr_ls/par_indepset.c
  parcsr_ls/par_interp.c
  parcsr_ls/par_jacobi_interp.c
  parcsr_ls/par_multi_interp.c
  parcsr_ls/par_laplace_27pt.c
  parcsr_ls/par_laplace_9pt.c
  parcsr_ls/par_laplace.c
  parcsr_ls/par_lr_interp.c
  parcsr_ls/par_nongalerkin.c
  parcsr_ls/par_nodal_systems.c
  parcsr_ls/par_rap.c
  parcsr_ls/par_rap_communication.c
  parcsr_ls/par_rotate_7pt.c
  parcsr_ls/par_vardifconv.c
  parcsr_ls/par_relax.c
  parcsr_ls/par_relax_more.c
  parcsr_ls/par_relax_interface.c
  parcsr_ls/par_scaled_matnorm.c
  parcsr_ls/par_schwarz.c
  parcsr_ls/par_stats.c
  parcsr_ls/par_strength.c
  parcsr_ls/par_sv_interp.c
  parcsr_ls/par_sv_interp_ln.c
  parcsr_ls/partial.c
  parcsr_ls/pcg_par.c
  parcsr_ls/schwarz.c
  parcsr_ls/block_tridiag.c
  parcsr_ls/ams.c
  parcsr_ls/ads.c
  parcsr_ls/ame.c
)

# Headers and sources: struct_mv
list (APPEND HYPRE_HEADERS
  struct_mv/HYPRE_struct_mv.h
  struct_mv/_hypre_struct_mv.h
)
list (APPEND HYPRE_SOURCES
  struct_mv/assumed_part.c
  struct_mv/box_algebra.c
  struct_mv/box_boundary.c
  struct_mv/box.c
  struct_mv/box_manager.c
  struct_mv/communication_info.c
  struct_mv/computation.c
  struct_mv/F90_HYPRE_struct_grid.c
  struct_mv/F90_HYPRE_struct_matrix.c
  struct_mv/F90_HYPRE_struct_stencil.c
  struct_mv/F90_HYPRE_struct_vector.c
  struct_mv/HYPRE_struct_grid.c
  struct_mv/HYPRE_struct_matrix.c
  struct_mv/HYPRE_struct_stencil.c
  struct_mv/HYPRE_struct_vector.c
  struct_mv/project.c
  struct_mv/struct_axpy.c
  struct_mv/struct_communication.c
  struct_mv/struct_copy.c
  struct_mv/struct_grid.c
  struct_mv/struct_innerprod.c
  struct_mv/struct_io.c
  struct_mv/struct_matrix.c
  struct_mv/struct_matrix_mask.c
  struct_mv/struct_matvec.c
  struct_mv/struct_scale.c
  struct_mv/struct_stencil.c
  struct_mv/struct_vector.c
)

# Headers and sources: struct_ls
list (APPEND HYPRE_HEADERS
  struct_ls/HYPRE_struct_ls.h
  struct_ls/_hypre_struct_ls.h
)
list (APPEND HYPRE_SOURCES
  struct_ls/coarsen.c
  struct_ls/cyclic_reduction.c
  struct_ls/F90_HYPRE_struct_bicgstab.c
  struct_ls/F90_HYPRE_struct_gmres.c
  struct_ls/F90_HYPRE_struct_hybrid.c
  struct_ls/F90_HYPRE_struct_int.c
  struct_ls/F90_HYPRE_struct_jacobi.c
  struct_ls/F90_HYPRE_struct_pcg.c
  struct_ls/F90_HYPRE_struct_pfmg.c
  struct_ls/F90_HYPRE_struct_smg.c
  struct_ls/F90_HYPRE_struct_sparse_msg.c
  struct_ls/general.c
  struct_ls/hybrid.c
  struct_ls/HYPRE_struct_bicgstab.c
  struct_ls/HYPRE_struct_hybrid.c
  struct_ls/HYPRE_struct_int.c
  struct_ls/HYPRE_struct_jacobi.c
  struct_ls/HYPRE_struct_pfmg.c
  struct_ls/HYPRE_struct_smg.c
  struct_ls/HYPRE_struct_sparse_msg.c
  struct_ls/HYPRE_struct_pcg.c
  struct_ls/HYPRE_struct_gmres.c
  struct_ls/HYPRE_struct_flexgmres.c
  struct_ls/HYPRE_struct_lgmres.c
  struct_ls/jacobi.c
  struct_ls/pcg_struct.c
  struct_ls/pfmg2_setup_rap.c
  struct_ls/pfmg3_setup_rap.c
  struct_ls/pfmg.c
  struct_ls/pfmg_relax.c
  struct_ls/pfmg_setup.c
  struct_ls/pfmg_setup_interp.c
  struct_ls/pfmg_setup_rap5.c
  struct_ls/pfmg_setup_rap7.c
  struct_ls/pfmg_setup_rap.c
  struct_ls/pfmg_solve.c
  struct_ls/point_relax.c
  struct_ls/red_black_constantcoef_gs.c
  struct_ls/red_black_gs.c
  struct_ls/semi.c
  struct_ls/semi_interp.c
  struct_ls/semi_restrict.c
  struct_ls/semi_setup_rap.c
  struct_ls/smg2_setup_rap.c
  struct_ls/smg3_setup_rap.c
  struct_ls/smg_axpy.c
  struct_ls/smg.c
  struct_ls/smg_relax.c
  struct_ls/smg_residual.c
  struct_ls/smg_setup.c
  struct_ls/smg_setup_interp.c
  struct_ls/smg_setup_rap.c
  struct_ls/smg_setup_restrict.c
  struct_ls/smg_solve.c
  struct_ls/sparse_msg2_setup_rap.c
  struct_ls/sparse_msg3_setup_rap.c
  struct_ls/sparse_msg.c
  struct_ls/sparse_msg_filter.c
  struct_ls/sparse_msg_interp.c
  struct_ls/sparse_msg_restrict.c
  struct_ls/sparse_msg_setup.c
  struct_ls/sparse_msg_setup_rap.c
  struct_ls/sparse_msg_solve.c
)

# Headers and sources: sstruct_mv
list (APPEND HYPRE_HEADERS
  sstruct_mv/HYPRE_sstruct_mv.h
  sstruct_mv/_hypre_sstruct_mv.h
)
list (APPEND HYPRE_SOURCES
  sstruct_mv/F90_HYPRE_sstruct_graph.c
  sstruct_mv/F90_HYPRE_sstruct_grid.c
  sstruct_mv/F90_HYPRE_sstruct_matrix.c
  sstruct_mv/F90_HYPRE_sstruct_stencil.c
  sstruct_mv/F90_HYPRE_sstruct_vector.c
  sstruct_mv/HYPRE_sstruct_graph.c
  sstruct_mv/HYPRE_sstruct_grid.c
  sstruct_mv/HYPRE_sstruct_matrix.c
  sstruct_mv/HYPRE_sstruct_stencil.c
  sstruct_mv/HYPRE_sstruct_vector.c
  sstruct_mv/sstruct_axpy.c
  sstruct_mv/sstruct_copy.c
  sstruct_mv/sstruct_graph.c
  sstruct_mv/sstruct_grid.c
  sstruct_mv/sstruct_innerprod.c
  sstruct_mv/sstruct_matrix.c
  sstruct_mv/sstruct_matvec.c
  sstruct_mv/sstruct_scale.c
  sstruct_mv/sstruct_stencil.c
  sstruct_mv/sstruct_vector.c
)

# Headers and sources: sstruct_ls
list (APPEND HYPRE_HEADERS
  sstruct_ls/HYPRE_sstruct_ls.h
  sstruct_ls/_hypre_sstruct_ls.h
)
list (APPEND HYPRE_SOURCES
  sstruct_ls/F90_HYPRE_sstruct_bicgstab.c
  sstruct_ls/F90_HYPRE_sstruct_gmres.c
  sstruct_ls/F90_HYPRE_sstruct_flexgmres.c
  sstruct_ls/F90_HYPRE_sstruct_lgmres.c
  sstruct_ls/F90_HYPRE_sstruct_InterFAC.c
  sstruct_ls/F90_HYPRE_sstruct_int.c
  sstruct_ls/F90_HYPRE_sstruct_maxwell.c
  sstruct_ls/F90_HYPRE_sstruct_pcg.c
  sstruct_ls/F90_HYPRE_sstruct_split.c
  sstruct_ls/F90_HYPRE_sstruct_sys_pfmg.c
  sstruct_ls/HYPRE_sstruct_bicgstab.c
  sstruct_ls/HYPRE_sstruct_gmres.c
  sstruct_ls/HYPRE_sstruct_flexgmres.c
  sstruct_ls/HYPRE_sstruct_lgmres.c
  sstruct_ls/HYPRE_sstruct_InterFAC.c
  sstruct_ls/HYPRE_sstruct_int.c
  sstruct_ls/HYPRE_sstruct_maxwell.c
  sstruct_ls/HYPRE_sstruct_pcg.c
  sstruct_ls/HYPRE_sstruct_split.c
  sstruct_ls/HYPRE_sstruct_sys_pfmg.c
  sstruct_ls/fac.c
  sstruct_ls/fac_amr_rap.c
  sstruct_ls/fac_amr_fcoarsen.c
  sstruct_ls/fac_amr_zero_data.c
  sstruct_ls/fac_cf_coarsen.c
  sstruct_ls/fac_cfstencil_box.c
  sstruct_ls/fac_CFInterfaceExtents.c
  sstruct_ls/fac_interp2.c
  sstruct_ls/fac_relax.c
  sstruct_ls/fac_restrict2.c
  sstruct_ls/fac_setup2.c
  sstruct_ls/fac_solve3.c
  sstruct_ls/fac_zero_cdata.c
  sstruct_ls/fac_zero_stencilcoef.c
  sstruct_ls/hypre_bsearch.c
  sstruct_ls/krylov.c
  sstruct_ls/krylov_sstruct.c
  sstruct_ls/eliminate_rowscols.c
  sstruct_ls/maxwell_grad.c
  sstruct_ls/maxwell_physbdy.c
  sstruct_ls/maxwell_PNedelec.c
  sstruct_ls/maxwell_PNedelec_bdy.c
  sstruct_ls/maxwell_semi_interp.c
  sstruct_ls/maxwell_TV.c
  sstruct_ls/maxwell_TV_setup.c
  sstruct_ls/maxwell_zeroBC.c
  sstruct_ls/hypre_MaxwellSolve.c
  sstruct_ls/hypre_MaxwellSolve2.c
  sstruct_ls/nd1_amge_interpolation.c
  sstruct_ls/node_relax.c
  sstruct_ls/sstruct_amr_intercommunication.c
  sstruct_ls/sstruct_owninfo.c
  sstruct_ls/sstruct_recvinfo.c
  sstruct_ls/sstruct_sendinfo.c
  sstruct_ls/sstruct_sharedDOFComm.c
  sstruct_ls/sys_pfmg.c
  sstruct_ls/sys_pfmg_relax.c
  sstruct_ls/sys_pfmg_setup.c
  sstruct_ls/sys_pfmg_setup_interp.c
  sstruct_ls/sys_pfmg_setup_rap.c
  sstruct_ls/sys_pfmg_solve.c
  sstruct_ls/sys_semi_interp.c
  sstruct_ls/sys_semi_restrict.c
)

# Configure a header file to pass CMake settings to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/config/HYPRE_config.h.cmake.in"
  "${PROJECT_BINARY_DIR}/HYPRE_config.h"
  )

# BINARY must be first in order to get the correct HYPRE_config.h file
include_directories(${hypre_BINARY_DIR})
include_directories(${hypre_SOURCE_DIR})
include_directories(utilities)
include_directories(multivector)
include_directories(krylov)
include_directories(seq_mv)
include_directories(parcsr_mv)
include_directories(parcsr_block_mv)
include_directories(distributed_matrix)
include_directories(IJ_mv)
include_directories(matrix_matrix)
include_directories(distributed_ls)
include_directories(distributed_ls/Euclid)
include_directories(distributed_ls/ParaSails)
include_directories(parcsr_ls)
include_directories(struct_mv)
include_directories(struct_ls)
include_directories(sstruct_mv)
include_directories(sstruct_ls)

# Set library build type
if (HYPRE_SHARED)
  set (BUILD_SHARED_LIBS ON CACHE INTERNAL "" FORCE)
else ()
  set (BUILD_SHARED_LIBS OFF CACHE INTERNAL "" FORCE)
endif ()

# Set MPI compile flags
if (NOT HYPRE_SEQUENTIAL)
  find_package (MPI)
  if ((MPI_C_FOUND) AND (NOT CMAKE_C_COMPILER STREQUAL MPI_C_COMPILER))
    include_directories (${MPI_C_INCLUDE_PATH})
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_COMPILE_FLAGS}")
  endif ()
  if ((MPI_CXX_FOUND) AND (NOT CMAKE_CXX_COMPILER STREQUAL MPI_CXX_COMPILER))
    include_directories (${MPI_CXX_INCLUDE_PATH})
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
  endif ()
endif (NOT HYPRE_SEQUENTIAL)

# Set OpenMP compile flags
if (HYPRE_USING_OPENMP)
  find_package (OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif (OPENMP_FOUND)
endif (HYPRE_USING_OPENMP)

# Turn optimization off for this file
set_source_files_properties (lapack/dlamch.c PROPERTIES COMPILE_FLAGS -O0)

if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()

if (HYPRE_USING_FEI)
  add_subdirectory(FEI_mv)
endif ()

add_library (HYPRE ${HYPRE_SOURCES} ${FEI_LIBS})

install (TARGETS HYPRE DESTINATION lib)
install (FILES ${HYPRE_HEADERS} DESTINATION include)

# add_subdirectory (test EXCLUDE_FROM_ALL)
 
# # does this system provide the log and exp functions?
# include (${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
#  
# check_function_exists (log HAVE_LOG)
# check_function_exists (exp HAVE_EXP)
 
# # add the MathFunctions library?
# if (USE_MYMATH)
#   include_directories ("${PROJECT_SOURCE_DIR}/MathFunctions")
#   add_subdirectory (MathFunctions)
#   set (EXTRA_LIBS ${EXTRA_LIBS} MathFunctions)
# endif (USE_MYMATH)
