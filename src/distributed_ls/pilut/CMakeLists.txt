# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(PILUT_HDRS
  HYPRE_DistributedMatrixPilutSolver_protos.h
  HYPRE_DistributedMatrixPilutSolver_types.h
)

set(PILUT_SRCS
  comm.c
  debug.c
  distributed_qsort.c
  distributed_qsort_si.c
  HYPRE_DistributedMatrixPilutSolver.c
  ilut.c
  parilut.c
  parutil.c
  pblas1.c
  serilut.c
  trifactor.c
  util.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${PILUT_SRCS}
          ${PILUT_HDRS}
)

convert_filenames_to_full_paths(PILUT_HDRS)
set(HDRS ${HDRS} ${PILUT_HDRS} PARENT_SCOPE)
