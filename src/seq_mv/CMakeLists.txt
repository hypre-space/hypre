# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_seq_mv.h
  seq_mv.h
)

set(SRCS
  csr_matop.c
  csr_matrix.c
  csr_matvec.c
  genpart.c
  HYPRE_csr_matrix.c
  HYPRE_mapped_matrix.c
  HYPRE_multiblock_matrix.c
  HYPRE_vector.c
  mapped_matrix.c
  multiblock_matrix.c
  vector_batched.c
  csr_matop_device.c
  csr_matrix_cuda_utils.c
  csr_matvec_device.c
  csr_matvec_oomp.c
  csr_spadd_device.c
  csr_spgemm_device.c
  csr_spgemm_device_cusparse.c
  csr_spgemm_device_numblocks.c
  csr_spgemm_device_numer.c
  csr_spgemm_device_numer1.c
  csr_spgemm_device_numer2.c
  csr_spgemm_device_numer3.c
  csr_spgemm_device_numer4.c
  csr_spgemm_device_numer5.c
  csr_spgemm_device_numer6.c
  csr_spgemm_device_numer7.c
  csr_spgemm_device_numer8.c
  csr_spgemm_device_numer9.c
  csr_spgemm_device_numer10.c
  csr_spgemm_device_onemklsparse.c
  csr_spgemm_device_rocsparse.c
  csr_spgemm_device_rowest.c
  csr_spgemm_device_symbl.c
  csr_spgemm_device_symbl1.c
  csr_spgemm_device_symbl2.c
  csr_spgemm_device_symbl3.c
  csr_spgemm_device_symbl4.c
  csr_spgemm_device_symbl5.c
  csr_spgemm_device_symbl6.c
  csr_spgemm_device_symbl7.c
  csr_spgemm_device_symbl8.c
  csr_spgemm_device_symbl9.c
  csr_spgemm_device_symbl10.c
  csr_spgemm_device_util.c
  csr_spmv_device.c
  csr_sptrans_device.c
  vector.c
  vector_device.c
)

# Autogenerate csr_spgemm_device_numer$ files
file(READ "csr_spgemm_device_numer.in" CONTENTS)
foreach(number RANGE 1 10)
  # Check if file exists, then don't recreate
  set(fn "csr_spgemm_device_numer${number}.c")
  if (NOT EXISTS ${fn})
    file(WRITE ${fn} "#define HYPRE_SPGEMM_BIN ${number}")
    file(APPEND ${fn} "${CONTENTS}")
  endif (NOT EXISTS ${fn})
endforeach(number RANGE 1 10)

# Autogenerate csr_spgemm_device_symbl$ files
file(READ "csr_spgemm_device_symbl.in" CONTENTS)
foreach(number RANGE 1 10)
  set(fn "csr_spgemm_device_symbl${number}.c")
  if (NOT EXISTS ${fn})
    file(WRITE ${fn} "#define HYPRE_SPGEMM_BIN ${number}")
    file(APPEND ${fn} "${CONTENTS}")
  endif (NOT EXISTS ${fn})
endforeach(number RANGE 1 10)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

if (HYPRE_USING_CUDA OR HYPRE_USING_SYCL)
  set(GPU_SRCS
    csr_matop_device.c
    csr_matrix_cuda_utils.c
    csr_matvec_device.c
    csr_matvec_oomp.c
    csr_spadd_device.c
    csr_spgemm_device.c
    csr_spgemm_device_cusparse.c
    csr_spgemm_device_numblocks.c
    csr_spgemm_device_numer.c
    csr_spgemm_device_numer1.c
    csr_spgemm_device_numer2.c
    csr_spgemm_device_numer3.c
    csr_spgemm_device_numer4.c
    csr_spgemm_device_numer5.c
    csr_spgemm_device_numer6.c
    csr_spgemm_device_numer7.c
    csr_spgemm_device_numer8.c
    csr_spgemm_device_numer9.c
    csr_spgemm_device_numer10.c
    csr_spgemm_device_onemklsparse.c
    csr_spgemm_device_rocsparse.c
    csr_spgemm_device_rowest.c
    csr_spgemm_device_symbl.c
    csr_spgemm_device_symbl1.c
    csr_spgemm_device_symbl2.c
    csr_spgemm_device_symbl3.c
    csr_spgemm_device_symbl4.c
    csr_spgemm_device_symbl5.c
    csr_spgemm_device_symbl6.c
    csr_spgemm_device_symbl7.c
    csr_spgemm_device_symbl8.c
    csr_spgemm_device_symbl9.c
    csr_spgemm_device_symbl10.c
    csr_spgemm_device_util.c
    csr_spmv_device.c
    csr_sptrans_device.c
    vector_device.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
