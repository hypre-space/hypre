# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(EXAMPLE_SRCS
  ex1.c
  ex2.c
  ex3.c
  ex4.c
  ex5.c
  ex6.c
  ex7.c
  ex8.c
  ex9.c
  ex11.c
  ex12.c
  ex13.c
  ex14.c
  ex15.c
  ex16.c
  ex17.c
  ex18.c
)

if (HYPRE_BIGINT)
  list(APPEND EXAMPLE_SRCS
    ex5big.c
    ex15big.c
  )
endif()

if (HYPRE_COMPLEX)
  list(APPEND EXAMPLE_SRCS
    ex18comp.c
  )
endif()

add_hypre_executables(EXAMPLE_SRCS)

if(HYPRE_ENABLE_FORTRAN)
  # Pure Fortran-enabled examples: ex5f and ex12f
  # Note: CUDA/Fortran (cudaf.f, ex5f_cptr.f, ex12f_cptr.f)
  #       paths are handled by GPU toolchains (skipped here)
  add_executable(ex5f ex5f.f)
  target_link_libraries(ex5f PUBLIC HYPRE)
  if(HYPRE_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS Fortran)
    target_link_libraries(ex5f PUBLIC MPI::MPI_Fortran)
  endif()

  add_executable(ex12f ex12f.f)
  target_link_libraries(ex12f PUBLIC HYPRE)
  if(HYPRE_ENABLE_MPI)
    target_link_libraries(ex12f PUBLIC MPI::MPI_Fortran)
  endif()
endif()
