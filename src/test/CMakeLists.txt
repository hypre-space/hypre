# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

add_definitions (-DHYPRE_TIMING)

if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()

set(TEST_SRCS
  ij.c
  sstruct.c
  struct.c
  ams_driver.c
  maxwell_unscaled.c
  struct_migrate.c
  sstruct_fac.c
  ij_mv.c
  ij_assembly.c
)

add_hypre_executables(TEST_SRCS)
 
