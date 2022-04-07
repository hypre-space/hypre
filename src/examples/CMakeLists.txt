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

