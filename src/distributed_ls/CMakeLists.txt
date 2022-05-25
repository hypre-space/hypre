# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS "")

add_subdirectory(Euclid)
add_subdirectory(ParaSails)
add_subdirectory(pilut)

set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
