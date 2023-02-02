# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include_directories(fei-hypre)

# option(HYPRE_USING_SUPERLU "Use internal SuperLU routines" ON)
# option(HYPRE_USING_MLI     "Use MLI" ON)
# 
# if(HYPRE_USING_SUPERLU)
#   include_directories(SuperLU/SRC)
#   add_definitions(-DHAVE_SUPERLU)
#   add_subdirectory(SuperLU)
#   set(FEI_LIBS ${FEI_LIBS} $<TARGET_OBJECTS:HYPRE_superlu>)
# endif()
# 
# if(HYPRE_USING_MLI)
#   include_directories(femli)
#   add_definitions(-DHAVE_MLI)
#   if(HYPRE_USING_SUPERLU)
#     add_definitions(-DMLI_SUPERLU)
#   endif()
#   add_subdirectory(femli)
#   set(FEI_LIBS ${FEI_LIBS} $<TARGET_OBJECTS:HYPRE_mli>)
# endif()

add_subdirectory(fei-hypre)
