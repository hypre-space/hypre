
project(HYPRE_FEI C CXX)

include_directories(fei-base)
include_directories(fei-hypre)
include_directories(ml/src/include)

option(HYPRE_USING_SUPERLU "Use internal SuperLU routines" ON)
option(HYPRE_USING_MLI     "Use MLI" ON)

if(HYPRE_USING_SUPERLU)
  include_directories(SuperLU/SRC)
  add_definitions(-DHAVE_SUPERLU)
  add_subdirectory(SuperLU)
  set(FEI_LIBS ${FEI_LIBS} $<TARGET_OBJECTS:HYPRE_superlu>)
endif()

if(HYPRE_USING_MLI)
  include_directories(femli)
  add_definitions(-DHAVE_MLI)
  if(HYPRE_USING_SUPERLU)
    add_definitions(-DMLI_SUPERLU)
  endif()
  add_subdirectory(femli)
  set(FEI_LIBS ${FEI_LIBS} $<TARGET_OBJECTS:HYPRE_mli>)
endif()

add_subdirectory(fei-hypre)
set(FEI_LIBS ${FEI_LIBS} $<TARGET_OBJECTS:HYPRE_fei>)

set(FEI_LIBS ${FEI_LIBS} PARENT_SCOPE)
