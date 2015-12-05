cmake_minimum_required (VERSION 2.8.8)
project (hypre-test)

if (${hypre-test_SOURCE_DIR} STREQUAL ${hypre-test_BINARY_DIR})
  message(FATAL_ERROR "In-place build not allowed! Please use a separate build directory. See the Users Manual or INSTALL file for details.")
endif ()

# Set the installation directory for hypre
set (HYPRE_INSTALL_DIR "${PROJECT_SOURCE_DIR}/../hypre" CACHE PATH
  "Installation directory for HYPRE")

# Set default compile optimization flag
set (CMAKE_BUILD_TYPE "RELEASE" CACHE INTERNAL "" FORCE)

# Include directories
set (HYPRE_INCLUDE_DIRS "${HYPRE_INSTALL_DIR}/include")

# Library directories
set (HYPRE_LIB_DIRS "${HYPRE_INSTALL_DIR}/lib")

# Link libraries
set (HYPRE_LIBS HYPRE)

# Link libraries for Unix systems
if (UNIX)
  list (APPEND HYPRE_LIBS m)
endif (UNIX)

# CDEFS = -DHYPRE_TIMING -DHYPRE_FORTRAN
# CXXDEFS = -DNOFEI -DHYPRE_TIMING -DMPICH_SKIP_MPICXX
# -lstdc++ -lm

# Set MPI compile flags
if (NOT HYPRE_SEQUENTIAL)
  find_package (MPI)
  if ((MPI_C_FOUND) AND (NOT CMAKE_C_COMPILER STREQUAL MPI_C_COMPILER))
    include_directories (${MPI_C_INCLUDE_PATH})
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_COMPILE_FLAGS}")
    list (APPEND HYPRE_LIBS ${MPI_C_LIBRARIES})
  endif ()
endif (NOT HYPRE_SEQUENTIAL)

include_directories (${HYPRE_INCLUDE_DIRS})
link_directories (${HYPRE_LIB_DIRS})
add_definitions (-DHYPRE_TIMING)

if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()

add_executable (ij ij.c)
target_link_libraries (ij ${HYPRE_LIBS})
 
add_executable (new_ij new_ij.c)
target_link_libraries (new_ij ${HYPRE_LIBS})
 
add_executable (sstruct sstruct.c)
target_link_libraries (sstruct ${HYPRE_LIBS})
 
add_executable (struct struct.c)
target_link_libraries (struct ${HYPRE_LIBS})
 
add_executable (ams_driver ams_driver.c)
target_link_libraries (ams_driver ${HYPRE_LIBS})
 
add_executable (maxwell_unscaled maxwell_unscaled.c)
target_link_libraries (maxwell_unscaled ${HYPRE_LIBS})
 
add_executable (sstruct_fac sstruct_fac.c)
target_link_libraries (sstruct_fac ${HYPRE_LIBS})
 
add_executable (ij_mv ij_mv.c)
target_link_libraries (ij_mv ${HYPRE_LIBS})
 
