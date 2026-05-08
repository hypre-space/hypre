# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# This handles the non-compiler aspect of the SYCL toolkit.
message(STATUS "Enabling SYCL toolkit")

# We enforce the use of Intel's oneAPI DPC++/C++ Compiler
if(NOT CMAKE_CXX_COMPILER MATCHES "dpcpp|icpx")
  message(FATAL_ERROR "SYCL requires DPC++ or Intel C++ compiler")
endif()

# limit C++ errors to one
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel|Clang")
  target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ferror-limit=1>)
endif()

# Find Intel SYCL
find_package(IntelSYCL REQUIRED)

# Sycl requires hypre streams
set(HYPRE_USING_CUDA_STREAMS ON CACHE BOOL "" FORCE)

# Set up SYCL flags
if(IntelSYCL_FOUND)
  # Standard SYCL flags
  target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fsycl>)
  target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fsycl-unnamed-lambda>)
  target_link_options(${PROJECT_NAME} PUBLIC -fsycl)
  target_link_options(${PROJECT_NAME} PUBLIC -fsycl-device-code-split=per_kernel)
  target_link_options(${PROJECT_NAME} PUBLIC -Wl,--no-relax)

  # Use either user-specified target or IntelSYCL's default
  if(HYPRE_SYCL_TARGET)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fsycl-targets=${HYPRE_SYCL_TARGET}>)
    target_link_options(${PROJECT_NAME} PUBLIC -fsycl-targets=${HYPRE_SYCL_TARGET})
  elseif(INTEL_SYCL_TARGETS)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fsycl-targets=${INTEL_SYCL_TARGETS}>)
    target_link_options(${PROJECT_NAME} PUBLIC -fsycl-targets=${INTEL_SYCL_TARGETS})
  endif()

  # Use either user-specified backend or IntelSYCL's default
  if(HYPRE_SYCL_TARGET_BACKEND)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Xsycl-target-backend=${HYPRE_SYCL_TARGET_BACKEND}>)
    target_link_options(${PROJECT_NAME} PUBLIC -Xsycl-target-backend=${HYPRE_SYCL_TARGET_BACKEND})
  elseif(INTEL_SYCL_BACKEND)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Xsycl-target-backend=${INTEL_SYCL_BACKEND}>)
    target_link_options(${PROJECT_NAME} PUBLIC -Xsycl-target-backend=${INTEL_SYCL_BACKEND})
  endif()
endif()

# Find Intel oneDPL
if(NOT DEFINED DPLROOT)
  if(DEFINED ENV{DPLROOT} AND EXISTS "$ENV{DPLROOT}")
    set(DPLROOT "$ENV{DPLROOT}")
  elseif(DEFINED ENV{ONEAPI_ROOT} AND EXISTS "$ENV{ONEAPI_ROOT}/dpl/latest")
    set(DPLROOT "$ENV{ONEAPI_ROOT}/dpl/latest")
  endif()
endif()

set(HYPRE_ONEDPL_HINTS)
if(DEFINED DPLROOT AND EXISTS "${DPLROOT}/lib/cmake/oneDPL")
  list(APPEND HYPRE_ONEDPL_HINTS "${DPLROOT}/lib/cmake/oneDPL")
endif()

find_package(oneDPL REQUIRED HINTS ${HYPRE_ONEDPL_HINTS})

# Check if DPL is found
if(NOT oneDPL_FOUND)
  message(FATAL_ERROR "Could not find oneDPL installation. Please set DPLROOT")
endif()

get_target_property(HYPRE_ONEDPL_INCLUDE_DIRS oneDPL INTERFACE_INCLUDE_DIRECTORIES)
if(HYPRE_ONEDPL_INCLUDE_DIRS)
  target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${HYPRE_ONEDPL_INCLUDE_DIRS})
  foreach(HYPRE_ONEDPL_INCLUDE_DIR IN LISTS HYPRE_ONEDPL_INCLUDE_DIRS)
    target_compile_options(${PROJECT_NAME} PUBLIC
      $<$<COMPILE_LANGUAGE:CXX>:-isystem>
      $<$<COMPILE_LANGUAGE:CXX>:${HYPRE_ONEDPL_INCLUDE_DIR}>)
  endforeach()
endif()

get_target_property(HYPRE_ONEDPL_COMPILE_DEFINITIONS oneDPL INTERFACE_COMPILE_DEFINITIONS)
if(HYPRE_ONEDPL_COMPILE_DEFINITIONS)
  target_compile_definitions(${PROJECT_NAME} PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${HYPRE_ONEDPL_COMPILE_DEFINITIONS}>)
endif()

get_target_property(HYPRE_ONEDPL_COMPILE_OPTIONS oneDPL INTERFACE_COMPILE_OPTIONS)
if(HYPRE_ONEDPL_COMPILE_OPTIONS)
  target_compile_options(${PROJECT_NAME} PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${HYPRE_ONEDPL_COMPILE_OPTIONS}>)
endif()

get_target_property(HYPRE_ONEDPL_LINK_LIBS oneDPL INTERFACE_LINK_LIBRARIES)
if(HYPRE_ONEDPL_LINK_LIBS)
  foreach(HYPRE_ONEDPL_LINK_ITEM IN LISTS HYPRE_ONEDPL_LINK_LIBS)
    target_link_libraries(${PROJECT_NAME} PUBLIC $<LINK_ONLY:${HYPRE_ONEDPL_LINK_ITEM}>)
  endforeach()
endif()

if (HYPRE_ENABLE_ONEMKLSPARSE)
  set(HYPRE_USING_ONEMKLSPARSE ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_ONEMKLBLAS)
  set(HYPRE_USING_ONEMKLBLAS ON CACHE BOOL "" FORCE)
endif()

if (HYPRE_ENABLE_ONEMKLRAND)
  set(HYPRE_USING_ONEMKLRAND ON CACHE BOOL "" FORCE)
endif()

# Setup MKL
if (HYPRE_USING_ONEMKLSPARSE OR HYPRE_USING_ONEMKLBLAS OR HYPRE_USING_ONEMKLRAND)
  # Find MKL
  set(MKL_LINK dynamic)
  set(MKL_THREADING sequential)
  #set(ENABLE_TRY_SYCL_COMPILE ON) # This option slows down the build
  find_package(MKL CONFIG REQUIRED HINTS "$ENV{MKLROOT}/lib/cmake/mkl")

  # Add all required MKL components explicitly
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
    $<LINK_ONLY:MKL::MKL_SYCL::BLAS>
    $<LINK_ONLY:MKL::MKL_SYCL::LAPACK>
    $<LINK_ONLY:MKL::MKL_SYCL::SPARSE>
    $<LINK_ONLY:MKL::MKL_SYCL::VM>
    $<LINK_ONLY:MKL::MKL_SYCL::RNG>
    $<LINK_ONLY:MKL::MKL_SYCL::STATS>
    $<LINK_ONLY:MKL::MKL_SYCL::DATA_FITTING>
  )

  # Ensure compile options and include directories are properly propagated
  target_compile_options(${PROJECT_NAME} PUBLIC
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
  )
  target_include_directories(${PROJECT_NAME} PUBLIC
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif()

# Set GPU warp size
set(HYPRE_WARP_SIZE 32 CACHE INTERNAL "GPU warp size")

# Patch Umpire's QuickPool C wrapper to use 256-byte alignment for SYCL.
# oneMKL requires 256-byte aligned USM pointers; Umpire's default is 16.
function(hypre_patch_umpire_sycl_pool_alignment umpire_src_dir)
  set(_wrap_rm "${umpire_src_dir}/src/umpire/interface/c_fortran/wrapResourceManager.cpp")
  if(NOT EXISTS "${_wrap_rm}")
    message(WARNING "Umpire wrapResourceManager.cpp not found at ${_wrap_rm} — skipping 256-byte alignment patch")
    return()
  endif()
  file(READ "${_wrap_rm}" _content)
  if(_content MATCHES "256UL")
    return()
  endif()
  set(_old "    *SHCXX_rv = SH_this->makeAllocator<umpire::strategy::QuickPool>(\n        SHCXX_name, *SHCXX_allocator, initial_size, block);\n    SHC_rv->addr = SHCXX_rv;\n    SHC_rv->idtor = 1;\n    return SHC_rv;\n    // splicer end class.ResourceManager.method.make_allocator_quick_pool")
  set(_new "#if defined(UMPIRE_ENABLE_SYCL)\n    *SHCXX_rv = SH_this->makeAllocator<umpire::strategy::QuickPool>(\n        SHCXX_name, *SHCXX_allocator, initial_size, block, 256UL);\n#else\n    *SHCXX_rv = SH_this->makeAllocator<umpire::strategy::QuickPool>(\n        SHCXX_name, *SHCXX_allocator, initial_size, block);\n#endif\n    SHC_rv->addr = SHCXX_rv;\n    SHC_rv->idtor = 1;\n    return SHC_rv;\n    // splicer end class.ResourceManager.method.make_allocator_quick_pool")
  string(REPLACE "${_old}" "${_new}" _content "${_content}")
  if(NOT _content MATCHES "256UL")
    message(WARNING "Failed to patch QuickPool alignment in ${_wrap_rm} — file format may have changed. Umpire allocations may not meet oneMKL's 256-byte alignment requirement.")
    return()
  endif()
  file(WRITE "${_wrap_rm}" "${_content}")
  message(STATUS "Patched Umpire wrapResourceManager.cpp: QuickPool SYCL pool alignment set to 256 bytes")
endfunction()
