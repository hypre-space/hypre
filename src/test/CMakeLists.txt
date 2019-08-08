add_definitions (-DHYPRE_TIMING)

if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()

list(APPEND TEST_SRCS
  ij.c
  sstruct.c
  struct.c
  ams_driver.c
  maxwell_unscaled.c
  struct_migrate.c
  sstruct_fac.c
  ij_mv.c
)

add_hypre_executables(TEST_SRCS)
 
