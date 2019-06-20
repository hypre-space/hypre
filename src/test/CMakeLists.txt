list(APPEND EXAMPLE_SRCS
  ij.c
  sstruct.c
  ams_driver.c
  maxwell_unscaled.c
  struct_migrate.c
  sstruct_fac.c
  ij_mv.c
  )

add_hypre_c_executables(EXAMPLE_SRCS)

