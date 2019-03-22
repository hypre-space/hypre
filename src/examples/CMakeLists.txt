list(APPEND EXAMPLE_SRCS
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

add_hypre_c_executables(EXAMPLE_SRCS)

