#ifndef HYPRE_INTERFACE_INTERPRETER
#define HYPRE_INTERFACE_INTERPRETER

#include "utilities.h"

typedef struct
{
  char * (*CAlloc)        ( int count, int elt_size );
  int    (*Free)          ( char *ptr );
  int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs );
  void * (*CreateVector)  ( void *vector );
  int    (*DestroyVector) ( void *vector );
  void * (*MatvecCreate)  ( void *A, void *x );
  int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
  int    (*MatvecDestroy) ( void *matvec_data );
  double (*InnerProd)     ( void *x, void *y );
  int    (*CopyVector)    ( void *x, void *y );
  int    (*ClearVector)   ( void *x );
  int    (*SetRandomValues)   ( void *x, int seed );
  int    (*ScaleVector)   ( double alpha, void *x );
  int    (*Axpy)          ( double alpha, void *x, void *y );
  int    (*PrintVector)   ( void* x, const char* file );
  void*  (*ReadVector)    ( MPI_Comm, const char* file );

} HYPRE_InterfaceInterpreter;

#endif
