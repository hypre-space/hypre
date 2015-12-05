#ifndef HYPRE_INTERFACE_INTERPRETER
#define HYPRE_INTERFACE_INTERPRETER

#include "utilities.h"

typedef struct
{
  char*  (*CAlloc)        ( int count, int elt_size );
  int    (*Free)          ( char *ptr );
  int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs );

  /* vector operations */

  void*  (*CreateVector)  ( void *vector );
  int    (*DestroyVector) ( void *vector );

  void*  (*MatvecCreate)  ( void *A, void *x );
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

  /* multivector operations */

  void*  (*CreateMultiVector)  ( void*, int n, void *vector );
  void*  (*CopyCreateMultiVector)  ( void *x, int );
  void    (*DestroyMultiVector) ( void *x );

  void*  (*MatMultiVecCreate)  ( void *A, void *x );
  int    (*MatMultiVec)        ( void *data, double alpha, void *A,
				 void *x, double beta, void *y );
  int    (*MatMultiVecDestroy)  ( void *data );

  int    (*Width)  ( void *x );
  int    (*Height) ( void *x );

  void   (*SetMask) ( void *x, int *mask );

  void   (*CopyMultiVector)    ( void *x, void *y );
  void   (*ClearMultiVector)   ( void *x );
  void   (*SetRandomVectors)   ( void *x, int seed );
  void   (*MultiInnerProd)     ( void *x, void *y, int, int, int, double* );
  void   (*MultiInnerProdDiag) ( void *x, void *y, int*, int, double* );
  void   (*MultiVecMat)        ( void *x, int, int, int, double*, void *y );
  void   (*MultiVecMatDiag)    ( void *x, int*, int, double*, void *y );
  void   (*MultiAxpy)          ( double alpha, void *x, void *y );
  void   (*MultiXapy)          ( void *x, int, int, int, double*, void *y );
  void   (*Eval)               ( void (*f)( void*, void*, void* ), void*, void *x, void *y );

  int    (*PrintMultiVector)   ( void* x, const char* file );
  void*  (*ReadMultiVector)    ( MPI_Comm, void*, const char* file );

} HYPRE_InterfaceInterpreter;

#endif
