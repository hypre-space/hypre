#ifndef HYPRE_MATVEC_FUNCTIONS
#define HYPRE_MATVEC_FUNCTIONS

typedef struct
{
  void*  (*MatvecCreate)  ( void *A, void *x );
  int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
  int    (*MatvecDestroy) ( void *matvec_data );

  void*  (*MatMultiVecCreate)  ( void *A, void *x );
  int    (*MatMultiVec)        ( void *data, double alpha, void *A,
				 void *x, double beta, void *y );
  int    (*MatMultiVecDestroy)  ( void *data );

} HYPRE_MatvecFunctions;

#endif
