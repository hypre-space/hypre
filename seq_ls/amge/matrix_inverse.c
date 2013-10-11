#include "headers.h" 

HYPRE_Int matrix_inverse(HYPRE_Real *X,

		   HYPRE_Real *A,

		   HYPRE_Int n)

{
  HYPRE_Int ierr = 0;

  HYPRE_Int i,j,k;
  HYPRE_Int matz = 1;

  HYPRE_Real *W, *Aux1, *Aux2;
  HYPRE_Real *Q; 

  HYPRE_Real diag;
  HYPRE_Real eps = 1.e-3;

  Q   = hypre_CTAlloc(HYPRE_Real, n*n);
  W   = hypre_CTAlloc(HYPRE_Real, n);
  Aux1= hypre_CTAlloc(HYPRE_Real, n);
  Aux2= hypre_CTAlloc(HYPRE_Real, n);
  

  if (n > 0)
    rs_(&n, &n, A, W, &matz, Q, Aux1, Aux2, &ierr);

  for (i=0; i < n; i++)
    for (j=0; j < n; j++)
      X[j+i*n] = 0.e0;

  for (k=0; k < n; k++)
    {
      if (W[k] < eps)
	diag = 0;
      else
	diag = 1.e0/W[k];

      for (i=0; i < n; i++)
	for (j=0; j < n; j++)
	  X[j+i*n] += Q[j+k*n] * diag * Q[i+k*n];
    }

  hypre_TFree(Q);
  hypre_TFree(W);
  hypre_TFree(Aux1);
  hypre_TFree(Aux2);

  return ierr;

}
      
