#include "headers.h" 

int matrix_inverse(double *X,

		   double *A,

		   int n)

{
  int ierr = 0;

  int i,j,k;
  int matz = 1;

  double *W, *Aux1, *Aux2;
  double *Q; 

  double diag;
  double eps = 1.e-3;

  Q   = hypre_CTAlloc(double, n*n);
  W   = hypre_CTAlloc(double, n);
  Aux1= hypre_CTAlloc(double, n);
  Aux2= hypre_CTAlloc(double, n);
  

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
      
