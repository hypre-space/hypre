#include <stdio.h>

extern int mli_computespectrum_(int *,int *,double *, double *, int *, double *,
                     double *, double *, int *);

main()
{
   int    i, mDim=24, ierr, matz=0;
   double *matrix, *evalues, *evectors, *daux1, *daux2;
   FILE   *fp;

   matrix = (double *) malloc( mDim * mDim * sizeof(double) );
   fp = fopen("test.m", "r");
   for ( i = 0; i < mDim*mDim; i++ )
      fscanf(fp, "%lg", &(matrix[i])); 
   evectors = (double *) malloc( mDim * mDim * sizeof(double) );
   evalues  = (double *) malloc( mDim * sizeof(double) );
   daux1    = (double *) malloc( mDim * sizeof(double) );
   daux2    = (double *) malloc( mDim * sizeof(double) );
/*
   for ( i = 0; i < mDim; i++ ) matrix[i*mDim+i] += 10.0;
*/
   mli_computespectrum_(&mDim, &mDim, matrix, evalues, &matz, evectors,
                        daux1, daux2, &ierr);
   for ( i = 0; i < mDim; i++ )
      printf("eigenvalue = %e\n", evalues[i]);
   free(matrix); 
   free(evectors); 
   free(evalues); 
   free(daux1); 
   free(daux2); 
}

