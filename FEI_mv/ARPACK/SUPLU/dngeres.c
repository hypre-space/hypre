#include <stdlib.h>

#define abs(x) ((x)>0.0? (x):-(x))
#define z(i,j) z[(*ldz)*( (j) -1 )+ (i) - 1]

extern double  dlapy2_(double *, double *);
extern double  dnrm2_(int *, double *, int *);

void dngeres(int *n,        int *nev,   int *aptr, int *aind, 
             double *avals, int *bptr,  int *bind, double *bvals,
             double *dr,    double *di, double *z, int *ldz,
             double *res)
{
   int     i, first=1, ione=1, j;
   double  md, rnrm, *ax, *bx;
 
   ax = (double*)malloc((*n)*sizeof(double)); 
   bx = (double*)malloc((*n)*sizeof(double)); 

   for (i = 0; i<(*nev); i++) {
      if (di[i] == 0.0) {
         dmvm_(n, avals, aind, aptr, &z(1,i+1), ax, &ione);
         dmvm_(n, bvals, bind, bptr, &z(1,i+1), bx, &ione);
         md = -dr[i];
         daxpy_(n, &md, bx, &ione, ax, &ione);
         res[i] = dnrm2_(n, ax, &ione);
         res[i] = res[i]/abs(dr[i]); 
      }
      else if (first) {
         dmvm_(n, avals, aind, aptr, &z(1,i+1), ax, &ione);
         dmvm_(n, bvals, bind, bptr, &z(1,i+1), bx, &ione);
         md = -dr[i];
         daxpy_(n, &md, bx, &ione, ax, &ione);
         dmvm_(n, bvals, bind, bptr, &z(1,i+2), bx, &ione);
         daxpy_(n, &di[i], bx, &ione, ax, &ione);
         rnrm = dnrm2_(n, ax, &ione);
         res[i] = rnrm*rnrm;

         dmvm_(n, avals, aind, aptr, &z(1,i+2), ax, &ione);
         dmvm_(n, bvals, bind, bptr, &z(1,i+2), bx, &ione);
         md = -dr[i];
         daxpy_(n, &md, bx, &ione, ax, &ione);
         dmvm_(n, bvals, bind, bptr, &z(1,i+1), bx, &ione);
         md = -di[i];
         daxpy_(n, &md, bx, &ione, ax, &ione);
         rnrm = dnrm2_(n, ax, &ione);
         res[i] = dlapy2_(&res[i], &rnrm);
         res[i] = res[i]/dlapy2_(&dr[i],&di[i]);  
         res[i+1] = res[i];
         first = 0; 
      }
      else {
         first = 1;
      }
   }
   free(ax);
   free(bx);
}
