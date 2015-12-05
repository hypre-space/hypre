#include <stdlib.h>

#define abs(x) ((x)>0.0? (x):-(x))

#define z(i,j) z[(*ldz)*( (j) -1 )+ (i) - 1]
extern double  dlapy2_(double *, double *);
extern double  dnrm2_(int *, double *, int *);

void dnstres(int *n,         int *nev,    int *colptr, int *rowind, 
             double *nzvals, double *dr,  double *di,  double *z, 
             int *ldz,       double *res)
{
   int     i, first=1, ione=1, j;
   double  md, rnrm, *work;
 
   work = (double*)malloc((*n)*sizeof(double)); 

   for (i = 0; i<(*nev); i++) {
      if (di[i] == 0.0) {
         dmvm_(n, nzvals, rowind, colptr, &z(1,i+1), work, &ione);
         md = -dr[i];
         daxpy_(n, &md, &z(1,i+1), &ione, work, &ione);
         res[i] = dnrm2_(n, work, &ione);
         res[i] = res[i]/abs(dr[i]); 
      }
      else if (first) {
         dmvm_(n, nzvals, rowind, colptr, &z(1,i+1), work, &ione);
         md = -dr[i];
         daxpy_(n, &md,    &z(1,i+1),   &ione, work, &ione);
         daxpy_(n, &di[i], &z(1,i+2), &ione, work, &ione);
         res[i] = dnrm2_(n, work, &ione);
         dmvm_(n, nzvals, rowind, colptr, &z(1,i+2), work, &ione);
         md = -di[i];
         daxpy_(n, &md, &z(1,i+1),   &ione, work, &ione);
         md = -dr[i];
         daxpy_(n, &md, &z(1,i+2), &ione, work, &ione);
         rnrm = dnrm2_(n, work, &ione);
         res[i] = dlapy2_(&res[i], &rnrm);
         res[i] = res[i]/dlapy2_(&dr[i],&di[i]);  
         res[i+1] = res[i];
         first = 0; 
      }
      else {
         first = 1;
      }
   }
   free(work);
}
