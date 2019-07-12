#include "spkernels.h"

void allocLevel(int n, struct level_t *lev) {
  lev->nlevL = 0;
  lev->jlevL = (int *) malloc(n*sizeof(int));
  lev->ilevL = (int *) malloc(n*sizeof(int));
  lev->nlevU = 0;
  lev->jlevU = (int *) malloc(n*sizeof(int));
  lev->ilevU = (int *) malloc(n*sizeof(int));
  lev->levL = (int *) malloc(n*sizeof(int));
  lev->levU = (int *) malloc(n*sizeof(int));
}

/*-------------------------------------------------------------*/
/* arrays in h_lev should be allocated before calling */
void makeLevelCSR(int n, int *ia, int *ja, struct level_t *h_lev)
{
   int *level;

   memset(h_lev->ilevL, 0, n*sizeof(int));
   memset(h_lev->ilevU, 0, n*sizeof(int));

   // L
   level = h_lev->levL;

   h_lev->ilevL[0] = 0;

   for (int i = 0; i < n; i++)
   {
      int l = -1;
      for (int j = ia[i]; j < ia[i+1]; j++)
      {
         if (ja[j] < i)
         {
            l = max(l, level[ja[j]]);
         }
      }
      level[i] = l+1;
      h_lev->ilevL[l+1+1] ++;
      h_lev->nlevL = max(h_lev->nlevL, l+1+1);
   }

   for (int i = 1; i <= h_lev->nlevL; i++)
   {
      h_lev->ilevL[i] += h_lev->ilevL[i-1];
   }

   assert(h_lev->ilevL[h_lev->nlevL] == n);

   for (int i = 0; i < n; i++)
   {
      int *k = &h_lev->ilevL[level[i]];
      h_lev->jlevL[(*k)] = i;
      (*k)++;
   }

   for (int i = h_lev->nlevL-1; i > 0; i--)
   {
      h_lev->ilevL[i] = h_lev->ilevL[i-1];
   }

   h_lev->ilevL[0] = 0;

   // U
   level = h_lev->levU;

   h_lev->ilevU[0] = 0;

   for (int i = n-1; i >= 0; i--)
   {
      int l = -1;
      for (int j = ia[i]; j < ia[i+1]; j++)
      {
         if (ja[j] > i)
         {
            l = max(l, level[ja[j]]);
         }
      }

      level[i] = l+1;
      h_lev->ilevU[l+1+1] ++;
      h_lev->nlevU = max(h_lev->nlevU, l+1+1);
   }

   for (int i = 1; i <= h_lev->nlevU; i++)
   {
      h_lev->ilevU[i] += h_lev->ilevU[i-1];
   }

   assert(h_lev->ilevU[h_lev->nlevU] == n);

   for (int i = 0; i < n; i++)
   {
      int *k = &h_lev->ilevU[level[i]];
      h_lev->jlevU[(*k)] = i;
      (*k)++;
   }

   for (int i = h_lev->nlevU-1; i > 0; i--)
   {
      h_lev->ilevU[i] = h_lev->ilevU[i-1];
   }

   h_lev->ilevU[0] = 0;
}

/*-------------------------------*/
void FreeLev(struct level_t *h_lev)
{
  free(h_lev->jlevL);
  free(h_lev->ilevL);
  free(h_lev->jlevU);
  free(h_lev->ilevU);
  free(h_lev->levL);
  free(h_lev->levU);
}
