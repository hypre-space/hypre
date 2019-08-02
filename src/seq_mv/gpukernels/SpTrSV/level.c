#include "spkernels.h"

void allocLevel(int n, struct level_t *lev)
{
   lev->nlevL = 0;
   lev->num_klevL = 0;
   lev->jlevL = (int *) malloc(n*sizeof(int));
   lev->ilevL = (int *) malloc((n+1)*sizeof(int));
   lev->klevL = (int *) malloc((n+1)*sizeof(int));

   lev->nlevU = 0;
   lev->num_klevU = 0;
   lev->jlevU = (int *) malloc(n*sizeof(int));
   lev->ilevU = (int *) malloc((n+1)*sizeof(int));
   lev->klevU = (int *) malloc((n+1)*sizeof(int));

   lev->levL = (int *) malloc(n*sizeof(int));
   lev->levU = (int *) malloc(n*sizeof(int));
}

/*-------------------------------------------------------------*/
/* arrays in h_lev should be allocated before calling */
void makeLevelCSR(int n, int *ia, int *ja, struct level_t *h_lev)
{
   int *level;

   memset(h_lev->ilevL, 0, (n+1)*sizeof(int));
   memset(h_lev->ilevU, 0, (n+1)*sizeof(int));

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

   /*------------------- make k-level*/
   int k = 0, pre_gDim = 0;
   for(int i = 0; i < h_lev->nlevL; i++ )
   {
      int l1 = h_lev->ilevL[i];
      int l2 = h_lev->ilevL[i+1];
      const int bDim = SPTRSV_BLOCKDIM;
      const HYPRE_Int group_size = 32;
      const HYPRE_Int num_groups_per_block = bDim / group_size;
      const HYPRE_Int gDim = (l2 - l1 + num_groups_per_block - 1) / num_groups_per_block;
      if (i == 0 || gDim > 1 || pre_gDim > 1)
      {
         h_lev->klevL[k++] = i;
      }
      pre_gDim = gDim;
   }
   h_lev->klevL[k] = h_lev->nlevL;

   /*
      for(int i = 0; i < h_lev->nlevL; i++)
      {
      h_lev->klevL[i]=0;
      }//set klev[] = 0
      int k = 1;
      for(int i = 0; i < h_lev->nlevL; i++ )
      {
      int l1 = h_lev->ilevL[i];
      int l2 = h_lev->ilevL[i+1];
      const int bDim = SPTRSV_BLOCKDIM;
      const HYPRE_Int group_size = 32;
      const HYPRE_Int num_groups_per_block = bDim / group_size;
      const HYPRE_Int gDim = (l2 - l1 + num_groups_per_block - 1) / num_groups_per_block;
      if (gDim == 1)
      {
      h_lev->klevL[k] ++;
   //  h_lev->block_klevL[k] = gDim;
   }
   else
   {
   //  h_lev->block_klevL[k+1] = gDim;
   h_lev->klevL[k+1] = h_lev->klevL[k]+1;
   h_lev->klevL[k+2] = h_lev->klevL[k]+2;
   k ++ ;
   }
   }
   */

   h_lev->num_klevL = k;

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

   /*------------------- make k-level*/
   k = 0;
   pre_gDim = 0;
   for(int i = 0; i < h_lev->nlevU; i++ )
   {
      int l1 = h_lev->ilevU[i];
      int l2 = h_lev->ilevU[i+1];
      const int bDim = SPTRSV_BLOCKDIM;
      const HYPRE_Int group_size = 32;
      const HYPRE_Int num_groups_per_block = bDim / group_size;
      const HYPRE_Int gDim = (l2 - l1 + num_groups_per_block - 1) / num_groups_per_block;
      if (i == 0 || gDim > 1 || pre_gDim > 1)
      {
         h_lev->klevU[k++] = i;
      }
      pre_gDim = gDim;
   }
   h_lev->klevU[k] = h_lev->nlevU;

   /*
      for(int i = 0; i < h_lev->nlevL; i++)
      {
      h_lev->klevL[i]=0;
      }//set klev[] = 0
      int k = 1;
      for(int i = 0; i < h_lev->nlevL; i++ )
      {
      int l1 = h_lev->ilevL[i];
      int l2 = h_lev->ilevL[i+1];
      const int bDim = SPTRSV_BLOCKDIM;
      const HYPRE_Int group_size = 32;
      const HYPRE_Int num_groups_per_block = bDim / group_size;
      const HYPRE_Int gDim = (l2 - l1 + num_groups_per_block - 1) / num_groups_per_block;
      if (gDim == 1)
      {
      h_lev->klevL[k] ++;
   //  h_lev->block_klevL[k] = gDim;
   }
   else
   {
   //  h_lev->block_klevL[k+1] = gDim;
   h_lev->klevL[k+1] = h_lev->klevL[k]+1;
   h_lev->klevL[k+2] = h_lev->klevL[k]+2;
   k ++ ;
   }
   }
   */

   h_lev->num_klevU = k;
}

/*-------------------------------*/
void FreeLev(struct level_t *h_lev)
{
  free(h_lev->jlevL);
  free(h_lev->ilevL);
  free(h_lev->klevL);
  free(h_lev->jlevU);
  free(h_lev->ilevU);
  free(h_lev->klevU);
  free(h_lev->levL);
  free(h_lev->levU);
}

