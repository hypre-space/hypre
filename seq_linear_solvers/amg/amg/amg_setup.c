 /* (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * AMG setup routine
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * HYPRE_AMGSetup
 *--------------------------------------------------------------------------*/

int      HYPRE_AMGSetup(A, data)
hypre_Matrix  *A;
void    *data;
{
   hypre_AMGData  *amg_data = data;

   int      num_variables;
   int      num_unknowns;
   int      num_points;
   int     *iu;
   int     *ip;
   int     *iv;
   
   int      num_levels;
   int      ndimu;
   int      ndimp;
   int      ndima;
   int      ndimb;
   hypre_Matrix  *P;
   int     *icdep;
   int     *imin;
   int     *imax;
   int     *ipmn;
   int     *ipmx;
   int     *icg;
   int     *ifg;
   hypre_Matrix **A_array;
   hypre_Matrix **P_array;
   hypre_VectorInt **IU_array;
   hypre_VectorInt **IP_array;
   hypre_VectorInt **IV_array;
   hypre_VectorInt **ICG_array;
   int     *leva;
   int     *levb;
   int     *levv;
   int     *levp;
   int     *levpi;
   int     *levi;
   int     *numa;
   int     *numb;
   int     *numv;
   int     *nump;
   
   double  *a;
   int     *ia;
   int     *ja; 
   
   double  *b;
   int     *ib;
   int     *jb;
   
   int      i, j, k;
   int      decr;

   double  *vtmp;
   hypre_Vector  *Vtemp;

   char     fnam[255];
   int Setup_err_flag;
   
   
   /*----------------------------------------------------------
    * Initialize problem part of hypre_AMGData
    *----------------------------------------------------------*/

   num_variables = hypre_MatrixSize(A);
   num_unknowns  = hypre_AMGDataNumUnknowns(amg_data);
   num_points    = hypre_AMGDataNumPoints(amg_data);
   iu   	 = hypre_AMGDataIU(amg_data);
   ip   	 = hypre_AMGDataIP(amg_data);
   iv   	 = hypre_AMGDataIV(amg_data);

   /* set default number of unknowns */
   if (!num_unknowns)
      num_unknowns = 1;

   /* set default number of points */
   if (!num_points)
   {
      if ((num_variables % num_unknowns) != 0)
      {
	 printf("Incompatible number of unknowns\n");
	 exit(1);
      }

      num_points = num_variables / num_unknowns;
   }

   /* set default variable/unknown/point mappings */
   if (!iu || !ip || !iv)
   {
/*****************
      if ((num_unknowns*num_points) != num_variables)
      {
	 printf("Incompatible number of unknowns and points\n");
	 exit(1);
      }
*****************/

      iu = hypre_CTAlloc(int, hypre_NDIMU(num_variables));
      ip = hypre_CTAlloc(int, hypre_NDIMU(num_variables));
      iv = hypre_CTAlloc(int, hypre_NDIMP(num_points+1));

      k = 0;
      for (i = 1; i <= num_points; i++)
	 for (j = 1; j <= num_unknowns; j++)
	 {
	    iu[k] = j;
	    ip[k] = i;
	    k++;
	 }
      i = 1;
      for (k = 0; k <= num_points; k++)
      {
	 iv[k] = i;
	 i += num_unknowns;
      }
   }


   hypre_AMGDataA(amg_data)            = A;
   hypre_AMGDataNumVariables(amg_data) = num_variables;
   hypre_AMGDataNumUnknowns(amg_data)  = num_unknowns;
   hypre_AMGDataNumPoints(amg_data)    = num_points;
   hypre_AMGDataIU(amg_data)    	 = iu;
   hypre_AMGDataIP(amg_data)    	 = ip;
   hypre_AMGDataIV(amg_data)    	 = iv;

   /*----------------------------------------------------------
    * Initialize remainder of hypre_AMGData
    *----------------------------------------------------------*/

   num_levels = hypre_AMGDataLevMax(amg_data);
   
   ndimu = hypre_NDIMU(num_variables);
   ndimp = hypre_NDIMP(num_points);
   ndima = hypre_NDIMA(hypre_MatrixIA(A)[num_variables]-1);
   ndimb = hypre_NDIMB(hypre_MatrixIA(A)[num_variables]-1);

   b  = hypre_CTAlloc(double, ndimb);
   ib = hypre_CTAlloc(int, ndimu);
   jb = hypre_CTAlloc(int, ndimb);
   P  = hypre_NewMatrix(b, ib, jb, num_variables);
   
   icdep = hypre_CTAlloc(int, num_levels*num_levels);
   imin  = hypre_CTAlloc(int, num_levels);
   imax  = hypre_CTAlloc(int, num_levels);
   ipmn  = hypre_CTAlloc(int, num_levels);
   ipmx  = hypre_CTAlloc(int, num_levels);
   icg   = hypre_CTAlloc(int, ndimu);
   ifg   = hypre_CTAlloc(int, ndimu);
   vtmp  = hypre_CTAlloc(double, num_variables);
   Vtemp = hypre_NewVector(vtmp,num_variables);
   
   /* set fine level point and variable bounds */
   ipmn[0] = 1;
   ipmx[0] = num_points;
   imin[0] = 1;
   imax[0] = num_variables;
   
   hypre_AMGDataNumLevels(amg_data) = num_levels;
   hypre_AMGDataNDIMU(amg_data)     = ndimu;
   hypre_AMGDataNDIMP(amg_data)     = ndimp;
   hypre_AMGDataNDIMA(amg_data)     = ndima;
   hypre_AMGDataNDIMB(amg_data)     = ndimb;   
   hypre_AMGDataP(amg_data)         = P;   
   hypre_AMGDataICDep(amg_data)     = icdep;
   hypre_AMGDataIMin(amg_data)      = imin;
   hypre_AMGDataIMax(amg_data)      = imax;
   hypre_AMGDataIPMN(amg_data)      = ipmn;
   hypre_AMGDataIPMX(amg_data)      = ipmx;
   hypre_AMGDataICG(amg_data)       = icg;
   hypre_AMGDataIFG(amg_data)       = ifg;

   hypre_AMGDataVecTemp(amg_data)   = vtmp;
   hypre_AMGDataVtemp(amg_data)     = Vtemp;
   
   /*----------------------------------------------------------
    * Call the setup phase code
    *----------------------------------------------------------*/

   hypre_WriteSetupParams(amg_data);   
   hypre_CALL_SETUP(Setup_err_flag, A, amg_data);
   if (Setup_err_flag != 0)
   {
      return(Setup_err_flag);
   }
   
   /*----------------------------------------------------------
    * Set some local variables
    *----------------------------------------------------------*/
   
   num_levels = hypre_AMGDataNumLevels(amg_data);
   
   a  = hypre_MatrixData(A);
   ia = hypre_MatrixIA(A);
   ja = hypre_MatrixJA(A);
   b  = hypre_MatrixData(P);
   ib = hypre_MatrixIA(P);
   jb = hypre_MatrixJA(P);
   
   /*----------------------------------------------------------
    * Create `lev' and `num' arrays
    *----------------------------------------------------------*/
   
   leva       = hypre_CTAlloc(int, num_levels);
   levb       = hypre_CTAlloc(int, num_levels);
   levv       = hypre_CTAlloc(int, num_levels);
   levp       = hypre_CTAlloc(int, num_levels);
   levpi      = hypre_CTAlloc(int, num_levels);
   levi       = hypre_CTAlloc(int, num_levels);
   numa       = hypre_CTAlloc(int, num_levels);
   numb       = hypre_CTAlloc(int, num_levels);
   numv       = hypre_CTAlloc(int, num_levels);
   nump       = hypre_CTAlloc(int, num_levels);
   for (j = 0; j < num_levels; j++)
   {
      leva[j]  = ia[imin[j]-1];
      levb[j]  = ib[imin[j]-1];
      levv[j]  = imin[j];
      levp[j]  = ipmn[j];
      levpi[j] = ipmn[j] + j;
      levi[j]  = imin[j] + j;
      numa[j]  = ia[imax[j]+1-1]-ia[imin[j]-1];
      numb[j]  = ib[imax[j]+1-1]-ib[imin[j]-1];
      numv[j]  = imax[j] - imin[j] + 1;
      nump[j]  = ipmx[j] - ipmn[j] + 1;
   }

/* fix for problem if iterative weights produces level with no
   rows */

   k = num_levels;
   for (j = k-1; j > 0; j--)
   {
       if (imax[j] == imax[j-1]) 
       {
           num_levels = j;
           hypre_AMGDataNumLevels(amg_data) = num_levels;
       }
   }

/* end iterative weight fix */ 

   hypre_AMGDataAArray(amg_data) = A_array;
   hypre_AMGDataPArray(amg_data) = P_array;
   hypre_AMGDataIUArray(amg_data) = IU_array;
   hypre_AMGDataIPArray(amg_data) = IP_array;
   hypre_AMGDataIVArray(amg_data) = IV_array;
   hypre_AMGDataICGArray(amg_data) = ICG_array;
   hypre_AMGDataLevA(amg_data)   = leva;
   hypre_AMGDataLevB(amg_data)   = levb;
   hypre_AMGDataLevV(amg_data)   = levv;
   hypre_AMGDataLevP(amg_data)   = levp;   
   hypre_AMGDataLevPI(amg_data)  = levpi;   
   hypre_AMGDataLevI(amg_data)   = levi;
   hypre_AMGDataNumA(amg_data)   = numa;
   hypre_AMGDataNumB(amg_data)   = numb;
   hypre_AMGDataNumV(amg_data)   = numv;
   hypre_AMGDataNumP(amg_data)   = nump;

   /*----------------------------------------------------------
    * Shift index arrays
    *----------------------------------------------------------*/
   
   /* make room for `num(j)+1' entry in `ia', `ib', and `iv' */
   for (j = num_levels-1; j > 0; j--)
   {
      for (k = numv[j]; k >= 0; k--)
      {
	 ia[levi[j]+k-1] = ia[levv[j]+k-1];
	 ib[levi[j]+k-1] = ib[levv[j]+k-1];
      }
      for (k = nump[j]; k >= 0; k--)
      {
	 iv[levpi[j]+k-1] = iv[levp[j]+k-1];
      }
   }
  
   /* shift `ja' and `jb' */
   decr = numv[0];
   for (j = 1; j < num_levels; j++)
   {
      for (k = leva[j]; k < leva[j] + numa[j]; k++)
      {
	 ja[k-1] -= decr;
      }
      decr += numv[j];
   }
   decr = numv[0];
   for (j = 1; j < num_levels; j++)
   {
      for (k = levb[j]; k < levb[j] + numb[j]; k++)
      {
	 jb[k-1] -= decr;
      }
      decr += numv[j];
   }
   
   /* shift `ia' and `ib' */
   decr = numa[0];
   for (j = 1; j < num_levels; j++)
   {
      for (k = levi[j]; k < levi[j] + numv[j] + 1; k++)
      {
	 ia[k-1] -= decr;
      }
      decr += numa[j];
   }
   decr = numb[0];
   for (j = 1; j < num_levels; j++)
   {
      for (k = levi[j]; k < levi[j] + numv[j] + 1; k++)
      {
	 ib[k-1] -= decr;
      }
      decr += numb[j];
   }
   
   /* shift `iv' */
   decr = numv[0];
   for (j = 1; j < num_levels; j++)
   {
      for (k = levpi[j]; k < levpi[j] + nump[j] + 1; k++)
      {
	 iv[k-1] -= decr;
      }
      decr += numv[j];
   }
   
   /* shift `ip' */
   decr = nump[0];
   for (j = 1; j < num_levels; j++)
   {
      for (k = levv[j]; k < levv[j] + numv[j]; k++)
      {
	 ip[k-1] -= decr;
      }
      decr += nump[j];
   }
   
   /* shift `icg' */
   decr = numv[0];
   for (j = 0; j < num_levels - 1; j++)
   {
      for (k = levv[j]; k < levv[j] + numv[j]; k++)
      {
	 icg[k-1] -= decr;
      }
      decr += numv[j+1];
   }
   
   /* shift `imin' and `imax' */
   decr = numv[0];  
   for (j = 1; j < num_levels; j++)
   {
      imin[j] -= decr;
      imax[j] -= decr;
      decr += numv[j];
   }
   
   /* shift `ipmn' and `ipmx' */
   decr = nump[0];  
   for (j = 1; j < num_levels; j++)
   {
      ipmn[j] -= decr;
      ipmx[j] -= decr;
      decr += nump[j];
   }

   /* shift `jb' array to allow use of matvec  */

   for (j = 0; j < num_levels; j++)
   {
     for (k = levb[j]; k < levb[j] + numb[j]; k++)
     {
         jb[k-1] = icg[jb[k-1]+levv[j]-2];
     }
   }  
  
   

 
 
   
   /*----------------------------------------------------------
    * Set up A_array and P_array
    *----------------------------------------------------------*/
   
   A_array = hypre_TAlloc(hypre_Matrix*, num_levels);
   P_array = hypre_TAlloc(hypre_Matrix*, num_levels-1);
   
   A_array[0] = A;
   P_array[0] = P;
   
   for (j = 1; j < num_levels; j++)
   {
      A_array[j] =
	 hypre_NewMatrix(&a[leva[j]-1], &ia[levi[j]-1], &ja[leva[j]-1], numv[j]);
   }
   
   for (j = 1; j < num_levels-1; j++)
   {
      P_array[j] =
	 hypre_NewMatrix(&b[levb[j]-1], &ib[levi[j]-1], &jb[levb[j]-1], numv[j]);
   }
   
   hypre_AMGDataAArray(amg_data) = A_array;
   hypre_AMGDataPArray(amg_data) = P_array;   
   

   /*----------------------------------------------------------
    * Set up  IU_array, IP_array, IV_array, and ICG_array
    *----------------------------------------------------------*/
   
   IU_array = hypre_TAlloc(hypre_VectorInt*, num_levels);
   IP_array = hypre_TAlloc(hypre_VectorInt*, num_levels);
   IV_array = hypre_TAlloc(hypre_VectorInt*, num_levels);
   ICG_array = hypre_TAlloc(hypre_VectorInt*, num_levels);
   
   for (j = 0; j < num_levels; j++)
   {
      IU_array[j] = hypre_NewVectorInt(&iu[levv[j]-1], numv[j]);
      IP_array[j] = hypre_NewVectorInt(&ip[levv[j]-1], numv[j]);
      IV_array[j] = hypre_NewVectorInt(&iv[levpi[j]-1],nump[j]+1);
      ICG_array[j] = hypre_NewVectorInt(&icg[levv[j]-1], numv[j]);
   }
   
   hypre_AMGDataIUArray(amg_data) = IU_array;
   hypre_AMGDataIPArray(amg_data) = IP_array;
   hypre_AMGDataIVArray(amg_data) = IV_array;
   hypre_AMGDataICGArray(amg_data) = ICG_array;

   if (hypre_AMGDataIOutDat(amg_data) <= -1)
   {
      for (j = 0; j < num_levels; j++)
      {
         sprintf(fnam,"A_%d.ysmp",j);
         hypre_WriteYSMP(fnam, A_array[j]);
      }
    }


   if (hypre_AMGDataIOutDat(amg_data) <= -2)
   {
      for (j=0; j < num_levels-1; j++)
      {
         sprintf(fnam,"P_%d.ysmp",j);
         hypre_WriteYSMP(fnam, P_array[j]);
      }
   }

   if (hypre_AMGDataIOutDat(amg_data) <= -3)
   {
      for (j=0; j < num_levels-1; j++)
      {
         sprintf(fnam,"ICG_%d.vec",j);
         hypre_WriteVecInt(fnam, ICG_array[j]);
      }
   }
   
   if (hypre_AMGDataIOutDat(amg_data) == -3)
            hypre_AMGDataIOutDat(amg_data) = 3;

   return(Setup_err_flag);
}


