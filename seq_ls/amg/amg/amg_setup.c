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
 * amg_Setup
 *--------------------------------------------------------------------------*/

int      amg_Setup(A, data)
Matrix  *A;
void    *data;
{
   AMGData  *amg_data = data;

   int      num_variables;
   int      num_unknowns;
   int      num_points;
   int     *iu;
   int     *ip;
   int     *iv;
   double  *xp;
   double  *yp;
   double  *zp;
   
   int      num_levels;
   int      ndimu;
   int      ndimp;
   int      ndima;
   int      ndimb;
   Matrix  *P;
   int     *icdep;
   int     *imin;
   int     *imax;
   int     *ipmn;
   int     *ipmx;
   int     *icg;
   int     *ifg;
   Matrix **A_array;
   Matrix **P_array;
   VectorInt **IU_array;
   VectorInt **IP_array;
   VectorInt **IV_array;
   VectorInt **ICG_array;
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
   Vector  *Vtemp;

   char     fnam[255];
   int Setup_err_flag;
   
   
   /*----------------------------------------------------------
    * Initialize problem part of AMGData
    *----------------------------------------------------------*/

   num_variables = MatrixSize(A);
   num_unknowns  = AMGDataNumUnknowns(amg_data);
   num_points    = AMGDataNumPoints(amg_data);
   iu   	 = AMGDataIU(amg_data);
   ip   	 = AMGDataIP(amg_data);
   iv   	 = AMGDataIV(amg_data);
   xp   	 = AMGDataXP(amg_data);
   yp   	 = AMGDataYP(amg_data);
   zp   	 = AMGDataZP(amg_data);

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
      if ((num_unknowns*num_points) != num_variables)
      {
	 printf("Incompatible number of unknowns and points\n");
	 exit(1);
      }

      iu = ctalloc(int, NDIMU(num_variables));
      ip = ctalloc(int, NDIMU(num_variables));
      iv = ctalloc(int, NDIMP(num_points+1));

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

   if (!xp || !yp || !zp)
   {
      xp = ctalloc(double, NDIMP(num_points));
      yp = ctalloc(double, NDIMP(num_points));
      zp = ctalloc(double, NDIMP(num_points));
   }

   AMGDataA(amg_data)            = A;
   AMGDataNumVariables(amg_data) = num_variables;
   AMGDataNumUnknowns(amg_data)  = num_unknowns;
   AMGDataNumPoints(amg_data)    = num_points;
   AMGDataIU(amg_data)    	 = iu;
   AMGDataIP(amg_data)    	 = ip;
   AMGDataIV(amg_data)    	 = iv;
   AMGDataXP(amg_data)    	 = xp;
   AMGDataYP(amg_data)    	 = yp;
   AMGDataZP(amg_data)    	 = zp;

   /*----------------------------------------------------------
    * Initialize remainder of AMGData
    *----------------------------------------------------------*/

   num_levels = AMGDataLevMax(amg_data);
   
   ndimu = NDIMU(num_variables);
   ndimp = NDIMP(num_points);
   ndima = NDIMA(MatrixIA(A)[num_variables]-1);
   ndimb = NDIMB(MatrixIA(A)[num_variables]-1);

   b  = ctalloc(double, ndimb);
   ib = ctalloc(int, ndimu);
   jb = ctalloc(int, ndimb);
   P  = NewMatrix(b, ib, jb, num_variables);
   
   icdep = ctalloc(int, num_levels*num_levels);
   imin  = ctalloc(int, num_levels);
   imax  = ctalloc(int, num_levels);
   ipmn  = ctalloc(int, num_levels);
   ipmx  = ctalloc(int, num_levels);
   icg   = ctalloc(int, ndimu);
   ifg   = ctalloc(int, ndimu);
   vtmp  = ctalloc(double, num_variables);
   Vtemp = NewVector(vtmp,num_variables);
   
   /* set fine level point and variable bounds */
   ipmn[0] = 1;
   ipmx[0] = num_points;
   imin[0] = 1;
   imax[0] = num_variables;
   
   AMGDataNumLevels(amg_data) = num_levels;
   AMGDataNDIMU(amg_data)     = ndimu;
   AMGDataNDIMP(amg_data)     = ndimp;
   AMGDataNDIMA(amg_data)     = ndima;
   AMGDataNDIMB(amg_data)     = ndimb;   
   AMGDataP(amg_data)         = P;   
   AMGDataICDep(amg_data)     = icdep;
   AMGDataIMin(amg_data)      = imin;
   AMGDataIMax(amg_data)      = imax;
   AMGDataIPMN(amg_data)      = ipmn;
   AMGDataIPMX(amg_data)      = ipmx;
   AMGDataICG(amg_data)       = icg;
   AMGDataIFG(amg_data)       = ifg;

   AMGDataVecTemp(amg_data)   = vtmp;
   AMGDataVtemp(amg_data)     = Vtemp;
   
   /*----------------------------------------------------------
    * Call the setup phase code
    *----------------------------------------------------------*/

   WriteSetupParams(amg_data);   
   CALL_SETUP(Setup_err_flag, A, amg_data);
   if (Setup_err_flag != 0)
   {
      return(Setup_err_flag);
   }
   
   /*----------------------------------------------------------
    * Set some local variables
    *----------------------------------------------------------*/
   
   num_levels = AMGDataNumLevels(amg_data);
   
   a  = MatrixData(A);
   ia = MatrixIA(A);
   ja = MatrixJA(A);
   b  = MatrixData(P);
   ib = MatrixIA(P);
   jb = MatrixJA(P);
   
   /*----------------------------------------------------------
    * Create `lev' and `num' arrays
    *----------------------------------------------------------*/
   
   leva       = ctalloc(int, num_levels);
   levb       = ctalloc(int, num_levels);
   levv       = ctalloc(int, num_levels);
   levp       = ctalloc(int, num_levels);
   levpi      = ctalloc(int, num_levels);
   levi       = ctalloc(int, num_levels);
   numa       = ctalloc(int, num_levels);
   numb       = ctalloc(int, num_levels);
   numv       = ctalloc(int, num_levels);
   nump       = ctalloc(int, num_levels);
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

   AMGDataAArray(amg_data) = A_array;
   AMGDataPArray(amg_data) = P_array;
   AMGDataIUArray(amg_data) = IU_array;
   AMGDataIPArray(amg_data) = IP_array;
   AMGDataIVArray(amg_data) = IV_array;
   AMGDataICGArray(amg_data) = ICG_array;
   AMGDataLevA(amg_data)   = leva;
   AMGDataLevB(amg_data)   = levb;
   AMGDataLevV(amg_data)   = levv;
   AMGDataLevP(amg_data)   = levp;   
   AMGDataLevPI(amg_data)  = levpi;   
   AMGDataLevI(amg_data)   = levi;
   AMGDataNumA(amg_data)   = numa;
   AMGDataNumB(amg_data)   = numb;
   AMGDataNumV(amg_data)   = numv;
   AMGDataNumP(amg_data)   = nump;

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
   
   A_array = talloc(Matrix*, num_levels);
   P_array = talloc(Matrix*, num_levels-1);
   
   A_array[0] = A;
   P_array[0] = P;
   
   for (j = 1; j < num_levels; j++)
   {
      A_array[j] =
	 NewMatrix(&a[leva[j]-1], &ia[levi[j]-1], &ja[leva[j]-1], numv[j]);
   }
   
   for (j = 1; j < num_levels-1; j++)
   {
      P_array[j] =
	 NewMatrix(&b[levb[j]-1], &ib[levi[j]-1], &jb[levb[j]-1], numv[j]);
   }
   
   AMGDataAArray(amg_data) = A_array;
   AMGDataPArray(amg_data) = P_array;   
   

   /*----------------------------------------------------------
    * Set up  IU_array, IP_array, IV_array, and ICG_array
    *----------------------------------------------------------*/
   
   IU_array = talloc(VectorInt*, num_levels);
   IP_array = talloc(VectorInt*, num_levels);
   IV_array = talloc(VectorInt*, num_levels);
   ICG_array = talloc(VectorInt*, num_levels);
   
   for (j = 0; j < num_levels; j++)
   {
      IU_array[j] = NewVectorInt(&iu[levv[j]-1], numv[j]);
      IP_array[j] = NewVectorInt(&ip[levv[j]-1], numv[j]);
      IV_array[j] = NewVectorInt(&iv[levpi[j]-1],nump[j]+1);
      ICG_array[j] = NewVectorInt(&icg[levv[j]-1], numv[j]);
   }
   
   AMGDataIUArray(amg_data) = IU_array;
   AMGDataIPArray(amg_data) = IP_array;
   AMGDataIVArray(amg_data) = IV_array;
   AMGDataICGArray(amg_data) = ICG_array;

   if (AMGDataIOutDat(amg_data) == -1)
   {
      for (j = 1; j < num_levels; j++)
      {
         sprintf(fnam,"A_%d.ysmp",j);
         WriteYSMP(fnam, A_array[j]);
      }
    }


   if (AMGDataIOutDat(amg_data) == -2)
   {
      for (j=0; j < num_levels-1; j++)
      {
         sprintf(fnam,"P_%d.ysmp",j);
         WriteYSMP(fnam, P_array[j]);
      }
   }

   if (AMGDataIOutDat(amg_data) == -3)
   {
      for (j=0; j < num_levels-1; j++)
      {
         sprintf(fnam,"ICG_%d.vec",j);
         WriteVecInt(fnam, ICG_array[j]);
      }
   }
   
   return(Setup_err_flag);
}


