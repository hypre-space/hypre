/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * AMGS01 functions
 *
 *****************************************************************************/

#include "amg.h"
#include "amgs01.h"


/*--------------------------------------------------------------------------
 * AMGS01
 *--------------------------------------------------------------------------*/

void         AMGS01(u, f, tol, data)
Vector      *u;
Vector      *f;
double       tol;
Data        *data;
{
   AMGS01Data  *amgs01_data = data;


   CALL_SOLVE(u, f, tol, amgs01_data);
}

/*--------------------------------------------------------------------------
 * AMGS01Setup
 *--------------------------------------------------------------------------*/

void         AMGS01Setup(problem, data)
Problem     *problem;
Data        *data;
{
   AMGS01Data  *amgs01_data = data;
   
   int      levmax        = AMGS01DataLevMax(amgs01_data);
   int      num_variables = ProblemNumVariables(problem);
   int      num_points    = ProblemNumPoints(problem);
   
   Matrix  *A;
   Matrix  *P;
   
   int      num_levels;
   int      ndimu;
   int      ndimp;
   int      ndima;
   int      ndimb;
   int     *icdep;
   int     *imin;
   int     *imax;
   int     *ipmn;
   int     *ipmx;
   int     *icg;
   int     *ifg;
   Matrix **A_array;
   Matrix **P_array;
   int     *leva;
   int     *levb;
   int     *levv;
   int     *levp;
   int     *levi;
/* veh test levpi */
   int     *levpi;
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
   
   int     *ip;
   int     *iv;

   int      j, k;
   int      decr;
   
   
   /*----------------------------------------------------------
    * Set new variables
    *----------------------------------------------------------*/
   
   num_levels = levmax;
   
   A  = ProblemA(problem);
   ia = MatrixIA(A);
   
   /* set size variables */
   ndimu = NDIMU(num_variables);
   ndimp = NDIMP(num_points);
   ndima = NDIMA(ia[num_variables]-1);
   ndimb = NDIMB(ia[num_variables]-1);
   
   
   b  = ctalloc(double, ndimb);
   ib = ctalloc(int, ndimu);
   jb = ctalloc(int, ndimb);
   P  = NewMatrix(b, ib, jb, num_variables);
   
   icdep      = ctalloc(int, levmax*levmax);
   imin       = ctalloc(int, levmax);
   imax       = ctalloc(int, levmax);
   ipmn       = ctalloc(int, levmax);
   ipmx       = ctalloc(int, levmax);
   icg        = ctalloc(int, ndimu);
   ifg        = ctalloc(int, ndimu);
   
   /* set fine level point and variable bounds */
   ipmn[0] = 1;
   ipmx[0] = num_points;
   imin[0] = 1;
   imax[0] = num_variables;
   
   /*----------------------------------------------------------
    * Fill in the remainder of the AMGS01Data structure
    *----------------------------------------------------------*/
   
   AMGS01DataProblem(amgs01_data)   = problem;
   
   AMGS01DataNumLevels(amgs01_data) = num_levels;
   AMGS01DataA(amgs01_data)         = A;
   AMGS01DataP(amgs01_data)         = P;   
   AMGS01DataNDIMU(amgs01_data)     = ndimu;
   AMGS01DataNDIMP(amgs01_data)     = ndimp;
   AMGS01DataNDIMA(amgs01_data)     = ndima;
   AMGS01DataNDIMB(amgs01_data)     = ndimb;   
   AMGS01DataICDep(amgs01_data)     = icdep;
   AMGS01DataIMin(amgs01_data)      = imin;
   AMGS01DataIMax(amgs01_data)      = imax;
   AMGS01DataIPMN(amgs01_data)      = ipmn;
   AMGS01DataIPMX(amgs01_data)      = ipmx;
   AMGS01DataICG(amgs01_data)       = icg;
   AMGS01DataIFG(amgs01_data)       = ifg;
   
   /*----------------------------------------------------------
    * Call the setup phase code
    *----------------------------------------------------------*/
   
   CALL_SETUP(problem, amgs01_data);
   
   /*----------------------------------------------------------
    * Set some local variables
    *----------------------------------------------------------*/
   
   num_levels = AMGS01DataNumLevels(amgs01_data);
   
   a  = MatrixData(A);
   ia = MatrixIA(A);
   ja = MatrixJA(A);
   b  = MatrixData(P);
   ib = MatrixIA(P);
   jb = MatrixJA(P);
   ip = ProblemIP(problem);
   iv = ProblemIV(problem);
   
   /*----------------------------------------------------------
    * Create `lev' and `num' arrays
    *----------------------------------------------------------*/
   
   leva       = ctalloc(int, num_levels);
   levb       = ctalloc(int, num_levels);
   levv       = ctalloc(int, num_levels);
   levp       = ctalloc(int, num_levels);
   levi       = ctalloc(int, num_levels);
   levpi      = ctalloc(int, num_levels);
   numa       = ctalloc(int, num_levels);
   numb       = ctalloc(int, num_levels);
   numv       = ctalloc(int, num_levels);
   nump       = ctalloc(int, num_levels);
   for (j = 0; j < num_levels; j++)
   {
      leva[j] = ia[imin[j]-1];
      levb[j] = ib[imin[j]-1];
      levv[j] = imin[j];
      levp[j] = ipmn[j];
      levpi[j] = ipmn[j] + j;
      levi[j] = imin[j] + j;
      numa[j] = ia[imax[j]+1-1]-ia[imin[j]-1];
      numb[j] = ib[imax[j]+1-1]-ib[imin[j]-1];
      numv[j] = imax[j] - imin[j] + 1;
      nump[j] = ipmx[j] - ipmn[j] + 1;
   }

   AMGS01DataAArray(amgs01_data) = A_array;
   AMGS01DataPArray(amgs01_data) = P_array;
   AMGS01DataLevA(amgs01_data)   = leva;
   AMGS01DataLevB(amgs01_data)   = levb;
   AMGS01DataLevV(amgs01_data)   = levv;
   AMGS01DataLevP(amgs01_data)   = levp;   
   AMGS01DataLevPI(amgs01_data)  = levpi;
   AMGS01DataLevI(amgs01_data)   = levi;
   AMGS01DataNumA(amgs01_data)   = numa;
   AMGS01DataNumB(amgs01_data)   = numb;
   AMGS01DataNumV(amgs01_data)   = numv;
   AMGS01DataNumP(amgs01_data)   = nump;

   /*----------------------------------------------------------
    * Shift index arrays
    *----------------------------------------------------------*/
   
   /* make room for `num(j)+1' entry in `ia' and `ib' */
   /* veh also make room for `num(j)+1' entry in iv */
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
   
   AMGS01DataAArray(amgs01_data) = A_array;
   AMGS01DataPArray(amgs01_data) = P_array;   
   
#if 0
{
   char     fnam[255];

   for (j = 1; j < num_levels; j++)
   {
      sprintf(fnam,"level_%d.ysmp",j);
      WriteYSMP(fnam, A_array[j]);
   }
}
#endif

}

/*--------------------------------------------------------------------------
 * ReadAMGS01Params
 *--------------------------------------------------------------------------*/

Data  *ReadAMGS01Params(fp)
FILE  *fp;
{
   Data    *data;

   /* setup params */
   int      levmax;
   int      ncg;
   double   ecg;
   int      nwt;
   double   ewt;
   int      nstr;

   /* solve params */
   int      ncyc;
   int     *mu;
   int     *ntrlx;
   int     *iprlx;
   int     *ierlx;
   int     *iurlx;

   /* output params */
   int      ioutdat;
   int      ioutgrd;
   int      ioutmat;
   int      ioutres;
   int      ioutsol;

   int      i;


   fscanf(fp, "%d", &levmax);
   fscanf(fp, "%d", &ncg);
   fscanf(fp, "%le", &ecg);
   fscanf(fp, "%d", &nwt);
   fscanf(fp, "%le", &ewt);
   fscanf(fp, "%d", &nstr);
   
   fscanf(fp, "%d", &ncyc);
   mu = ctalloc(int, levmax);
   for (i = 0; i < 10; i++)
      fscanf(fp, "%d", &mu[i]);
   ntrlx = ctalloc(int, 4);
   for (i = 0; i < 4; i++)
      fscanf(fp, "%d", &ntrlx[i]);
   iprlx = ctalloc(int, 4);
   for (i = 0; i < 4; i++)
      fscanf(fp, "%d", &iprlx[i]);
   ierlx = ctalloc(int, 4);
   for (i = 0; i < 4; i++)
      fscanf(fp, "%d", &ierlx[i]);
   iurlx = ctalloc(int, 4);
   for (i = 0; i < 4; i++)
      fscanf(fp, "%d", &iurlx[i]);
   
   fscanf(fp, "%d", &ioutdat);
   fscanf(fp, "%d", &ioutgrd);
   fscanf(fp, "%d", &ioutmat);
   fscanf(fp, "%d", &ioutres);
   fscanf(fp, "%d", &ioutsol);

   data = NewAMGS01Data(levmax, ncg, ecg, nwt, ewt, nstr,
			ncyc, mu, ntrlx, iprlx, ierlx, iurlx,
			ioutdat, ioutgrd, ioutmat, ioutres, ioutsol,
			GlobalsLogFileName);

   return data;
}

/*--------------------------------------------------------------------------
 * NewAMGS01Data
 *--------------------------------------------------------------------------*/

Data   *NewAMGS01Data(levmax, ncg, ecg, nwt, ewt, nstr,
		      ncyc, mu, ntrlx, iprlx, ierlx, iurlx,
		      ioutdat, ioutgrd, ioutmat, ioutres, ioutsol,
		      log_file_name)
int     levmax;
int     ncg;
double  ecg;
int     nwt;
double  ewt;
int     nstr;
int     ncyc;
int    *mu;
int    *ntrlx;
int    *iprlx;
int    *ierlx;
int    *iurlx;
int     ioutdat;
int     ioutgrd;
int     ioutmat;
int     ioutres;
int     ioutsol;
char   *log_file_name;
{
   AMGS01Data  *amgs01_data;

   amgs01_data = ctalloc(AMGS01Data, 1);

   AMGS01DataLevMax(amgs01_data)  = levmax;
   AMGS01DataNCG(amgs01_data)     = ncg;
   AMGS01DataECG(amgs01_data)     = ecg;
   AMGS01DataNWT(amgs01_data)     = nwt;
   AMGS01DataEWT(amgs01_data)     = ewt;
   AMGS01DataNSTR(amgs01_data)    = nstr;
   				    
   AMGS01DataNCyc(amgs01_data)    = ncyc;
   AMGS01DataMU(amgs01_data)      = mu;
   AMGS01DataNTRLX(amgs01_data)   = ntrlx;
   AMGS01DataIPRLX(amgs01_data)   = iprlx;
   AMGS01DataIERLX(amgs01_data)   = ierlx;
   AMGS01DataIURLX(amgs01_data)   = iurlx;
   				    
   AMGS01DataIOutDat(amgs01_data) = ioutdat;
   AMGS01DataIOutGrd(amgs01_data) = ioutgrd;
   AMGS01DataIOutMat(amgs01_data) = ioutmat;
   AMGS01DataIOutRes(amgs01_data) = ioutres;
   AMGS01DataIOutSol(amgs01_data) = ioutsol;

   AMGS01DataLogFileName(amgs01_data) = log_file_name;
   
   return (Data *)amgs01_data;
}

/*--------------------------------------------------------------------------
 * FreeAMGS01Data
 *--------------------------------------------------------------------------*/

void         FreeAMGS01Data(data)
Data        *data;
{
   AMGS01Data  *amgs01_data = data;


   if (amgs01_data)
   {
      tfree(AMGS01DataMU(amgs01_data));
      tfree(AMGS01DataNTRLX(amgs01_data));
      tfree(AMGS01DataIPRLX(amgs01_data));
      tfree(AMGS01DataIERLX(amgs01_data));
      tfree(AMGS01DataIURLX(amgs01_data));
      tfree(amgs01_data);
   }
}

