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

   int      num_levels;
   Matrix  *A;
   Matrix  *P;
   int     *icdep;
   int     *imin;
   int     *imax;
   int     *ipmn;
   int     *ipmx;
   int     *icg;
   int     *ifg;
   int     *ifc;

   int     *ia;

   double  *b;
   int     *ib;
   int     *jb;


   /*----------------------------------------------------------
    * Set new variables
    *----------------------------------------------------------*/

   num_levels = levmax;

   A  = ProblemA(problem);
   ia = MatrixIA(A);

   b  = talloc(double, NDIMA(ia[num_variables]-1));
   ib = talloc(int, NDIMU(num_variables+1));
   jb = talloc(int, NDIMA(ia[num_variables]-1));
   P  = NewMatrix(b, ib, jb, num_variables);

   icdep      = ctalloc(int, levmax*levmax);
   imin       = ctalloc(int, levmax);
   imax       = ctalloc(int, levmax);
   ipmn       = ctalloc(int, levmax);
   ipmx       = ctalloc(int, levmax);
   icg        = ctalloc(int, NDIMU(num_variables));
   ifg        = ctalloc(int, NDIMU(num_variables));
   ifc        = ctalloc(int, NDIMU(num_variables));

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
   AMGS01DataICDep(amgs01_data)     = icdep;
   AMGS01DataIMin(amgs01_data)      = imin;
   AMGS01DataIMax(amgs01_data)      = imax;
   AMGS01DataIPMN(amgs01_data)      = ipmn;
   AMGS01DataIPMX(amgs01_data)      = ipmx;
   AMGS01DataICG(amgs01_data)       = icg;
   AMGS01DataIFG(amgs01_data)       = ifg;
   AMGS01DataIFC(amgs01_data)       = ifc;

   /*----------------------------------------------------------
    * Call the setup phase code
    *----------------------------------------------------------*/

   CALL_SETUP(problem, amgs01_data);
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
			ncyc, mu,
			ntrlx, iprlx, ierlx, iurlx,
			ioutdat, ioutgrd, ioutmat,
			ioutres, ioutsol, GlobalsLogFileName);

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

   amgs01_data = talloc(AMGS01Data, 1);

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
