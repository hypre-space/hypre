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
 * Write routines
 *
 *****************************************************************************/

#include "amg.h"
#include "pcg.h"
#include "wjacobi.h"
#include "amgs01.h"


/*--------------------------------------------------------------------------
 * WriteYSMP
 *--------------------------------------------------------------------------*/

void     WriteYSMP(file_name, matrix)
char    *file_name;
Matrix  *matrix;
{
   FILE    *fp;

   double  *data;
   int     *ia;
   int     *ja;
   int      size;
   
   int      j;


   /*----------------------------------------------------------
    * Write the matrix data
    *----------------------------------------------------------*/

   data = MatrixData(matrix);
   ia   = MatrixIA(matrix);
   ja   = MatrixJA(matrix);
   size = MatrixSize(matrix);

   fp = fopen(file_name, "w");

   /* write junk line */
   fprintf(fp, "1 1\n");

   fprintf(fp, "%d\n", size);

   for (j = 0; j < size+1; j++)
      fprintf(fp, "%d\n", ia[j]);

   for (j = 0; j < ia[size]-1; j++)
      fprintf(fp, "%d\n", ja[j]);

   for (j = 0; j < ia[size]-1; j++)
      fprintf(fp, "%le\n", data[j]);

   fclose(fp);

   return;
}

/*--------------------------------------------------------------------------
 * WriteVec
 *--------------------------------------------------------------------------*/

void     WriteVec(file_name, vector)
char    *file_name;
Vector  *vector;
{
   FILE    *fp;

   double  *data;
   int      size;
   
   int      j;


   /*----------------------------------------------------------
    * Write in the data
    *----------------------------------------------------------*/

   data = VectorData(vector);
   size = VectorSize(vector);

   fp = fopen(file_name, "w");

   /* write junk line */
   fprintf(fp, "1 1\n");

   fprintf(fp, "%d\n", size);

   for (j = 0; j < size; j++)
      fprintf(fp, "%le\n", data[j]);

   fclose(fp);

   return;
}


/*--------------------------------------------------------------------------
 * WriteSolver
 *--------------------------------------------------------------------------*/

void     WriteSolver(file_name, solver)
char    *file_name;
Solver  *solver;

{
   FILE    *fp;

   int          type;
   double       stop_tolerance;
   PCGData     *pcg_data;
   WJacobiData *wjacobi_data; 
   AMGS01Data  *amgs01_data;

   /* PCG params */
   int          max_iter;
   int          two_norm;

   /* weighted Jacobi params */
   double       weight;
   int          jac_max_iter;
   
   /* amg setup params */
   int      levmax;
   int      ncg;
   double   ecg;
   int      nwt;
   double   ewt;
   int      nstr;
 
   /* amg solve params */
   int      ncyc;
   int     *mu;
   int     *ntrlx;
   int     *iprlx;
   int     *ierlx;
   int     *iurlx;
 
   /* amg output params */
   int      ioutdat;
   int      ioutgrd;
   int      ioutmat;
   int      ioutres;
   int      ioutsol;

   int      j;



   /*----------------------------------------------------------
    * Write the solver data
    *----------------------------------------------------------*/

   type = SolverType(solver);
   stop_tolerance = SolverStopTolerance(solver);
   pcg_data = SolverPCGData(solver);
   wjacobi_data = SolverWJacobiData(solver);
   amgs01_data = SolverAMGS01Data(solver);   

   max_iter = PCGDataMaxIter(pcg_data);
   two_norm = PCGDataTwoNorm(pcg_data);

   weight = WJacobiDataWeight(wjacobi_data);
   jac_max_iter = WJacobiDataMaxIter(wjacobi_data);

   levmax = AMGS01DataLevMax(amgs01_data);
   ncg = AMGS01DataNCG(amgs01_data);
   ecg =  AMGS01DataECG(amgs01_data);
   nwt = AMGS01DataNWT(amgs01_data);
   ewt = AMGS01DataEWT(amgs01_data);
   nstr = AMGS01DataNSTR(amgs01_data);
		  
   ncyc = AMGS01DataNCyc(amgs01_data);
   mu = AMGS01DataMU(amgs01_data);
   ntrlx = AMGS01DataNTRLX(amgs01_data);
   iprlx = AMGS01DataIPRLX(amgs01_data);
   ierlx = AMGS01DataIERLX(amgs01_data);
   iurlx = AMGS01DataIURLX(amgs01_data);
		  
   ioutdat = AMGS01DataIOutDat(amgs01_data);
   ioutgrd = AMGS01DataIOutGrd(amgs01_data);
   ioutmat = AMGS01DataIOutMat(amgs01_data);
   ioutres = AMGS01DataIOutRes(amgs01_data);
   ioutsol = AMGS01DataIOutSol(amgs01_data);


   fp = fopen(file_name, "a");

   fprintf(fp,"\nSOLVER PARAMETERS:\n\n");
   fprintf(fp, "  Solver Type:  %d - ", type);

   if (type == 0)
   {
      fprintf(fp, "AMG \n\n");
   }
   else if (type == 1)
   {
      fprintf(fp, "AMGCG \n\n");
   }
   else if (type ==2)
   {
      fprintf(fp, "JCG \n\n");
   }

   if (type == 1 | type == 2)
   {
       fprintf(fp, "  Preconditioned Conjugate Gradient Parameters:\n");
       fprintf(fp, "    Solver Stop Tolerance (PCG):  %e \n", stop_tolerance);
       fprintf(fp, "    Maximum Iterations (PCG): %d \n", max_iter);
       fprintf(fp, "    Two Norm Flag (PCG): %d \n\n", two_norm);
       
       if (type == 2)
       {
        fprintf(fp, "  Jacobi Preconditioner Parameters:\n");
        fprintf(fp, "    Weight for Jacobi Relaxation: %f \n", weight);
        fprintf(fp, "    Maximum Jacobi Iterations: %d \n\n", jac_max_iter);
       }
   }
   fprintf(fp, "  AMG Parameters:\n");
   fprintf(fp, "    Maximum number of levels:            %d \n", levmax);
   fprintf(fp, "    Coarsening controls (ncg, ecg):      %d   %f \n", ncg, ecg);
   fprintf(fp, "    Interpolation controls (nwt, ewt):   %d   %f \n", nwt, ewt);
   fprintf(fp, "    Strong connection definition (nstr): %d \n", nstr);
   fprintf(fp, "    Number and type of cycles (ncyc):    %d \n", ncyc); 
   fprintf(fp, "    W-cycling parameter (mu): ");
   for (j = 0; j < 10; j++)
        fprintf(fp, "%d ", mu[j]);
   fprintf(fp, "\n");
   fprintf(fp, "    Relaxation Parameters:\n");
   fprintf(fp, "       ntr(f,d,u,c): %d  %d  %d  %d \n", ntrlx[0], ntrlx[1],
                                                        ntrlx[2], ntrlx[3]);
   fprintf(fp, "       ipr(f,d,u,c): %d  %d  %d  %d \n", iprlx[0], iprlx[1],
                                                        iprlx[2], iprlx[3]);
   fprintf(fp, "       ier(f,d,u,c): %d  %d  %d  %d \n", ierlx[0], ierlx[1],
                                                        ierlx[2], ierlx[3]);
   fprintf(fp, "       iur(f,d,u,c): %d  %d  %d  %d \n", iurlx[0], iurlx[1],
                                                        iurlx[2], iurlx[3]);
 
   fprintf(fp, "    Output flag (ioutdat): %d \n", ioutdat);
   fprintf(fp, "    ioutgrd: (unused) %d \n", ioutgrd);
   fprintf(fp, "    Matrix output flag (ioutmat): %d \n", ioutmat); 
   fprintf(fp, "    Residual report (ioutres): %d \n", ioutres); 
   fprintf(fp, "    Graphical solution flag (ioutsol): %d \n", ioutsol);
 

   fclose(fp);

   return;
}



/*--------------------------------------------------------------------------
 * WriteProblem
 *--------------------------------------------------------------------------*/

void     WriteProblem(infile_name, outfile_name)
char    *infile_name;
char    *outfile_name;
/*Problem *problem;*/

{
   FILE    *fp;
   FILE    *fpi;

   int      num_variables;
   int      num_unknowns;
   int      num_points;

   int     *iu;
   int     *ip;
   int     *iv;

   char     temp_file_name[256];
   FILE    *temp_fp;
   int      flag;
   double  *data;
   double   dtemp;
   int      j;

   fp = fopen(outfile_name, "a");
   fprintf(fp, "\nPROBLEM INFORMATION: \n\n");
   fprintf(fp, "    Input problem file: %s \n", infile_name); 
  

   /*----------------------------------------------------------
    * Open the problem file
    *----------------------------------------------------------*/
   
   fpi = fopen(infile_name, "r");
   fscanf(fpi, "%d", &num_variables);
   fprintf(fp, "    Number of variables: %d \n", num_variables);
  
   fscanf(fpi, "%s", temp_file_name);
   fprintf(fp, "    Input matrix file: %s \n", temp_file_name);

   fscanf(fpi, "%d", &flag);
   
   /* Right-hand side */
   if (flag == 0)
   {
       fscanf(fpi, "%s", temp_file_name); 
       fprintf(fp, "    Right-hand side file name: %s \n", temp_file_name);
   }
   else
   {
       if (flag == 1)
       {
          fscanf(fpi, "%le", &dtemp);   
          fprintf(fp, "    Right-hand side constant with value: %e \n", dtemp);
       }
       else if (flag == 2)
       {
           fprintf(fp, "    Right-hand side is random vector. \n");
       }
   }

   /* Initial guess */
   fscanf(fpi, "%d", &flag);
   if (flag == 0)
   {
       fscanf(fpi, "%s", temp_file_name); 
       fprintf(fp, "    Initial guess file name: %s \n", temp_file_name);
   }
   else
   {
       if (flag == 1)
       {
          fscanf(fpi, "%le", &dtemp);   
          fprintf(fp, "    Initial guess constant with value: %e \n", dtemp);
       }
       else if (flag == 2)
       {
           fprintf(fp, "    Initial guess is random vector. \n");
       }
   }

   fscanf(fpi, "%d%d", &num_unknowns, &num_points);
   fprintf(fp, "    Number of unknown functions: %d \n", num_unknowns);
   fprintf(fp, "    Number of unknown points: %d \n", num_points);

   /* iu, ip, iv */
   fscanf(fpi, "%d", &flag);
   if (flag == 0)
   { 
      fprintf(fp, "     iu, iv, ip read from file: %s \n", infile_name);  
      iu = ctalloc(int, NDIMU(num_variables));
      ip = ctalloc(int, NDIMU(num_variables));
      iv = ctalloc(int, NDIMP(num_points+1));

      for (j = 0; j < num_variables; j++)
	 fscanf(fp, "%d", &iu[j]);
      for (j = 0; j < num_variables; j++)
	 fscanf(fp, "%d", &ip[j]);
      for (j = 0; j < num_points+1; j++)
	 fscanf(fp, "%d", &iv[j]);
   }
   else
   {
      fprintf(fp, "    Pointers iu, iv, ip defined in standard way. \n");
   }

   /*----------------------------------------------------------
    * xp, yp, zp
    *----------------------------------------------------------*/

   fscanf(fpi, "%d", &flag);
   if (flag == 0)
   {
      fscanf(fpi, "%s", temp_file_name);
      fprintf(fp, "    x, y, z data read from file: %s \n", temp_file_name);
   }
 
   fclose(fpi);
   fclose(fp);

   return;
}


