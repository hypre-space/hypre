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

#include "headers.h"


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
 * WriteVecInt
 *--------------------------------------------------------------------------*/

void     WriteVecInt(file_name, vector)
char    *file_name;
VectorInt  *vector;
{
   FILE    *fp;

   int     *data;
   int      size;
   
   int      j;


   /*----------------------------------------------------------
    * Write the data
    *----------------------------------------------------------*/

   data = VectorIntData(vector);
   size = VectorIntSize(vector);

   fp = fopen(file_name, "w");

   /* write junk line */
   fprintf(fp, "1 1\n");

   fprintf(fp, "%d\n", size);

   for (j = 0; j < size; j++)
      fprintf(fp, "%d\n", data[j]);

   fclose(fp);

   return;
}

/*---------------------------------------------------------------
 * WriteSetupParams
 *---------------------------------------------------------------*/


void     WriteSetupParams(data)
void    *data;

{
   FILE    *fp;
   char    *file_name;

   AMGData  *amg_data = data;


   int      type;

   /* amg setup params */
   int      amg_levmax;
   int      amg_ncg;
   double   amg_ecg;
   int      amg_nwt;
   double   amg_ewt;
   int      amg_nstr;


   /* amg output params */
   int      amg_ioutdat;

   int      j;


   /*----------------------------------------------------------
    * Get the amg_data data
    *----------------------------------------------------------*/

   file_name = AMGDataLogFileName(amg_data);


   amg_levmax  = AMGDataLevMax(amg_data);
   amg_ncg     = AMGDataNCG(amg_data);
   amg_ecg     = AMGDataECG(amg_data);
   amg_nwt     = AMGDataNWT(amg_data);
   amg_ewt     = AMGDataEWT(amg_data);
   amg_nstr    = AMGDataNSTR(amg_data);

   amg_ioutdat = AMGDataIOutDat(amg_data);

   /*----------------------------------------------------------
    * Open the output file
    *----------------------------------------------------------*/
   if (amg_ioutdat > 1)
   { 
      fp = fopen(file_name, "a");

      fprintf(fp,"\n AMG SETUP PARAMETERS:\n\n");  
   
   /*----------------------------------------------------------
    * AMG info
    *----------------------------------------------------------*/


      fprintf(fp, "  AMG Parameters:\n");
      fprintf(fp, "    Maximum number of levels:            %d \n",
	      amg_levmax);
      fprintf(fp, "    Coarsening controls (ncg, ecg):      %d   %f \n",
	      amg_ncg, amg_ecg);
      fprintf(fp, "    Interpolation controls (nwt, ewt):   %d   %f \n",
	      amg_nwt, amg_ewt);
      fprintf(fp, "    Strong connection definition (nstr): %d \n", 
              amg_nstr); 
      fprintf(fp, "    Output flag (ioutdat): %d \n", amg_ioutdat);

   /*----------------------------------------------------------
    * Close the output file
    *----------------------------------------------------------*/

      fclose(fp);
   }
   return;
}


/*---------------------------------------------------------------
 * WriteSolverParams
 *---------------------------------------------------------------*/


void     WriteSolverParams(tol,data)
void    *data;
double   tol;

{
   FILE    *fp;
   char    *file_name;

   AMGData  *amg_data = data;


   int      type;

   double   stop_tolerance;

 
   /* amg solve params */
   int      amg_ncyc;
   int     *amg_mu;
   int     *amg_ntrlx;
   int     *amg_iprlx;
   int     *amg_ierlx;
   int     *amg_iurlx;

   /* amg output params */
   int      amg_ioutdat;

   int      j;


   /*----------------------------------------------------------
    * Get the amg_data data
    *----------------------------------------------------------*/

   file_name = AMGDataLogFileName(amg_data);

   stop_tolerance = tol;


   amg_ncyc    = AMGDataNCyc(amg_data);
   amg_mu      = AMGDataMU(amg_data);
   amg_ntrlx   = AMGDataNTRLX(amg_data);
   amg_iprlx   = AMGDataIPRLX(amg_data);
   amg_ierlx   = AMGDataIERLX(amg_data);
   amg_iurlx   = AMGDataIURLX(amg_data);

   amg_ioutdat = AMGDataIOutDat(amg_data);

   /*----------------------------------------------------------
    * Open the output file
    *----------------------------------------------------------*/

   if (amg_ioutdat == 1 || amg_ioutdat == 3)
   { 
      fp = fopen(file_name, "a");

      fprintf(fp,"\nAMG SOLVER PARAMETERS:\n\n");
      fprintf(fp, "  Solver Type: ");
      fprintf(fp, "AMG \n\n");
  
   
   /*----------------------------------------------------------
    * AMG info
    *----------------------------------------------------------*/

      fprintf(fp, "    Number and type of cycles (ncyc):    %d \n", amg_ncyc);
      fprintf(fp, "    Stopping Tolerance:                  %e \n",
                   stop_tolerance); 
      fprintf(fp, "    W-cycling parameter (mu): ");
      for (j = 0; j < 10; j++)
	 fprintf(fp, "%d ", amg_mu[j]);
      fprintf(fp, "\n");
      fprintf(fp, "    Relaxation Parameters:\n");
      fprintf(fp, "       ntr(f,d,u,c): %d  %d  %d  %d \n",
	      amg_ntrlx[0], amg_ntrlx[1], amg_ntrlx[2], amg_ntrlx[3]);
      fprintf(fp, "       ipr(f,d,u,c): %d  %d  %d  %d \n",
	      amg_iprlx[0], amg_iprlx[1], amg_iprlx[2], amg_iprlx[3]);
      fprintf(fp, "       ier(f,d,u,c): %d  %d  %d  %d \n",
	      amg_ierlx[0], amg_ierlx[1], amg_ierlx[2], amg_ierlx[3]);
      fprintf(fp, "       iur(f,d,u,c): %d  %d  %d  %d \n",
	      amg_iurlx[0], amg_iurlx[1], amg_iurlx[2], amg_iurlx[3]);
 
      fprintf(fp, "    Output flag (ioutdat): %d \n", amg_ioutdat);

   /*----------------------------------------------------------
    * Close the output file
    *----------------------------------------------------------*/

      fclose(fp);
   }

   return;
}

