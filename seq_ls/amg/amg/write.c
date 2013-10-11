/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Write routines
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_WriteYSMP
 *--------------------------------------------------------------------------*/

void     hypre_WriteYSMP(file_name, matrix)
char    *file_name;
hypre_Matrix  *matrix;
{
   FILE    *fp;

   HYPRE_Real  *data;
   HYPRE_Int     *ia;
   HYPRE_Int     *ja;
   HYPRE_Int      size;
   
   HYPRE_Int      j;


   /*----------------------------------------------------------
    * Write the matrix data
    *----------------------------------------------------------*/

   data = hypre_MatrixData(matrix);
   ia   = hypre_MatrixIA(matrix);
   ja   = hypre_MatrixJA(matrix);
   size = hypre_MatrixSize(matrix);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%d\n", size);

   for (j = 0; j < size+1; j++)
      hypre_fprintf(fp, "%d\n", ia[j]);

   for (j = 0; j < ia[size]-1; j++)
      hypre_fprintf(fp, "%d\n", ja[j]);

   for (j = 0; j < ia[size]-1; j++)
      hypre_fprintf(fp, "%le\n", data[j]);

   fclose(fp);

   return;
}

/*--------------------------------------------------------------------------
 * hypre_WriteVec
 *--------------------------------------------------------------------------*/

void     hypre_WriteVec(file_name, vector)
char    *file_name;
hypre_Vector  *vector;
{
   FILE    *fp;

   HYPRE_Real  *data;
   HYPRE_Int      size;
   
   HYPRE_Int      j;


   /*----------------------------------------------------------
    * Write in the data
    *----------------------------------------------------------*/

   data = hypre_VectorData(vector);
   size = hypre_VectorSize(vector);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%d\n", size);

   for (j = 0; j < size; j++)
      hypre_fprintf(fp, "%le\n", data[j]);

   fclose(fp);

   return;
}


/*--------------------------------------------------------------------------
 * hypre_WriteVecInt
 *--------------------------------------------------------------------------*/

void     hypre_WriteVecInt(file_name, vector)
char    *file_name;
hypre_VectorInt  *vector;
{
   FILE    *fp;

   HYPRE_Int     *data;
   HYPRE_Int      size;
   
   HYPRE_Int      j;


   /*----------------------------------------------------------
    * Write the data
    *----------------------------------------------------------*/

   data = hypre_VectorIntData(vector);
   size = hypre_VectorIntSize(vector);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%d\n", size);

   for (j = 0; j < size; j++)
      hypre_fprintf(fp, "%d\n", data[j]);

   fclose(fp);

   return;
}

/*---------------------------------------------------------------
 * hypre_WriteSetupParams
 *---------------------------------------------------------------*/


void     hypre_WriteSetupParams(data)
void    *data;

{
   FILE    *fp;
   char    *file_name;

   hypre_AMGData  *amg_data = data;


   HYPRE_Int      type;

   /* amg setup params */
   HYPRE_Int      amg_levmax;
   HYPRE_Int      amg_ncg;
   HYPRE_Real   amg_ecg;
   HYPRE_Int      amg_nwt;
   HYPRE_Real   amg_ewt;
   HYPRE_Int      amg_nstr;


   /* amg output params */
   HYPRE_Int      amg_ioutdat;

   HYPRE_Int      j;


   /*----------------------------------------------------------
    * Get the amg_data data
    *----------------------------------------------------------*/

   file_name = hypre_AMGDataLogFileName(amg_data);


   amg_levmax  = hypre_AMGDataLevMax(amg_data);
   amg_ncg     = hypre_AMGDataNCG(amg_data);
   amg_ecg     = hypre_AMGDataECG(amg_data);
   amg_nwt     = hypre_AMGDataNWT(amg_data);
   amg_ewt     = hypre_AMGDataEWT(amg_data);
   amg_nstr    = hypre_AMGDataNSTR(amg_data);

   amg_ioutdat = hypre_AMGDataIOutDat(amg_data);

   /*----------------------------------------------------------
    * Open the output file
    *----------------------------------------------------------*/
   if (amg_ioutdat > 1)
   { 
      fp = fopen(file_name, "a");

      hypre_fprintf(fp,"\n AMG SETUP PARAMETERS:\n\n");  
   
   /*----------------------------------------------------------
    * AMG info
    *----------------------------------------------------------*/


      hypre_fprintf(fp, "  AMG Parameters:\n");
      hypre_fprintf(fp, "    Maximum number of levels:            %d \n",
	      amg_levmax);
      hypre_fprintf(fp, "    Coarsening controls (ncg, ecg):      %d   %f \n",
	      amg_ncg, amg_ecg);
      hypre_fprintf(fp, "    Interpolation controls (nwt, ewt):   %d   %f \n",
	      amg_nwt, amg_ewt);
      hypre_fprintf(fp, "    Strong connection definition (nstr): %d \n", 
              amg_nstr); 
      hypre_fprintf(fp, "    Output flag (ioutdat): %d \n", amg_ioutdat);

   /*----------------------------------------------------------
    * Close the output file
    *----------------------------------------------------------*/

      fclose(fp);
   }
   return;
}


/*---------------------------------------------------------------
 * hypre_WriteSolverParams
 *---------------------------------------------------------------*/


void     hypre_WriteSolverParams(tol,data)
void    *data;
HYPRE_Real   tol;

{
   FILE    *fp;
   char    *file_name;

   hypre_AMGData  *amg_data = data;


   HYPRE_Int      type;

   HYPRE_Real   stop_tolerance;

 
   /* amg solve params */
   HYPRE_Int      amg_ncyc;
   HYPRE_Int      amg_levmax;
   HYPRE_Int     *amg_mu;
   HYPRE_Int     *amg_ntrlx;
   HYPRE_Int     *amg_iprlx;

   /* amg output params */
   HYPRE_Int      amg_ioutdat;

   HYPRE_Int      j;


   /*----------------------------------------------------------
    * Get the amg_data data
    *----------------------------------------------------------*/

   file_name = hypre_AMGDataLogFileName(amg_data);

   stop_tolerance = tol;


   amg_ncyc    = hypre_AMGDataNCyc(amg_data);
   amg_levmax  = hypre_AMGDataLevMax(amg_data);
   amg_mu      = hypre_AMGDataMU(amg_data);
   amg_ntrlx   = hypre_AMGDataNTRLX(amg_data);
   amg_iprlx   = hypre_AMGDataIPRLX(amg_data);

   amg_ioutdat = hypre_AMGDataIOutDat(amg_data);

   /*----------------------------------------------------------
    * Open the output file
    *----------------------------------------------------------*/

   if (amg_ioutdat == 1 || amg_ioutdat == 3)
   { 
      fp = fopen(file_name, "a");

      hypre_fprintf(fp,"\nAMG SOLVER PARAMETERS:\n\n");
      hypre_fprintf(fp, "  Solver Type: ");
      hypre_fprintf(fp, "AMG \n\n");
  
   
   /*----------------------------------------------------------
    * AMG info
    *----------------------------------------------------------*/

      hypre_fprintf(fp, "    Number and type of cycles (ncyc):    %d \n", amg_ncyc);
      hypre_fprintf(fp, "    Stopping Tolerance:                  %e \n",
                   stop_tolerance); 
      hypre_fprintf(fp, "    W-cycling parameter (mu): ");
      for (j = 0; j < amg_levmax; j++)
	 hypre_fprintf(fp, "%d ", amg_mu[j]);
      hypre_fprintf(fp, "\n");
      hypre_fprintf(fp, "    Relaxation Parameters:\n");
      hypre_fprintf(fp, "       ntr(f,d,u,c): %d  %d  %d  %d \n",
	      amg_ntrlx[0], amg_ntrlx[1], amg_ntrlx[2], amg_ntrlx[3]);
      hypre_fprintf(fp, "       ipr(f,d,u,c): %d  %d  %d  %d \n",
	      amg_iprlx[0], amg_iprlx[1], amg_iprlx[2], amg_iprlx[3]);
      hypre_fprintf(fp, "    Output flag (ioutdat): %d \n", amg_ioutdat);

   /*----------------------------------------------------------
    * Close the output file
    *----------------------------------------------------------*/

      fclose(fp);
   }

   return;
}

