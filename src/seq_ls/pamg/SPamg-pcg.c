/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Serial-Parallel AMG driver   (SPamg)
 *
 *****************************************************************************/

#include "headers.h"
#include "amg.h"
#include "pcg.h"

/*--------------------------------------------------------------------------
 *Spamg-pcg
 *--------------------------------------------------------------------------*/

HYPRE_Int
main( HYPRE_Int   argc,
      char *argv[] )
{
   void             *amg_data;

   PCGData          *pcg_data;

   hypre_CSRMatrix  *A;

   hypre_Vector     *f;
   hypre_Vector     *u;

   double            strong_threshold;
   char              *filename;
   HYPRE_Int               num_fine;

   HYPRE_Int               cycle_type;
   double           *tmp;

   double   stop_tol;
   char     pcg_logfilename[256];

   HYPRE_Int     *num_grid_sweeps;  
   HYPRE_Int     *grid_relax_type;   
   HYPRE_Int    **grid_relax_points; 

   HYPRE_Int      j; 

   num_grid_sweeps = hypre_CTAlloc(HYPRE_Int,4);
   grid_relax_type = hypre_CTAlloc(HYPRE_Int,4); 
   grid_relax_points = hypre_CTAlloc(HYPRE_Int *,4); 

   for (j = 0; j < 2; j++)
   {
      num_grid_sweeps[j] = 3;
      grid_relax_type[j] = 1;  
      grid_relax_points[j] = hypre_CTAlloc(HYPRE_Int,3); 
      grid_relax_points[j][0] = 1;
      grid_relax_points[j][1] = -1;
      grid_relax_points[j][2] = 1;
   }

   num_grid_sweeps[2] = 3;
   grid_relax_type[2] = 1; 
   grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int,2); 
   grid_relax_points[2][0] = -1;
   grid_relax_points[2][1] = 1;
   grid_relax_points[2][2] = -1;

   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 9;
   grid_relax_points[3] = hypre_CTAlloc(HYPRE_Int,1);
   grid_relax_points[3][0] = 0;



   if (argc < 4)
   {
      hypre_fprintf(stderr, "Usage:  SPamg-pcg <file> <strong_threshold>  <mu>\n");
      exit(1);
   }

  /*-------------------------------------------------------
    * Set up debugging tools
    *-------------------------------------------------------*/
   
   hypre_InitMemoryDebug(0); 

  /*-------------------------------------------------------
    * Begin AMG-PCG driver
    *-------------------------------------------------------*/
           
   strong_threshold = atof(argv[2]);
   cycle_type = atoi(argv[3]);
   

   amg_data = hypre_AMGInitialize();
   hypre_AMGSetStrongThreshold(amg_data, strong_threshold);
   hypre_AMGSetLogging(amg_data,3,"amg.out.log");
   hypre_AMGSetCycleType(amg_data, cycle_type);

   filename = argv[1];
   A = hypre_CSRMatrixRead(filename);

   num_fine = hypre_CSRMatrixNumRows(A);

   f = hypre_SeqVectorCreate(num_fine);
   hypre_SeqVectorInitialize(f);
   hypre_SeqVectorSetConstantValues(f, 1.0);
                              
   u = hypre_SeqVectorCreate(num_fine);
   hypre_SeqVectorInitialize(u);

   tmp = hypre_CTAlloc(double, num_fine);

   for (j = 0; j < num_fine; j++)
   {
       tmp[j] = hypre_Rand();
   }
   hypre_VectorData(u) = tmp;  


/*   hypre_SeqVectorSetConstantValues(u, 0.0); */

   /* Set the relaxation parameters for symmetric cycle */

   hypre_AMGSetNumGridSweeps(num_grid_sweeps, amg_data);
   hypre_AMGSetGridRelaxType(grid_relax_type, amg_data);
   hypre_AMGSetGridRelaxPoints(grid_relax_points, amg_data);

   /* Initialize the PCG data structure */

   pcg_data = hypre_CTAlloc(PCGData, 1);
   PCGDataMaxIter(pcg_data) = 50;
   PCGDataTwoNorm(pcg_data) = 1;
   PCGDataA(pcg_data)  = A;

   hypre_sprintf(pcg_logfilename,"pcg.out.log");

   PCGDataLogFileName(pcg_data) = pcg_logfilename;

/*   stop_tol = hypre_AMGDataTol(amg_data); */
   stop_tol = 1.0e-7;
   hypre_AMGSetTol(amg_data, 0.0);
   hypre_AMGSetMaxIter(amg_data, 1);
   

   /* Perform the PCG and AMG setups */

   hypre_AMGSetup(amg_data, A, f, u);
   
   PCGSetup(A, hypre_AMGSolve, amg_data, pcg_data);

   /* Perform the PCG Solve */

   PCG(u, f, stop_tol, pcg_data);

   hypre_FinalizeMemoryDebug(); 
                
   return 0;
}













