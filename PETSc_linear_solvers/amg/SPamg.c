
/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Serial-Parallel AMG driver   (SPamg)
 *
 *****************************************************************************/


#include "headers.h"


/*--------------------------------------------------------------------------
 *Spamg
 *--------------------------------------------------------------------------*/

int
main( int   argc,
      char *argv[] )
{
   void             *amg_data;

   hypre_CSRMatrix  *A;

   hypre_Vector     *f;
   hypre_Vector     *u;

   double            strong_threshold;
   char              *filename;
   int               solve_err_flag;
   int               num_fine;

   int               cycle_type;

/*----------------------------------------------------
   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points; 

   int      j;

   num_grid_sweeps = hypre_CTAlloc(int,4);
   grid_relax_type = hypre_CTAlloc(int,4);
   grid_relax_points = hypre_CTAlloc(int *,4);

   for (j = 0; j < 3; j++)
   {
      num_grid_sweeps[j] = 2;
      grid_relax_type[j] = 1; 
      grid_relax_points[j] = hypre_CTAlloc(int,2); 
      grid_relax_points[j][0] = 1;
      grid_relax_points[j][1] = -1;
   }
   num_grid_sweeps[3] = 100;
   grid_relax_type[3] = 1;
   grid_relax_points[3] = hypre_CTAlloc(int,100);
   for (j=0;j<100;j++)
       grid_relax_points[3][j] = 0;
----------------------------------------------------*/

   if (argc < 4)
   {
      fprintf(stderr, "Usage:  SPamg <file> <strong_threshold>  <mu>\n");
      exit(1);
   }

  /*-------------------------------------------------------
    * Set up debugging tools
    *-------------------------------------------------------*/
   
   hypre_InitMemoryDebug(0); 

  /*-------------------------------------------------------
    * Begin AMG driver
    *-------------------------------------------------------*/
           
   strong_threshold = atof(argv[2]);
   cycle_type = atoi(argv[3]);
   

   amg_data = hypre_AMGInitialize();
   hypre_AMGSetStrongThreshold(strong_threshold, amg_data);
   hypre_AMGSetLogging(3,"amg.out.log",amg_data);
   hypre_AMGSetCycleType(cycle_type, amg_data);

/*--------------  
   hypre_AMGSetNumGridSweeps(num_grid_sweeps, amg_data);
   hypre_AMGSetGridRelaxType(grid_relax_type, amg_data);
   hypre_AMGSetGridRelaxPoints(grid_relax_points, amg_data);
--------------*/

   filename = argv[1];
   A = hypre_ReadCSRMatrix(filename);

   num_fine = hypre_CSRMatrixNumRows(A);

   f = hypre_CreateVector(num_fine);
   hypre_InitializeVector(f);
   hypre_SetVectorConstantValues(f, 0.0);
                              
   u = hypre_CreateVector(num_fine);
   hypre_InitializeVector(u);
   hypre_SetVectorConstantValues(u, 1.0);

   hypre_AMGSetup(amg_data,A);

   solve_err_flag = hypre_AMGSolve(amg_data, f, u);

   hypre_FinalizeMemoryDebug(); 
                
   return 0;
}













