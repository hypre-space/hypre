
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
 * Routine for testing hypre_AMGInterp
 *
 *****************************************************************************/


#include "headers.h"

#ifdef AMG_MALLOC_DEBUG
/* malloc debug stuff */
char amg_malloclog[256];
#endif  


/*--------------------------------------------------------------------------
 * amgdrv
 *--------------------------------------------------------------------------*/

void main(argc, argv)

int   argc;
char *argv[];

{
   void             *amg_data;

   hypre_CSRMatrix  *A;

   hypre_Vector     *f;
   hypre_Vector     *u;

   double            strong_threshold;
   char              *filename;
   int               solve_err_flag;
   int               num_fine;



   if (argc < 3)
   {
      fprintf(stderr, "Usage:  interpdrive <file> <strong_threshold>\n");
      exit(1);
   }

  /*-------------------------------------------------------
    * Set up debugging tools
    *-------------------------------------------------------*/

#ifdef AMG_MALLOC_DEBUG
   /* malloc debug stuff */
   malloc_logpath = amg_malloclog;
   sprintf(malloc_logpath, "malloc.log");
#endif
           

   strong_threshold = atof(argv[2]);

   amg_data = hypre_AMGInitialize();
   hypre_AMGSetStrongThreshold(strong_threshold, amg_data);
   hypre_AMGSetLogging(3,"amg.out.log",amg_data);
   
   filename = argv[1];
   A = hypre_ReadCSRMatrix(filename);

   num_fine = hypre_CSRMatrixNumRows(A);

   f = hypre_CreateVector(num_fine);
   hypre_InitializeVector(f);
   hypre_SetVectorConstantValues(f, 1.0);
                              
   u = hypre_CreateVector(num_fine);
   hypre_InitializeVector(u);
   hypre_SetVectorConstantValues(u, 0.0);

   hypre_AMGSetup(amg_data,A);

   solve_err_flag = hypre_AMGSolve(amg_data, f, u);
                 

   /*-------------------------------------------------------
    * Debugging prints
    *-------------------------------------------------------*/

#ifdef AMG_MALLOC_DEBUG
   /* malloc debug stuff */
   malloc_verify(0);
   malloc_shutdown();
#endif
                                        
}













