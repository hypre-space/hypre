/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include <stdio.h>
 
#include "general.h"
#include "matrix.h"
#include "io.h"

#define  hypre_NDIMU(nv)  (50*nv)
#define  hypre_NDIMA(na)  (6*na)


/*--------------------------------------------------------------------------
 * RYSMP reads old-style ysmp files (with `1 1')
 *--------------------------------------------------------------------------*/
 
hypre_Matrix   *RYSMP(file_name)
char     *file_name;
{
   hypre_Matrix  *matrix;
 
   FILE    *fp;
 
   double  *data;
   int     *ia;
   int     *ja;
   int      size;
   
   int      j;
 
 
   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/
 
   fp = fopen(file_name, "r");
 
    /* read in junk line */
    fscanf(fp, "%*[^\n]\n");

   fscanf(fp, "%d", &size);
 
   ia = hypre_CTAlloc(int, hypre_NDIMU(size+1));
   for (j = 0; j < size+1; j++)
      fscanf(fp, "%d", &ia[j]);
 
   ja = hypre_CTAlloc(int, hypre_NDIMA(ia[size]-1));
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%d", &ja[j]);
 
   data = hypre_CTAlloc(double, hypre_NDIMA(ia[size]-1));
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%le", &data[j]);
 
   fclose(fp);
 
   /*----------------------------------------------------------
    * Create the matrix structure
    *----------------------------------------------------------*/
 
   matrix = hypre_NewMatrix(data, ia, ja, size);
 
   return matrix;
}

/*--------------------------------------------------------------------------
 *  Program kills the junk line `1 1' in old-style ysmp  files
 *--------------------------------------------------------------------------*/

int   main(argc, argv)
int   argc;
char *argv[];
{
   char    *run_name;

   char     file_name_old[255];
   char     file_name_new[255];

   FILE    *fp;

   hypre_Matrix  *A;

   /*-------------------------------------------------------
    * Check that the number of command args is correct
    *-------------------------------------------------------*/

   if (argc < 2)
   {
      fprintf(stderr, "Usage:  kill_ones  <ysmp file name>\n");
      exit(1);
   }

   run_name = argv[1];

   sprintf(file_name_old, "%s", run_name);
   sprintf(file_name_new, "%s.new", run_name);

   A = RYSMP(file_name_old);
   hypre_WriteYSMP(file_name_new,A);

   return 0;
}

