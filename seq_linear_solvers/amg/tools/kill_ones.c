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

#define  NDIMU(nv)  (50*nv)
#define  NDIMA(na)  (6*na)


/*--------------------------------------------------------------------------
 * RYSMP reads old-style ysmp files (with `1 1')
 *--------------------------------------------------------------------------*/
 
Matrix   *RYSMP(file_name)
char     *file_name;
{
   Matrix  *matrix;
 
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
 
   ia = ctalloc(int, NDIMU(size+1));
   for (j = 0; j < size+1; j++)
      fscanf(fp, "%d", &ia[j]);
 
   ja = ctalloc(int, NDIMA(ia[size]-1));
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%d", &ja[j]);
 
   data = ctalloc(double, NDIMA(ia[size]-1));
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%le", &data[j]);
 
   fclose(fp);
 
   /*----------------------------------------------------------
    * Create the matrix structure
    *----------------------------------------------------------*/
 
   matrix = NewMatrix(data, ia, ja, size);
 
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

   Matrix  *A;

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
   WriteYSMP(file_name_new,A);

   return 0;
}

