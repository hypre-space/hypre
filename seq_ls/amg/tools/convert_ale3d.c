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


/*--------------------------------------------------------------------------
 * Convert ale3d data to AMG data
 *--------------------------------------------------------------------------*/

int   main(argc, argv)
int   argc;
char *argv[];
{
   char    *run_name;

   char     file_name[255];
   FILE    *fp;

   long int ft;
   int      idum;
   int      jdiag;
   double   adum;
   double   atemp;
   int      ctemp;
   int      ret;

   double  *S_data;
   int     *S_ia;
   int     *S_ja;
   Matrix  *A;

   double  *x;
   double  *y;
   double  *z;
   
   int      n, m, i, j;



   /*-------------------------------------------------------
    * Check that the number of command args is correct
    *-------------------------------------------------------*/

   if (argc < 2)
   {
      fprintf(stderr, "Usage:  convert_ale3d <run name>\n");
      exit(1);
   }

   run_name = argv[1];

   /*-------------------------------------------------------
    * Read in the ale3d data
    *-------------------------------------------------------*/

   sprintf(file_name, "%s.raw.mat", run_name);

   fp = fopen(file_name, "r");

   fscanf(fp, "%d", &n);

   /* save current file position */
   ft = ftell(fp);

   /* compute S_ia */
   S_ia = talloc(int, (n+1));

   m = 1;
   i = 0;
   while (m < n+1)
   {
      fscanf(fp, "%d%d%le", &j,&idum,&adum);
      i++;
      if (j == m)
      {
	 S_ia[m-1] = i;
	 m++;
      }
      
   }
   do
   { 
       ret = fscanf(fp, "%d%d%le", &j,&idum,&adum);
       i++;
   } while (ret != EOF);
   S_ia[m-1] = i;

   /* seek to the previously saved file position */
   fseek(fp, ft, 0);

   /* read in column info, S_ja, and values S_data */
   S_ja = talloc(int, S_ia[n]);
   S_data = talloc(double, S_ia[n]);
 
  
   for (i = 0; i < S_ia[n]; i++)
   {
       fscanf(fp, "%d%d%le", &j,&S_ja[i],&S_data[i]);
   }
 
   /* place diagonal first on each row */
   for (i = 0; i < n; i++)
   {
       jdiag = -1;
       for (j = S_ia[i]-1; j < S_ia[i+1]-1; j++)
           if (S_ja[j] == i+1) jdiag = j;

       if (jdiag != -1)
       {
          ctemp = S_ja[jdiag];
          atemp = S_data[jdiag];
          S_ja[jdiag] = S_ja[S_ia[i]-1];
          S_data[jdiag] = S_data[S_ia[i]-1];
          S_ja[S_ia[i]-1] = ctemp;
          S_data[S_ia[i]-1] = atemp;
       }
    }   
   

    fclose(fp);

  
   /* create the matrix */
   A = NewMatrix(S_data, S_ia, S_ja, n); 

   /*-------------------------------------------------------
    * Write out the matrix data
    *-------------------------------------------------------*/

   sprintf(file_name, "%s.in.ysmp", run_name);
   WriteYSMP(file_name, A);

   return 0;
}

