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

#include "io.h"


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

