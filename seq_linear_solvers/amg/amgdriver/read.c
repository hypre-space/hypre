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
 * Read routines
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * ReadYSMP
 *--------------------------------------------------------------------------*/

Matrix   *ReadYSMP(file_name)
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
 * ReadVec
 *--------------------------------------------------------------------------*/

Vector   *ReadVec(file_name)
char     *file_name;
{
   Vector  *vector;

   FILE    *fp;

   double  *data;
   int      size;
   
   int      j;


   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   /* read in junk line */
   fscanf(fp, "%*[^\n]\n");

   fscanf(fp, "%d", &size);

   data = ctalloc(double, NDIMU(size));
   for (j = 0; j < size; j++)
      fscanf(fp, "%le", &data[j]);

   fclose(fp);

   /*----------------------------------------------------------
    * Create the vector structure
    *----------------------------------------------------------*/

   vector = NewVector(data, size);

   return vector;
}

