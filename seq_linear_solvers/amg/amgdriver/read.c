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

hypre_Matrix   *ReadYSMP(file_name)
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
 * ReadVec
 *--------------------------------------------------------------------------*/

hypre_Vector   *ReadVec(file_name)
char     *file_name;
{
   hypre_Vector  *vector;

   FILE    *fp;

   double  *data;
   int      size;
   
   int      j;


   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   fscanf(fp, "%d", &size);

   data = hypre_CTAlloc(double, hypre_NDIMU(size));
   for (j = 0; j < size; j++)
      fscanf(fp, "%le", &data[j]);

   fclose(fp);

   /*----------------------------------------------------------
    * Create the vector structure
    *----------------------------------------------------------*/

   vector = hypre_NewVector(data, size);

   return vector;
}

