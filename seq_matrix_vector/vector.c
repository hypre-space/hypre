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
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CreateVector
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_CreateVector( int size )
{
   hypre_Vector  *vector;

   vector = hypre_CTAlloc(hypre_Vector, 1);

   hypre_VectorData(vector) = NULL;
   hypre_VectorSize(vector) = size;

   /* set defaults */
   hypre_VectorOwnsData(vector) = 1;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_DestroyVector
 *--------------------------------------------------------------------------*/

int 
hypre_DestroyVector( hypre_Vector *vector )
{
   int  ierr=0;

   if (vector)
   {
      if ( hypre_VectorOwnsData(vector) )
      {
         hypre_TFree(hypre_VectorData(vector));
      }
      hypre_TFree(vector);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_InitializeVector
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeVector( hypre_Vector *vector )
{
   int  size = hypre_VectorSize(vector);
   int  ierr = 0;

   if ( ! hypre_VectorData(vector) )
      hypre_VectorData(vector) = hypre_CTAlloc(double, size);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetVectorDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_SetVectorDataOwner( hypre_Vector *vector,
                          int           owns_data   )
{
   int    ierr=0;

   hypre_VectorOwnsData(vector) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * ReadVector
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_ReadVector( char *file_name )
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

   vector = hypre_CreateVector(size);
   hypre_InitializeVector(vector);

   data = hypre_VectorData(vector);
   for (j = 0; j < size; j++)
   {
      fscanf(fp, "%le", &data[j]);
   }

   fclose(fp);

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_PrintVector
 *--------------------------------------------------------------------------*/

int
hypre_PrintVector( hypre_Vector *vector,
                   char         *file_name )
{
   FILE    *fp;

   double  *data;
   int      size;
   
   int      j;

   int      ierr = 0;

   /*----------------------------------------------------------
    * Print in the data
    *----------------------------------------------------------*/

   data = hypre_VectorData(vector);
   size = hypre_VectorSize(vector);

   fp = fopen(file_name, "w");

   fprintf(fp, "%d\n", size);

   for (j = 0; j < size; j++)
   {
      fprintf(fp, "%e\n", data[j]);
   }

   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetVectorConstantValues
 *--------------------------------------------------------------------------*/

int
hypre_SetVectorConstantValues( hypre_Vector *v,
                               double        value )
{
   double  *vector_data = hypre_VectorData(v);
   int      size        = hypre_VectorSize(v);
           
   int      i;
           
   int      ierr  = 0;


   for (i = 0; i < size; i++)
      vector_data[i] = value;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CopyVector
 *--------------------------------------------------------------------------*/

int
hypre_CopyVector( hypre_Vector *x,
                  hypre_Vector *y )
{
   double  *x_data = hypre_VectorData(x);
   double  *y_data = hypre_VectorData(y);
   int      size   = hypre_VectorSize(x);
           
   int      i;
           
   int      ierr = 0;

   for (i = 0; i < size; i++)
      y_data[i] = x_data[i];

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ScaleVector
 *--------------------------------------------------------------------------*/

int
hypre_ScaleVector( double        alpha,
                   hypre_Vector *y     )
{
   double  *y_data = hypre_VectorData(y);
   int      size   = hypre_VectorSize(y);
           
   int      i;
           
   int      ierr = 0;

   for (i = 0; i < size; i++)
      y_data[i] *= alpha;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_Axpy
 *--------------------------------------------------------------------------*/

int
hypre_Axpy( double        alpha,
            hypre_Vector *x,
            hypre_Vector *y     )
{
   double  *x_data = hypre_VectorData(x);
   double  *y_data = hypre_VectorData(y);
   int      size   = hypre_VectorSize(x);
           
   int      i;
           
   int      ierr = 0;

   for (i = 0; i < size; i++)
      y_data[i] += alpha * x_data[i];

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_InnerProd
 *--------------------------------------------------------------------------*/

double   hypre_InnerProd( hypre_Vector *x,
                          hypre_Vector *y )
{
   double  *x_data = hypre_VectorData(x);
   double  *y_data = hypre_VectorData(y);
   int      size   = hypre_VectorSize(x);
           
   int      i;

   double      result = 0.0;

   for (i = 0; i < size; i++)
      result += y_data[i] * x_data[i];
  
   return result;
}

