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
 * hypre_VectorCreate
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_VectorCreate( int size )
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
 * hypre_VectorDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_VectorDestroy( hypre_Vector *vector )
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
 * hypre_VectorInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_VectorInitialize( hypre_Vector *vector )
{
   int  size = hypre_VectorSize(vector);
   int  ierr = 0;

   if ( ! hypre_VectorData(vector) )
      hypre_VectorData(vector) = hypre_CTAlloc(double, size);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_VectorSetDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_VectorSetDataOwner( hypre_Vector *vector,
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
hypre_VectorRead( char *file_name )
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

   vector = hypre_VectorCreate(size);
   hypre_VectorInitialize(vector);

   data = hypre_VectorData(vector);
   for (j = 0; j < size; j++)
   {
      fscanf(fp, "%le", &data[j]);
   }

   fclose(fp);

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_VectorPrint
 *--------------------------------------------------------------------------*/

int
hypre_VectorPrint( hypre_Vector *vector,
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
 * hypre_VectorSetConstantValues
 *--------------------------------------------------------------------------*/

int
hypre_VectorSetConstantValues( hypre_Vector *v,
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
 * hypre_VectorSetRandomValues
 *
 *     returns vector of values randomly distributed between -1.0 and +1.0
 *--------------------------------------------------------------------------*/

int
hypre_VectorSetRandomValues( hypre_Vector *v,
                             int           seed )
{
   double  *vector_data = hypre_VectorData(v);
   int      size        = hypre_VectorSize(v);
           
   int      i;
           
   int      ierr  = 0;
   hypre_SeedRand(seed);

   for (i = 0; i < size; i++)
      vector_data[i] = 2.0 * hypre_Rand() - 1.0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_VectorCopy
 *--------------------------------------------------------------------------*/

int
hypre_VectorCopy( hypre_Vector *x,
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
 * hypre_VectorScale
 *--------------------------------------------------------------------------*/

int
hypre_VectorScale( double        alpha,
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
 * hypre_VectorAxpy
 *--------------------------------------------------------------------------*/

int
hypre_VectorAxpy( double        alpha,
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
 * hypre_VectorInnerProd
 *--------------------------------------------------------------------------*/

double   hypre_VectorInnerProd( hypre_Vector *x,
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

