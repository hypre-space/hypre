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
 * hypre_SeqVectorCreate
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorCreate( int size )
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
 * hypre_SeqVectorDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_SeqVectorDestroy( hypre_Vector *vector )
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
 * hypre_SeqVectorInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_SeqVectorInitialize( hypre_Vector *vector )
{
   int  size = hypre_VectorSize(vector);
   int  ierr = 0;

   if ( ! hypre_VectorData(vector) )
      hypre_VectorData(vector) = hypre_CTAlloc(double, size);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_SeqVectorSetDataOwner( hypre_Vector *vector,
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
hypre_SeqVectorRead( char *file_name )
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

   vector = hypre_SeqVectorCreate(size);
   hypre_SeqVectorInitialize(vector);

   data = hypre_VectorData(vector);
   for (j = 0; j < size; j++)
   {
      fscanf(fp, "%le", &data[j]);
   }

   fclose(fp);

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorPrint
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorPrint( hypre_Vector *vector,
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
      fprintf(fp, "%le\n", data[j]);
   }

   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetConstantValues
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorSetConstantValues( hypre_Vector *v,
                               double        value )
{
   double  *vector_data = hypre_VectorData(v);
   int      size        = hypre_VectorSize(v);
           
   int      i;
           
   int      ierr  = 0;


#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < size; i++)
      vector_data[i] = value;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetRandomValues
 *
 *     returns vector of values randomly distributed between -1.0 and +1.0
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorSetRandomValues( hypre_Vector *v,
                             int           seed )
{
   double  *vector_data = hypre_VectorData(v);
   int      size        = hypre_VectorSize(v);
           
   int      i;
           
   int      ierr  = 0;
   hypre_SeedRand(seed);

/* RDF: threading this loop may cause problems because of hypre_Rand() */
   for (i = 0; i < size; i++)
      vector_data[i] = 2.0 * hypre_Rand() - 1.0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCopy
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorCopy( hypre_Vector *x,
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
 * hypre_SeqVectorScale
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorScale( double        alpha,
                   hypre_Vector *y     )
{
   double  *y_data = hypre_VectorData(y);
   int      size   = hypre_VectorSize(y);
           
   int      i;
           
   int      ierr = 0;

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < size; i++)
      y_data[i] *= alpha;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpy
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorAxpy( double        alpha,
            hypre_Vector *x,
            hypre_Vector *y     )
{
   double  *x_data = hypre_VectorData(x);
   double  *y_data = hypre_VectorData(y);
   int      size   = hypre_VectorSize(x);
           
   int      i;
           
   int      ierr = 0;

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < size; i++)
      y_data[i] += alpha * x_data[i];

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInnerProd
 *--------------------------------------------------------------------------*/

double   hypre_SeqVectorInnerProd( hypre_Vector *x,
                          hypre_Vector *y )
{
   double  *x_data = hypre_VectorData(x);
   double  *y_data = hypre_VectorData(y);
   int      size   = hypre_VectorSize(x);
           
   int      i;

   double      result = 0.0;

#define HYPRE_SMP_PRIVATE i
#define HYPRE_SMP_REDUCTION_OP +
#define HYPRE_SMP_REDUCTION_VARS result
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < size; i++)
      result += y_data[i] * x_data[i];

   return result;
}

