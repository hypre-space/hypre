/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * hypre_IJVector interface
 *
 *****************************************************************************/

#include "./IJ_mv.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * hypre_IJVectorDistribute
 *--------------------------------------------------------------------------*/

int 
hypre_IJVectorDistribute( HYPRE_IJVector vector, const int *vec_starts )
{
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (vec == NULL)
   {
      printf("Vector variable is NULL -- hypre_IJVectorDistribute\n");
      exit(1);
   } 

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )

      return( hypre_IJVectorDistributePar(vec, vec_starts) );

   else
   {
      printf("Unrecognized object type -- hypre_IJVectorDistribute\n");
      exit(1);
   }

   return -99;
}

/*--------------------------------------------------------------------------
 * hypre_IJVectorZeroValues
 *--------------------------------------------------------------------------*/

int 
hypre_IJVectorZeroValues( HYPRE_IJVector vector )
{
   int ierr = 0;
   hypre_IJVector *vec = (hypre_IJVector *) vector;

   if (vec == NULL)
   {
      printf("Vector variable is NULL -- hypre_IJVectorZeroValues\n");
      exit(1);
   } 

   /*  if ( hypre_IJVectorObjectType(vec) == HYPRE_PETSC )

      return( hypre_IJVectorZeroValuesPETSc(vec) );

   else if ( hypre_IJVectorObjectType(vec) == HYPRE_ISIS )

      return( hypre_IJVectorZeroValuesISIS(vec) );

   else */

   if ( hypre_IJVectorObjectType(vec) == HYPRE_PARCSR )

      return( hypre_IJVectorZeroValuesPar(vec) );

   else
   {
      printf("Unrecognized object type -- hypre_IJVectorZeroValues\n");
      exit(1);
   }

   return -99;
}
