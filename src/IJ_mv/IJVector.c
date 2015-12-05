/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * hypre_IJVector interface
 *
 *****************************************************************************/

#include "./_hypre_IJ_mv.h"

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
