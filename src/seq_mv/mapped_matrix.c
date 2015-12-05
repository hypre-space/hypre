/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Member functions for hypre_MappedMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_MappedMatrix *
hypre_MappedMatrixCreate(  )
{
   hypre_MappedMatrix  *matrix;


   matrix = hypre_CTAlloc(hypre_MappedMatrix, 1);

   return ( matrix );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_MappedMatrixDestroy( hypre_MappedMatrix *matrix )
{
   HYPRE_Int  ierr=0;

   if (matrix)
   {
      hypre_TFree(hypre_MappedMatrixMatrix(matrix));
      hypre_TFree(hypre_MappedMatrixMapData(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_MappedMatrixLimitedDestroy( hypre_MappedMatrix *matrix )
{
   HYPRE_Int  ierr=0;

   if (matrix)
   {
      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_MappedMatrixInitialize( hypre_MappedMatrix *matrix )
{
   HYPRE_Int    ierr=0;

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixAssemble
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_MappedMatrixAssemble( hypre_MappedMatrix *matrix )
{
   HYPRE_Int    ierr=0;

   if( matrix == NULL )
      return ( -1 ) ;

   if( hypre_MappedMatrixMatrix(matrix) == NULL )
      return ( -1 ) ;

   if( hypre_MappedMatrixColMap(matrix) == NULL )
      return ( -1 ) ;

   if( hypre_MappedMatrixMapData(matrix) == NULL )
      return ( -1 ) ;

   return(ierr);
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_MappedMatrixPrint(hypre_MappedMatrix *matrix  )
{
   hypre_printf("Stub for hypre_MappedMatrix\n");
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixGetColIndex
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixGetColIndex(hypre_MappedMatrix *matrix, HYPRE_Int j  )
{
   return( hypre_MappedMatrixColIndex(matrix,j) );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixGetMatrix
 *--------------------------------------------------------------------------*/

void *
hypre_MappedMatrixGetMatrix(hypre_MappedMatrix *matrix )
{
   return( hypre_MappedMatrixMatrix(matrix) );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetMatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixSetMatrix(hypre_MappedMatrix *matrix, void *matrix_data  )
{
   HYPRE_Int ierr=0;

   hypre_MappedMatrixMatrix(matrix) = matrix_data;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetColMap
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixSetColMap(hypre_MappedMatrix *matrix, 
                          HYPRE_Int (*ColMap)(HYPRE_Int, void *)  )
{
   HYPRE_Int ierr=0;

   hypre_MappedMatrixColMap(matrix) = ColMap;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetMapData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixSetMapData(hypre_MappedMatrix *matrix, 
                          void *map_data )
{
   HYPRE_Int ierr=0;

   hypre_MappedMatrixMapData(matrix) = map_data;

   return(ierr);
}

