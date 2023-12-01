/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_MappedMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_MappedMatrix *
hypre_MappedMatrixCreate( void )
{
   hypre_MappedMatrix  *matrix;


   matrix = hypre_CTAlloc(hypre_MappedMatrix,  1, HYPRE_MEMORY_HOST);

   return ( matrix );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixDestroy( hypre_MappedMatrix *matrix )
{
   HYPRE_Int  ierr = 0;

   if (matrix)
   {
      hypre_TFree(hypre_MappedMatrixMatrix(matrix), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_MappedMatrixMapData(matrix), HYPRE_MEMORY_HOST);

      hypre_TFree(matrix, HYPRE_MEMORY_HOST);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixLimitedDestroy( hypre_MappedMatrix *matrix )
{
   HYPRE_Int  ierr = 0;

   if (matrix)
   {
      hypre_TFree(matrix, HYPRE_MEMORY_HOST);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixInitialize( hypre_MappedMatrix *matrix )
{
   HYPRE_Int ierr = 0;

   HYPRE_UNUSED_VAR(matrix);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixAssemble
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixAssemble( hypre_MappedMatrix *matrix )
{
   HYPRE_Int    ierr = 0;

   if ( matrix == NULL )
   {
      return ( -1 ) ;
   }

   if ( hypre_MappedMatrixMatrix(matrix) == NULL )
   {
      return ( -1 ) ;
   }

   if ( hypre_MappedMatrixColMap(matrix) == NULL )
   {
      return ( -1 ) ;
   }

   if ( hypre_MappedMatrixMapData(matrix) == NULL )
   {
      return ( -1 ) ;
   }

   return (ierr);
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_MappedMatrixPrint(hypre_MappedMatrix *matrix  )
{
   HYPRE_UNUSED_VAR(matrix);

   hypre_printf("Stub for hypre_MappedMatrix\n");
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixGetColIndex
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixGetColIndex(hypre_MappedMatrix *matrix, HYPRE_Int j  )
{
   return ( hypre_MappedMatrixColIndex(matrix, j) );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixGetMatrix
 *--------------------------------------------------------------------------*/

void *
hypre_MappedMatrixGetMatrix(hypre_MappedMatrix *matrix )
{
   return ( hypre_MappedMatrixMatrix(matrix) );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetMatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixSetMatrix(hypre_MappedMatrix *matrix, void *matrix_data  )
{
   HYPRE_Int ierr = 0;

   hypre_MappedMatrixMatrix(matrix) = matrix_data;

   return (ierr);
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetColMap
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixSetColMap(hypre_MappedMatrix *matrix,
                            HYPRE_Int (*ColMap)(HYPRE_Int, void *)  )
{
   HYPRE_Int ierr = 0;

   hypre_MappedMatrixColMap(matrix) = ColMap;

   return (ierr);
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetMapData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MappedMatrixSetMapData(hypre_MappedMatrix *matrix,
                             void *map_data )
{
   HYPRE_Int ierr = 0;

   hypre_MappedMatrixMapData(matrix) = map_data;

   return (ierr);
}
