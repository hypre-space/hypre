/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_MappedMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewMappedMatrix
 *--------------------------------------------------------------------------*/

hypre_MappedMatrix *
hypre_NewMappedMatrix(  )
{
   hypre_MappedMatrix  *matrix;


   matrix = hypre_CTAlloc(hypre_MappedMatrix, 1);

   return ( matrix );
}

/*--------------------------------------------------------------------------
 * hypre_FreeMappedMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_FreeMappedMatrix( hypre_MappedMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      hypre_TFree(hypre_MappedMatrixMatrix(matrix));
      hypre_TFree(hypre_MappedMatrixMapData(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_LimitedFreeMappedMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_LimitedFreeMappedMatrix( hypre_MappedMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_InitializeMappedMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeMappedMatrix( hypre_MappedMatrix *matrix )
{
   int    ierr=0;

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_AssembleMappedMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleMappedMatrix( hypre_MappedMatrix *matrix )
{
   int    ierr=0;

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
 * hypre_PrintMappedMatrix
 *--------------------------------------------------------------------------*/

void
hypre_PrintMappedMatrix(hypre_MappedMatrix *matrix  )
{
   printf("Stub for hypre_MappedMatrix\n");
}

/*--------------------------------------------------------------------------
 * hypre_GetMappedMatrixColIndex
 *--------------------------------------------------------------------------*/

int
hypre_GetMappedMatrixColIndex(hypre_MappedMatrix *matrix, int j  )
{
   return( hypre_MappedMatrixColIndex(matrix,j) );
}

/*--------------------------------------------------------------------------
 * hypre_GetMappedMatrixMatrix
 *--------------------------------------------------------------------------*/

void *
hypre_GetMappedMatrixMatrix(hypre_MappedMatrix *matrix )
{
   return( hypre_MappedMatrixMatrix(matrix) );
}

/*--------------------------------------------------------------------------
 * hypre_SetMappedMatrixMatrix
 *--------------------------------------------------------------------------*/

int
hypre_SetMappedMatrixMatrix(hypre_MappedMatrix *matrix, void *matrix_data  )
{
   int ierr=0;

   hypre_MappedMatrixMatrix(matrix) = matrix_data;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SetMappedMatrixColMap
 *--------------------------------------------------------------------------*/

int
hypre_SetMappedMatrixColMap(hypre_MappedMatrix *matrix, 
                          int (*ColMap)(int, void *)  )
{
   int ierr=0;

   hypre_MappedMatrixColMap(matrix) = ColMap;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SetMappedMatrixMapData
 *--------------------------------------------------------------------------*/

int
hypre_SetMappedMatrixMapData(hypre_MappedMatrix *matrix, 
                          void *map_data )
{
   int ierr=0;

   hypre_MappedMatrixMapData(matrix) = map_data;

   return(ierr);
}

