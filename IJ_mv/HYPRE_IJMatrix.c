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
 * HYPRE_IJMatrix interface
 *
 *****************************************************************************/

#include "./IJ_mv.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixCreate
 *--------------------------------------------------------------------------*/

int HYPRE_IJMatrixCreate( MPI_Comm comm, int ilower, int iupper, 
			  int jlower, int jupper, HYPRE_IJMatrix *matrix) 

{
   int ierr=0;
   int *row_partitioning;
   int *col_partitioning;
   int *info;
   int *recv_buf;
   int num_procs;
   int i, i4;
   int square;

   hypre_IJMatrix *ijmatrix;

   ijmatrix = hypre_CTAlloc(hypre_IJMatrix, 1);

   hypre_IJMatrixComm(ijmatrix)         = comm;
   hypre_IJMatrixObject(ijmatrix)       = NULL;
   hypre_IJMatrixTranslator(ijmatrix)   = NULL;
   hypre_IJMatrixObjectType(ijmatrix)   = HYPRE_UNITIALIZED;
   hypre_IJMatrixAssembleFlag(ijmatrix) = 0;

   MPI_Comm_size(comm,&num_procs);
 
   info = hypre_CTAlloc(int,4);
   recv_buf = hypre_CTAlloc(int,4*num_procs);
   row_partitioning = hypre_CTAlloc(int, num_procs+1);

   info[0] = ilower;
   info[1] = iupper;
   info[2] = jlower;
   info[3] = jupper;

   /* Generate row- and column-partitioning through information exchange
      across all processors, check whether the matrix is square, and
      if the partitionings match. i.e. no overlaps or gaps,
      if there are overlaps or gaps in the row partitioning or column
      partitioning , ierr will be set to -9 or -10, respectively */

   MPI_Allgather(info,4,MPI_INT,recv_buf,4,MPI_INT,comm);

   row_partitioning[0] = recv_buf[0];
   square = 1;
   for (i=0; i < num_procs-1; i++)
   {
      i4 = 4*i;
      if ( recv_buf[i4+1] != (recv_buf[i4+4]-1) )
      {
         printf("Warning -- row partitioning does not line up!\n");
	 ierr = -9;
	 break;
      }
      else
	 row_partitioning[i+1] = recv_buf[i4+4];
	 
      if (square && (recv_buf[i4]   != recv_buf[i4+2]) ||
                    (recv_buf[i4+1] != recv_buf[i4+3])  )
      {
         square = 0;
      }
   }	
   i4 = (num_procs-1)*4;
   row_partitioning[num_procs] = recv_buf[i4+1]+1;

   if ((recv_buf[i4] != recv_buf[i4+2]) || (recv_buf[i4+1] != recv_buf[i4+3])) 
      square = 0;

   if (square)
      col_partitioning = row_partitioning;
   else
   {   
      col_partitioning = hypre_CTAlloc(int,num_procs+1);
      col_partitioning[0] = recv_buf[2];
      for (i=0; i < num_procs-1; i++)
      {
         i4 = 4*i;
         if (recv_buf[i4+3] != recv_buf[i4+6]-1)
         {
           printf("Warning -- col partitioning does not line up!\n");
   	   ierr = -10;
   	   break;
         }
         else
   	   col_partitioning[i+1] = recv_buf[i4+6];
      }
      col_partitioning[num_procs] = recv_buf[num_procs*4-1]+1;
   }

   hypre_TFree(info);
   hypre_TFree(recv_buf);
   
   hypre_IJMatrixRowPartitioning(ijmatrix) = row_partitioning;
   hypre_IJMatrixColPartitioning(ijmatrix) = col_partitioning;

   *matrix = (HYPRE_IJMatrix) ijmatrix;
  
   return ierr; 
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixDestroy( HYPRE_IJMatrix matrix )
{
   int ierr = 0;
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixDestroy\n");
      exit(1);
   }

   if (ijmatrix)
   {
      if (hypre_IJMatrixRowPartitioning(ijmatrix) ==
                      hypre_IJMatrixColPartitioning(ijmatrix))
         hypre_TFree(hypre_IJMatrixRowPartitioning(ijmatrix));
      else
      {
         hypre_TFree(hypre_IJMatrixRowPartitioning(ijmatrix));
         hypre_TFree(hypre_IJMatrixColPartitioning(ijmatrix));
      }

      /*
      if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
         ierr = hypre_IJMatrixDestroyPETSc( ijmatrix );
      else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
         ierr = hypre_IJMatrixDestroyISIS( ijmatrix );
      else */ 

      if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
         ierr = hypre_IJMatrixDestroyParCSR( ijmatrix );
      else
      {
         /* Tong : to be able to destroy IJ without destroying the underlying
                   matrix 
         printf("Unrecognized object type -- HYPRE_IJMatrixDestroy\n");
         exit(1);
         */
         ierr = 1;
      }
   }

   hypre_TFree(ijmatrix); 

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixInitialize( HYPRE_IJMatrix matrix )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;
   int ierr = 0;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixInitialize\n");
      return 1;
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixInitializePETSc( ijmatrix ) ;
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixInitializeISIS( ijmatrix ) ;
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixInitializeParCSR( ijmatrix ) ;
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixInitialize\n");
      return 1;
   }
   return ierr; 
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetValues( HYPRE_IJMatrix matrix, int nrows,
                         int *ncols, const int *rows,
                         const int *cols, const double *values)
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixSetValues\n");
      exit(1);
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      return( hypre_IJMatrixSetValuesPETSc( ijmatrix, nrows, ncols, 
                                            rows, cols, values ) );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      return( hypre_IJMatrixSetValuesISIS( ijmatrix, nrows, ncols, 
                                           rows, cols, values ) );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      return( hypre_IJMatrixSetValuesParCSR( ijmatrix, nrows, ncols,
                                             rows, cols, values ) );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixSetValues\n");
      exit(1);
   }
    
   return -99;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToValues
 *--------------------------------------------------------------------------*/


int 
HYPRE_IJMatrixAddToValues( HYPRE_IJMatrix matrix, int nrows,
                           int *ncols, const int *rows,
                           const int *cols, const double *values)
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixAddToValues\n");
      exit(1);
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      return( hypre_IJMatrixAddToValuesPETSc( ijmatrix, nrows, ncols, 
                                              rows, cols, values ) );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      return( hypre_IJMatrixAddToValuesISIS( ijmatrix, nrows, ncols, 
                                             rows, cols, values ) );
   else */ if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      return( hypre_IJMatrixAddToValuesParCSR( ijmatrix, nrows, ncols,
                                               rows, cols, values ) );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixAddToValues\n");
      exit(1);
   }

   return -99;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixAssemble( HYPRE_IJMatrix matrix )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixAssemble\n");
      exit(1);
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      return( hypre_IJMatrixAssemblePETSc( ijmatrix ) );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      return( hypre_IJMatrixAssembleISIS( ijmatrix ) );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      return( hypre_IJMatrixAssembleParCSR( ijmatrix ) );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixAssemble\n");
      exit(1);
   }

   return -99;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixGetValues( HYPRE_IJMatrix matrix, int nrows, int *ncols,
                         int *rows, int *cols, double *values)
{
   int ierr = 0;
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixGetValues\n");
      exit(1);
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixGetValuesPETSc( ijmatrix, nrows, ncols, 
					   rows, cols, values );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixGetValuesISIS( ijmatrix, nrows, ncols, 
					  rows, cols, values );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixGetValuesParCSR( ijmatrix, nrows, ncols,
					    rows, cols, values );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixGetValues\n");
      exit(1);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetObjectType
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetObjectType( HYPRE_IJMatrix matrix, int type )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixSetObjectType\n");
      exit(1);
   }

   hypre_IJMatrixObjectType(ijmatrix) = type;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetObjectType
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixGetObjectType( HYPRE_IJMatrix matrix, int *type )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixGetObjectType\n");
      exit(1);
   }

   *type = hypre_IJMatrixObjectType(ijmatrix);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetObject
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to an underlying ijmatrix type used to implement IJMatrix.
Assumes that the implementation has an underlying matrix, so it would not
work with a direct implementation of IJMatrix. 

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

int
HYPRE_IJMatrixGetObject( HYPRE_IJMatrix matrix, void **object )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixGetObject\n");
      exit(1);
   }

   *object = hypre_IJMatrixObject( ijmatrix );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetRowSizes
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetRowSizes( HYPRE_IJMatrix matrix, const int *sizes )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixSetRowSizes\n");
      exit(1);
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      return( hypre_IJMatrixSetRowSizesPETSc( ijmatrix , sizes ) );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      return( hypre_IJMatrixSetRowSizesISIS( ijmatrix , sizes ) );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      return( hypre_IJMatrixSetRowSizesParCSR( ijmatrix , sizes ) );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixSetRowSizes\n");
      exit(1);
   }

   return -99;
}


/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetDiagOffdSizes
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetDiagOffdSizes( HYPRE_IJMatrix matrix, 
				const int *diag_sizes,
				const int *offdiag_sizes )
{
   int ierr = 0;
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixSetDiagOffdSizes\n");
      exit(1);
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixSetDiagOffdSizesPETSc( ijmatrix , diag_sizes ,
						offdiag_sizes );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixSetDiagOffdSizesISIS( ijmatrix , diag_sizes ,
						offdiag_sizes );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      ierr  = hypre_IJMatrixSetDiagOffdSizesParCSR( ijmatrix , diag_sizes ,
							offdiag_sizes );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixSetDiagOffdSizes\n");
      exit(1);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixRead
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixRead( const char *filename, MPI_Comm comm, int type,
		    HYPRE_IJMatrix *matrix)
{
   int ierr = 0;
   hypre_IJMatrix *ijmatrix;

   /* if ( type == HYPRE_PETSC )
      ierr = hypre_IJMatrixReadPETSc( comm, filename, &ijmatrix );
   else if ( type == HYPRE_ISIS )
      ierr = hypre_IJMatrixReadISIS( comm, filename, &ijmatrix );
   else */ if ( type == HYPRE_PARCSR )
      ierr = hypre_IJMatrixReadParCSR( comm, filename, &ijmatrix );
   else 
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixRead\n");
      exit(1);
   }

   *matrix = (HYPRE_IJMatrix ) ijmatrix;
   hypre_IJMatrixAssembleFlag(ijmatrix) = 1;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixPrint
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixPrint( HYPRE_IJMatrix matrix, const char *filename)
{
   int ierr = 0;
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixPrint\n");
      exit(1);
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixPrintPETSc( ijmatrix , filename );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixPrintISIS( ijmatrix , filename );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixPrintParCSR( ijmatrix , filename );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixPrint\n");
      exit(1);
   }

   return ierr;
}
