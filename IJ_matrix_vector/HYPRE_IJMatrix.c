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

#include "./IJ_matrix_vector.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixCreate
 *--------------------------------------------------------------------------*/

/*---------------------------------------------------------------------- */
/** 
"IJ" represents the "linear algebraic" conceptual view of a matrix. 
The "I" and "J" in the name
are meant to be reminiscent of traditional matrix notation like A(I,J).

Collective.

{\bf Note:} Must be the first function called using "matrix" as an actual argument.
@return integer error code
@param HYPRE_IJMatrix &IJmatrix: the matrix to be initialized.
@param MPI_Comm comm: a single MPI_Communicator that contains exactly the MPI processes that are to
participate in any collective operations.
@param int global_m, global_n: the dimensions of the entire, global matrix.
@param int local_m, local_n: the dimensions of the locally stored matrix.
*/
/*---------------------------------------------------------------------- */

int HYPRE_IJMatrixCreate( MPI_Comm comm, HYPRE_IJMatrix *in_matrix_ptr, 
          int global_m, int global_n)

{
   int ierr=0;

   hypre_IJMatrix    *matrix;

   matrix = hypre_CTAlloc(hypre_IJMatrix, 1);

   hypre_IJMatrixContext(matrix) = comm;
   hypre_IJMatrixM(matrix)    = global_m;
   hypre_IJMatrixN(matrix)    = global_n;
   hypre_IJMatrixLocalStorage(matrix) = NULL;
   hypre_IJMatrixTranslator(matrix) = NULL;
   hypre_IJMatrixLocalStorageType(matrix) = HYPRE_UNITIALIZED;
   hypre_IJMatrixInsertionSemantics(matrix) = 0;
   hypre_IJMatrixReferenceCount(matrix) = 1;

   *in_matrix_ptr = (HYPRE_IJMatrix) matrix;
  
   return( ierr ); 
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixDestroy( HYPRE_IJMatrix IJmatrix )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   if (matrix)
   {
      hypre_IJMatrixReferenceCount( matrix ) --;
   
      if ( hypre_IJMatrixReferenceCount( matrix ) <= 0 )
      {
	/*
         if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
            ierr = hypre_IJMatrixDestroyPETSc( matrix );
         else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
            ierr = hypre_IJMatrixDestroyISIS( matrix );
         else */ 
	 if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
            ierr = hypre_IJMatrixDestroyParCSR( matrix );
         else
            ierr = -1;

         hypre_TFree(matrix);
      }
   }
   else
   {
      ierr = -1;
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixInitialize( HYPRE_IJMatrix IJmatrix )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixInitializePETSc( matrix );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixInitializeISIS( matrix );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixInitializeParCSR( matrix );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixAssemble( HYPRE_IJMatrix IJmatrix )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   if( (hypre_IJMatrixM(matrix) < 0 ) ||
       (hypre_IJMatrixN(matrix) < 0 ) )
      return(-1);

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixAssemblePETSc( matrix );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixAssembleISIS( matrix );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixAssembleParCSR( matrix );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixDistribute
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixDistribute( HYPRE_IJMatrix IJmatrix, 
			  const int     *row_starts,
			  const int     *col_starts )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixDistributeParCSR( matrix, row_starts, col_starts );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetLocalStorageType
 *--------------------------------------------------------------------------*/

/**
Tells "matrix" which underlying "storage type" it should build from the IJMatrix interface
calls. Conceptually, "IJMatrix" is just an "interface" (an abstract base class in C++-speak),
with no implementation. Specific concrete classes implement this interface. One way for a user
to choose which concrete class to use is by *instantiating* the class of choice (in an OO
implementation) or by *linking* in the implementation of choice (a "compile time" choice). An
intermediate method is through the use of HYPRE_SetIJMatrixStorageType that *may* be provided
by specific implementations. For example, it is possible to build either a PETSc MPIAIJ matrix
or a ParCSR matrix from the IJ matrix interface, and this choice can be controlled through this
function.

Not collective, but must be same on all processors in group that stores matrix.
@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
 the matrix to be operated on. 
@param int type [IN]
possible types should be documented with each implementation. The first HYPRE implementation
will list its possible values in HYPRE.h (at least for now). Note that this function is optional,
as all implementations must have a default. Eventually the plan is to let solvers choose a storage
type that they can work with automatically, but this is in the future.
*/

int 
HYPRE_IJMatrixSetLocalStorageType( HYPRE_IJMatrix IJmatrix, int type )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   hypre_IJMatrixLocalStorageType(matrix) = type;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetLocalSize
 *--------------------------------------------------------------------------*/

/**
Tells "matrix" how many rows and columns are locally owned.
Not collective.
REQUIREMENTS: HYPRE_IJMatrixSetLocalStorageType needs to be called before this routine

@return integer error code
@param HYPRE_IJMatrix matrix [IN]
 the matrix to be operated on. 
@param int local_m [IN]
 local number of rows
@param int local_n [IN]
 local number of columns
*/

int 
HYPRE_IJMatrixSetLocalSize( HYPRE_IJMatrix IJmatrix, int local_m, int local_n )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixSetLocalSizePETSc (matrix, local_m, local_n);
   if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixSetLocalSizeISIS (matrix, local_m, local_n);
      */
   if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixSetLocalSizeParCSR (matrix, local_m, local_n);
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetRowSizes
 *--------------------------------------------------------------------------*/

/**
* HYPRE_IJMatrixSetRowSizes( HYPRE_IJMatrix IJmatrix, const int *sizes); 

Not collective.
Tells "matrix" how many nonzeros to expect in each row.
Knowing this quantity apriori may have a significant impact on the time needed for the Assemble phase,
and this option should always be utilized if the information is available.
@return integer error code
@param HYPRE_IJMatrix &matrix [IN] 
the matrix to be operated on.
@param int *sizes [IN]
a vector of length = local_m giving the estimated sizes for the diagonal parts of
all local_m rows, in order from lowest globally numbered local row to highest.
*/

int 
HYPRE_IJMatrixSetRowSizes( HYPRE_IJMatrix IJmatrix, const int *sizes )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixSetRowSizesPETSc( matrix , sizes );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixSetRowSizesISIS( matrix , sizes );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixSetRowSizesParCSR( matrix , sizes );
   else
      ierr = -1;

   return(ierr);
}


/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetDiagRowSizes
 *--------------------------------------------------------------------------*/

/**
* HYPRE_IJMatrixSetDiagRowSizes( HYPRE_IJMatrix IJmatrix, const int *sizes); 

Not collective.
Tells "matrix" how many nonzeros to expect in each row corresponding to other rows also
on this processor, also known as the "diagonal block" corresponding to this processor. Knowing
this quantity apriori may have a significant impact on the time needed for the Assemble phase,
and this option should always be utilized if the information is available.It is most useful in conjunction
with the next function.
@return integer error code
@param HYPRE_IJMatrix &matrix [IN] 
the matrix to be operated on.
@param int *sizes [IN]
a vector of length = local_m giving the exact sizes for the diagonal parts of
all local_m rows, in order from lowest globally numbered local row to highest.
*/

int 
HYPRE_IJMatrixSetDiagRowSizes( HYPRE_IJMatrix IJmatrix, const int *sizes )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixSetDiagRowSizesPETSc( matrix , sizes );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixSetDiagRowSizesISIS( matrix , sizes );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixSetDiagRowSizesParCSR( matrix , sizes );
   else
      ierr = -1;

   return(ierr);
}


/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetOffDiagRowSizes
 *--------------------------------------------------------------------------*/

/**
* HYPRE_IJMatrixSetOffDiagRowSizes( HYPRE_IJMatrix matrix, const int *sizes);

Tells "matrix" how many nonzeros to expect in each corresponding to rows NOT
on this processor, i.e. the off-diagonal block corresponding to this processor. As above, this option
should always be utilized if the information is available, as the setting of coefficients and assemble
operations can be very expensive without them.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on
@param int *sizes [IN]
a vector of length >= local_m giving the exact sizes for the off-diagonal parts of
all local_m rows, in order from lowest globally numbered local row to highest
*/

int 
HYPRE_IJMatrixSetOffDiagRowSizes( HYPRE_IJMatrix IJmatrix, const int *sizes )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixSetOffDiagRowSizesPETSc( matrix , sizes );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixSetOffDiagRowSizesISIS( matrix , sizes );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixSetOffDiagRowSizesParCSR( matrix , sizes );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixQueryInsertionSemantics
 *--------------------------------------------------------------------------*/

/** 
{\bf Note:} It is probably best to ignore  HYPRE_IJMatrixQueryInsertionSemantics for the time being as it is really
planned as a possible future contingency and is confusing as it is now. -AJC, 6/99

There are three possible semantic "levels" in the integer parameter "row" used in Set and Add
functions, and each implementation supports one of them. This function returns a value indicating
the semantics supported by the instantiated IJMatrix. 
level = -1: processors may include values for "row" for any row number in the global matrix.
level = 0: processors may only include values for row representing locally stored rows.
level = > 0: in addition to the above, processors may also include values for "row" representing
rows within the set of locally stored rows plus the next "level" levels of nearest neighbors, 
also known as "level sets" or "nearest neighbors".
Since level 0 is the most restrictive, it is also the easiest to implement, and the safest to use, as
all IJ matrices MUST support level 0.
In contrast, level 1 is the most general and most difficult to implement, least safe to use, and
potentially least efficient.
Levels greater than 0 represent a compromise that is appropriate for many engineering applications,
like finite element applications, where contributions to a particular matrix row may be made from
more than one processor, typically in some small neighborhood around that row. 

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be initialized.
@param int *level [OUT]
level of off-processor value-setting that this implementation supports
*/

int 
HYPRE_IJMatrixQueryInsertionSemantics( HYPRE_IJMatrix IJmatrix, int *level )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   *level = hypre_IJMatrixInsertionSemantics(matrix);

   return(ierr);

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInsertBlock
 *--------------------------------------------------------------------------*/

/** 
Inserts a block of coefficients into an IJMatrix, overwriting any coefficients
in the event of a collision.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int m, n [IN]
the size of the block of values to be added.
@param int *rows [IN]
an integer vector of length m giving the indices in the global matrix
corresponding to the rows in "values".
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be inserted into the matrix, stored in a dense, row-major
block of size m X n.

*/

int 
HYPRE_IJMatrixInsertBlock( HYPRE_IJMatrix IJmatrix, int m, int n,
                           const int *rows, const int *cols, 
			   const double *values)
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /*  if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixInsertBlockPETSc( matrix, m, n, rows, cols, values );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixInsertBlockISIS( matrix, m, n, rows, cols, values );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixInsertBlockParCSR( matrix, m, n, rows, cols, values );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToBlock
 *--------------------------------------------------------------------------*/

/** 
Modifies the values stored in matrix by adding in a block.
If there is no value already in a particular matrix position, the structure is augmented
with a new entry. In the event of a collision, the corresponding values are summed.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int m, n [IN]
the size of the block of values to be added.
@param int *rows [IN]
an integer vector of length m giving the indices in the global matrix
corresponding to the rows in "values".
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be inserted into the matrix, stored in a dense, row-major
block of size m X n.

*/

int 
HYPRE_IJMatrixAddToBlock( HYPRE_IJMatrix IJmatrix, int m, int n,
                          const int *rows, const int *cols, 
			  const double *values)
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixAddToBlockPETSc( matrix, m, n, rows, cols, values );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixAddToBlockISIS( matrix, m, n, rows, cols, values );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixAddToBlockParCSR( matrix, m, n, rows, cols, values );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInsertRow
 *--------------------------------------------------------------------------*/

/** 
Inserts a row into the matrix. This is generally a high-speed but inflexible method to build the
matrix. This call replaces any previously existing row structure with the structure represented by
indices and coeffs.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int n [IN]
the number of values in the row to be inserted.
@param int row [IN]
index of row to be inserted.
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be inserted into the matrix.
*/

int 
HYPRE_IJMatrixInsertRow( HYPRE_IJMatrix IJmatrix, int n,
                           int row, const int *cols, const double *values)
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixInsertRowPETSc( matrix, n, row, cols, values );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixInsertRowISIS( matrix, n, row, cols, values );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
     /* Currently a slight mismatch between "Insert" and "Set" */
      ierr = hypre_IJMatrixInsertRowParCSR( matrix, n, row, cols, values );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToRow
 *--------------------------------------------------------------------------*/

/** 
Adds a row to the row of a matrix before assembly. 

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int n [IN]
the number of values in the row to be added.
@param int row [IN]
index of row to be added.
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be added to the matrix.
*/

int 
HYPRE_IJMatrixAddToRow( HYPRE_IJMatrix IJmatrix, int n,
                           int row, const int *cols, const double *values)
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixAddToRowPETSc( matrix, n, row, cols, values );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixAddToRowISIS( matrix, n, row, cols, values );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
     /* Currently a slight mismatch between "Insert" and "Set" */
      ierr = hypre_IJMatrixAddToRowParCSR( matrix, n, row, cols, values );
   else
      ierr = -1;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToRowAfter
 *--------------------------------------------------------------------------*/

/** 
Adds a row to the row of a matrix after assembly.
Note: Adds only to already existing elements.

Not collective.

@return integer error code
@param HYPRE_IJMatrix &matrix [IN]
the matrix to be operated on.
@param int n [IN]
the number of values in the row to be added.
@param int row [IN]
index of row to be added.
@param int *cols [IN]
an integer vector of length n giving the indices in the global matrix
corresponding to the columns in "values".
@param double *values {IN]
The values to be added to the matrix.
*/

int 
HYPRE_IJMatrixAddToRowAfter( HYPRE_IJMatrix IJmatrix, int n,
                           int row, const int *cols, const double *values)
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixAddToRowafterPETSc( matrix, n, row, cols, values );
   else if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixAddToRowAfterISIS( matrix, n, row, cols, values );
   else */ if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
     /* Currently a slight mismatch between "Insert" and "Set" */
      ierr = hypre_IJMatrixAddToRowAfterParCSR( matrix, n, row, cols, values );
   else
      ierr = -1;

   return(ierr);
}
/*********************************************************************************/
/* The following are routines that are not generally used by or supported for users */


/*--------------------------------------------------------------------------
 * hypre_RefIJMatrix
 *--------------------------------------------------------------------------*/

/**
Sets a reference to point to an IJMatrix.

@return integer error code
@param IJMatrix [IN]
The matrix to be pointed to.
@param reference [OUT]
The pointer to be set to point to IJMatrix.
*/

int 
hypre_RefIJMatrix( HYPRE_IJMatrix IJmatrix, HYPRE_IJMatrix *reference )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   hypre_IJMatrixReferenceCount(matrix) ++;

   *reference = IJmatrix;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetLocalStorage
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to an underlying matrix type used to implement IJMatrix.
Assumes that the implementation has an underlying matrix, so it would not
work with a direct implementation of IJMatrix. 

@return integer error code
@param IJMatrix [IN]
The matrix to be pointed to.
*/

void *
HYPRE_IJMatrixGetLocalStorage( HYPRE_IJMatrix IJmatrix )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   return( hypre_IJMatrixLocalStorage( matrix ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the row partitioning if IJmatrix has an underlying
parcsr matrix

@return integer error code
@param IJMatrix [IN]
The matrix to be pointed to.
*/

int
HYPRE_IJMatrixGetRowPartitioning( HYPRE_IJMatrix IJmatrix ,
				  const int    **row_partitioning )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixGetRowPartitioningParCSR( matrix ,
                                                     row_partitioning );
   else
      ierr = -1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the column partitioning if IJmatrix has an underlying
parcsr matrix

@return integer error code
@param IJMatrix [IN]
The matrix to be pointed to.
*/

int
HYPRE_IJMatrixGetColPartitioning( HYPRE_IJMatrix IJmatrix ,
				  const int    **col_partitioning )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   if ( hypre_IJMatrixLocalStorageType(matrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixGetColPartitioningParCSR( matrix ,
                                                     col_partitioning );
   else
      ierr = -1;

   return ierr;
}
