/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.3 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "headers.h"
#include <assert.h>

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixMatvec( double           alpha,
              	 hypre_ParCSRMatrix *A,
                 hypre_ParVector    *x,
                 double           beta,
                 hypre_ParVector    *y     )
{
   hypre_ParCSRCommHandle	**comm_handle;
   hypre_ParCSRCommPkg	*comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix      *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *offd   = hypre_ParCSRMatrixOffd(A);
   hypre_Vector         *x_local  = hypre_ParVectorLocalVector(x);   
   hypre_Vector         *y_local  = hypre_ParVectorLocalVector(y);   
   int         num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   int         num_cols = hypre_ParCSRMatrixGlobalNumCols(A);

   hypre_Vector      *x_tmp;
   int        x_size = hypre_ParVectorGlobalSize(x);
   int        y_size = hypre_ParVectorGlobalSize(y);
   int        num_vectors = hypre_VectorNumVectors(x_local);
   int	      num_cols_offd = hypre_CSRMatrixNumCols(offd);
   int        ierr = 0;
   int	      num_sends, i, j, jv, index, start;

   int        vecstride = hypre_VectorVectorStride( x_local );
   int        idxstride = hypre_VectorIndexStride( x_local );

   double     *x_tmp_data, **x_buf_data;
   double     *x_local_data = hypre_VectorData(x_local);
   /*---------------------------------------------------------------------
    *  Check for size compatibility.  ParMatvec returns ierr = 11 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 12 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 13 if both are true.
    *
    *  Because temporary vectors are often used in ParMatvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
   assert( idxstride>0 );

    if (num_cols != x_size)
              ierr = 11;

    if (num_rows != y_size)
              ierr = 12;

    if (num_cols != x_size && num_rows != y_size)
              ierr = 13;

    assert( hypre_VectorNumVectors(y_local)==num_vectors );

    if ( num_vectors==1 )
       x_tmp = hypre_SeqVectorCreate( num_cols_offd );
    else
    {
       assert( num_vectors>1 );
       x_tmp = hypre_SeqMultiVectorCreate( num_cols_offd, num_vectors );
    }
   hypre_SeqVectorInitialize(x_tmp);
   x_tmp_data = hypre_VectorData(x_tmp);
   
   comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle*,num_vectors);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
	hypre_MatvecCommPkgCreate(A);
	comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   x_buf_data = hypre_CTAlloc( double*, num_vectors );
   for ( jv=0; jv<num_vectors; ++jv )
      x_buf_data[jv] = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart
                                    (comm_pkg, num_sends));

   if ( num_vectors==1 )
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            x_buf_data[0][index++] 
               = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
   }
   else
      for ( jv=0; jv<num_vectors; ++jv )
      {
         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
               x_buf_data[jv][index++] 
                  = x_local_data[
                     jv*vecstride +
                     idxstride*hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j) ];
         }
      }

   assert( idxstride==1 );
   /* >>> ... The assert is because the following loop only works for 'column' storage of a multivector <<<
      >>> This needs to be fixed to work more generally, at least for 'row' storage. <<<
      >>> This in turn, means either change CommPkg so num_sends is no.zones*no.vectors (not no.zones)
      >>> or, less dangerously, put a stride in the logic of CommHandleCreate (stride either from a
      >>> new arg or a new variable inside CommPkg).  Or put the num_vector iteration inside
      >>> CommHandleCreate (perhaps a new multivector variant of it).
   */
   for ( jv=0; jv<num_vectors; ++jv )
   {
      comm_handle[jv] = hypre_ParCSRCommHandleCreate
         ( 1, comm_pkg, x_buf_data[jv], &(x_tmp_data[jv*num_cols_offd]) );
   }

   hypre_CSRMatrixMatvec( alpha, diag, x_local, beta, y_local);
   
   for ( jv=0; jv<num_vectors; ++jv )
   {
      hypre_ParCSRCommHandleDestroy(comm_handle[jv]);
      comm_handle[jv] = NULL;
   }
   hypre_TFree(comm_handle);

   if (num_cols_offd) hypre_CSRMatrixMatvec( alpha, offd, x_tmp, 1.0, y_local);    

   hypre_SeqVectorDestroy(x_tmp);
   x_tmp = NULL;
   for ( jv=0; jv<num_vectors; ++jv ) hypre_TFree(x_buf_data[jv]);
   hypre_TFree(x_buf_data);
  
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixMatvecT( double           alpha,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *x,
                  double           beta,
                  hypre_ParVector    *y     )
{
   hypre_ParCSRCommHandle	**comm_handle;
   hypre_ParCSRCommPkg	*comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   hypre_Vector *y_tmp;
   int           vecstride = hypre_VectorVectorStride( y_local );
   int           idxstride = hypre_VectorIndexStride( y_local );
   double       *y_tmp_data, **y_buf_data;
   double       *y_local_data = hypre_VectorData(y_local);

   int         num_rows  = hypre_ParCSRMatrixGlobalNumRows(A);
   int         num_cols  = hypre_ParCSRMatrixGlobalNumCols(A);
   int	       num_cols_offd = hypre_CSRMatrixNumCols(offd);
   int         x_size = hypre_ParVectorGlobalSize(x);
   int         y_size = hypre_ParVectorGlobalSize(y);
   int         num_vectors = hypre_VectorNumVectors(y_local);

   int         i, j, jv, index, start, num_sends;

   int         ierr  = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of 
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
    if (num_rows != x_size)
              ierr = 1;

    if (num_cols != y_size)
              ierr = 2;

    if (num_rows != x_size && num_cols != y_size)
              ierr = 3;
   /*-----------------------------------------------------------------------
    *-----------------------------------------------------------------------*/

    comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle*,num_vectors);

    if ( num_vectors==1 )
    {
       y_tmp = hypre_SeqVectorCreate(num_cols_offd);
    }
    else
    {
       y_tmp = hypre_SeqMultiVectorCreate(num_cols_offd,num_vectors);
    }
    hypre_SeqVectorInitialize(y_tmp);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
	hypre_MatvecCommPkgCreate(A);
	comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   y_buf_data = hypre_CTAlloc( double*, num_vectors );
   for ( jv=0; jv<num_vectors; ++jv )
      y_buf_data[jv] = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart
                                     (comm_pkg, num_sends));
   y_tmp_data = hypre_VectorData(y_tmp);
   y_local_data = hypre_VectorData(y_local);

   assert( idxstride==1 ); /* >>> only 'column' storage of multivectors implemented so far */

   if (num_cols_offd) hypre_CSRMatrixMatvecT(alpha, offd, x_local, 0.0, y_tmp);

   for ( jv=0; jv<num_vectors; ++jv )
   {
      /* >>> this is where we assume multivectors are 'column' storage */
      comm_handle[jv] = hypre_ParCSRCommHandleCreate
         ( 2, comm_pkg, &(y_tmp_data[jv*num_cols_offd]), y_buf_data[jv] );
   }

   hypre_CSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);

   for ( jv=0; jv<num_vectors; ++jv )
   {
      hypre_ParCSRCommHandleDestroy(comm_handle[jv]);
      comm_handle[jv] = NULL;
   }
   hypre_TFree(comm_handle);

   if ( num_vectors==1 )
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            y_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
               += y_buf_data[0][index++];
      }
   }
   else
      for ( jv=0; jv<num_vectors; ++jv )
      {
         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
               y_local_data[ jv*vecstride +
                             idxstride*hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j) ]
                  += y_buf_data[jv][index++];
         }
      }
	
   hypre_SeqVectorDestroy(y_tmp);
   y_tmp = NULL;
   for ( jv=0; jv<num_vectors; ++jv ) hypre_TFree(y_buf_data[jv]);
   hypre_TFree(y_buf_data);

   return ierr;
}
