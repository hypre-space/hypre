/*
 * File:          bHYPRE_IJParCSRMatrix_Impl.c
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.IJParCSRMatrix" (version 1.0.0)
 * 
 * The IJParCSR matrix class.
 * 
 * Objects of this type can be cast to IJMatrixView, Operator, or
 * CoefficientAccess objects using the {\tt \_\_cast} methods.
 * 
 */

#include "bHYPRE_IJParCSRMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix._includes) */
/* Put additional includes or other arbitrary code here... */

/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include <assert.h>
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "HYPRE_parcsr_mv.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_IJParCSRMatrix__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix._load) */
  /* Insert-Code-Here {bHYPRE.IJParCSRMatrix._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_IJParCSRMatrix__ctor(
  /* in */ bHYPRE_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* Note: User calls of __create are DEPRECATED.
      Use Create(), which also calls this function */

   struct bHYPRE_IJParCSRMatrix__data * data;

   data = hypre_CTAlloc(struct bHYPRE_IJParCSRMatrix__data,1);
   /* data = (struct bHYPRE_IJParCSRMatrix__data *)
      malloc( sizeof ( struct bHYPRE_IJParCSRMatrix__data ) ); */

   data -> comm = MPI_COMM_NULL;
   data -> ij_A = NULL;
   data -> owns_matrix = 1;

   bHYPRE_IJParCSRMatrix__set_data( self, data );
   
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_IJParCSRMatrix__dtor(
  /* in */ bHYPRE_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   if ( ij_A && data->owns_matrix ) ierr += HYPRE_IJMatrixDestroy( ij_A );
   hypre_assert( ierr == 0 );

   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix._dtor) */
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_IJParCSRMatrix
impl_bHYPRE_IJParCSRMatrix_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Create) */
  /* Insert-Code-Here {bHYPRE.IJParCSRMatrix.Create} (Create method) */

   int ierr = 0;
   HYPRE_IJMatrix dummy;
   HYPRE_IJMatrix * Hmat = &dummy;
   struct bHYPRE_IJParCSRMatrix__data * data;
   bHYPRE_IJParCSRMatrix mat = bHYPRE_IJParCSRMatrix__create();

   data = bHYPRE_IJParCSRMatrix__get_data( mat );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   ierr += HYPRE_IJMatrixCreate( data -> comm,
                                ilower, iupper, jlower, jupper, Hmat );
   ierr += HYPRE_IJMatrixSetObjectType( *Hmat, HYPRE_PARCSR );
   hypre_assert( ierr == 0 );
   data -> ij_A = *Hmat;

   return mat;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Create) */
}

/*
 * Method:  GenerateLaplacian[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GenerateLaplacian"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_IJParCSRMatrix
impl_bHYPRE_IJParCSRMatrix_GenerateLaplacian(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t nx,
  /* in */ int32_t ny,
  /* in */ int32_t nz,
  /* in */ int32_t Px,
  /* in */ int32_t Py,
  /* in */ int32_t Pz,
  /* in */ int32_t p,
  /* in */ int32_t q,
  /* in */ int32_t r,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* in */ int32_t discretization)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GenerateLaplacian) */
  /* Insert-Code-Here {bHYPRE.IJParCSRMatrix.GenerateLaplacian} (GenerateLaplacian method) */

   /* The returned matrix represents a Laplacian with 7,9,or 27 point discretization
      as specified.   Initialize but not Assemble is called before returning. */

   int ierr = 0;
   bHYPRE_IJParCSRMatrix bHA;
   HYPRE_ParCSRMatrix HA;
   int first_local_row, last_local_row, first_local_col, last_local_col;
   int local_num_rows, size, i;
   int * row_sizes;
   int * col_inds;
   double * row_values;
   int stride[1];
   MPI_Comm comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
;

   hypre_assert( nvalues == 4 );
   hypre_assert( discretization==7 || discretization==9 || discretization==27 );
   hypre_assert( discretization==7 ); /* only 7-point 3D example implemented */
   HA = (HYPRE_ParCSRMatrix) GenerateLaplacian(
      comm, nx, ny, nz, Px, Py, Pz, p, q, r, values );

   /* We need to return a bHYPRE_IJParCSRMatrix.  Make a one and copy HA to it... */

   ierr += HYPRE_ParCSRMatrixGetLocalRange(
      HA, &first_local_row, &last_local_row , &first_local_col, &last_local_col );
   local_num_rows = last_local_row - first_local_row + 1;

   bHA = bHYPRE_IJParCSRMatrix_Create(
      mpi_comm, first_local_row, last_local_row, first_local_col, last_local_col );

   row_sizes = hypre_CTAlloc( int, local_num_rows );
   size = discretization;
   for (i=0; i < local_num_rows; i++)
   {
      row_sizes[i] = size;
   }
   ierr = bHYPRE_IJParCSRMatrix_SetRowSizes( bHA, row_sizes, local_num_rows );
   hypre_TFree( row_sizes );

   ierr = bHYPRE_IJParCSRMatrix_Initialize( bHA );

   row_sizes = hypre_CTAlloc( int, 1 );
   stride[0] = 1;

   /* Copy row data to the new matrix... */
   for (i=first_local_row; i<= last_local_row; i++)
   {
      ierr += HYPRE_ParCSRMatrixGetRow( HA, i, &size, &col_inds, &row_values );
      ierr += bHYPRE_IJParCSRMatrix_SetValues(
         bHA, 1, &size, &i, col_inds, row_values, size );
      ierr += HYPRE_ParCSRMatrixRestoreRow( HA, i, &size, &col_inds, &row_values );
   }

   hypre_assert( ierr == 0 );
   return bHA;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GenerateLaplacian) */
}

/*
 * (Optional) Set the max number of nonzeros to expect in each
 * row of the diagonal and off-diagonal blocks.  The diagonal
 * block is the submatrix whose column numbers correspond to
 * rows owned by this process, and the off-diagonal block is
 * everything else.  The arrays {\tt diag\_sizes} and {\tt
 * offdiag\_sizes} contain estimated sizes for each row of the
 * diagonal and off-diagonal blocks, respectively.  This routine
 * can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[local_nrows] */ int32_t* diag_sizes,
  /* in rarray[local_nrows] */ int32_t* offdiag_sizes,
  /* in */ int32_t local_nrows)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetDiagOffdSizes) */
  /* Insert the implementation of the SetDiagOffdSizes method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixSetDiagOffdSizes( ij_A, 
                                          diag_sizes, 
                                          offdiag_sizes );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetDiagOffdSizes) */
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetCommunicator(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_MPICommunicator mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   /* The data type of the last argument, mpi_comm, should be MPI_Comm */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

#ifdef HYPRE_DEBUG
   printf("impl_bHYPRE_IJParCSRMatrix_SetCommunicator\n");
#endif
   
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Initialize"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_Initialize(
  /* in */ bHYPRE_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixInitialize( ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Initialize) */
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Assemble"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_Assemble(
  /* in */ bHYPRE_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixAssemble( ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Assemble) */
}

/*
 * Set the local range for a matrix object.  Each process owns
 * some unique consecutive range of rows, indicated by the
 * global row indices {\tt ilower} and {\tt iupper}.  The row
 * data is required to be such that the value of {\tt ilower} on
 * any process $p$ be exactly one more than the value of {\tt
 * iupper} on process $p-1$.  Note that the first row of the
 * global matrix may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * For square matrices, {\tt jlower} and {\tt jupper} typically
 * should match {\tt ilower} and {\tt iupper}, respectively.
 * For rectangular matrices, {\tt jlower} and {\tt jupper}
 * should define a partitioning of the columns.  This
 * partitioning must be used for any vector $v$ that will be
 * used in matrix-vector products with the rectangular matrix.
 * The matrix data structure may use {\tt jlower} and {\tt
 * jupper} to store the diagonal blocks (rectangular in general)
 * of the matrix separately from the rest of the matrix.
 * 
 * Collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetLocalRange"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetLocalRange(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetLocalRange) */
  /* Insert the implementation of the SetLocalRange method here... */

   /* DEPRECATED  Use Create */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   if ( data -> comm == MPI_COMM_NULL )    
   {
#ifdef HYPRE_DEBUG
      printf("Set Communicator must be called before Create in IJParCSRMatrix\n");
#endif
      return( -1 );
   }
   else
   {
      ierr = HYPRE_IJMatrixCreate( data -> comm,
                                   ilower, iupper, jlower, jupper, &ij_A );

      ierr = HYPRE_IJMatrixSetObjectType( ij_A, HYPRE_PARCSR );

      data -> ij_A = ij_A;
   
      return( ierr );
   }

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetLocalRange) */
}

/*
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
 * ncols} and {\tt rows} are of dimension {\tt nrows} and
 * contain the number of columns in each row and the row
 * indices, respectively.  The array {\tt cols} contains the
 * column indices for each of the {\tt rows}, and is ordered by
 * rows.  The data in the {\tt values} array corresponds
 * directly to the column entries in {\tt cols}.  The last argument
 * is the size of the cols and values arrays, i.e. the total number
 * of nonzeros being provided, i.e. the sum of all values in ncols.
 * This functin erases any previous values at the specified locations and
 * replaces them with new ones, or, if there was no value there before,
 * inserts a new one.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* in rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixSetValues( ij_A, nrows,
                                   ncols,
                                   rows,
                                   cols,
                                   values ); 

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetValues) */
}

/*
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_AddToValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_AddToValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* in rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixAddToValues( ij_A, nrows,
                                     ncols ,
                                     rows ,
                                     cols ,
                                     values  ); 
   
   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.AddToValues) */
}

/*
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetLocalRange"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_GetLocalRange(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ int32_t* ilower,
  /* out */ int32_t* iupper,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetLocalRange) */
  /* Insert the implementation of the GetLocalRange method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixGetLocalRange( ij_A, ilower, iupper, jlower, jupper );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetLocalRange) */
}

/*
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetRowCounts"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_GetRowCounts(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* rows,
  /* inout rarray[nrows] */ int32_t* ncols)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetRowCounts) */
  /* Insert the implementation of the GetRowCounts method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixGetRowCounts( ij_A, nrows,
                                      rows,
                                      ncols );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetRowCounts) */
}

/*
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_GetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* inout rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetValues) */
  /* Insert the implementation of the GetValues method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixGetValues( ij_A, nrows,
                                   ncols,
                                   rows,
                                   cols,
                                   values ); 

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetValues) */
}

/*
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  The integer nrows is the number of rows in
 * the local matrix.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetRowSizes"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetRowSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[nrows] */ int32_t* sizes,
  /* in */ int32_t nrows)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetRowSizes) */
  /* Insert the implementation of the SetRowSizes method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixSetRowSizes( ij_A, sizes );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetRowSizes) */
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Print"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_Print(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* filename)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Print) */
  /* Insert the implementation of the Print method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixPrint( ij_A, filename);

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Print) */
}

/*
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Read"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_Read(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* filename,
  /* in */ bHYPRE_MPICommunicator comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Read) */
  /* Insert the implementation of the Read method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;
   MPI_Comm mpicomm = bHYPRE_MPICommunicator__get_data(comm)->mpi_comm;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixRead( filename, mpicomm, HYPRE_PARCSR, &ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Read) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetStringParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_GetIntValue(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */


   int ierr = 0;
   int M, N;
   void * object;
   HYPRE_IJMatrix HijA;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_ParCSRMatrix HA ;

   data = bHYPRE_IJParCSRMatrix__get_data( self );
   HijA = data->ij_A;

   ierr += HYPRE_IJMatrixGetObject( HijA, &object );
   HA = (HYPRE_ParCSRMatrix) object;

   if ( strcmp(name,"GlobalNumRows")==0 )
   {
      ierr += HYPRE_ParCSRMatrixGetDims( HA, &M, &N );
      *value = M;
   }
   else if ( strcmp(name,"GlobalNumCols")==0 )
   {
      ierr += HYPRE_ParCSRMatrixGetDims( HA, &M, &N );
      *value = N;
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_GetDoubleValue(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* out */ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_Setup(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixAssemble( ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_Apply(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */

   /* Apply means to multiply by a vector, y = A*x .  Here, we call
    * the HYPRE Matvec function which performs y = a*A*x + b*y (we set
    * a=1 and b=0).  */
   int ierr=0;
   void * object;
   struct bHYPRE_IJParCSRMatrix__data * data;
   struct bHYPRE_IJParCSRVector__data * data_x, * data_b;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_IJMatrix ij_A;
   HYPRE_IJVector ij_x, ij_b;
   HYPRE_ParVector xx, bb;
   HYPRE_ParCSRMatrix A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );
   ij_A = data -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &object );
   A = (HYPRE_ParCSRMatrix) object;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_b = bHYPRE_IJParCSRVector__cast( b );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)b );
   }

   if ( bHYPRE_Vector_queryInt( *x, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_x = bHYPRE_IJParCSRVector__cast( *x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }

   data_x = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   ij_x = data_x -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   data_b = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   ij_b = data_b -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &object );
   bb = (HYPRE_ParVector) object;

   ierr += HYPRE_ParCSRMatrixMatvec( 1.0, A, bb, 0.0, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b ); /* ref was created by queryInt */
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x ); /* ref was created by queryInt */

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Apply) */
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_ApplyAdjoint(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.IJParCSRMatrix.ApplyAdjoint} (ApplyAdjoint method) */

   /* ApplyAdjoing means to multiply by a vector, y = A'*x , where A' is the
    * adjoint of A (=transpose, this is a real matrix).  Here, we call
    * the HYPRE Matvec function which performs y = a*A*x + b*y (we set
    * a=1 and b=0).  */
   int ierr=0;
   void * object;
   struct bHYPRE_IJParCSRMatrix__data * data;
   struct bHYPRE_IJParCSRVector__data * data_x, * data_b;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_IJMatrix ij_A;
   HYPRE_IJVector ij_x, ij_b;
   HYPRE_ParVector xx, bb;
   HYPRE_ParCSRMatrix A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );
   ij_A = data -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &object );
   A = (HYPRE_ParCSRMatrix) object;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_b = bHYPRE_IJParCSRVector__cast( b );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)b );
   }

   if ( bHYPRE_Vector_queryInt( *x, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_x = bHYPRE_IJParCSRVector__cast( *x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }

   data_x = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   ij_x = data_x -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   data_b = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   ij_b = data_b -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &object );
   bb = (HYPRE_ParVector) object;

   ierr += HYPRE_ParCSRMatrixMatvecT( 1.0, A, bb, 0.0, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b ); /* ref was created by queryInt */
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x ); /* ref was created by queryInt */

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.ApplyAdjoint) */
}

/*
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetRow"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRMatrix_GetRow(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t row,
  /* out */ int32_t* size,
  /* out array<int,column-major> */ struct sidl_int__array** col_ind,
  /* out array<double,column-major> */ struct sidl_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetRow) */
  /* Insert the implementation of the GetRow method here... */

   int ierr=0;
   void * object;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHYPREP_A;
   int * iindices[1];
   double * dvalues[1];

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &object );
   bHYPREP_A = (HYPRE_ParCSRMatrix) object;

   *col_ind = sidl_int__array_create1d( size[0] );
   *values = sidl_double__array_create1d( size[0] );

   *iindices = sidlArrayAddr1( *col_ind, 0 );
   *dvalues = sidlArrayAddr1( *values, 0 );

   /* RestoreRow doesn't do anything but reset a parameter.  Its
    * function is to make sure the user who calls GetRow is aware that
    * the data in the output arrays will be changed. */
   HYPRE_ParCSRMatrixRestoreRow( bHYPREP_A, row, size, iindices, dvalues );
   ierr += HYPRE_ParCSRMatrixGetRow( bHYPREP_A, row, size, iindices, dvalues );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetRow) */
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_CoefficientAccess__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_CoefficientAccess__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_CoefficientAccess(struct 
  bHYPRE_CoefficientAccess__object* obj) {
  return bHYPRE_CoefficientAccess__getURL(obj);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) {
  return bHYPRE_MPICommunicator__getURL(obj);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) {
  return bHYPRE_Operator__getURL(obj);
}
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_IJParCSRMatrix__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj) {
  return bHYPRE_IJParCSRMatrix__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) {
  return bHYPRE_Vector__getURL(obj);
}
struct bHYPRE_IJMatrixView__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJMatrixView(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_IJMatrixView__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJMatrixView(struct 
  bHYPRE_IJMatrixView__object* obj) {
  return bHYPRE_IJMatrixView__getURL(obj);
}
struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_ProblemDefinition__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj) {
  return bHYPRE_ProblemDefinition__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MatrixVectorView__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj) {
  return bHYPRE_MatrixVectorView__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
