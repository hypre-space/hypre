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
 * ParCSRMatrix Fortran interface to macros
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGlobalNumRows
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixglobalnumrows, HYPRE_PARCSRMATRIXGLOBALNUMROWS)( long int *matrix,
                                                  int      *num_rows,
                                                  int      *ierr      )
{
   *num_rows = (int) ( hypre_ParCSRMatrixGlobalNumRows
                          ( (hypre_ParCSRMatrix *) *matrix ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixRowStarts
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixrowstarts, HYPRE_PARCSRMATRIXROWSTARTS)( long int *matrix,
                                              long int *row_starts,
                                              int      *ierr      )
{
   *row_starts = (long int) ( hypre_ParCSRMatrixRowStarts
                                 ( (hypre_ParCSRMatrix *) *matrix ) );

   *ierr = 0;
}

