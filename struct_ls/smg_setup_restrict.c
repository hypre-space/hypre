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
 *
 *****************************************************************************/

#include "headers.h"
#include "smg.h"

/*--------------------------------------------------------------------------
 * hypre_SMGNewRestrictOp
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_SMGNewRestrictOp( hypre_StructMatrix *A,
                        hypre_StructGrid   *cgrid,
                        int                 cdir  )
{
   hypre_StructMatrix *R;

   return R;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetupRestrictOp
 *--------------------------------------------------------------------------*/

int
hypre_SMGSetupRestrictOp( hypre_StructMatrix *A,
                          hypre_StructMatrix *R,
                          hypre_StructVector *temp_vec,
                          int                 cdir,
                          hypre_Index         cindex,
                          hypre_Index         cstride  )
{
   int ierr;

   return ierr;
}
