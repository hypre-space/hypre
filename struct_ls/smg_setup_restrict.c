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
 * zzz_SMGNewRestrictOp
 *--------------------------------------------------------------------------*/

zzz_StructMatrix *
zzz_SMGNewRestrictOp( zzz_StructMatrix *A,
                      zzz_StructGrid   *cgrid,
                      int               cdir  )
{
   zzz_StructMatrix *R;

   return R;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetupRestrictOp
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetupRestrictOp( zzz_StructMatrix *A,
                        zzz_StructMatrix *R,
                        zzz_StructVector *temp_vec,
                        int               cdir,
                        zzz_Index        *cindex,
                        zzz_Index        *cstride  )
{
   int ierr;

   return ierr;
}
