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
#include "smg3.h"

/*--------------------------------------------------------------------------
 * zzz_SMG3RelaxPlanes
 *--------------------------------------------------------------------------*/

int
zzz_SMG3RelaxPlanes( void             *relax_vdata,
                     zzz_StructMatrix *A,
                     zzz_StructVector *x,
                     zzz_StructVector *b           )
{
   zzz_SMGRelaxData  *relax_data = relax_vdata;

   zzz_ResidualData  *residual_data = (relax_data -> residual_data);
   zzz_StructMatrix  *Aoff          = (relax_data -> Aoff);
   zzz_StructVector  *r             = (relax_data -> r);
   zzz_SMG2Data      *smg2_data     = (relax_data -> smg2_data);

   int ierr;

   /* Compute right-hand-side for plane solves */
   zzz_SMGResidual(residual_data, Aoff, x, b, r);

   /* Call 2D SMG code */
   zzz_SMG2Solve(smg2_data, A, r, x);

   return ierr;
}

