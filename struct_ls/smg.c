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
 * zzz_SMGInitialize
 *--------------------------------------------------------------------------*/

zzz_SMGData *
zzz_SMGInitialize( MPI_Comm *comm )
{
   zzz_SMGData *smg_data;

   smg_data = zzz_CTAlloc(zzz_SMGData, 1);

   (smg_data -> cindex)  = zzz_NewIndex();
   (smg_data -> cstride) = zzz_NewIndex();
   (smg_data -> findex)  = zzz_NewIndex();
   (smg_data -> fstride) = zzz_NewIndex();

   /* set default parameters ... */
   zzz_SetIndex((smg_data -> cindex), 0, 0, 0);
   zzz_SetIndex((smg_data -> cstride), 1, 1, 2);
   zzz_SetIndex((smg_data -> findex), 0, 0, 1);
   zzz_SetIndex((smg_data -> fstride), 1, 1, 2);

   return smg_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSet ...
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * zzz_SMGGet ...
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * zzz_SMGFinalize
 *--------------------------------------------------------------------------*/

int
zzz_SMGFinalize( zzz_SMGData *smg_data )
{
   int  ierr;

   /* free up solver structure */

   return(ierr);
}

