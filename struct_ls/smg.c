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

void *
zzz_SMGInitialize( MPI_Comm *comm )
{
   zzz_SMGData *smg_data;

   smg_data = zzz_CTAlloc(zzz_SMGData, 1);

   (smg_data -> comm)        = comm;
   (smg_data -> base_index)  = zzz_NewIndex();
   (smg_data -> base_stride) = zzz_NewIndex();

   /* set defaults */
   (smg_data -> tol)        = 1.0e-06;
   (smg_data -> max_iter)   = 200;
   (smg_data -> zero_guess) = 0;
   (smg_data -> max_levels) = 0;
   (smg_data -> cdir) = 2;
   (smg_data -> ci) = 0;
   (smg_data -> fi) = 1;
   (smg_data -> cs) = 2;
   (smg_data -> fs) = 2;
   zzz_SetIndex((smg_data -> base_index), 0, 0, 0);
   zzz_SetIndex((smg_data -> base_stride), 1, 1, 1);

   return (void *) smg_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetTol
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetTol( void   *smg_vdata,
               double  tol       )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetMaxIter( void *smg_vdata,
                   int   max_iter  )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
zzz_SMGSetZeroGuess( void *smg_vdata )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          ierr = 0;
 
   (smg_data -> zero_guess) = 1;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetBase
 *--------------------------------------------------------------------------*/
 
int
zzz_SMGSetBase( void      *smg_vdata,
                zzz_Index *base_index,
                zzz_Index *base_stride )
{
   zzz_SMGData *smg_data = smg_vdata;
   int          d;
   int          ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      zzz_IndexD((smg_data -> base_index),  d) = zzz_IndexD(base_index,  d);
      zzz_IndexD((smg_data -> base_stride), d) = zzz_IndexD(base_stride, d);
   }
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGGet ...
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * zzz_SMGFinalize
 *--------------------------------------------------------------------------*/

int
zzz_SMGFinalize( void *smg_vdata )
{
   zzz_SMGData *smg_data = smg_vdata;
   int ierr;


   return(ierr);
}

