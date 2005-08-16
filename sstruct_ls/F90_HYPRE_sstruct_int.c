/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructInt Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"
#include "HYPRE_sstruct_int.h"
#include "HYPRE_MatvecFunctions.h"



/*--------------------------------------------------------------------------
 *  HYPRE_SStructPVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpvectorsetrandomva, HYPRE_SSTRUCTPVECTORSETRANDOMVA)
               (long int *pvector, int *seed, int *ierr)
{
   *ierr = (int) ( hypre_SStructPVectorSetRandomValues( (hypre_SStructPVector *) pvector,
                                                        (int)                   *seed ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetrandomval, HYPRE_SSTRUCTVECTORSETRANDOMVAL)
               (long int *vector, int *seed, int *ierr)
{
   *ierr = (int) ( hypre_SStructVectorSetRandomValues( (hypre_SStructVector *) vector,
                                                       (int)                  *seed ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsetrandomvalues, HYPRE_SSTRUCTSETRANDOMVALUES)
               (long int *v, int *seed, int *ierr)
{
   *ierr = (int) ( hypre_SStructSetRandomValues( (void *) v, (int) *seed )); 
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsetupinterpreter, HYPRE_SSTRUCTSETUPINTERPRETER)
               (long int *i, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructSetupInterpreter( (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSetupMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsetupmatvec, HYPRE_SSTRUCTSETUPMATVEC)
               (long int *mv, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructSetupMatvec( (HYPRE_MatvecFunctions *) mv));
}
