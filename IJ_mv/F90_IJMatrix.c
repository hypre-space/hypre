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
 * hypre_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./IJ_mv.h"
#include "fortran.h"


/*--------------------------------------------------------------------------
 * hypre_IJMatrixSetObject
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixsetobject, HYPRE_IJMATRIXSETOBJECT)(
                                                     long int *matrix,
                                                     long int *object,
                                                     int      *ierr    )
{
   *ierr = (int) ( hypre_IJMatrixSetObject( (HYPRE_IJMatrix) *matrix,
                                            (void *)         *object  ) );
}

