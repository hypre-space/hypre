/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * AMG functions (Fortran 90 interface)
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * HYPRE_AMGInitialize
 *--------------------------------------------------------------------------*/

void   hypre_NAME_C_FOR_FORTRAN(amg_initialize)(data, port_data)
int   *data;
int   *port_data;
{
   *data = (int) HYPRE_AMGInitialize((void *) *port_data);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGFinalize
 *--------------------------------------------------------------------------*/

void   hypre_NAME_C_FOR_FORTRAN(amg_finalize)(data)
int   *data;
{
   HYPRE_AMGFinalize((void *) *data);
}

