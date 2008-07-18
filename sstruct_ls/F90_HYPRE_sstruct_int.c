/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/





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
 *  HYPRE_SStructVectorSetupInterpreter
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
