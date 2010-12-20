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
#include "HYPRE_MatvecFunctions.h"



/*--------------------------------------------------------------------------
 *  HYPRE_SStructPVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpvectorsetrandomva, HYPRE_SSTRUCTPVECTORSETRANDOMVA)
               (hypre_F90_Obj *pvector, HYPRE_Int *seed, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( hypre_SStructPVectorSetRandomValues( (hypre_SStructPVector *) pvector,
                                                        (HYPRE_Int)                   *seed ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetrandomval, HYPRE_SSTRUCTVECTORSETRANDOMVAL)
               (hypre_F90_Obj *vector, HYPRE_Int *seed, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( hypre_SStructVectorSetRandomValues( (hypre_SStructVector *) vector,
                                                       (HYPRE_Int)                  *seed ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsetrandomvalues, HYPRE_SSTRUCTSETRANDOMVALUES)
               (hypre_F90_Obj *v, HYPRE_Int *seed, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( hypre_SStructSetRandomValues( (void *) v, (HYPRE_Int) *seed )); 
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsetupinterpreter, HYPRE_SSTRUCTSETUPINTERPRETER)
               (hypre_F90_Obj *i, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructSetupInterpreter( (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSetupMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsetupmatvec, HYPRE_SSTRUCTSETUPMATVEC)
               (hypre_F90_Obj *mv, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructSetupMatvec( (HYPRE_MatvecFunctions *) mv));
}
