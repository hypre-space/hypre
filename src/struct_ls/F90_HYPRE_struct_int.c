/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorsetrandomvalu, HYPRE_STRUCTVECTORSETRANDOMVALU)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *seed,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( hypre_StructVectorSetRandomValues(
           (hypre_StructVector *) vector,
           hypre_F90_PassInt (seed) ));
}


/*--------------------------------------------------------------------------
 * HYPRE_StructSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsetrandomvalues, HYPRE_STRUCTSETRANDOMVALUES)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *seed,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( hypre_StructSetRandomValues(
           (hypre_StructVector *) vector,
           hypre_F90_PassInt (seed) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsetupinterpreter, HYPRE_STRUCTSETUPINTERPRETER)
   (hypre_F90_Obj *i,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSetupInterpreter(
           (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSetupMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsetupmatvec, HYPRE_STRUCTSETUPMATVEC)
   (hypre_F90_Obj *mv,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSetupMatvec(
           hypre_F90_PassObjRef (HYPRE_MatvecFunctions, mv)));
}
