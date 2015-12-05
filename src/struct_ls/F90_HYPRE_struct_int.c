/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/





#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorsetrandomvalu, HYPRE_STRUCTVECTORSETRANDOMVALU)
               (long int *vector, int *seed, int *ierr)

{
   *ierr = (int) ( hypre_StructVectorSetRandomValues( (hypre_StructVector *) vector,
                                                      (int)                 *seed ));
}


/*--------------------------------------------------------------------------
 * HYPRE_StructSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsetrandomvalues, HYPRE_STRUCTSETRANDOMVALUES)
               (long int *vector, int *seed, int *ierr)

{
   *ierr = (int) ( hypre_StructSetRandomValues( (hypre_StructVector *) vector,
                                                (int)                 *seed ));
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsetupinterpreter, HYPRE_STRUCTSETUPINTERPRETER)
               (long int *i, int *ierr)

{
   *ierr = (int) ( HYPRE_StructSetupInterpreter( (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSetupMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsetupmatvec, HYPRE_STRUCTSETUPMATVEC)
               (long int *mv, int *ierr)

{
   *ierr = (int) ( HYPRE_StructSetupMatvec( (HYPRE_MatvecFunctions *) mv));
}
