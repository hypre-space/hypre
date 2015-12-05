/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * hypre_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./_hypre_IJ_mv.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * hypre_IJMatrixSetObject
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixsetobject, HYPRE_IJMATRIXSETOBJECT)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Obj *object,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( hypre_IJMatrixSetObject(
           hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
           (void *)         *object  ) );
}

