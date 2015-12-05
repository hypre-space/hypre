/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/

#include <HYPRE_config.h>
#include "fortran.h"
#ifndef HYPRE_SEQUENTIAL
#include <mpi.h>
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#if 0 /* This function is problematic and no longer needed anyway. */
void
hypre_F90_IFACE(hypre_mpi_comm_f2c, HYPRE_MPI_COMM_F2C)
   (hypre_F90_Obj  *c_comm,
    hypre_F90_Comm *f_comm,
    hypre_F90_Int  *ierr)
{
   *c_comm = (hypre_F90_Obj) hypre_MPI_Comm_f2c( (hypre_int) *f_comm );
   *ierr = 0;
}
#endif
