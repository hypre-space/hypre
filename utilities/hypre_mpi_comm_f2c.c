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



#include <HYPRE_config.h>
#include "fortran.h"
#ifndef HYPRE_SEQUENTIAL
#include <mpi.h>
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mpi_comm_f2c, HYPRE_MPI_COMM_F2C)(int *c_comm,
                                                        int *f_comm,
                                                        int *ierr)
{
#ifdef HYPRE_HAVE_MPI_COMM_F2C

   *c_comm = (int)MPI_Comm_f2c(*f_comm);

   if (sizeof(MPI_Comm) > sizeof(int))
      *ierr = 1;
   else
      *ierr = 0;

#else

   *c_comm = *f_comm;
   *ierr = 0;

#endif
}
