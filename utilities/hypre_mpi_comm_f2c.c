/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
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
