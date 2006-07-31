/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
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

#ifndef EUCLID_USER_INTERFAACE
#define EUCLID_USER_INTERFAACE

/* collection of everthing that should be included for user apps */

#include "Euclid_dh.h"
#include "SubdomainGraph_dh.h"
#include "Mat_dh.h"
#include "Factor_dh.h"
#include "MatGenFD.h"
#include "Vec_dh.h"
#include "Parser_dh.h"
#include "Mem_dh.h"
#include "Timer_dh.h"
#include "TimeLog_dh.h"
#include "guards_dh.h"
#include "krylov_dh.h"
#include "io_dh.h"
#include "mat_dh_private.h"

#ifdef PETSC_MODE
#include "euclid_petsc.h"
#endif

#ifdef FAKE_MPI
#include "fake_mpi.h"
#endif

#endif
