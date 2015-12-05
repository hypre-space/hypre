/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
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
