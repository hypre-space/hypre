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
 * Main header file
 *
 *****************************************************************************/

#ifndef HYPRE_HEADERS_HEADER
#define HYPRE_HEADERS_HEADER


#include <stdio.h>
#include <math.h>

#include <cvode/spgmr.h>

#include <amg.h>

#include "general.h"

#include "globals.h"

#include "problem.h"
#include "solver.h"

#include "protos.h"

/* malloc debug stuff */
#ifdef AMG_MALLOC_DEBUG
#include <gmalloc.h>
#endif


#endif
