/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header file to include all header information for amg.
 *
 *****************************************************************************/

#ifndef _AMG_HEADER
#define _AMG_HEADER


#include <stdio.h>
#include <math.h>

#include "general.h"

#include "matrix.h"
#include "vector.h"

#include "globals.h"

#include "problem.h"
#include "solver.h"

#include "amg_proto.h"
#include "fortran.h"


#define  NDIMU(nv)  (3*nv)
#define  NDIMP(np)  (3*np)
#define  NDIMA(na)  (5*na)
#define  NDIMB(na)  (1*na)


#endif
