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
 * Routines for manipulating global structures.
 *
 *****************************************************************************/

#define AMG_GLOBALS

#include "headers.h"


/*--------------------------------------------------------------------------
 * NewGlobals
 *--------------------------------------------------------------------------*/

void   NewGlobals(run_name)
char  *run_name;
{
   globals = hypre_CTAlloc(Globals, 1);

   sprintf(GlobalsRunName,     "%s",     run_name);
   sprintf(GlobalsInFileName,  "%s.in",  run_name);
   sprintf(GlobalsOutFileName, "%s.out", run_name);
   sprintf(GlobalsLogFileName, "%s.log", GlobalsOutFileName);
}


/*--------------------------------------------------------------------------
 * FreeGlobals
 *--------------------------------------------------------------------------*/

void  FreeGlobals()
{
   hypre_TFree(globals);
}


