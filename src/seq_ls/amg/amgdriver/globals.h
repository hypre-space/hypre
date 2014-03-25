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





#ifndef _GLOBALS_HEADER
#define _GLOBALS_HEADER


/*----------------------------------------------------------------
 * Globals structure
 *----------------------------------------------------------------*/

typedef struct
{
   char     run_name[256];
   char     in_file_name[256];
   char     out_file_name[256];
   char     log_file_name[256];

} Globals;

#ifdef AMG_GLOBALS
Globals  *globals;
#else
extern Globals  *globals;
#endif


/*--------------------------------------------------------------------------
 * Accessor macros: Globals
 *--------------------------------------------------------------------------*/

#define GlobalsRunName        (globals -> run_name)
#define GlobalsInFileName     (globals -> in_file_name)
#define GlobalsOutFileName    (globals -> out_file_name)
#define GlobalsLogFileName    (globals -> log_file_name)
			      

#endif
