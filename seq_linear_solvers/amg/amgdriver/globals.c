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
   globals = ctalloc(Globals, 1);

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
   tfree(globals);
}


