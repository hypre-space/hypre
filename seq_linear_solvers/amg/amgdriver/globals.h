/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

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
