/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
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
