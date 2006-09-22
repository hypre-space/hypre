/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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



#include "sig_dh.h"
#include "Parser_dh.h"


#undef __FUNC__
#define __FUNC__ "sigHandler_dh"
void sigHandler_dh(int sig)
{
  fprintf(stderr, "\n[%i] Euclid Signal Handler got: %s\n", myid_dh, SIGNAME[sig]);
  fprintf(stderr, "[%i] ========================================================\n", myid_dh);
  fprintf(stderr, "[%i] function calling sequence that led to the exception:\n", myid_dh);
  fprintf(stderr, "[%i] ========================================================\n", myid_dh);
  printFunctionStack(stderr);
  fprintf(stderr, "\n\n");

  if (logFile != NULL) {
    fprintf(logFile, "\n[%i] Euclid Signal Handler got: %s\n", myid_dh, SIGNAME[sig]);
    fprintf(logFile, "[%i] ========================================================\n", myid_dh);
    fprintf(logFile, "[%i] function calling sequence that led to the exception:\n", myid_dh);
    fprintf(logFile, "[%i] ========================================================\n", myid_dh);
    printFunctionStack(logFile);
    fprintf(logFile, "\n\n");
  }

  EUCLID_EXIT;
}

#undef __FUNC__
#define __FUNC__ "sigRegister_dh"
void sigRegister_dh()
{
  if (Parser_dhHasSwitch(parser_dh, "-sig_dh")) {
    int i;
    for (i=0; i<euclid_signals_len; ++i) {
      signal(euclid_signals[i], sigHandler_dh);
    }
  }
}
