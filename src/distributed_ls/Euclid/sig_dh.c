/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




#include "sig_dh.h"
#include "Parser_dh.h"


#undef __FUNC__
#define __FUNC__ "sigHandler_dh"
void sigHandler_dh(hypre_int sig)
{
  hypre_fprintf(stderr, "\n[%i] Euclid Signal Handler got: %s\n", myid_dh, SIGNAME[sig]);
  hypre_fprintf(stderr, "[%i] ========================================================\n", myid_dh);
  hypre_fprintf(stderr, "[%i] function calling sequence that led to the exception:\n", myid_dh);
  hypre_fprintf(stderr, "[%i] ========================================================\n", myid_dh);
  printFunctionStack(stderr);
  hypre_fprintf(stderr, "\n\n");

  if (logFile != NULL) {
    hypre_fprintf(logFile, "\n[%i] Euclid Signal Handler got: %s\n", myid_dh, SIGNAME[sig]);
    hypre_fprintf(logFile, "[%i] ========================================================\n", myid_dh);
    hypre_fprintf(logFile, "[%i] function calling sequence that led to the exception:\n", myid_dh);
    hypre_fprintf(logFile, "[%i] ========================================================\n", myid_dh);
    printFunctionStack(logFile);
    hypre_fprintf(logFile, "\n\n");
  }

  EUCLID_EXIT;
}

#undef __FUNC__
#define __FUNC__ "sigRegister_dh"
void sigRegister_dh()
{
  if (Parser_dhHasSwitch(parser_dh, "-sig_dh")) {
    hypre_int i;
    for (i=0; i<euclid_signals_len; ++i) {
      signal(euclid_signals[i], sigHandler_dh);
    }
  }
}
