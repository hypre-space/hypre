/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_Euclid.h"
/* #include "sig_dh.h" */
/* #include "Parser_dh.h" */
/* #include "euclid_common.h" */

/* RDF: This next code was in 'sig_dh.h' but only used in this file.  Because of
 * the global variables 'euclid_signals_len' and 'euclid_signals', it was easier
 * to put the source directly here instead. */
/* END 'sig_dh.h' code */

#include <signal.h>

extern void sigRegister_dh(void);
extern void sigHandler_dh(hypre_int sig);

/* 
  list of signals the Euclid will handle
*/
#ifdef WIN32
hypre_int euclid_signals_len = 2;
hypre_int euclid_signals[] = { SIGSEGV, SIGFPE };
#else
hypre_int euclid_signals_len = 3;
hypre_int euclid_signals[] = { SIGSEGV, SIGFPE, SIGBUS };
#endif

/* 
   signal names and explanatory messages 
*/
static const char *SIGNAME[] = {
    "Unknown signal",
    "HUP (Hangup detected on controlling terminal or death of controlling process)",
    "INT: Interrupt from keyboard",
    "QUIT: Quit from keyboard",
    "ILL: Illegal Instruction",
    "TRAP",
    "ABRT: Abort signal",
    "EMT",
    "FPE (Floating Point Exception)",
    "KILL: Kill signal",
    "BUS (Bus Error, possibly illegal memory access)",
    "SEGV (Segmentation Violation (memory access out of range?))",
    "SYS",
    "PIPE: Broken pipe: write to pipe with no readers",
    "ALRM: Timer signal",
    "TERM: Termination signal",
    "URG",
    "STOP",
    "TSTP",
    "CONT",
    "CHLD"
};

/* END 'sig_dh.h' code */

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
void sigRegister_dh(void)
{
  if (Parser_dhHasSwitch(parser_dh, "-sig_dh")) {
    hypre_int i;
    for (i=0; i<euclid_signals_len; ++i) {
      signal(euclid_signals[i], sigHandler_dh);
    }
  }
}
