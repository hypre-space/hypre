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
