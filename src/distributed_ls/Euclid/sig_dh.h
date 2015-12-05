#ifndef MY_SIG_DH
#define MY_SIG_DH

#include "euclid_common.h"
#include <signal.h>

extern void sigRegister_dh();
extern void sigHandler_dh(int sig);

/* 
  list of signals the Euclid will handle
*/
int euclid_signals_len = 3;
int euclid_signals[] = { SIGSEGV, SIGFPE, SIGBUS };

/* 
   signal names and explanatory messages 
*/
static char *SIGNAME[] = {
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

#endif

