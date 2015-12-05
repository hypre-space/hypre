/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/




/* Contains definitions of globally scoped  objects; 
 * Also, functions for error handling and message logging.
 */

#include "euclid_common.h"
#include "Parser_dh.h"
#include "Mem_dh.h"
#include "TimeLog_dh.h"
extern void sigRegister_dh(); /* use sig_dh.h if not for euclid_signals_len */

/*-------------------------------------------------------------------------
 * Globally scoped variables, flags, and objects
 *-------------------------------------------------------------------------*/
bool        errFlag_dh = false; /* set to "true" by functions encountering errors */
Parser_dh   parser_dh = NULL;   /* for setting/getting runtime options */
TimeLog_dh  tlog_dh = NULL;     /* internal timing  functionality */
Mem_dh      mem_dh = NULL;      /* memory management */
FILE        *logFile = NULL;
char        msgBuf_dh[MSG_BUF_SIZE_DH]; /* for internal use */
HYPRE_Int         np_dh = 1;     /* number of processors and subdomains */
HYPRE_Int         myid_dh = 0;   /* rank of this processor (and subdomain) */
MPI_Comm    comm_dh = 0;


  /* Each processor (may) open a logfile.
   * The bools are switches for controlling the amount of informational 
   * output, and where it gets written to.  Function logging is only enabled
   * when compiled with the debugging (-g) option.
   */
FILE *logFile;
void openLogfile_dh(HYPRE_Int argc, char *argv[]);
void closeLogfile_dh();
bool logInfoToStderr  = false;
bool logInfoToFile    = true;
bool logFuncsToStderr = false;
bool logFuncsToFile   = false;

bool ignoreMe = true;
HYPRE_Int  ref_counter = 0;


/*-------------------------------------------------------------------------
 * End of global definitions. 
 * Error and info functions follow.
 *-------------------------------------------------------------------------*/

#define MAX_MSG_SIZE 1024
#define MAX_STACK_SIZE 20

static  char errMsg_private[MAX_STACK_SIZE][MAX_MSG_SIZE];
static  HYPRE_Int errCount_private = 0;

static  char calling_stack[MAX_STACK_SIZE][MAX_MSG_SIZE];
/* static  HYPRE_Int  priority_private[MAX_STACK_SIZE]; */
static  HYPRE_Int calling_stack_count = 0;

/* static  char errMsg[MAX_MSG_SIZE];    */

void  openLogfile_dh(HYPRE_Int argc, char *argv[])
{
  char buf[1024];

  /* this doesn't really belong here, but it's gotta go someplace! */
/*  strcpy(errMsg, "error msg was never set -- ??"); */

  if (logFile != NULL) return; 

  /* set default logging filename */
  hypre_sprintf(buf, "logFile");

  /* set user supplied logging filename, if one was specified */
  if (argc && argv != NULL) {
    HYPRE_Int j;
    for (j=1; j<argc; ++j) {
      if (strcmp(argv[j],"-logFile") == 0) { 
        if (j+1 < argc) {
          hypre_sprintf(buf, "%s", argv[j+1]);
          break;
        }
      }
    }
  }

  /* attempt to open logfile, unless the user entered "-logFile none" */
  if (strcmp(buf, "none")) {
    char a[5];
    hypre_sprintf(a, ".%i", myid_dh);
    strcat(buf, a);

    if ((logFile = fopen(buf, "w")) == NULL ) {
      hypre_fprintf(stderr, "can't open >%s< for writing; continuing anyway\n", buf);
    }
  } 
}

void  closeLogfile_dh()
{
  if (logFile != NULL) {
    if (fclose(logFile)) {
      hypre_fprintf(stderr, "Error closing logFile\n");
    }
    logFile = NULL;
  }
}

void  setInfo_dh(char *msg, char *function, char *file, HYPRE_Int line)
{
  if (logInfoToFile && logFile != NULL) {
    hypre_fprintf(logFile, "INFO: %s;\n       function= %s  file=%s  line=%i\n", 
                                          msg, function, file, line);
    fflush(logFile);
  }
  if (logInfoToStderr) {
    hypre_fprintf(stderr, "INFO: %s;\n       function= %s  file=%s  line=%i\n", 
                                          msg, function, file, line);
  }
}

/*----------------------------------------------------------------------
 *  Error handling stuph follows
 *----------------------------------------------------------------------*/

void dh_StartFunc(char *function, char *file, HYPRE_Int line, HYPRE_Int priority)
{
  if (priority == 1) {
    hypre_sprintf(calling_stack[calling_stack_count], 
          "[%i]   %s  file= %s  line= %i", myid_dh, function, file, line);
    /* priority_private[calling_stack_count] = priority; */
    ++calling_stack_count;

    if (calling_stack_count == MAX_STACK_SIZE) {
      hypre_fprintf(stderr, "_____________ dh_StartFunc: OVERFLOW _____________________\n");
      if (logFile != NULL) {
        hypre_fprintf(logFile, "_____________ dh_StartFunc: OVERFLOW _____________________\n");
      }
      --calling_stack_count;
    }
  }
}

void dh_EndFunc(char *function, HYPRE_Int priority)
{
  if (priority == 1) {
    --calling_stack_count;

    if (calling_stack_count < 0) {
      calling_stack_count = 0;
      hypre_fprintf(stderr, "_____________ dh_EndFunc: UNDERFLOW _____________________\n");
      if (logFile != NULL) {
        hypre_fprintf(logFile, "_____________ dh_EndFunc: UNDERFLOW _____________________\n");
      }
    }
  }
}


void  setError_dh(char *msg, char *function, char *file, HYPRE_Int line)
{
  errFlag_dh = true;
  if (! strcmp(msg, "")) {
    hypre_sprintf(errMsg_private[errCount_private], 
        "[%i] called from: %s  file= %s  line= %i", 
                                        myid_dh, function, file, line);
  } else {
    hypre_sprintf(errMsg_private[errCount_private], 
        "[%i] ERROR: %s\n       %s  file= %s  line= %i\n", 
                                           myid_dh, msg, function, file, line);
  }
  ++errCount_private;

  /* shouldn't do things like this; but we're not building
     for the ages: all the world's a stage, this is merely a
     prop to be bonfired at play's end.
   */
  if (errCount_private == MAX_STACK_SIZE) --errCount_private;
}

void  printErrorMsg(FILE *fp)
{
  if (! errFlag_dh) {
    hypre_fprintf(fp, "errFlag_dh is not set; nothing to print!\n");
    fflush(fp);
  } else {
    HYPRE_Int i;
    hypre_fprintf(fp, "\n============= error stack trace ====================\n");
    for (i=0; i<errCount_private; ++i) {
      hypre_fprintf(fp, "%s\n", errMsg_private[i]);
    }
    hypre_fprintf(fp, "\n");
    fflush(fp);
  }
}

void  printFunctionStack(FILE *fp)
{
  HYPRE_Int i;
  for (i=0; i<calling_stack_count; ++i) {
    hypre_fprintf(fp, "%s\n", calling_stack[i]);
  }
  hypre_fprintf(fp, "\n");
  fflush(fp);
}


/*----------------------------------------------------------------------
 *  function call tracing support follows
 *----------------------------------------------------------------------*/

#define MAX_ERROR_SPACES   200
static char spaces[MAX_ERROR_SPACES];
static HYPRE_Int nesting = 0;
static bool initSpaces = true;
#define INDENT_DH 3

void Error_dhStartFunc(char *function, char *file, HYPRE_Int line)
{
  if (initSpaces) {
    memset(spaces, ' ', MAX_ERROR_SPACES*sizeof(char));
    initSpaces = false;
  }

  /* get rid of string null-terminator from last
   * call (if any) to Error_dhStartFunc()
  */
  spaces[INDENT_DH*nesting] = ' ';  

  /* add null-terminator, so the correct number of spaces will be printed */
  ++nesting; 
  if (nesting > MAX_ERROR_SPACES-1) nesting = MAX_ERROR_SPACES-1;
  spaces[INDENT_DH*nesting] = '\0';

  if (logFuncsToStderr) {
    hypre_fprintf(stderr, "%s(%i) %s  [file= %s  line= %i]\n", 
                            spaces, nesting, function, file, line);
  }
  if (logFuncsToFile && logFile != NULL) {
    hypre_fprintf(logFile, "%s(%i) %s  [file= %s  line= %i]\n", 
                            spaces, nesting, function, file, line);
    fflush(logFile);
  }
}

void Error_dhEndFunc(char *function)
{ 
  nesting -= 1;
  if (nesting < 0) nesting = 0;
  spaces[INDENT_DH*nesting] = '\0';
}

/*----------------------------------------------------------------------
 *  Euclid initialization and shutdown
 *----------------------------------------------------------------------*/

static bool EuclidIsActive = false;

#undef __FUNC__
#define __FUNC__ "EuclidIsInitialized"
bool EuclidIsInitialized()
{
  return EuclidIsActive;
}

#undef __FUNC__
#define __FUNC__ "EuclidInitialize"
void EuclidInitialize(HYPRE_Int argc, char *argv[], char *help)
{
  if (! EuclidIsActive) {
    hypre_MPI_Comm_size(comm_dh, &np_dh);
    hypre_MPI_Comm_rank(comm_dh, &myid_dh);
    openLogfile_dh(argc, argv); 
    if (mem_dh == NULL) { Mem_dhCreate(&mem_dh); CHECK_V_ERROR; }
    if (tlog_dh == NULL) { TimeLog_dhCreate(&tlog_dh); CHECK_V_ERROR; }
    if (parser_dh == NULL) { Parser_dhCreate(&parser_dh); CHECK_V_ERROR; }
    Parser_dhInit(parser_dh, argc, argv); CHECK_V_ERROR;
    if (Parser_dhHasSwitch(parser_dh, "-sig_dh")) {
      sigRegister_dh(); CHECK_V_ERROR;
    }
    if (Parser_dhHasSwitch(parser_dh, "-help")) {
      if (myid_dh == 0) hypre_printf("%s\n\n", help);
      EUCLID_EXIT;
    }
    if (Parser_dhHasSwitch(parser_dh, "-logFuncsToFile")) { 
      logFuncsToFile = true;
    }
    if (Parser_dhHasSwitch(parser_dh, "-logFuncsToStderr")) {
      logFuncsToStderr = true;
    }

    EuclidIsActive = true;
  }

}


/* to do: should restore the signal handler that we preempted above! */
#undef __FUNC__
#define __FUNC__ "EuclidFinalize"
void EuclidFinalize()
{
  if (ref_counter) return;

  if (EuclidIsActive) {
    if (parser_dh != NULL) { Parser_dhDestroy(parser_dh); CHECK_V_ERROR; }
    if (tlog_dh != NULL) { TimeLog_dhDestroy(tlog_dh); CHECK_V_ERROR; }
    if (logFile != NULL) { Mem_dhPrint(mem_dh, logFile, true); CHECK_V_ERROR; }
/*  Mem_dhPrint(mem_dh, stderr, false); CHECK_V_ERROR; */
    if (mem_dh != NULL) { Mem_dhDestroy(mem_dh); CHECK_V_ERROR; }
    if (logFile != NULL) { closeLogfile_dh(); CHECK_V_ERROR; }
    EuclidIsActive = false;
  }
}


/*----------------------------------------------------------------------
 *  msc. support functions
 *----------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "printf_dh"
void printf_dh(char *fmt, ...)
{
  START_FUNC_DH
  va_list args;
  char *buf = msgBuf_dh;

  va_start(args, fmt);
  vsprintf(buf, fmt, args);
  if (myid_dh == 0) {
    hypre_fprintf(stdout, "%s", buf);
  }
  va_end(args);
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "fprintf_dh"
void fprintf_dh(FILE *fp, char *fmt, ...)
{
  START_FUNC_DH
  va_list args;
  char *buf = msgBuf_dh;

  va_start(args, fmt);
  vsprintf(buf, fmt, args);
  if (myid_dh == 0) {
    hypre_fprintf(fp, "%s", buf);
  }
  va_end(args);
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "echoInvocation_dh"
void echoInvocation_dh(MPI_Comm comm, char *prefix, HYPRE_Int argc, char *argv[])
{
  START_FUNC_DH
  HYPRE_Int i, id;

  hypre_MPI_Comm_rank(comm, &id);

  if (prefix != NULL) {
    printf_dh("\n%s ", prefix);
  } else {
    printf_dh("\n");
  }

  printf_dh("program invocation: ");
  for (i=0; i<argc; ++i) {
    printf_dh("%s ", argv[i]);
  }
  printf_dh("\n");
  END_FUNC_DH
}
