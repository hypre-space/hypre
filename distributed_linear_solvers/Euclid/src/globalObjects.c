/* Contains definitions of globally scoped  objects; 
 * Also, functions for error handling and message logging.
 */

#include "euclid_common.h"

/*-------------------------------------------------------------------------
 * Globally scoped variables, flags, and objects
 *-------------------------------------------------------------------------*/
bool        errFlag_dh = false; /* set to "true" by functions encountering errors */
Parser_dh   parser_dh = NULL;   /* for setting/getting runtime options */
TimeLog_dh  tlog_dh = NULL;     /* internal timing  functionality */
Mem_dh      mem_dh = NULL;      /* memory management */
FILE        *logFile = NULL;
char        msgBuf_dh[MSG_BUF_SIZE_DH]; /* for internal use */
int         np_dh = 1;     /* number of processors and subdomains */
int         myid_dh = 0;   /* rank of this processor (and subdomain) */
MPI_Comm    comm_dh = 0;


  /* Each processor (may) open a logfile.
   * The bools are switches for controlling the amount of informational 
   * output, and where it gets written to.  Function logging is only enabled
   * when compiled with the debugging (-g) option.
   */
FILE *logFile;
void openLogfile_dh(int argc, char *argv[]);
void closeLogfile_dh();
bool logInfoToStderr  = false;
bool logInfoToFile    = true;
bool logFuncsToStderr = false;
bool logFuncsToFile   = false;


/*-------------------------------------------------------------------------
 * End of global definitions. 
 * Error and info functions follow.
 *-------------------------------------------------------------------------*/

#define MAX_MSG_SIZE 1024
static char errMsg[MAX_MSG_SIZE];   

void  openLogfile_dh(int argc, char *argv[])
{
  char buf[1024];

  /* this doesn't really belong here, but it's gotta go someplace! */
  strcpy(errMsg, "error msg was never set -- ??");

  if (logFile != NULL) return; 

  /* set default logging filename */
  sprintf(buf, "logFile.%i", myid_dh);

#if 0
  sprintf(testBuf, "logFile");

  /* set user supplied logging filename, if one was specified */
  for (j=1; j<argc; ++j) {
    if (strcmp(argv[j],"-logFile") == 0) { 
      if (j+1 < argc) {
        sprintf(buf, "%s.%i", argv[j+1], myid_dh);
        sprintf(testBuf, "%s", argv[j+1]);
        break;
      }
    }
  }

  /* attempt to open logfile, unless the user entered "-logFile none" */
  if (strcmp(testBuf, "none")) {

#endif

    if ((logFile = fopen(buf, "w")) == NULL ) {
      fprintf(stderr, "can't open >%s< for writing; continuing anyway\n", buf);
    }
 /* } */
}

void  closeLogfile_dh()
{
  if (logFile != NULL) {
    if (fclose(logFile)) {
      fprintf(stderr, "Error closing logFile\n");
    }
    logFile = NULL;
  }
}

void  setInfo_dh(char *msg, char *function, char *file, int line)
{
  if (logInfoToFile && logFile != NULL) {
    fprintf(logFile, "INFO: %s;\n       function= %s  file=%s  line=%i\n", 
                                          msg, function, file, line);
    fflush(logFile);
  }
  if (logInfoToStderr) {
    fprintf(stderr, "INFO: %s;\n       function= %s  file=%s  line=%i\n", 
                                          msg, function, file, line);
  }
}

/*----------------------------------------------------------------------
 *  Error handling stuph follows
 *----------------------------------------------------------------------*/

#define MAX_STACK_SIZE 20
static  char errMsg_private[MAX_STACK_SIZE][MAX_MSG_SIZE];
static  int errCount_private = 0;

void  setError_dh(char *msg, char *function, char *file, int line)
{
  errFlag_dh = true;
  if (! strcmp(msg, "")) {
    sprintf(errMsg_private[errCount_private], 
        "[%i] called from: %s  file= %s  line= %i", 
                                        myid_dh, function, file, line);
  } else {
    sprintf(errMsg_private[errCount_private], 
        "[%i] ERROR: %s\n       %s  file= %s  line= %i\n", 
                                           myid_dh, msg, function, file, line);
  }
  ++errCount_private;

  /* shouldn't do things like this; but we're not building
     for the ages: all the world's a stage, this is merely a
     prop to be bonfired at play's close.
   */
  if (errCount_private == MAX_STACK_SIZE) --errCount_private;
}

void  printErrorMsg(FILE *fp)
{
  if (! errFlag_dh) {
    fprintf(fp, "errFlag_dh is not set; nothing to print!\n");
    fflush(fp);
  } else {
    int i;
    fprintf(fp, "\n============= error stack trace ====================\n");
    for (i=0; i<errCount_private; ++i) {
      fprintf(fp, "%s\n", errMsg_private[i]);
    }
    fprintf(fp, "\n");
    fflush(fp);
  }
}


/*----------------------------------------------------------------------
 *  function call tracing support follows
 *----------------------------------------------------------------------*/

#define MAX_ERROR_SPACES   200
static char spaces[MAX_ERROR_SPACES];
static int nesting = 0;
static bool initSpaces = true;
#define INDENT_DH 3

void Error_dhStartFunc(char *function, char *file, int line)
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
    fprintf(stderr, "%s(%i) %s  [file= %s  line= %i]\n", 
                            spaces, nesting, function, file, line);
  }
  if (logFuncsToFile && logFile != NULL) {
    fprintf(logFile, "%s(%i) %s  [file= %s  line= %i]\n", 
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
