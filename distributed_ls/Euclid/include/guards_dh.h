#ifndef GUARDS_DH
#define GUARDS_DH

#include "euclid_common.h"

/* note: this header file isn't used by any Euclid objects; 
         therefore, this can be changed without recompiling
         the entire library.
*/

/*
   This file defines the INITIALIZE_DH(a,b,c) and FINALIZE_DH macros.
   All Euclid library usage must be surrounded by these macros.

   The definitions depend on whether PETSC_MODE, MPI_MODE, or 
   SEQUENTIAL_MODE is defined; one of these is defined during
   the make process (see, for example, PCPACK_DIR/bmake_mpi/common).
*/

/* ------------------- Code common to all modes ------------------ */

/*
   WARNING: There are several dependencies here; changing the
            order of the following calls can break things!
           
*/

#define MY_INIT(argc, argv)  \
            openLogfile_dh(argc, argv); \
            Mem_dhCreate(&mem_dh); ERRCHKA; \
            TimeLog_dhCreate(&tlog_dh); ERRCHKA; \
            Parser_dhCreate(&parser_dh); ERRCHKA; \
            Parser_dhInit(parser_dh, argc, argv); ERRCHKA; \
            if (Parser_dhHasSwitch(parser_dh, "-help")) { \
              fprintf(stderr, help);  \
              exit(-1); \
            }  \
            if (Parser_dhHasSwitch(parser_dh, "-logFuncsToFile")) { \
              logFuncsToFile = true; \
            } \
            if (Parser_dhHasSwitch(parser_dh, "-logFuncsToStderr")) { \
              logFuncsToStderr = true; \
            } \
              {

#define MY_FINALIZE   \
             } \
            Parser_dhDestroy(parser_dh); ERRCHKA; \
            TimeLog_dhDestroy(tlog_dh); ERRCHKA;  \
            if (logFile != NULL) Mem_dhPrint(mem_dh, logFile, true); \
            Mem_dhPrint(mem_dh, stderr, false); \
            Mem_dhDestroy(mem_dh); ERRCHKA; \
            closeLogfile_dh(); ERRCHKA; 

/*
              } \
            Parser_dhDestroy(parser_dh); ERRCHKA; \
            if (logFile != NULL) TimeLog_dhPrint(tlog_dh, logFile, true); \
            TimeLog_dhPrint(tlog_dh, stderr, false); \
            TimeLog_dhDestroy(tlog_dh); ERRCHKA;  \
            if (logFile != NULL) Mem_dhPrint(mem_dh, logFile, true); \
            Mem_dhPrint(mem_dh, stderr, false); \
            Mem_dhDestroy(mem_dh); ERRCHKA; \
            closeLogfile_dh(); ERRCHKA; 
*/

#ifdef SEQUENTIAL_MODE
#define INITIALIZE_DH  INIT_SEQ_DH
#define FINALIZE_DH    FINALIZE_SEQ_DH
#endif

#ifdef MPI_MODE
#define INITIALIZE_DH  INIT_MPI_DH
#define FINALIZE_DH    FINALIZE_MPI_DH
#endif

#ifdef PETSC_MODE
#define INITIALIZE_DH INIT_PETSC_DH
#define FINALIZE_DH    FINALIZE_PETSC_DH
#endif

/* --------------------------- PETSC_MODE ------------------------ */


#ifdef PETSC_MODE

#define  INIT_PETSC_DH(argc, argv, help) \
            PetscInitialize(&argc,&argv,(char*)0,help); \
            comm_dh = PETSC_COMM_WORLD;  \
            MPI_Comm_size(comm_dh, &np_dh);  \
            MPI_Comm_rank(comm_dh, &myid_dh);  \
            MY_INIT(argc, argv)


#define  FINALIZE_PETSC_DH     \
            MY_FINALIZE \
            PetscFinalize(); 

#endif 

/* --------------------------- MPI_MODE ------------------------ */

#ifdef MPI_MODE
#include "mpi.h"
#define  INIT_MPI_DH(argc, argv, help) \
            MPI_Init(&argc,&argv);  \
            comm_dh = MPI_COMM_WORLD;    \
            MPI_Comm_size(comm_dh, &np_dh);  \
            MPI_Comm_rank(comm_dh, &myid_dh);  \
            MY_INIT(argc, argv)


#define  FINALIZE_MPI_DH     \
            MY_FINALIZE   \
            MPI_Finalize(); 
#endif

/* --------------------------- SEQUENTIAL_MODE ------------------------ */

#define  INIT_SEQ_DH(argc, argv, help)  MY_INIT(argc, argv)
#define  FINALIZE_SEQ_DH                    MY_FINALIZE

#endif
