#ifndef EUCLID_USER_INTERFAACE
#define EUCLID_USER_INTERFAACE

/* collection of everthing that should be included for user apps */

#include "Euclid_dh.h"
#include "SubdomainGraph_dh.h"
#include "Mat_dh.h"
#include "Factor_dh.h"
#include "MatGenFD.h"
#include "Vec_dh.h"
#include "Parser_dh.h"
#include "Mem_dh.h"
#include "Timer_dh.h"
#include "TimeLog_dh.h"
#include "guards_dh.h"
#include "krylov_dh.h"
#include "io_dh.h"
#include "mat_dh_private.h"

#ifdef PETSC_MODE
#include "euclid_petsc.h"
#endif

#ifdef FAKE_MPI
#include "fake_mpi.h"
#endif

#endif
