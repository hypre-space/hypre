#ifndef GUARDS_DH
#define GUARDS_DH

#include "euclid_common.h"


/*
   This file defines the INITIALIZE_DH and FINALIZE_DH macros
*/


/* --------------------------- PETSC_MODE ------------------------ */


#if defined(PETSC_MODE)

#define  INITIALIZE_DH(argc, argv, help) \
            PetscInitialize(&argc,&argv,(char*)0,help); \
            comm_dh = PETSC_COMM_WORLD;  \
            EuclidInitialize(argc, argv, help); \
            dh_StartFunc(__FUNC__, __FILE__, __LINE__, 1); \
            {


#define  FINALIZE_DH \
            } \
            dh_EndFunc(__FUNC__, 1); \
            EuclidFinalize(); \
            PetscFinalize(); 



/* --------------------------- MPI_MODE ------------------------ */

#elif defined(MPI_MODE)

#define  INITIALIZE_DH(argc, argv, help) \
            MPI_Init(&argc,&argv);  \
            comm_dh = MPI_COMM_WORLD;    \
            MPI_Errhandler_set(comm_dh, MPI_ERRORS_RETURN); \
            EuclidInitialize(argc, argv, help); \
            dh_StartFunc(__FUNC__, __FILE__, __LINE__, 1); \
            {


#define  FINALIZE_DH \
            } \
            dh_EndFunc(__FUNC__, 1); \
            EuclidFinalize(); \
            MPI_Finalize(); 

/* --------------------------- SEQUENTIAL_MODE ------------------------ */

/* for now, this is identical to MPI_MODE */

#else

#define  INITIALIZE_DH(argc, argv, help) \
            MPI_Init(&argc,&argv);  \
            comm_dh = MPI_COMM_WORLD;    \
            MPI_Errhandler_set(comm_dh, MPI_ERRORS_RETURN); \
            EuclidInitialize(argc, argv, help); \
            dh_StartFunc(__FUNC__, __FILE__, __LINE__, 1); \
            {


#define  FINALIZE_DH \
            } \
            dh_EndFunc(__FUNC__, 1); \
            EuclidFinalize(); \
            MPI_Finalize(); 

#endif

#endif /* #ifndef GUARDS_DH */
