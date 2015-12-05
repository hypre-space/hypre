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



/* --------------------------- hypre_MPI_MODE ------------------------ */

#elif defined(hypre_MPI_MODE)

#define  INITIALIZE_DH(argc, argv, help) \
            hypre_MPI_Init(&argc,&argv);  \
            comm_dh = hypre_MPI_COMM_WORLD;    \
            hypre_MPI_Errhandler_set(comm_dh, hypre_MPI_ERRORS_RETURN); \
            EuclidInitialize(argc, argv, help); \
            dh_StartFunc(__FUNC__, __FILE__, __LINE__, 1); \
            {


#define  FINALIZE_DH \
            } \
            dh_EndFunc(__FUNC__, 1); \
            EuclidFinalize(); \
            hypre_MPI_Finalize(); 

/* --------------------------- SEQUENTIAL_MODE ------------------------ */

/* for now, this is identical to hypre_MPI_MODE */

#else

#define  INITIALIZE_DH(argc, argv, help) \
            hypre_MPI_Init(&argc,&argv);  \
            comm_dh = hypre_MPI_COMM_WORLD;    \
            hypre_MPI_Errhandler_set(comm_dh, hypre_MPI_ERRORS_RETURN); \
            EuclidInitialize(argc, argv, help); \
            dh_StartFunc(__FUNC__, __FILE__, __LINE__, 1); \
            {


#define  FINALIZE_DH \
            } \
            dh_EndFunc(__FUNC__, 1); \
            EuclidFinalize(); \
            hypre_MPI_Finalize(); 

#endif

#endif /* #ifndef GUARDS_DH */
