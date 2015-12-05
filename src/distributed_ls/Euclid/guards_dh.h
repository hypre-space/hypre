/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.3 $
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
