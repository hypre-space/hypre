/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Fortran <-> C interface macros
 *
 *****************************************************************************/

#ifndef HYPRE_FORTRAN_HEADER
#define HYPRE_FORTRAN_HEADER

#if defined(HYPRE_SOLARIS) || defined(HYPRE_ALPHA)
#define hypre_NAME_C_FOR_FORTRAN(name) name##_
#define hypre_NAME_FORTRAN_FOR_C(name) name##_
#elif defined(HYPRE_RS6000)
#define hypre_NAME_C_FOR_FORTRAN(name) name##__
#define hypre_NAME_FORTRAN_FOR_C(name) name##
#else
#define hypre_NAME_C_FOR_FORTRAN(name) name##__
#define hypre_NAME_FORTRAN_FOR_C(name) name##_
#endif

#define hypre_F90_IFACE(iface_name) hypre_NAME_FORTRAN_FOR_C(iface_name)

#endif
