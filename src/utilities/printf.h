/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_PRINTF_HEADER
#define hypre_PRINTF_HEADER

#include <stdio.h>

/* printf.c */
// #ifdef HYPRE_BIGINT
HYPRE_Int hypre_ndigits( HYPRE_BigInt number );
HYPRE_Int hypre_printf( const char *format, ... );
HYPRE_Int hypre_fprintf( FILE *stream, const char *format, ... );
HYPRE_Int hypre_sprintf( char *s, const char *format, ... );
HYPRE_Int hypre_snprintf( char *s, size_t size, const char *format, ...);
HYPRE_Int hypre_scanf( const char *format, ... );
HYPRE_Int hypre_fscanf( FILE *stream, const char *format, ... );
HYPRE_Int hypre_sscanf( char *s, const char *format, ... );
HYPRE_Int hypre_ParPrintf(MPI_Comm comm, const char *format, ...);
// #else
// #define hypre_printf  printf
// #define hypre_fprintf fprintf
// #define hypre_sprintf sprintf
// #define hypre_scanf   scanf
// #define hypre_fscanf  fscanf
// #define hypre_sscanf  sscanf
// #endif

#endif
