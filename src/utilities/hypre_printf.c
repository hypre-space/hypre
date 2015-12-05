/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.1 $
 ***********************************************************************EHEADER*/

#include "_hypre_utilities.h"
#include <stdarg.h>
#include <stdio.h>

#ifdef HYPRE_BIGINT

/* these prototypes are missing by default for some compilers */
int vscanf( const char *format , va_list arg );
int vfscanf( FILE *stream , const char *format, va_list arg );
int vsscanf( const char *s , const char *format, va_list arg );

HYPRE_Int
new_format( const char *format,
            char **newformat_ptr )
{
   const char *fp;
   char       *newformat, *nfp;
   HYPRE_Int   newformatlen;
   HYPRE_Int   foundpercent = 0;

   newformatlen = 2*strlen(format)+1; /* worst case is all %d's to %lld's */
   newformat = hypre_TAlloc(char, newformatlen);

   nfp = newformat;
   for (fp = format; *fp != '\0'; fp++)
   {
      if (*fp == '%')
      {
         foundpercent = 1;
      }
      else if (foundpercent)
      {
         switch(*fp)
         {
            case 'd':
               *nfp = 'l'; nfp++;
               *nfp = 'l'; nfp++;
            case 'c':
            case 'e':
            case 'E':
            case 'f':
            case 'g':
            case 'G':
            case 'i':
            case 'n':
            case 'o':
            case 'p':
            case 's':
            case 'u':
            case 'x':
            case 'S':
            case '%':
               foundpercent = 0;
         }
      }
      *nfp = *fp; nfp++;
   }
   *nfp = *fp; nfp++;

   *newformat_ptr = newformat;

/*   printf("\nNEWFORMAT: %s\n", *newformat_ptr);*/

   return 0;
}

HYPRE_Int
free_format( char *newformat )
{
#ifdef HYPRE_BIGINT
   hypre_TFree(newformat);
#endif

   return 0;
}

/* printf functions */

HYPRE_Int
hypre_printf( const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vprintf(newformat, ap);
   free_format(newformat);
   va_end(ap);

   return ierr;
}

HYPRE_Int
hypre_fprintf( FILE *stream, const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vfprintf(stream, newformat, ap);
   free_format(newformat);
   va_end(ap);

   return ierr;
}

HYPRE_Int
hypre_sprintf( char *s, const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vsprintf(s, newformat, ap);
   free_format(newformat);
   va_end(ap);

   return ierr;
}

/* scanf functions */

HYPRE_Int
hypre_scanf( const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vscanf(newformat, ap);
   free_format(newformat);
   va_end(ap);

   return ierr;
}

HYPRE_Int
hypre_fscanf( FILE *stream, const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vfscanf(stream, newformat, ap);
   free_format(newformat);
   va_end(ap);

   return ierr;
}

HYPRE_Int
hypre_sscanf( char *s, const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vsscanf(s, newformat, ap);
   free_format(newformat);
   va_end(ap);

   return ierr;
}

#else

/* this is used only to eliminate compiler warnings */
HYPRE_Int hypre_printf_empty;

#endif
