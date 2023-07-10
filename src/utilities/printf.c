/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include <stdarg.h>
#include <stdio.h>

#define hypre_printf_buffer_len 4096
char hypre_printf_buffer[hypre_printf_buffer_len];

// #ifdef HYPRE_BIGINT

/* these prototypes are missing by default for some compilers */
/*
int vscanf( const char *format , va_list arg );
int vfscanf( FILE *stream , const char *format, va_list arg );
int vsscanf( const char *s , const char *format, va_list arg );
*/

HYPRE_Int
new_format( const char *format,
            char **newformat_ptr )
{
   const char *fp;
   char       *newformat, *nfp;
   HYPRE_Int   newformatlen;
   HYPRE_Int   copychar;
   HYPRE_Int   foundpercent = 0;

   newformatlen = 2 * strlen(format) + 1; /* worst case is all %d's to %lld's */

   if (newformatlen > hypre_printf_buffer_len)
   {
      newformat = hypre_TAlloc(char, newformatlen, HYPRE_MEMORY_HOST);
   }
   else
   {
      newformat = hypre_printf_buffer;
   }

   nfp = newformat;
   for (fp = format; *fp != '\0'; fp++)
   {
      copychar = 1;
      if (*fp == '%')
      {
         foundpercent = 1;
      }
      else if (foundpercent)
      {
         if (*fp == 'l')
         {
            fp++; /* remove 'l' and maybe add it back in switch statement */
            if (*fp == 'l')
            {
               fp++; /* remove second 'l' if present */
            }
         }
         switch (*fp)
         {
            case 'b': /* used for BigInt type in hypre */
#if defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT)
               *nfp = 'l'; nfp++;
               *nfp = 'l'; nfp++;
#endif
               *nfp = 'd'; nfp++; copychar = 0;
               foundpercent = 0; break;
            case 'd':
            case 'i':
#if defined(HYPRE_BIGINT)
               *nfp = 'l'; nfp++;
               *nfp = 'l'; nfp++;
#endif
               foundpercent = 0; break;
            case 'f':
            case 'e':
            case 'E':
            case 'g':
            case 'G':
#if defined(HYPRE_SINGLE)          /* no modifier */
#elif defined(HYPRE_LONG_DOUBLE)   /* modify with 'L' */
               *nfp = 'L'; nfp++;
#else                              /* modify with 'l' (default is _double_) */
               *nfp = 'l'; nfp++;
#endif
               foundpercent = 0; break;
            case 'c':
            case 'n':
            case 'o':
            case 'p':
            case 's':
            case 'u':
            case 'x':
            case 'X':
            case '%':
               foundpercent = 0; break;
         }
      }
      if (copychar)
      {
         *nfp = *fp; nfp++;
      }
   }
   *nfp = *fp;

   *newformat_ptr = newformat;

   /*   printf("\nNEWFORMAT: %s\n", *newformat_ptr);*/

   return 0;
}

HYPRE_Int
free_format( char *newformat )
{
   if (newformat != hypre_printf_buffer)
   {
      hypre_TFree(newformat, HYPRE_MEMORY_HOST);
   }

   return 0;
}

HYPRE_Int
hypre_ndigits( HYPRE_BigInt number )
{
   HYPRE_Int     ndigits = 0;

   while (number)
   {
      number /= 10;
      ndigits++;
   }

   return ndigits;
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

   fflush(stdout);

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

HYPRE_Int
hypre_snprintf( char *s, size_t size, const char *format, ...)
{
   va_list   ap;
   char     *newformat;
   HYPRE_Int ierr = 0;

   va_start(ap, format);
   new_format(format, &newformat);
   ierr = vsnprintf(s, size, newformat, ap);
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

HYPRE_Int
hypre_ParPrintf(MPI_Comm comm, const char *format, ...)
{
   HYPRE_Int my_id;
   HYPRE_Int ierr = hypre_MPI_Comm_rank(comm, &my_id);

   if (ierr)
   {
      return ierr;
   }

   if (!my_id)
   {
      va_list ap;
      char   *newformat;

      va_start(ap, format);
      new_format(format, &newformat);
      ierr = vprintf(newformat, ap);
      free_format(newformat);
      va_end(ap);

      fflush(stdout);
   }

   return ierr;
}
// #else
//
// /* this is used only to eliminate compiler warnings */
// HYPRE_Int hypre_printf_empty;
//
// #endif
