/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
//#define __USE_GNU
#include <signal.h>
#include <execinfo.h>
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

/*--------------------------------------------------------------------------
 * hypre_multmod
 *--------------------------------------------------------------------------*/

/* This function computes (a*b) % mod, which can avoid overflow in large value of (a*b) */
HYPRE_Int
hypre_multmod(HYPRE_Int a,
              HYPRE_Int b,
              HYPRE_Int mod)
{
   HYPRE_Int res = 0; // Initialize result
   a %= mod;
   while (b)
   {
      // If b is odd, add a with result
      if (b & 1)
      {
         res = (res + a) % mod;
      }
      // Here we assume that doing 2*a
      // doesn't cause overflow
      a = (2 * a) % mod;
      b >>= 1;  // b = b / 2
   }
   return res;
}

/*--------------------------------------------------------------------------
 * hypre_partition1D
 *--------------------------------------------------------------------------*/
void
hypre_partition1D(HYPRE_Int  n, /* total number of elements */
                  HYPRE_Int  p, /* number of partitions */
                  HYPRE_Int  j, /* index of this partition */
                  HYPRE_Int *s, /* first element in this partition */
                  HYPRE_Int *e  /* past-the-end element */ )

{
   if (1 == p)
   {
      *s = 0;
      *e = n;
      return;
   }

   HYPRE_Int size = n / p;
   HYPRE_Int rest = n - size * p;
   if (j < rest)
   {
      *s = j * (size + 1);
      *e = (j + 1) * (size + 1);
   }
   else
   {
      *s = j * size + rest;
      *e = (j + 1) * size + rest;
   }
}

/*--------------------------------------------------------------------------
 * hypre_strcpy
 *
 * Note: strcpy that allows overlapping in memory
 *--------------------------------------------------------------------------*/

char *
hypre_strcpy(char *destination, const char *source)
{
   size_t len = strlen(source);

   /* no overlapping */
   if (source > destination + len || destination > source + len)
   {
      return strcpy(destination, source);
   }
   else
   {
      /* +1: including the terminating null character */
      return ((char *) memmove(destination, source, len + 1));
   }
}

/*--------------------------------------------------------------------------
 * hypre_CheckDirExists
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CheckDirExists(const char *path)
{
#ifndef _WIN32
   DIR *dir = opendir(path);

   if (dir)
   {
      closedir(dir);
      return 1;
   }
#else
   DWORD att = GetFileAttributesA(path);

   if (att == INVALID_FILE_ATTRIBUTES)
   {
      return 0;
   }

   if (att & FILE_ATTRIBUTE_DIRECTORY)
   {
      return 1;
   }
#endif

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CreateDir
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CreateDir(const char *path)
{
   char msg[HYPRE_MAX_MSG_LEN];

   if (mkdir(path, 0777))
   {
      hypre_sprintf(msg, "Could not create directory: %s", path);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CreateNextDirOfSequence
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CreateNextDirOfSequence(const char *basepath, const char *prefix, char **fullpath_ptr)
{
   HYPRE_Int       max_suffix = -1;
   char           *fullpath;

#ifndef _WIN32
   HYPRE_Int       suffix;
   char            msg[HYPRE_MAX_MSG_LEN];
   DIR            *dir;
   struct dirent  *entry;

   if ((dir = opendir(basepath)) == NULL)
   {
      hypre_sprintf(msg, "Could not open directory: %s", basepath);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
      return hypre_error_flag;
   }

   max_suffix = -1;
   while ((entry = readdir(dir)) != NULL)
   {
      if (strncmp(entry->d_name, prefix, strlen(prefix)) == 0)
      {
         if (hypre_sscanf(entry->d_name + strlen(prefix), "%d", &suffix) == 1)
         {
            if (suffix > max_suffix)
            {
               max_suffix = suffix;
            }
         }
      }
   }
   closedir(dir);
#else
   /* TODO (VPM) */
#endif

   /* Create directory */
   fullpath = hypre_TAlloc(char, strlen(basepath) + 10, HYPRE_MEMORY_HOST);
   hypre_sprintf(fullpath, "%s/%s%05d", basepath, prefix, max_suffix + 1);
   hypre_CreateDir(fullpath);

   /* Set output pointer */
   *fullpath_ptr = fullpath;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_ConvertIndicesToString
 *
 * Converts an array of integers (indices) into a formatted string.
 * The function creates a string representing the array in a comma-
 * separated format, enclosed within square brackets ("[]").
 *
 * - If the input array is empty (size = 0), it returns a string "[]".
 * - The resulting string includes the list of integers with proper
 *   formatting: each integer is separated by a comma and a space.
 *
 * Parameters:
 * - size: Number of elements in the input array.
 * - indices: Pointer to the array of integers (HYPRE_Int) to convert.
 *
 * Returns:
 * - A dynamically allocated string representing the integer array.
 *--------------------------------------------------------------------*/

char*
hypre_ConvertIndicesToString(HYPRE_Int  size,
                             HYPRE_Int *indices)
{
   HYPRE_Int    max_length;
   HYPRE_Int    i, length;
   char        *string;
   char        *pos;

   if (!size)
   {
      string = hypre_TAlloc(char, 3, HYPRE_MEMORY_HOST);
      hypre_sprintf(string, "[]");

      return string;
   }

   /* Estimate maximum string needed */
   max_length = 12 * size + 3;
   string = hypre_TAlloc(char, max_length, HYPRE_MEMORY_HOST);

   pos    = string;
   length = hypre_sprintf(pos, "[");
   pos    += length;

   for (i = 0; i < size; i++)
   {
      /* Add comma before all but the first element */
      if (i > 0)
      {
         length = hypre_sprintf(pos, ", ");
         pos += length;
      }

      /* Write integer as string */
      length = hypre_sprintf(pos, "%d", indices[i]);
      pos += length;
   }

   hypre_sprintf(pos, "]");

   return string;
}

/*--------------------------------------------------------------------------
 * hypre_PrintStackTrace
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PrintStackTrace(HYPRE_Int rank)
{
#ifdef _WIN32
   // Windows implementation not provided
   hypre_printf("Stack trace not implemented for Windows\n");
#else
   void *stack[64];
   int frames = backtrace(stack, 64);

   // Convert raw addresses into an array of strings that describe them
   char **symbols = backtrace_symbols(stack, frames);

   /* Print a fancy header for readability */
   hypre_printf("\n");
   hypre_printf("============================================================");
   hypre_printf("============================================================");
   hypre_printf("\n Stack trace for rank %d\n", rank);
   hypre_printf("============================================================");
   hypre_printf("============================================================\n");

   for (int i = 0; i < frames; i++)
   {
#ifndef __USE_GNU
      hypre_printf(" [%02d]: %s\n", i, symbols[i]);
#else
      Dl_info info;
      if (dladdr(stack[i], &info) && info.dli_fname)
      {
         // We have a shared object or executable file name
         const char *obj_path = info.dli_fname;

         // Calculate offset from the base of this object
         unsigned long base_addr  = (unsigned long)info.dli_fbase;
         unsigned long frame_addr = (unsigned long)stack[i];
         unsigned long offset     = frame_addr - base_addr;

         // Attempt to get file and line information using addr2line
         // The offset alone is typically enough for most debugging setups
         char cmd[1024];
         snprintf(cmd, sizeof(cmd),
                  "addr2line -Cfe %s 0x%lx 2>/dev/null", 
                  obj_path, offset);

         FILE *pipe = popen(cmd, "r");
         if (pipe)
         {
            char func_line[512] = {0};
            char file_line[512] = {0};

            // addr2line -f prints the (demangled) function name on the first line
            // and file:line on the second line
            if (fgets(func_line, sizeof(func_line), pipe) &&
                fgets(file_line, sizeof(file_line), pipe))
            {
               // Remove trailing newline
               func_line[strcspn(func_line, "\n")] = 0;
               file_line[strcspn(file_line, "\n")] = 0;

               if (strcmp(file_line, "??:0") != 0 && strcmp(file_line, "??:?") != 0)
               {
                  hypre_printf(" [%02d]: %s (%s)\n", i, file_line, func_line);
               }
               else if (strcmp(func_line, "??") != 0)
               {
                  // We got some function name but no file:line
                  hypre_printf(" [%02d]: %s\n", i, func_line);
               }
            }
            pclose(pipe);
         }
      }
#endif
   }

   /* Print a closing footer */
   hypre_printf("============================================================");
   hypre_printf("============================================================\n\n");

   free(symbols);
#endif

   return hypre_error_flag;
}
