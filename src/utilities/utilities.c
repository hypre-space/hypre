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
