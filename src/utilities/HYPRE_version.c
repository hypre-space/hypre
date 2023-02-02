/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_Version utility functions
 *
 *****************************************************************************/

#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_Version( char **version_ptr )
{
   HYPRE_Int  len = 30;
   char      *version;

   /* compute string length */
   len += strlen(HYPRE_RELEASE_VERSION);

   version = hypre_CTAlloc(char, len, HYPRE_MEMORY_HOST);

   hypre_sprintf(version, "HYPRE Release Version %s", HYPRE_RELEASE_VERSION);

   *version_ptr = version;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_VersionNumber( HYPRE_Int  *major_ptr,
                     HYPRE_Int  *minor_ptr,
                     HYPRE_Int  *patch_ptr,
                     HYPRE_Int  *single_ptr )
{
   HYPRE_Int  major, minor, patch, single;
   HYPRE_Int  nums[3], i, j;
   char      *ptr = (char *) HYPRE_RELEASE_VERSION;

   /* get major/minor/patch numbers */
   for (i = 0; i < 3; i++)
   {
      char str[4];

      for (j = 0; (j < 3) && (*ptr != '.') && (*ptr != '\0'); j++)
      {
         str[j] = *ptr;
         ptr++;
      }
      str[j] = '\0';
      nums[i] = atoi((char *)str);
      ptr++;
   }
   major = nums[0];
   minor = nums[1];
   patch = nums[2];

   single = (HYPRE_Int) HYPRE_RELEASE_NUMBER;

   if (major_ptr)   {*major_ptr   = major;}
   if (minor_ptr)   {*minor_ptr   = minor;}
   if (patch_ptr)   {*patch_ptr   = patch;}
   if (single_ptr)  {*single_ptr  = single;}

   return hypre_error_flag;
}

