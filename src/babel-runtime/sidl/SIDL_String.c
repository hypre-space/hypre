/*
 * File:        SIDL_String.c
 * Copyright:   (c) 2001 The Regents of the University of California
 * Release:     $Name: V1-9-0b $
 * Revision:    @(#) $Revision: 1.4 $
 * Date:        $Date: 2003/04/07 21:44:31 $
 * Description: convenience string manipulation functions for C clients
 * Copyright (c) 2000-2001, The Regents of the University of Calfornia.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * UCRL-CODE-2002-054
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include "SIDL_String.h"
#include <string.h>

#ifndef NULL
#define NULL 0
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

/*
 * Allocate a string of the specified size with an additional location for
 * the string null character.  Strings allocated using this method should be
 * freed using <code>SIDL_String_free</code>.
 */
char* SIDL_String_alloc(size_t size)
{
   return (char*) malloc(size + 1);
}

/*
 * Free the memory associated with the specified string.  Nothing is done if
 * the string pointer is null.
 */
void SIDL_String_free(char* s)
{
   if (s) {
      free((void*) s);
   }
}

/*
 * Return the length of the string.  If the string is null, then its length
 * is zero.  Note the string length does not include the terminating null
 * character.
 */
size_t SIDL_String_strlen(const char* s)
{
   size_t len = 0;
   if (s) {
      len = strlen(s);
   }
   return len;
}

/*
 * Copy the string <code>s2</code> into <code>s1</code> and include the
 * terminating null character.  Note that this routine does not check whether
 * there is sufficient space in the destination string.
 */
void SIDL_String_strcpy(char* s1, const char* s2)
{
   if (s1) {
      if (s2) {
         strcpy(s1, s2);
      } else {
        *s1 = '\0';
      }
   }
}

/*
 * Duplicate the string.  If the argument is null, then the return value is
 * null.  This new string should be deallocated by a call to the string free
 * function <code>SIDL_String_free</code>.
 */
char* SIDL_String_strdup(const char* s)
{
   char* str = NULL;
   if (s) {
      str = SIDL_String_alloc(SIDL_String_strlen(s));
      SIDL_String_strcpy(str, s);
   }
   return str;
}

/*
 * Return whether the two strings are equal.  Either or both of the two
 * argument strings may be null.
 */
int SIDL_String_equals(const char* s1, const char* s2)
{
   int eq = FALSE;

   if (s1 == s2) {
      eq = TRUE;
   } else if ((s1 != NULL) && (s2 != NULL)) {
      eq = strcmp(s1, s2) ? FALSE : TRUE;
   }

   return eq;
}

/*
 * Return whether the first string ends with the second string.  If either
 * of the two strings is null, then return false.
 */
int SIDL_String_endsWith(const char* s, const char* end)
{
   int ends_with = FALSE;

   if ((s != NULL) && (end != NULL)) {
      int offset = SIDL_String_strlen(s) - SIDL_String_strlen(end);
      if ((offset >= 0) && !strcmp(&s[offset], end)) {
         ends_with = TRUE;
      }
   }

   return ends_with;
}

/*
 * Return whether the first string starts with the second string.  If either
 * of the two strings is null, then return false.
 */
int SIDL_String_startsWith(const char* s, const char* start)
{
   int match = FALSE;

   if ((s != NULL) && (start != NULL)) {
      match = strncmp(s, start, SIDL_String_strlen(start)) ? FALSE : TRUE;
   }

   return match;
}

/*
 * Return the substring starting at the specified index and continuing to
 * the end of the string.  If the index is past the end of the string or
 * if the first argument is null, then null is returned.  The return string
 * should be freed by a call to <code>SIDL_String_free</code>.
 */
char* SIDL_String_substring(const char* s, const int index)
{
   char* substring = NULL;

   if (s != NULL) {
      size_t len = SIDL_String_strlen(s);
      if (index < len) {
         substring = SIDL_String_strdup(&s[index]);
      }
   }

   return substring;
}

/*
 * Concatenate the two strings and return the resulting string.  Null string
 * arguments are ignored.  The return string should be freed by calling routine
 * <code>SIDL_String_free</code>.
 */
char* SIDL_String_concat2(const char* s1, const char* s2)
{
   size_t len1 = SIDL_String_strlen(s1);
   size_t len2 = SIDL_String_strlen(s2);
   size_t lenN = len1 + len2;

   char* s = SIDL_String_alloc(lenN);

   SIDL_String_strcpy(s, s1);
   SIDL_String_strcpy(&s[len1], s2);

   return s;
}

/*
 * Concatenate the three strings and return the resulting string.  Null string
 * arguments are ignored.  The return string should be freed by calling routine
 * <code>SIDL_String_free</code>.
 */
char* SIDL_String_concat3(const char* s1, const char* s2, const char* s3)
{
   size_t len1 = SIDL_String_strlen(s1);
   size_t len2 = SIDL_String_strlen(s2);
   size_t len3 = SIDL_String_strlen(s3);
   size_t lenN = len1 + len2 + len3;

   char* s = SIDL_String_alloc(lenN);

   SIDL_String_strcpy(s, s1);
   SIDL_String_strcpy(&s[len1], s2);
   SIDL_String_strcpy(&s[len1+len2], s3);

   return s;
}

/*
 * Concatenate the four strings and return the resulting string.  Null string
 * arguments are ignored.  The return string should be freed by calling routine
 * <code>SIDL_String_free</code>.
 */
char* SIDL_String_concat4(
   const char* s1, const char* s2, const char* s3, const char* s4)
{
   size_t len1 = SIDL_String_strlen(s1);
   size_t len2 = SIDL_String_strlen(s2);
   size_t len3 = SIDL_String_strlen(s3);
   size_t len4 = SIDL_String_strlen(s4);
   size_t lenN = len1 + len2 + len3 + len4;

   char* s = SIDL_String_alloc(lenN);

   SIDL_String_strcpy(s, s1);
   SIDL_String_strcpy(&s[len1], s2);
   SIDL_String_strcpy(&s[len1+len2], s3);
   SIDL_String_strcpy(&s[len1+len2+len3], s4);

   return s;
}

/*
 * Replace instances of oldchar with newchar in the provided string.  Null
 * string arguments are ignored.
 */
void SIDL_String_replace(char* s, char oldchar, char newchar)
{
  if (s != NULL) {
    char* ptr = s;
    while (*ptr) {
      if (*ptr == oldchar) {
        *ptr = newchar;
      }
      ptr++;
    }
  }
}
