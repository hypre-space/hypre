/*
 * File:        sidl_String.c
 * Copyright:   (c) 2001 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.8 $
 * Date:        $Date: 2007/09/27 19:35:44 $
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

#include "sidl_String.h"
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
 * freed using <code>sidl_String_free</code>.
 */
char* sidl_String_alloc(size_t size)
{
   return (char*) malloc(size + 1);
}

/*
 * Free the memory associated with the specified string.  Nothing is done if
 * the string pointer is null.
 */
void sidl_String_free(char* s)
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
size_t sidl_String_strlen(const char* s)
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
void sidl_String_strcpy(char* s1, const char* s2)
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
 * function <code>sidl_String_free</code>.
 */
char* sidl_String_strdup(const char* s)
{
   char* str = NULL;
   if (s) {
      str = sidl_String_alloc(sidl_String_strlen(s));
      sidl_String_strcpy(str, s);
   }
   return str;
}


/*
 * Duplicate the string.  If the argument is null, then the return value is
 * null.  This new string should be deallocated by a call to the string free
 * function <code>sidl_String_free</code>.
 */
char* sidl_String_strndup(const char* s, size_t n)
{
   char* str = NULL;
   const char* p = s;
   if (s && n>0) {
     /* find  len=min(strlen(s),n) safely! */
     int len=1;
     while( (*p!='\0') && (len<n) ) { 
       p++; len++;
     }
     if (len < n) { 
       str = sidl_String_alloc(len);
       memcpy(str,s,len-1);
       str[len-1]='\0';
     } else { 
       str = sidl_String_alloc(n+1);
       memcpy(str, s, n);
       str[n]='\0';
     }
   }
   return str;
}

/*
 * Return whether the two strings are equal.  Either or both of the two
 * argument strings may be null.
 */
int sidl_String_equals(const char* s1, const char* s2)
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
int sidl_String_endsWith(const char* s, const char* end)
{
   int ends_with = FALSE;

   if ((s != NULL) && (end != NULL)) {
      int offset = sidl_String_strlen(s) - sidl_String_strlen(end);
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
int sidl_String_startsWith(const char* s, const char* start)
{
   int match = FALSE;

   if ((s != NULL) && (start != NULL)) {
      match = strncmp(s, start, sidl_String_strlen(start)) ? FALSE : TRUE;
   }

   return match;
}

/*
 * Return the substring starting at the specified index and continuing to
 * the end of the string.  If the index is past the end of the string or
 * if the first argument is null, then null is returned.  The return string
 * should be freed by a call to <code>sidl_String_free</code>.
 */
char* sidl_String_substring(const char* s, const int index)
{
   char* substring = NULL;

   if (s != NULL) {
      size_t len = sidl_String_strlen(s);
      if (index < len) {
         substring = sidl_String_strdup(&s[index]);
      }
   }

   return substring;
}

/*
 * Concatenate the two strings and return the resulting string.  Null string
 * arguments are ignored.  The return string should be freed by calling routine
 * <code>sidl_String_free</code>.
 */
char* sidl_String_concat2(const char* s1, const char* s2)
{
   size_t len1 = sidl_String_strlen(s1);
   size_t len2 = sidl_String_strlen(s2);
   size_t lenN = len1 + len2;

   char* s = sidl_String_alloc(lenN);

   sidl_String_strcpy(s, s1);
   sidl_String_strcpy(&s[len1], s2);

   return s;
}

/*
 * Concatenate the three strings and return the resulting string.  Null string
 * arguments are ignored.  The return string should be freed by calling routine
 * <code>sidl_String_free</code>.
 */
char* sidl_String_concat3(const char* s1, const char* s2, const char* s3)
{
   size_t len1 = sidl_String_strlen(s1);
   size_t len2 = sidl_String_strlen(s2);
   size_t len3 = sidl_String_strlen(s3);
   size_t lenN = len1 + len2 + len3;

   char* s = sidl_String_alloc(lenN);

   sidl_String_strcpy(s, s1);
   sidl_String_strcpy(&s[len1], s2);
   sidl_String_strcpy(&s[len1+len2], s3);

   return s;
}

/*
 * Concatenate the four strings and return the resulting string.  Null string
 * arguments are ignored.  The return string should be freed by calling routine
 * <code>sidl_String_free</code>.
 */
char* sidl_String_concat4(
   const char* s1, const char* s2, const char* s3, const char* s4)
{
   size_t len1 = sidl_String_strlen(s1);
   size_t len2 = sidl_String_strlen(s2);
   size_t len3 = sidl_String_strlen(s3);
   size_t len4 = sidl_String_strlen(s4);
   size_t lenN = len1 + len2 + len3 + len4;

   char* s = sidl_String_alloc(lenN);

   sidl_String_strcpy(s, s1);
   sidl_String_strcpy(&s[len1], s2);
   sidl_String_strcpy(&s[len1+len2], s3);
   sidl_String_strcpy(&s[len1+len2+len3], s4);

   return s;
}

/*
 * Replace instances of oldchar with newchar in the provided string.  Null
 * string arguments are ignored.
 */
void sidl_String_replace(char* s, char oldchar, char newchar)
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
