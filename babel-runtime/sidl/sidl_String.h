/*
 * File:        sidl_String.h
 * Copyright:   (c) 2001 The Regents of the University of California
 * Revision:    @(#) $Revision$
 * Date:        $Date$
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

#ifndef included_sidl_String_h
#define included_sidl_String_h

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate a string of the specified size with an additional location for
 * the string null character.  Strings allocated using this method may be
 * freed using <code>sidl_String_free</code> or the standard <code>free</code>
 * library call.
 */
char* sidl_String_alloc(size_t size);

/**
 * Free the memory associated with the specified string.  Nothing is done if
 * the string pointer is null.
 */
void sidl_String_free(char* s);

/**
 * Return the length of the string.  If the string is null, then its length
 * is zero.  Note the string length does not include the terminating null
 * character.
 */
size_t sidl_String_strlen(const char* s);

/**
 * Copy the string <code>s2</code> into <code>s1</code> and include the
 * terminating null character.  Note that this routine does not check whether
 * there is sufficient space in the destination string.
 */
void sidl_String_strcpy(char* s1, const char* s2);

/**
 * Duplicate the string.  If the argument is null, then the return value is
 * null.  This new string should be deallocated by a call to the string free
 * function <code>sidl_String_free</code>.
 */
char* sidl_String_strdup(const char* s);

/**
 * Return whether the two strings are equal.  Either or both of the two
 * argument strings may be null.
 */
int sidl_String_equals(const char* s1, const char* s2);

/**
 * Return whether the first string ends with the second string.  If either
 * of the two strings is null, then return false.
 */
int sidl_String_endsWith(const char* s, const char* end);

/**
 * Return whether the first string starts with the second string.  If either
 * of the two strings is null, then return false.
 */
int sidl_String_startsWith(const char* s, const char* start);

/**
 * Return the substring starting at the specified index and continuing to
 * the end of the string.  If the index is past the end of the string or
 * if the first argument is null, then null is returned.  The return string
 * should be freed by a call to <code>sidl_String_free</code>.
 */
char* sidl_String_substring(const char* s, const int index);

/**
 * Concatenate the two strings and return the resulting string.  Null string
 * arguments are ignored.  The return string should be freed by calling routine
 * <code>sidl_String_free</code>.
 */
char* sidl_String_concat2(const char* s1,
                          const char* s2);

/**
 * Concatenate the three strings and return the resulting string.  Null string
 * arguments are ignored.  The return string should be freed by calling routine
 * <code>sidl_String_free</code>.
 */
char* sidl_String_concat3(const char* s1,
                          const char* s2,
                          const char* s3);

/**
 * Concatenate the four strings and return the resulting string.  Null string
 * arguments are ignored.  The return string should be freed by calling routine
 * <code>sidl_String_free</code>.
 */
char* sidl_String_concat4(const char* s1,
                          const char* s2,
                          const char* s3,
                          const char* s4);

/**
 * Replace instances of oldchar with newchar in the provided string.  Null
 * string arguments are ignored.
 */
void sidl_String_replace(char* s, char oldchar, char newchar);

#ifdef __cplusplus
}
#endif
#endif
