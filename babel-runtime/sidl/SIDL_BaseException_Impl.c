/*
 * File:          SIDL_BaseException_Impl.c
 * Symbol:        SIDL.BaseException-v0.8.1
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for SIDL.BaseException
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
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
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "SIDL.BaseException" (version 0.8.1)
 * 
 * Every exception inherits from <code>BaseException</code>.  This class
 * provides basic functionality to get and set error messages and stack
 * traces.
 */

#include "SIDL_BaseException_Impl.h"

/* DO-NOT-DELETE splicer.begin(SIDL.BaseException._includes) */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct SIDL_BaseException_Trace {
  struct SIDL_BaseException_Trace  *next;
  char                             *line;
};
/* DO-NOT-DELETE splicer.end(SIDL.BaseException._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseException__ctor"

void
impl_SIDL_BaseException__ctor(
  SIDL_BaseException self)
{
   /* DO-NOT-DELETE splicer.begin(SIDL.BaseException._ctor) */
  struct SIDL_BaseException__data *data = 
    (struct SIDL_BaseException__data *)
    malloc(sizeof(struct SIDL_BaseException__data));

  if (data) 
  {
    data->d_message      = NULL;
    data->d_trace_head   = NULL;
    data->d_trace_tail   = NULL;
    data->d_trace_length = 0UL;
  }

  SIDL_BaseException__set_data(self, data);
   /* DO-NOT-DELETE splicer.end(SIDL.BaseException._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseException__dtor"

void
impl_SIDL_BaseException__dtor(
  SIDL_BaseException self)
{
   /* DO-NOT-DELETE splicer.begin(SIDL.BaseException._dtor) */
  struct SIDL_BaseException__data *data = 
    (self ? SIDL_BaseException__get_data(self) : NULL);

  if (data) 
  {
    if (data->d_message) {
      free((void *)(data->d_message));
      data->d_message = NULL;
    }

    if (data->d_trace_head) 
    {
      struct SIDL_BaseException_Trace* curr;
      while (data->d_trace_head) 
      {
        curr               = data->d_trace_head;
        data->d_trace_head = data->d_trace_head->next;
        if (curr->line != NULL) {
          free((void*)curr->line);
        }
        free((void *)(curr));
      }
      data->d_trace_head   = NULL;
      data->d_trace_tail   = NULL;
      data->d_trace_length = 0UL;
    }

    free((void*)data);
    SIDL_BaseException__set_data(self, NULL);
  }
   /* DO-NOT-DELETE splicer.end(SIDL.BaseException._dtor) */
}

/*
 * Return the message associated with the exception.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseException_getNote"

char*
impl_SIDL_BaseException_getNote(
  SIDL_BaseException self)
{
   /* DO-NOT-DELETE splicer.begin(SIDL.BaseException.getNote) */
  struct SIDL_BaseException__data *data = 
    (self ? SIDL_BaseException__get_data(self) : NULL);
  char *result = 
    ((data && data->d_message) 
     ? strcpy((char*)malloc(strlen(data->d_message)+1),
              data->d_message)
     : NULL);
  return result;
   /* DO-NOT-DELETE splicer.end(SIDL.BaseException.getNote) */
}

/*
 * Set the message associated with the exception.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseException_setNote"

void
impl_SIDL_BaseException_setNote(
  SIDL_BaseException self, const char* message)
{
   /* DO-NOT-DELETE splicer.begin(SIDL.BaseException.setNote) */
  struct SIDL_BaseException__data *data = 
    (self ? SIDL_BaseException__get_data(self) : NULL);
  if (data) {
    if (data->d_message) {
      free((void*)data->d_message);
    }
    data->d_message = 
      (message 
       ? strcpy((char *)malloc(strlen(message)+1), message)
       : NULL);
  }
   /* DO-NOT-DELETE splicer.end(SIDL.BaseException.setNote) */
}

/*
 * Returns formatted string containing the concatenation of all 
 * tracelines.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseException_getTrace"

char*
impl_SIDL_BaseException_getTrace(
  SIDL_BaseException self)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseException.getTrace) */
  char* tmp = NULL;
  struct SIDL_BaseException__data *data = 
    (self ? SIDL_BaseException__get_data(self) : NULL);

  if (data)
  {
    tmp = (char*)malloc(data->d_trace_length+1);
    if (tmp) {
      struct SIDL_BaseException_Trace* curr = data->d_trace_head;
      char *ptr = tmp;
      while (curr) {
        if (curr->line) {
          (void)strcpy(ptr, curr->line);
          ptr += strlen(curr->line);
          *(ptr++) = '\n';      /* overwrite '\0' with '\n' */
          curr = curr->next;
        }
      }
      *ptr = '\0';              /* nul terminate the string */
    }
  }
  return tmp;
  /* DO-NOT-DELETE splicer.end(SIDL.BaseException.getTrace) */
}

/*
 * Adds a stringified entry/line to the stack trace.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseException_addLine"

void
impl_SIDL_BaseException_addLine(
  SIDL_BaseException self, const char* traceline)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseException.addLine) */
  struct SIDL_BaseException__data *data = 
    (self ? SIDL_BaseException__get_data(self) : NULL);

  if (data)
  {
    struct SIDL_BaseException_Trace* tmp = (struct SIDL_BaseException_Trace*)
      malloc(sizeof(struct SIDL_BaseException_Trace));

    if (tmp)
    {
      unsigned long int linelen = (unsigned long int)strlen(traceline);
      if ((tmp->line = (char*)malloc(linelen+1)))
      {
        strcpy(tmp->line, traceline);
        tmp->next = NULL;

        if (data->d_trace_tail) {
          data->d_trace_tail->next = tmp; 
        }
        data->d_trace_tail = tmp; 
        if (data->d_trace_head == NULL) {
          data->d_trace_head = tmp; 
        }

        /* 
         * NOTE:  Add 1 more char to the length to allow for the addition
         * of the newline character (per line) when returning the trace
         * as a single string.
         */
        data->d_trace_length = data->d_trace_length + linelen + 1UL; 
      }
    }
  }
  /* DO-NOT-DELETE splicer.end(SIDL.BaseException.addLine) */
}

/*
 * Formats and adds an entry to the stack trace based on the 
 * file name, line number, and method name.
 */

#undef __FUNC__
#define __FUNC__ "impl_SIDL_BaseException_add"

void
impl_SIDL_BaseException_add(
  SIDL_BaseException self, const char* filename, int32_t lineno,
    const char* methodname)
{
  /* DO-NOT-DELETE splicer.begin(SIDL.BaseException.add) */
  /*
   *  The estimated length of the trace line is the sum of the lengths of
   *  the method name, file name, and hard-coded string contents plus a
   *  rough allowance for the line number.  Since we're using int for lineno,
   *  it is assumed the maximum int is 2^64, or 18446744073709551616 (i.e.,
   *  an allowance of 20 characters is made.)  Hence,
   *
   *    # bytes = filename + methodname + characters + lineno + 1
   *            = filename + methodname +      8     +   20   + 1
   *            = filename + methodname + 29
   *
   *  Of course, a more accurate approach would be to calculate the number 
   *  of digits in lineno prior to the malloc but, at first blush, it was
   *  assumed it wasn't worth the extra cycles given the purpose of the 
   *  stack trace.
   */
  const char* tmpfn;
  size_t filelen;
  const char*  tmpmn;
  size_t methlen;
  char*  tmpline;

  if (filename) {
    tmpfn = filename;
  } else {
    tmpfn = "UnspecifiedFile";
  } 
  if (methodname) {
    tmpmn = methodname;
  } else {
    tmpmn = "UnspecifiedMethod";
  } 

  filelen = strlen(tmpfn);
  methlen = strlen(tmpmn);
  tmpline = (char*) malloc(filelen + methlen + 29);

  if (tmpline) {
    sprintf(tmpline, "in %s at %s:%d", tmpmn, tmpfn, lineno), 
    SIDL_BaseException_addLine(self,tmpline); 
    free((void*)tmpline);
  }
  /* DO-NOT-DELETE splicer.end(SIDL.BaseException.add) */
}
