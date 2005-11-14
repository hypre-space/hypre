/*
 * File:          sidl_SIDLException_Impl.c
 * Symbol:        sidl.SIDLException-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.SIDLException
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
 * babel-version = 0.10.12
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.SIDLException" (version 0.9.3)
 * 
 * <code>SIDLException</code> provides the basic functionality of the
 * <code>BaseException</code> interface for getting and setting error
 * messages and stack traces.
 */

#include "sidl_SIDLException_Impl.h"

#line 53 "../../../babel/runtime/sidl/sidl_SIDLException_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidl.SIDLException._includes) */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct sidl_SIDLException_Trace {
  struct sidl_SIDLException_Trace  *next;
  char                             *line;
};
/* DO-NOT-DELETE splicer.end(sidl.SIDLException._includes) */
#line 64 "sidl_SIDLException_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_SIDLException__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_SIDLException__load(
  void)
{
#line 78 "../../../babel/runtime/sidl/sidl_SIDLException_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.SIDLException._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidl.SIDLException._load) */
#line 84 "sidl_SIDLException_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_SIDLException__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_SIDLException__ctor(
  /* in */ sidl_SIDLException self)
{
#line 96 "../../../babel/runtime/sidl/sidl_SIDLException_Impl.c"
   /* DO-NOT-DELETE splicer.begin(sidl.SIDLException._ctor) */
  struct sidl_SIDLException__data *data = 
    (struct sidl_SIDLException__data *)
    malloc(sizeof(struct sidl_SIDLException__data));

  if (data) 
  {
    data->d_message      = NULL;
    data->d_trace_head   = NULL;
    data->d_trace_tail   = NULL;
    data->d_trace_length = 0UL;
  }

  sidl_SIDLException__set_data(self, data);
   /* DO-NOT-DELETE splicer.end(sidl.SIDLException._ctor) */
#line 116 "sidl_SIDLException_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_SIDLException__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_SIDLException__dtor(
  /* in */ sidl_SIDLException self)
{
#line 127 "../../../babel/runtime/sidl/sidl_SIDLException_Impl.c"
   /* DO-NOT-DELETE splicer.begin(sidl.SIDLException._dtor) */
  struct sidl_SIDLException__data *data = 
    (self ? sidl_SIDLException__get_data(self) : NULL);

  if (data) 
  {
    if (data->d_message) {
      free((void *)(data->d_message));
      data->d_message = NULL;
    }

    if (data->d_trace_head) 
    {
      struct sidl_SIDLException_Trace* curr;
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
    sidl_SIDLException__set_data(self, NULL);
  }
   /* DO-NOT-DELETE splicer.end(sidl.SIDLException._dtor) */
#line 166 "sidl_SIDLException_Impl.c"
}

/*
 * Return the message associated with the exception.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_SIDLException_getNote"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_SIDLException_getNote(
  /* in */ sidl_SIDLException self)
{
#line 175 "../../../babel/runtime/sidl/sidl_SIDLException_Impl.c"
   /* DO-NOT-DELETE splicer.begin(sidl.SIDLException.getNote) */
  struct sidl_SIDLException__data *data = 
    (self ? sidl_SIDLException__get_data(self) : NULL);
  char *result = 
    ((data && data->d_message) 
     ? strcpy((char*)malloc(strlen(data->d_message)+1),
              data->d_message)
     : NULL);
  return result;
   /* DO-NOT-DELETE splicer.end(sidl.SIDLException.getNote) */
#line 194 "sidl_SIDLException_Impl.c"
}

/*
 * Set the message associated with the exception.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_SIDLException_setNote"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_SIDLException_setNote(
  /* in */ sidl_SIDLException self,
  /* in */ const char* message)
{
#line 202 "../../../babel/runtime/sidl/sidl_SIDLException_Impl.c"
   /* DO-NOT-DELETE splicer.begin(sidl.SIDLException.setNote) */
  struct sidl_SIDLException__data *data = 
    (self ? sidl_SIDLException__get_data(self) : NULL);
  if (data) {
    if (data->d_message) {
      free((void*)data->d_message);
    }
    data->d_message = 
      (message 
       ? strcpy((char *)malloc(strlen(message)+1), message)
       : NULL);
  }
   /* DO-NOT-DELETE splicer.end(sidl.SIDLException.setNote) */
#line 226 "sidl_SIDLException_Impl.c"
}

/*
 * Returns formatted string containing the concatenation of all 
 * tracelines.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_SIDLException_getTrace"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_SIDLException_getTrace(
  /* in */ sidl_SIDLException self)
{
#line 232 "../../../babel/runtime/sidl/sidl_SIDLException_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.SIDLException.getTrace) */
  char* tmp = NULL;
  struct sidl_SIDLException__data *data = 
    (self ? sidl_SIDLException__get_data(self) : NULL);

  if (data)
  {
    tmp = (char*)malloc(data->d_trace_length+1);
    if (tmp) {
      struct sidl_SIDLException_Trace* curr = data->d_trace_head;
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
  /* DO-NOT-DELETE splicer.end(sidl.SIDLException.getTrace) */
#line 269 "sidl_SIDLException_Impl.c"
}

/*
 * Adds a stringified entry/line to the stack trace.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_SIDLException_addLine"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_SIDLException_addLine(
  /* in */ sidl_SIDLException self,
  /* in */ const char* traceline)
{
#line 273 "../../../babel/runtime/sidl/sidl_SIDLException_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.SIDLException.addLine) */
  struct sidl_SIDLException__data *data = 
    (self ? sidl_SIDLException__get_data(self) : NULL);

  if (data)
  {
    struct sidl_SIDLException_Trace* tmp = (struct sidl_SIDLException_Trace*)
      malloc(sizeof(struct sidl_SIDLException_Trace));

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
  /* DO-NOT-DELETE splicer.end(sidl.SIDLException.addLine) */
#line 323 "sidl_SIDLException_Impl.c"
}

/*
 * Formats and adds an entry to the stack trace based on the 
 * file name, line number, and method name.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_SIDLException_add"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_SIDLException_add(
  /* in */ sidl_SIDLException self,
  /* in */ const char* filename,
  /* in */ int32_t lineno,
  /* in */ const char* methodname)
{
#line 328 "../../../babel/runtime/sidl/sidl_SIDLException_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.SIDLException.add) */
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
    sidl_SIDLException_addLine(self,tmpline); 
    free((void*)tmpline);
  }
  /* DO-NOT-DELETE splicer.end(sidl.SIDLException.add) */
#line 389 "sidl_SIDLException_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_SIDLException__object* 
  impl_sidl_SIDLException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_SIDLException__connect(url, _ex);
}
char * impl_sidl_SIDLException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) {
  return sidl_SIDLException__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidl_SIDLException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidl_SIDLException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidl_SIDLException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseException__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseException__connect(url, _ex);
}
char * impl_sidl_SIDLException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj) {
  return sidl_BaseException__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidl_SIDLException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
