/*
 * File:          sidl_SIDLException_Impl.c
 * Symbol:        sidl.SIDLException-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_SIDLException_Impl.c,v 1.7 2006/08/29 22:29:50 painter Exp $
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
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.SIDLException" (version 0.9.15)
 * 
 * <code>SIDLException</code> provides the basic functionality of the
 * <code>BaseException</code> interface for getting and setting error
 * messages and stack traces.
 */

#include "sidl_SIDLException_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidl.SIDLException._includes) */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "sidl_Exception.h"

struct sidl_SIDLException_Trace {
  struct sidl_SIDLException_Trace  *next;
  char                             *line;
};
/* DO-NOT-DELETE splicer.end(sidl.SIDLException._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
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
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.SIDLException._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidl.SIDLException._load) */
  }
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
  /* in */ sidl_SIDLException self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
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
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_SIDLException__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_SIDLException__ctor2(
  /* in */ sidl_SIDLException self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.SIDLException._ctor2) */
  /* Insert-Code-Here {sidl.SIDLException._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidl.SIDLException._ctor2) */
  }
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
  /* in */ sidl_SIDLException self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
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
  }
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
  /* in */ sidl_SIDLException self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
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
  }
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
  /* in */ const char* message,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
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
  }
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
  /* in */ sidl_SIDLException self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
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
  }
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
  /* in */ const char* traceline,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
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
  }
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
  /* in */ const char* methodname,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
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
    sidl_SIDLException_addLine(self,tmpline, _ex); 
    free((void*)tmpline);
  }
  /* DO-NOT-DELETE splicer.end(sidl.SIDLException.add) */
  }
}

/*
 * Method:  packObj[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_SIDLException_packObj"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_SIDLException_packObj(
  /* in */ sidl_SIDLException self,
  /* in */ sidl_io_Serializer ser,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.SIDLException.packObj) */
  struct sidl_SIDLException__data *data = NULL;
  int32_t i = 0;
  struct sidl_SIDLException_Trace *cur = NULL;

  data  = sidl_SIDLException__get_data(self);
  if(data) {
    sidl_io_Serializer_packString(ser, "d_message", data->d_message, _ex);SIDL_CHECK(*_ex); 
    cur = data->d_trace_head;
    while(cur) {
      ++i;
      cur = cur->next;
    }
    sidl_io_Serializer_packInt(ser, "traceSize", i, _ex);SIDL_CHECK(*_ex);   /*Serialize length of the trace*/
    cur = data->d_trace_head;
    while(cur) {
      sidl_io_Serializer_packString(ser, "traceLine", cur->line, _ex);SIDL_CHECK(*_ex);  /*Serialize each line*/
      cur = cur->next;
    }
  } else {
    sidl_io_Serializer_packString(ser, "d_message", NULL, _ex); SIDL_CHECK(*_ex); 
    sidl_io_Serializer_packInt(ser, "traceSize", 0, _ex); SIDL_CHECK(*_ex); 

  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidl.SIDLException.packObj) */
  }
}

/*
 * Method:  unpackObj[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_SIDLException_unpackObj"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_SIDLException_unpackObj(
  /* in */ sidl_SIDLException self,
  /* in */ sidl_io_Deserializer des,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.SIDLException.unpackObj) */
  struct sidl_SIDLException__data *data = NULL;
  int32_t size = 0;
  int32_t i;
  char* tmp_line = NULL;

  data  = sidl_SIDLException__get_data(self);
  if(!data) {
    data = (struct sidl_SIDLException__data *)
      malloc(sizeof (struct sidl_SIDLException__data));
    sidl_SIDLException__set_data(self,data);
  }
  sidl_io_Deserializer_unpackString(des, "d_message", &(data->d_message), _ex);SIDL_CHECK(*_ex); 
  sidl_io_Deserializer_unpackInt(des, "traceSize", &size, _ex); SIDL_CHECK(*_ex);
  for(i=0;i<size;++i) {
    sidl_io_Deserializer_unpackString(des, "d_message", &tmp_line, _ex);SIDL_CHECK(*_ex);
    impl_sidl_SIDLException_addLine(self, tmp_line, _ex);SIDL_CHECK(*_ex);
  }
  
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidl.SIDLException.unpackObj) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_sidl_SIDLException_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseException__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseException__connectI(url, ar, _ex);
}
struct sidl_BaseException__object* 
  impl_sidl_SIDLException_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseException__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidl_SIDLException_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidl_SIDLException_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidl_SIDLException_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidl_SIDLException_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidl_SIDLException_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidl_SIDLException__object* 
  impl_sidl_SIDLException_fconnect_sidl_SIDLException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_SIDLException__connectI(url, ar, _ex);
}
struct sidl_SIDLException__object* 
  impl_sidl_SIDLException_fcast_sidl_SIDLException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_SIDLException__cast(bi, _ex);
}
struct sidl_io_Deserializer__object* 
  impl_sidl_SIDLException_fconnect_sidl_io_Deserializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Deserializer__connectI(url, ar, _ex);
}
struct sidl_io_Deserializer__object* 
  impl_sidl_SIDLException_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_io_Deserializer__cast(bi, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidl_SIDLException_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Serializable__connectI(url, ar, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidl_SIDLException_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_io_Serializable__cast(bi, _ex);
}
struct sidl_io_Serializer__object* 
  impl_sidl_SIDLException_fconnect_sidl_io_Serializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Serializer__connectI(url, ar, _ex);
}
struct sidl_io_Serializer__object* 
  impl_sidl_SIDLException_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_io_Serializer__cast(bi, _ex);
}
