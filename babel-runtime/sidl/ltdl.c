/* ltdl.c -- system independent dlopen wrapper
   Copyright (C) 1998, 1999, 2000 Free Software Foundation, Inc.
   Originally by Thomas Tanner <tanner@ffii.org>
   This file is part of GNU Libtool.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

As a special exception to the GNU Lesser General Public License,
if you distribute this file as part of a program or library that
is built using GNU libtool, you may include it under the same
distribution terms that you use for the rest of that program.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
02111-1307  USA
*/

#include "babel_config.h"

#if HAVE_STDIO_H
#  include <stdio.h>
#endif

#if HAVE_STDLIB_H
#  include <stdlib.h>
#endif

#if HAVE_STRING_H
#  include <string.h>
#else
#  if HAVE_STRINGS_H
#    include <strings.h>
#  endif
#endif

#if HAVE_CTYPE_H
#  include <ctype.h>
#endif

#if HAVE_MALLOC_H
#  include <malloc.h>
#endif

#if HAVE_MEMORY_H
#  include <memory.h>
#endif

#include "ltdl.h"

#ifdef __CYGWIN__
#include <sys/cygwin.h>
#endif



/* --- WINDOWS SUPPORT --- */


#ifdef DLL_EXPORT
#  define LT_GLOBAL_DATA	__declspec(dllexport)
#else
#  define LT_GLOBAL_DATA
#endif

/* fopen() mode flags for reading a text file */
#undef	LT_READTEXT_MODE
#ifdef __WINDOWS__
#  define LT_READTEXT_MODE "rt"
#else
#  define LT_READTEXT_MODE "r"
#endif




/* --- MANIFEST CONSTANTS --- */


/* max. filename length */
#ifndef LT_FILENAME_MAX
#  define LT_FILENAME_MAX	1024
#endif

/* This is the maximum symbol size that won't require malloc/free */
#undef	LT_SYMBOL_LENGTH
#define LT_SYMBOL_LENGTH	128

/* This accounts for the _LTX_ separator */
#undef	LT_SYMBOL_OVERHEAD
#define LT_SYMBOL_OVERHEAD	5




/* --- TYPE DEFINITIONS -- */


/* This type is used for the array of caller data sets in each handler. */
typedef struct {
  lt_dlcaller_id	key;
  lt_ptr		data;
} lt_caller_data;




/* --- OPAQUE STRUCTURES DECLARED IN LTDL.H --- */


/* Extract the diagnostic strings from the error table macro in the same
   order as the enumberated indices in ltdl.h. */

static const char *lt_dlerror_strings[] =
  {
#define LT_ERROR(name, diagnostic)	(diagnostic),
    lt_dlerror_table
#undef LT_ERROR

    0
  };

/* This structure is used for the list of registered loaders. */
struct lt_dlloader {
  struct lt_dlloader   *next;
  const char	       *loader_name;	/* identifying name for each loader */
  const char	       *sym_prefix;	/* prefix for symbols */
  lt_module_open       *module_open;
  lt_module_close      *module_close;
  lt_find_sym	       *find_sym;
  lt_dlloader_exit     *dlloader_exit;
  lt_user_data		dlloader_data;
};

struct lt_dlhandle_struct {
  struct lt_dlhandle_struct   *next;
  lt_dlloader	       *loader;		/* dlopening interface */
  lt_dlinfo		info;
  int			depcount;	/* number of dependencies */
  lt_dlhandle	       *deplibs;	/* dependencies */
  lt_module		module;		/* system module handle */
  lt_ptr		system;		/* system specific data */
  lt_caller_data       *caller_data;	/* per caller associated data */
  int			flags;		/* various boolean stats */
};

/* Various boolean flags can be stored in the flags field of an
   lt_dlhandle_struct... */
#define LT_DLGET_FLAG(handle, flag) (((handle)->flags & (flag)) == (flag))
#define LT_DLSET_FLAG(handle, flag) ((handle)->flags |= (flag))

#define LT_DLRESIDENT_FLAG	    (0x01 << 0)
/* ...add more flags here... */

#define LT_DLIS_RESIDENT(handle)    LT_DLGET_FLAG(handle, LT_DLRESIDENT_FLAG)


#define LT_DLSTRERROR(name)	lt_dlerror_strings[LT_CONC(LT_ERROR_,name)]

static	const char	objdir[]		= LTDL_OBJDIR;
#ifdef	LTDL_SHLIB_EXT
static	const char	shlib_ext[]		= LTDL_SHLIB_EXT;
#endif
#ifdef	LTDL_SYSSEARCHPATH
static	const char	sys_search_path[]	= LTDL_SYSSEARCHPATH;
#endif




/* --- MUTEX LOCKING --- */


/* Macros to make it easier to run the lock functions only if they have 
   been registered.  The reason for the complicated lock macro is to
   ensure that the stored error message from the last error is not 
   accidentally erased if the current function doesn't generate an
   error of its own.  */
#define MUTEX_LOCK()				LT_STMT_START {	\
	if (mutex_lock) (*mutex_lock)();	} LT_STMT_END
#define MUTEX_UNLOCK()				LT_STMT_START { \
	if (mutex_unlock) (*mutex_unlock)();	} LT_STMT_END
#define MUTEX_SETERROR(errormsg)		LT_STMT_START {	\
	if (mutex_seterror) (*mutex_seterror) (errormsg);	\
	else last_error = (errormsg);		} LT_STMT_END
#define MUTEX_GETERROR(errormsg)		LT_STMT_START {	\
	if (mutex_seterror) errormsg = (*mutex_geterror)();	\
	else (errormsg) = last_error;		} LT_STMT_END

/* The mutex functions stored here are global, and are necessarily the
   same for all threads that wish to share access to libltdl.  */
static	lt_dlmutex_lock	    *mutex_lock	    = 0;
static	lt_dlmutex_unlock   *mutex_unlock   = 0;
static	lt_dlmutex_seterror *mutex_seterror = 0;
static	lt_dlmutex_geterror *mutex_geterror = 0;
static	const char	    *last_error	    = 0;


/* Either set or reset the mutex functions.  Either all the arguments must
   be valid functions, or else all can be NULL to turn off locking entirely.
   The registered functions should be manipulating a static global lock
   from the lock() and unlock() callbacks, which needs to be reentrant.  */
int
lt_dlmutex_register (lock, unlock, seterror, geterror)
     lt_dlmutex_lock *lock;
     lt_dlmutex_unlock *unlock;
     lt_dlmutex_seterror *seterror;
     lt_dlmutex_geterror *geterror;
{
  lt_dlmutex_unlock *old_unlock = unlock;
  int		     errors	= 0;

  /* Lock using the old lock() callback, if any.  */
  MUTEX_LOCK ();

  if ((lock && unlock && seterror && geterror) 
      || !(lock || unlock || seterror || geterror))
    {
      mutex_lock     = lock;
      mutex_unlock   = unlock;
      mutex_geterror = geterror;
    }
  else
    {
      MUTEX_SETERROR (LT_DLSTRERROR (INVALID_MUTEX_ARGS));
      ++errors;
    }

  /* Use the old unlock() callback we saved earlier, if any.  Otherwise
     record any errors using internal storage.  */
  if (old_unlock)
    (*old_unlock) ();

  /* Return the number of errors encountered during the execution of
     this function.  */
  return errors;
}




/* --- MEMORY HANDLING --- */


LT_GLOBAL_DATA    lt_ptr	(*lt_dlmalloc)	LT_PARAMS((size_t size))
 				    = (lt_ptr (*) LT_PARAMS((size_t))) malloc;
LT_GLOBAL_DATA    void		(*lt_dlfree)	LT_PARAMS((lt_ptr ptr))
 				    = (void (*) LT_PARAMS((lt_ptr))) free;

static		  lt_ptr	rpl_realloc	LT_PARAMS((lt_ptr ptr,
							   size_t size));

#define LT_DLMALLOC(tp, n)	((tp *) lt_dlmalloc ((n) * sizeof(tp)))
#define LT_DLREALLOC(tp, p, n)	((tp *) rpl_realloc ((p), (n) * sizeof(tp)))
#define LT_DLFREE(p)						\
	LT_STMT_START { if (p) (p) = (lt_dlfree (p), (lt_ptr) 0); } LT_STMT_END

#define LT_DLMEM_REASSIGN(p, q)			LT_STMT_START {	\
	if ((p) != (q)) { lt_dlfree (p); (p) = (q); }		\
						} LT_STMT_END



/* --- ERROR MESSAGES --- */


static	const char    **user_error_strings	= 0;
static	int		errorcount		= LT_ERROR_MAX;

int
lt_dladderror (diagnostic)
     const char *diagnostic;
{
  int		index	 = 0;
  int		result	 = -1;
  const char  **temp     = (const char **) 0;

  MUTEX_LOCK ();

  index	 = errorcount - LT_ERROR_MAX;
  temp = LT_DLREALLOC (const char *, user_error_strings, 1 + index);
  if (temp == 0)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
    }
  else
    {
      user_error_strings	= temp;
      user_error_strings[index] = diagnostic;
      result			= errorcount++;
    }

  MUTEX_UNLOCK ();

  return result;
}

int
lt_dlseterror (index)
     int index;
{
  int		errors	 = 0;

  MUTEX_LOCK ();

  if (index >= errorcount || index < 0)
    {
      /* Ack!  Error setting the error message! */
      MUTEX_SETERROR (LT_DLSTRERROR (INVALID_ERRORCODE));
      ++errors;
    }
  else if (index < LT_ERROR_MAX)
    {
      /* No error setting the error message! */
      MUTEX_SETERROR (lt_dlerror_strings[errorcount]);
    }
  else
    {
      /* No error setting the error message! */
      MUTEX_SETERROR (user_error_strings[errorcount - LT_ERROR_MAX]);
    }

  MUTEX_UNLOCK ();

  return errors;
}




/* --- REPLACEMENT FUNCTIONS --- */


#undef strdup
#define strdup rpl_strdup

static char *
strdup(const char *str)
{
  char *tmp = 0;

  if (str)
    {
      tmp = LT_DLMALLOC (char, 1+ strlen (str));
      if (tmp)
	{
	  strcpy(tmp, str);
	}
    }

  return tmp;
}


#if ! HAVE_STRCMP

#undef strcmp
#define strcmp rpl_strcmp

static int
strcmp (str1, str2)
     const char *str1;
     const char *str2;
{
  if (str1 == str2)
    return 0;
  if (str1 == 0)
    return -1;
  if (str2 == 0)
    return 1;

  for (;*str1 && *str2; ++str1, ++str2)
    {
      if (*str1 != *str2)
	break;
    }

  return (int)(*str1 - *str2);
}
#endif


#if ! HAVE_STRCHR

#  if HAVE_INDEX
#    define strchr index
#  else
#    define strchr rpl_strchr

static const char*
strchr(str, ch)
     const char *str;
     int ch;
{
  const char *p;

  for (p = str; *p != (char)ch && *p != '\0'; ++p)
    /*NOWORK*/;

  return (*p == (char)ch) ? p : 0;
}

#  endif
#endif /* !HAVE_STRCHR */

#if ! HAVE_STRRCHR

#  if HAVE_RINDEX
#    define strrchr rindex
#  else
#    define strrchr rpl_strrchr

static const char*
strrchr(str, ch)
     const char *str;
     int ch;
{
  const char *p, *q = 0;

  for (p = str; *p != '\0'; ++p)
    {
      if (*p == (char) ch)
	{
	  q = p;
	}
    }

  return q;
}

# endif
#endif

/* NOTE:  Neither bcopy nor the memcpy implementation below can
          reliably handle copying in overlapping areas of memory, so
          do not rely on this behaviour when invoking memcpy later.  */
#if ! HAVE_MEMCPY

#  if HAVE_BCOPY
#    define memcpy(dest, src, size)	bcopy (src, dest, size)
#  else
#    define memcpy rpl_memcpy

static char *
memcpy (dest, src, size)
     char *dest;
     const char *src;
     size_t size;
{
  size_t i = 0;

  for (i = 0; i < size; ++i)
    {
      dest[i] = src[i];
    }

  return dest;
}

#  endif
#endif

/* According to Alexandre Oliva <oliva@lsd.ic.unicamp.br>,
    ``realloc is not entirely portable''
   In any case we want to use the allocator supplied by the user without
   burdening them with an lt_dlrealloc function pointer to maintain.
   Instead implement our own version (with known boundary conditions)
   using lt_dlmalloc and lt_dlfree. */
static lt_ptr
rpl_realloc (ptr, size)
     lt_ptr ptr;
     size_t size;
{
  if (size < 1)
    {
      /* For zero or less bytes, free the original memory */
      if (ptr != 0)
	{
	  lt_dlfree (ptr);
	}

      return (lt_ptr) 0;
    }
  else if (ptr == 0)
    {
      /* Allow reallocation of a NULL pointer.  */
      return lt_dlmalloc (size);
    }
  else
    {
      /* Allocate a new block, copy and free the old block.  */
      lt_ptr mem = lt_dlmalloc (size);

      if (mem)
	{
	  memcpy (mem, ptr, size);
	  lt_dlfree (ptr);
	}

      /* Note that the contents of PTR are not damaged if there is
	 insufficient memory to realloc.  */
      return mem;
    }
}




/* --- DLOPEN() INTERFACE LOADER --- */


/* The Cygwin dlopen implementation prints a spurious error message to
   stderr if its call to LoadLibrary() fails for any reason.  We can
   mitigate this by not using the Cygwin implementation, and falling
   back to our own LoadLibrary() wrapper. */
#if HAVE_LIBDL && !defined(__CYGWIN__)

/* dynamic linking with dlopen/dlsym */

#if HAVE_DLFCN_H
#  include <dlfcn.h>
#endif

#if HAVE_SYS_DL_H
#  include <sys/dl.h>
#endif

#ifdef RTLD_GLOBAL
#  define LT_GLOBAL		RTLD_GLOBAL
#else
#  ifdef DL_GLOBAL
#    define LT_GLOBAL		DL_GLOBAL
#  endif
#endif /* !RTLD_GLOBAL */
#ifndef LT_GLOBAL
#  define LT_GLOBAL		0
#endif /* !LT_GLOBAL */

/* We may have to define LT_LAZY_OR_NOW in the command line if we
   find out it does not work in some platform. */
#ifndef LT_LAZY_OR_NOW
#  ifdef RTLD_LAZY
#    define LT_LAZY_OR_NOW	RTLD_LAZY
#  else
#    ifdef DL_LAZY
#      define LT_LAZY_OR_NOW	DL_LAZY
#    endif
#  endif /* !RTLD_LAZY */
#endif
#ifndef LT_LAZY_OR_NOW
#  ifdef RTLD_NOW
#    define LT_LAZY_OR_NOW	RTLD_NOW
#  else
#    ifdef DL_NOW
#      define LT_LAZY_OR_NOW	DL_NOW
#    endif
#  endif /* !RTLD_NOW */
#endif
#ifndef LT_LAZY_OR_NOW
#  define LT_LAZY_OR_NOW	0
#endif /* !LT_LAZY_OR_NOW */

#if HAVE_DLERROR
#  define DLERROR(arg)	dlerror ()
#else
#  define DLERROR(arg)	LT_DLSTRERROR (arg)
#endif

static lt_module
sys_dl_open(lt_user_data loader_data, const char *filename)
{
#if 0
  lt_module   module   = dlopen (filename, LT_GLOBAL | LT_LAZY_OR_NOW);
#else
  /*
   * HACK - change the default open mode to local.
   */
#ifdef RTLD_LOCAL
  lt_module   module   = dlopen (filename, RTLD_LOCAL | LT_LAZY_OR_NOW);
#else
  lt_module   module   = dlopen (filename, LT_LAZY_OR_NOW);
#endif
#endif

  if (!module)
    {
      MUTEX_SETERROR (DLERROR (CANNOT_OPEN));
    }

  return module;
}

static int
sys_dl_close(lt_user_data loader_data, lt_module module)
{
  int errors = 0;

  if (dlclose (module) != 0)
    {
      MUTEX_SETERROR (DLERROR (CANNOT_CLOSE));
      ++errors;
    }

  return errors;
}

static lt_ptr
sys_dl_sym (lt_user_data loader_data, lt_module module, const char *symbol)
{
  lt_ptr address = dlsym (module, symbol);

  if (!address)
    {
      MUTEX_SETERROR (DLERROR (SYMBOL_NOT_FOUND));
    }

  return address;
}

static struct lt_user_dlloader sys_dl =
  {
#  ifdef NEED_USCORE
    "_",
#  else
    0,
#  endif
    sys_dl_open, sys_dl_close, sys_dl_sym, 0, 0 };


#endif /* HAVE_LIBDL */



/* --- SHL_LOAD() INTERFACE LOADER --- */

#if HAVE_SHL_LOAD

/* dynamic linking with shl_load (HP-UX) (comments from gmodule) */

#ifdef HAVE_DL_H
#  include <dl.h>
#endif

/* some flags are missing on some systems, so we provide
 * harmless defaults.
 *
 * Mandatory:
 * BIND_IMMEDIATE  - Resolve symbol references when the library is loaded.
 * BIND_DEFERRED   - Delay code symbol resolution until actual reference.
 *
 * Optionally:
 * BIND_FIRST	   - Place the library at the head of the symbol search
 * 		     order.
 * BIND_NONFATAL   - The default BIND_IMMEDIATE behavior is to treat all
 * 		     unsatisfied symbols as fatal.  This flag allows
 * 		     binding of unsatisfied code symbols to be deferred
 * 		     until use.
 *		     [Perl: For certain libraries, like DCE, deferred
 *		     binding often causes run time problems. Adding
 *		     BIND_NONFATAL to BIND_IMMEDIATE still allows
 *		     unresolved references in situations like this.]
 * BIND_NOSTART	   - Do not call the initializer for the shared library
 *		     when the library is loaded, nor on a future call to
 *		     shl_unload().
 * BIND_VERBOSE	   - Print verbose messages concerning possible
 *		     unsatisfied symbols.
 *
 * hp9000s700/hp9000s800:
 * BIND_RESTRICTED - Restrict symbols visible by the library to those
 *		     present at library load time.
 * DYNAMIC_PATH	   - Allow the loader to dynamically search for the
 *		     library specified by the path argument.
 */

#ifndef	DYNAMIC_PATH
#  define DYNAMIC_PATH		0
#endif
#ifndef	BIND_RESTRICTED
#  define BIND_RESTRICTED	0
#endif

#define	LT_BIND_FLAGS	(BIND_IMMEDIATE | BIND_NONFATAL | DYNAMIC_PATH)

static lt_module
sys_shl_open (loader_data, filename)
     lt_user_data loader_data;
     const char *filename;
{
  static shl_t self = (shl_t) 0;
  lt_module module = shl_load (filename, LT_BIND_FLAGS, 0L);
  
  /* Since searching for a symbol against a NULL module handle will also
     look in everything else that was already loaded and exported with 
     the -E compiler flag, we always cache a handle saved before any
     modules are loaded.  */
  if (!self)
    {
      lt_ptr address;
      shl_findsym (&self, "main", TYPE_UNDEFINED, &address);
    }
  
  if (!filename)
    {
      module = self;
    }
  else
    {
      module = shl_load (filename, LT_BIND_FLAGS, 0L);

      if (!module)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (CANNOT_OPEN));
	}
    }
  
  return module;
}

static int
sys_shl_close (loader_data, module)
     lt_user_data loader_data;
     lt_module module;
{
  int errors = 0;

  if (module && (shl_unload ((shl_t) (module)) != 0))
    {
      MUTEX_SETERROR (LT_DLSTRERROR (CANNOT_CLOSE));
      ++errors;
    }

  return errors;
}

static lt_ptr
sys_shl_sym (loader_data, module, symbol)
     lt_user_data loader_data;
     lt_module module;
     const char *symbol;
{
  lt_ptr address = 0;

  /* sys_shl_open should never return a NULL module handle */
  if (module == (lt_module) 0)
  {
    MUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
  }
  else if (!shl_findsym((shl_t*) &module, symbol, TYPE_UNDEFINED, &address))
    {
      if (!address)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));
	}
    }
  
  return address;
}

static struct lt_user_dlloader sys_shl = {
  0, sys_shl_open, sys_shl_close, sys_shl_sym, 0, 0
};

#endif /* HAVE_SHL_LOAD */




/* --- LOADLIBRARY() INTERFACE LOADER --- */

#ifdef __WINDOWS__

/* dynamic linking for Win32 */

#include <windows.h>

/* Forward declaration; required to implement handle search below. */
static lt_dlhandle handles;

static lt_module
sys_wll_open(lt_user_data loader_data, const char *filename)
{
#if 0
  lt_dlhandle	cur;
#endif
  lt_module	module	   = 0;
  const char   *errormsg   = 0;
  char	       *searchname = 0;
  char	       *ext;
  char		self_name_buf[MAX_PATH];

  (void) errormsg;

  if (!filename)
    {
      /* Get the name of main module */
      *self_name_buf = 0;
      GetModuleFileName (NULL, self_name_buf, sizeof (self_name_buf));
      filename = ext = self_name_buf;
    }
  else
    {
      ext = strrchr (filename, '.');
    }

  if (ext)
    {
      /* FILENAME already has an extension. */
      searchname = strdup (filename);
    }
  else
    {
      /* Append a `.' to stop Windows from adding an
	 implicit `.dll' extension. */
      searchname = LT_DLMALLOC (char, 2+ strlen (filename));
      if (!searchname)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  return 0;
	}
      strcpy (searchname, filename);
      strcat (searchname, ".");
    }

#if __CYGWIN__
  {
    char wpath[MAX_PATH];
    cygwin_conv_to_full_win32_path(searchname, wpath);
    module = LoadLibrary(wpath);
  }
#else
  module = LoadLibrary (searchname);
#endif
  LT_DLFREE (searchname);

#if 0
  /* libltdl expects this function to fail if it is unable
     to physically load the library.  Sadly, LoadLibrary
     will search the loaded libraries for a match and return
     one of them if the path search load fails.

     We check whether LoadLibrary is returning a handle to
     an already loaded module, and simulate failure if we
     find one. */
  MUTEX_LOCK ();
  cur = handles;
  while (cur)
    {
      if (!cur->module)
	{
	  cur = 0;
	  break;
	}

      if (cur->module == module)
	{
	  break;
	}

      cur = cur->next;
  }
  MUTEX_UNLOCK ();

  if (cur || !module)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (CANNOT_OPEN));
      module = 0;
    }
#endif

  return module;
}

static int
sys_wll_close(lt_user_data loader_data, lt_module module)
{
  int	      errors   = 0;

  if (FreeLibrary(module) == 0)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (CANNOT_CLOSE));
      ++errors;
    }

  return errors;
}

static lt_ptr
sys_wll_sym(lt_user_data loader_data, lt_module module, const char *symbol)
{
  lt_ptr      address  = GetProcAddress (module, symbol);

  if (!address)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));
    }

  return address;
}

static struct lt_user_dlloader sys_wll = {
  0, sys_wll_open, sys_wll_close, sys_wll_sym, 0, 0
};

#endif /* __WINDOWS__ */




/* --- LOAD_ADD_ON() INTERFACE LOADER --- */


#ifdef __BEOS__

/* dynamic linking for BeOS */

#include <kernel/image.h>

static lt_module
sys_bedl_open (loader_data, filename)
     lt_user_data loader_data;
     const char *filename;
{
  image_id image = 0;

  if (filename)
    {
      image = load_add_on (filename);
    }
  else
    {
      image_info info;
      int32 cookie = 0;
      if (get_next_image_info (0, &cookie, &info) == B_OK)
	image = load_add_on (info.name);
    }

  if (image <= 0)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (CANNOT_OPEN));
      image = 0;
    }

  return (lt_module) image;
}

static int
sys_bedl_close (loader_data, module)
     lt_user_data loader_data;
     lt_module module;
{
  int errors = 0;

  if (unload_add_on ((image_id) module) != B_OK)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (CANNOT_CLOSE));
      ++errors;
    }

  return errors;
}

static lt_ptr
sys_bedl_sym (loader_data, module, symbol)
     lt_user_data loader_data;
     lt_module module;
     const char *symbol;
{
  lt_ptr address = 0;
  image_id image = (image_id) module;

  if (get_image_symbol (image, symbol, B_SYMBOL_TYPE_ANY, address) != B_OK)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));
      address = 0;
    }

  return address;
}

static struct lt_user_dlloader sys_bedl = {
  0, sys_bedl_open, sys_bedl_close, sys_bedl_sym, 0, 0
};

#endif /* __BEOS__ */




/* --- DLD_LINK() INTERFACE LOADER --- */


#if HAVE_DLD

/* dynamic linking with dld */

#if HAVE_DLD_H
#include <dld.h>
#endif

static lt_module
sys_dld_open (loader_data, filename)
     lt_user_data loader_data;
     const char *filename;
{
  lt_module module = strdup (filename);

  if (!module)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
      module = 0;
    }
  else if (dld_link (filename) != 0)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (CANNOT_OPEN));
      LT_DLFREE (module);
      module = 0;
    }

  return module;
}

static int
sys_dld_close (loader_data, module)
     lt_user_data loader_data;
     lt_module module;
{
  int errors = 0;

  if (dld_unlink_by_file ((char*)(module), 1) != 0)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (CANNOT_CLOSE));
      ++errors;
    }
  else
    {
      LT_DLFREE (module);
    }

  return errors;
}

static lt_ptr
sys_dld_sym (loader_data, module, symbol)
     lt_user_data loader_data;
     lt_module module;
     const char *symbol;
{
  lt_ptr address = dld_get_func (symbol);

  if (!address)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));
    }

  return address;
}

static struct lt_user_dlloader sys_dld = {
  0, sys_dld_open, sys_dld_close, sys_dld_sym, 0, 0
};

#endif /* HAVE_DLD */




/* --- DLPREOPEN() INTERFACE LOADER --- */


/* emulate dynamic linking using preloaded_symbols */

typedef struct lt_dlsymlists_t
{
  struct lt_dlsymlists_t       *next;
  const lt_dlsymlist	       *syms;
} lt_dlsymlists_t;

static	const lt_dlsymlist     *default_preloaded_symbols	= 0;
static	lt_dlsymlists_t	       *preloaded_symbols		= 0;

static int
presym_init(lt_user_data loader_data)
{
  int errors = 0;

  MUTEX_LOCK ();

  preloaded_symbols = 0;
  if (default_preloaded_symbols)
    {
      errors = lt_dlpreload (default_preloaded_symbols);
    }

  MUTEX_UNLOCK ();

  return errors;
}

static int
presym_free_symlists(void)
{
  lt_dlsymlists_t *lists;

  MUTEX_LOCK ();

  lists = preloaded_symbols;
  while (lists)
    {
      lt_dlsymlists_t	*tmp = lists;

      lists = lists->next;
      LT_DLFREE (tmp);
    }
  preloaded_symbols = 0;

  MUTEX_UNLOCK ();

  return 0;
}

static int
presym_exit(lt_user_data loader_data)
{
  presym_free_symlists ();
  return 0;
}

static int
presym_add_symlist(const lt_dlsymlist *preloaded)
{
  lt_dlsymlists_t *tmp;
  lt_dlsymlists_t *lists;
  int		   errors   = 0;

  MUTEX_LOCK ();

  lists = preloaded_symbols;
  while (lists)
    {
      if (lists->syms == preloaded)
	{
	  goto done;
	}
      lists = lists->next;
    }

  tmp = LT_DLMALLOC (lt_dlsymlists_t, 1);
  if (tmp)
    {
      tmp->syms = preloaded;
      tmp->next = preloaded_symbols;
      preloaded_symbols = tmp;
    }
  else
    {
      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
      ++errors;
    }

 done:
  MUTEX_UNLOCK ();
  return errors;
}

static lt_module
presym_open(lt_user_data loader_data, const char *filename)
{
  lt_dlsymlists_t *lists;
  lt_module	   module = (lt_module) 0;

  MUTEX_LOCK ();
  lists = preloaded_symbols;

  if (!lists)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (NO_SYMBOLS));
      goto done;
    }

  if (!filename)
    {
      filename = "@PROGRAM@";
    }

  while (lists)
    {
      const lt_dlsymlist *syms = lists->syms;

      while (syms->name)
	{
	  if (!syms->address && strcmp(syms->name, filename) == 0)
	    {
	      module = (lt_module) syms;
	      goto done;
	    }
	  ++syms;
	}

      lists = lists->next;
    }

  MUTEX_SETERROR (LT_DLSTRERROR (FILE_NOT_FOUND));

 done:
  MUTEX_UNLOCK ();
  return module;
}

static int
presym_close(lt_user_data loader_data, lt_module module)
{
  /* Just to silence gcc -Wall */
  module = 0;
  return 0;
}

static lt_ptr
presym_sym(lt_user_data loader_data, lt_module module, const char *symbol)
{
  lt_dlsymlist *syms = (lt_dlsymlist*) module;

  ++syms;
  while (syms->address)
    {
      if (strcmp(syms->name, symbol) == 0)
	{
	  return syms->address;
	}

    ++syms;
  }

  MUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));

  return 0;
}

static struct lt_user_dlloader presym = {
  0, presym_open, presym_close, presym_sym, presym_exit, 0
};





/* --- DYNAMIC MODULE LOADING --- */


static	char	       *user_search_path= 0;
static	lt_dlloader    *loaders		= 0;
static	lt_dlhandle	handles 	= 0;
static	int		initialized 	= 0;

/* Initialize libltdl. */
int
lt_dlinit ()
{
  int	      errors   = 0;

  MUTEX_LOCK ();

  /* Initialize only at first call. */
  if (++initialized == 1)
    {
      handles = 0;
      user_search_path = 0; /* empty search path */

#if HAVE_LIBDL && !defined(__CYGWIN__)
      errors += lt_dlloader_add (lt_dlloader_next (0), &sys_dl, "dlopen");
#endif
#if HAVE_SHL_LOAD
      errors += lt_dlloader_add (lt_dlloader_next (0), &sys_shl, "dlopen");
#endif
#ifdef __WINDOWS__
      errors += lt_dlloader_add (lt_dlloader_next (0), &sys_wll, "dlopen");
#endif
#ifdef __BEOS__
      errors += lt_dlloader_add (lt_dlloader_next (0), &sys_bedl, "dlopen");
#endif
#if HAVE_DLD
      errors += lt_dlloader_add (lt_dlloader_next (0), &sys_dld, "dld");
#endif
      errors += lt_dlloader_add (lt_dlloader_next (0), &presym, "dlpreload");

      if (presym_init (presym.dlloader_data))
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (INIT_LOADER));
	  ++errors;
	}
      else if (errors != 0)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (DLOPEN_NOT_SUPPORTED));
	  ++errors;
	}
    }

  MUTEX_UNLOCK ();

  return errors;
}

int
lt_dlpreload (preloaded)
     const lt_dlsymlist *preloaded;
{
  int errors = 0;

  if (preloaded)
    {
      errors = presym_add_symlist (preloaded);
    }
  else
    {
      const char *errormsg = 0;
      (void) errormsg;

      presym_free_symlists();
  
      MUTEX_LOCK ();
      if (default_preloaded_symbols)
	{
	  errors = lt_dlpreload (default_preloaded_symbols);
	}
      MUTEX_UNLOCK ();
    }

  return errors;
}

int
lt_dlpreload_default (preloaded)
     const lt_dlsymlist *preloaded;
{
  MUTEX_LOCK ();
  default_preloaded_symbols = preloaded;
  MUTEX_UNLOCK ();
  return 0;
}

int
lt_dlexit ()
{
  /* shut down libltdl */
  lt_dlloader *loader;
  const char  *errormsg;
  int	       errors   = 0;

  (void) errormsg;

  MUTEX_LOCK ();
  loader = loaders;

  if (!initialized)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (SHUTDOWN));
      ++errors;
      goto done;
    }

  /* shut down only at last call. */
  if (--initialized == 0)
    {
      int	level;

      while (handles && LT_DLIS_RESIDENT (handles))
	{
	  handles = handles->next;
	}

      /* close all modules */
      for (level = 1; handles; ++level)
	{
	  lt_dlhandle cur = handles;
	  int saw_nonresident = 0;

	  while (cur)
	    {
	      lt_dlhandle tmp = cur;
	      cur = cur->next;
	      if (!LT_DLIS_RESIDENT (tmp))
		saw_nonresident = 1;
	      if (!LT_DLIS_RESIDENT (tmp) && tmp->info.ref_count <= level)
		{
		  if (lt_dlclose (tmp))
		    {
		      ++errors;
		    }
		}
	    }
	  /* done if only resident modules are left */
	  if (!saw_nonresident)
	    break;
	}

      /* close all loaders */
      while (loader)
	{
	  lt_dlloader *next = loader->next;
	  lt_user_data data = loader->dlloader_data;
	  if (loader->dlloader_exit && loader->dlloader_exit (data))
	    {
	      ++errors;
	    }

	  LT_DLMEM_REASSIGN (loader, next);
	}
      loaders = 0;
    }

 done:
  MUTEX_UNLOCK ();
  return errors;
}

static int
tryall_dlopen(lt_dlhandle *handle, const char *filename)
{
  lt_dlhandle	 cur;
  lt_dlloader   *loader;
  const char	*saved_error;
  int		 errors		= 0;

  MUTEX_GETERROR (saved_error);
  MUTEX_LOCK ();

  cur	 = handles;
  loader = loaders;

  /* check whether the module was already opened */
  while (cur)
    {
      /* try to dlopen the program itself? */
      if (!cur->info.filename && !filename)
	{
	  break;
	}

      if (cur->info.filename && filename
	  && strcmp (cur->info.filename, filename) == 0)
	{
	  break;
	}

      cur = cur->next;
    }

  if (cur)
    {
      ++cur->info.ref_count;
      *handle = cur;
      goto done;
    }

  cur = *handle;
  if (filename)
    {
      cur->info.filename = strdup (filename);
      if (!cur->info.filename)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  ++errors;
	  goto done;
	}
    }
  else
    {
      cur->info.filename = 0;
    }

  while (loader)
    {
      lt_user_data data = loader->dlloader_data;

      cur->module = loader->module_open (data, filename);

      if (cur->module != 0)
	{
	  break;
	}
      loader = loader->next;
    }

  if (!loader)
    {
      LT_DLFREE (cur->info.filename);
      ++errors;
      goto done;
    }

  cur->loader	= loader;
  last_error	= saved_error;
  
 done:
  MUTEX_UNLOCK ();

  return errors;
}

static int
find_module(
  lt_dlhandle *handle,
  const char *dir,
  const char *libdir,
  const char *dlname,
  const char *old_name,
  int installed)
{
  int	error;
  char	*filename;

  /* try to open the old library first; if it was dlpreopened,
     we want the preopened version of it, even if a dlopenable
     module is available */
  if (old_name && tryall_dlopen(handle, old_name) == 0)
    {
      return 0;
    }

  /* try to open the dynamic library */
  if (dlname)
    {
      size_t len;

      /* try to open the installed module */
      if (installed && libdir)
	{
	  len	    = strlen (libdir) + 1 + strlen (dlname);
	  filename  = LT_DLMALLOC (char, 1+ len);

	  if (!filename)
	    {
	      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	      return 1;
	    }

	  sprintf (filename, "%s/%s", libdir, dlname);
	  error = (tryall_dlopen (handle, filename) != 0);
	  LT_DLFREE (filename);

	  if (!error)
	    {
	      return 0;
	    }
	}

      /* try to open the not-installed module */
      if (!installed)
	{
	  len = (dir ? strlen (dir) : 0) + strlen (objdir) + strlen (dlname);
	  filename = LT_DLMALLOC (char, 1+ len);

	  if (!filename)
	    {
	      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	      return 1;
	    }

	  if (dir)
	    {
	      strcpy (filename, dir);
	    }
	  else
	    {
	      *filename = 0;
	    }
	  strcat(filename, objdir);
	  strcat(filename, dlname);

	  error = tryall_dlopen (handle, filename) != 0;
	  LT_DLFREE (filename);
	  if (!error)
	    {
	      return 0;
	    }
	}

      /* maybe it was moved to another directory */
      {
	len	 = (dir ? strlen (dir) : 0) + strlen (dlname);
	filename = LT_DLMALLOC (char, 1+ len);

	if (dir)
	  {
	    strcpy (filename, dir);
	  }
	else
	  {
	    *filename = 0;
	  }
	strcat(filename, dlname);

	error = (tryall_dlopen (handle, filename) != 0);
	LT_DLFREE (filename);
	if (!error)
	  {
	    return 0;
	  }
      }
    }

  return 1;
}

static char*
canonicalize_path(const char *path)
{
  char *canonical = 0;

  if (path && *path)
    {
      char *ptr = strdup (path);
      canonical = ptr;

#ifdef LT_DIRSEP_CHAR
      /* Avoid this overhead where '/' is the only separator. */
      while (ptr = strchr (ptr, LT_DIRSEP_CHAR))
	{
	  *ptr++ = '/';
	}
#endif
    }

  return canonical;
}

static lt_ptr
find_file(
  const char *basename,
  const char *search_path,
  char **pdir,
  lt_dlhandle *handle)
{
  /* When handle != NULL search a library, otherwise a file
     return NULL on failure, otherwise the file/handle.  */

  lt_ptr    result	= 0;
  char	   *filename	= 0;
  int	    filenamesize= 0;
  int	    lenbase	= strlen (basename);
  char	   *canonical	= 0;
  char	   *next	= 0;

  MUTEX_LOCK ();

  if (!search_path || !*search_path)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (FILE_NOT_FOUND));
      goto cleanup;
    }

  canonical = canonicalize_path (search_path);
  if (!canonical)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
      goto cleanup;
    }

  next = canonical;
  while (next)
    {
      int lendir;
      char *cur = next;

      next = strchr (cur, LT_PATHSEP_CHAR);
      if (!next)
	{
	  next = cur + strlen (cur);
	}

      lendir = next - cur;
      if (*next == LT_PATHSEP_CHAR)
	{
	  ++next;
	}
      else
	{
	  next = 0;
	}

      if (lendir == 0)
	{
	  continue;
	}

      if (lendir + 1 + lenbase >= filenamesize)
	{
	  LT_DLFREE (filename);
	  filenamesize = lendir + 1 + lenbase + 1;
	  filename = LT_DLMALLOC (char, filenamesize);

	  if (!filename)
	    {
	      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	      goto cleanup;
	    }
	}

      strncpy(filename, cur, lendir);
      if (filename[lendir-1] != '/')
	{
	  filename[lendir++] = '/';
	}
      strcpy(filename+lendir, basename);
      if (handle)
	{
	  if (tryall_dlopen (handle, filename) == 0)
	    {
	      result = (lt_ptr) handle;
	      goto cleanup;
	    }
	}
      else
	{
	  FILE *file = fopen (filename, LT_READTEXT_MODE);
	  if (file)
	    {
	      LT_DLFREE (*pdir);

	      filename[lendir] = '\0';
	      *pdir = strdup(filename);
	      if (!*pdir)
		{
		  /* We could have even avoided the strdup,
		     but there would be some memory overhead. */
		  *pdir = filename;
		  filename = 0;
		}

	      result = (lt_ptr) file;
	      goto cleanup;
	    }
	}
    }

  MUTEX_SETERROR (LT_DLSTRERROR (FILE_NOT_FOUND));

 cleanup:
  LT_DLFREE (filename);
  LT_DLFREE (canonical);

  MUTEX_UNLOCK ();

  return result;
}

static int
load_deplibs(lt_dlhandle handle, char *deplibs)
{
#if LTDL_DLOPEN_DEPLIBS
  char	*p, *save_search_path;
  int   depcount = 0;
  int	i;
  char	**names = 0;
#endif
  int	errors = 0;

  handle->depcount = 0;

#if LTDL_DLOPEN_DEPLIBS
  if (!deplibs)
    {
      return errors;
    }
  ++errors;

  MUTEX_LOCK ();
  save_search_path = strdup (user_search_path);
  if (user_search_path && !save_search_path)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
      goto cleanup;
    }

  /* extract search paths and count deplibs */
  p = deplibs;
  while (*p)
    {
      if (!isspace ((int) *p))
	{
	  char *end = p+1;
	  while (*end && !isspace((int) *end))
	    {
	      ++end;
	    }

	  if (strncmp(p, "-L", 2) == 0 || strncmp(p, "-R", 2) == 0)
	    {
	      char save = *end;
	      *end = 0; /* set a temporary string terminator */
	      if (lt_dladdsearchdir(p+2))
		{
		  goto cleanup;
		}
	      *end = save;
	    }
	  else
	    {
	      ++depcount;
	    }

	  p = end;
	}
      else
	{
	  ++p;
	}
    }

  /* restore the old search path */
  LT_DLFREE (user_search_path);
  user_search_path = save_search_path;

  MUTEX_UNLOCK ();

  if (!depcount)
    {
      errors = 0;
      goto cleanup;
    }

  names = LT_DLMALLOC (char *, depcount * sizeof (char*));
  if (!names)
    {
      goto cleanup;
    }

  /* now only extract the actual deplibs */
  depcount = 0;
  p = deplibs;
  while (*p)
    {
      if (isspace ((int) *p))
	{
	  ++p;
	}
      else
	{
	  char *end = p+1;
	  while (*end && !isspace ((int) *end))
	    {
	      ++end;
	    }

	  if (strncmp(p, "-L", 2) != 0 && strncmp(p, "-R", 2) != 0)
	    {
	      char *name;
	      char save = *end;
	      *end = 0; /* set a temporary string terminator */
	      if (strncmp(p, "-l", 2) == 0)
		{
		  name = LT_DLMALLOC (char, 3+ /* "lib" */ strlen (p+2) + 1);
		  if (name)
		    {
		      sprintf (name, "lib%s", p+2);
		    }
		}
	      else
		{
		  name = strdup(p);
		}

	      if (name)
		{
		  names[depcount++] = name;
		}
	      else
		{
		  goto cleanup_names;
		}
	      *end = save;
	    }
	  p = end;
	}
    }

  /* load the deplibs (in reverse order)
     At this stage, don't worry if the deplibs do not load correctly,
     they may already be statically linked into the loading application
     for instance.  There will be a more enlightening error message
     later on if the loaded module cannot resolve all of its symbols.  */
  if (depcount)
    {
      int	j = 0;

      handle->deplibs = (lt_dlhandle*) LT_DLMALLOC (lt_dlhandle *, depcount);
      if (!handle->deplibs)
	    {
	  goto cleanup;
	    }

      for (i = 0; i < depcount; ++i)
	{
	  handle->deplibs[j] = lt_dlopenext(names[depcount-1-i]);
	  if (handle->deplibs[j])
	    {
	      ++j;
	    }
	}

      handle->depcount	= j;	/* Number of successfully loaded deplibs */
      errors		= 0;
    }

 cleanup_names:
  for (i = 0; i < depcount; ++i)
    {
      LT_DLFREE (names[i]);
    }

 cleanup:
  LT_DLFREE (names);
#endif

  return errors;
}

static int
unload_deplibs(lt_dlhandle handle)
{
  int i;
  int errors = 0;

  if (handle->depcount)
    {
      for (i = 0; i < handle->depcount; ++i)
	{
	  if (!LT_DLIS_RESIDENT (handle->deplibs[i]))
	    {
	      errors += lt_dlclose (handle->deplibs[i]);
	    }
	}
    }

  return errors;
}

static int
trim (char **dest, const char *str)
{
  /* remove the leading and trailing "'" from str
     and store the result in dest */
  const char *end   = strrchr (str, '\'');
  int	len	    = strlen  (str);
  char *tmp;

  LT_DLFREE (*dest);

  if (len > 3 && str[0] == '\'')
    {
      tmp = LT_DLMALLOC (char, end - str);
      if (!tmp)
	{
	  last_error = LT_DLSTRERROR (NO_MEMORY);
	  return 1;
	}

      strncpy(tmp, &str[1], (end - str) - 1);
      tmp[len-3] = '\0';
      *dest = tmp;
    }
  else
    {
      *dest = 0;
    }

  return 0;
}

static int
free_vars(
  char *dlname,
  char *oldname,
  char *libdir,
  char *deplibs)
{
  LT_DLFREE (dlname);
  LT_DLFREE (oldname);
  LT_DLFREE (libdir);
  LT_DLFREE (deplibs);

  return 0;
}

lt_dlhandle
lt_dlopen (filename)
     const char *filename;
{
  lt_dlhandle handle = 0, newhandle;
  const char *ext;
  const char *saved_error;
  char	*canonical = 0, *basename = 0, *dir = 0, *name = 0;

  MUTEX_GETERROR (saved_error);

  /* dlopen self? */
  if (!filename)
    {
      handle = (lt_dlhandle) LT_DLMALLOC (struct lt_dlhandle_struct, 1);
      if (!handle)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  return 0;
	}

      handle->info.ref_count	= 0;
      handle->depcount		= 0;
      handle->deplibs		= 0;
      handle->caller_data	= 0;
      newhandle			= handle;

      /* lt_dlclose()ing yourself is very bad!  Disallow it.  */
      LT_DLSET_FLAG (handle, LT_DLRESIDENT_FLAG);

      if (tryall_dlopen (&newhandle, 0) != 0)
	{
	  LT_DLFREE (handle);
	  return 0;
	}
      goto register_handle;
    }

  canonical = canonicalize_path (filename);
  if (!canonical)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
      LT_DLFREE (handle);
      return 0;
    }

  /* If the canonical module name is a path (relative or absolute)
     then split it into a directory part and a name part.  */
  basename = strrchr (canonical, '/');
  if (basename)
    {
      ++basename;
      dir = LT_DLMALLOC (char, basename - canonical + 1);
      if (!dir)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  handle = 0;
	  goto cleanup;
	}

      strncpy (dir, canonical, basename - canonical);
      dir[basename - canonical] = '\0';
    }
  else
    {
      basename = canonical;
    }

  /* Check whether we are opening a libtool module (.la extension).  */
  ext = strrchr(basename, '.');
  if (ext && strcmp(ext, ".la") == 0)
    {
      /* this seems to be a libtool module */
      FILE     *file = 0;
      int	i;
      char     *dlname = 0, *old_name = 0;
      char     *libdir = 0, *deplibs = 0;
      char     *line;
      size_t	line_len;
      int	error = 0;

      /* if we can't find the installed flag, it is probably an
	 installed libtool archive, produced with an old version
	 of libtool */
      int	installed = 1;

      /* extract the module name from the file name */
      name = LT_DLMALLOC (char, ext - basename + 1);
      if (!name)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  handle = 0;
	  goto cleanup;
	}

      /* canonicalize the module name */
      for (i = 0; i < ext - basename; ++i)
	{
	  if (isalnum ((int)(basename[i])))
	    {
	      name[i] = basename[i];
	    }
	  else
	    {
	      name[i] = '_';
	    }
	}

      name[ext - basename] = '\0';

    /* Now try to open the .la file.  If there is no directory name
       component, try to find it first in user_search_path and then other
       prescribed paths.  Otherwise (or in any case if the module was not
       yet found) try opening just the module name as passed.  */
      if (!dir)
	{
	  file = (FILE*) find_file(basename, user_search_path, &dir, 0);
	  if (!file)
	    {
	      file = (FILE*) find_file(basename, getenv("LTDL_LIBRARY_PATH"),
				       &dir, 0);
	    }

#ifdef LTDL_SHLIBPATH_VAR
	  if (!file)
	    {
	      file = (FILE*) find_file(basename, getenv(LTDL_SHLIBPATH_VAR),
				       &dir, 0);
	    }
#endif
#ifdef LTDL_SYSSEARCHPATH
	  if (!file)
	    {
	      file = (FILE*) find_file(basename, sys_search_path, &dir, 0);
	    }
#endif
	}
      if (!file)
	{
	  file = fopen (filename, LT_READTEXT_MODE);
	}
      if (!file)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (FILE_NOT_FOUND));
	}

      if (!file)
	{
	  handle = 0;
	  goto cleanup;
	}

      line_len = LT_FILENAME_MAX;
      line = LT_DLMALLOC (char, line_len);
      if (!line)
	{
	  fclose (file);
	  MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  handle = 0;
	  goto cleanup;
	}

      /* read the .la file */
      while (!feof(file))
	{
	  if (!fgets (line, line_len, file))
	    {
	      break;
	    }


	  /* Handle the case where we occasionally need to read a line 
	     that is longer than the initial buffer size.  */
	  while (line[strlen(line) -1] != '\n')
	    {
	      line = LT_DLREALLOC (char, line, line_len *2);
	      if (!fgets (&line[line_len -1], line_len +1, file))
		{
		  break;
		}
	      line_len *= 2;
	    }

	  if (line[0] == '\n' || line[0] == '#')
	    {
	      continue;
	    }

#undef  STR_DLNAME
#define STR_DLNAME	"dlname="
	  if (strncmp (line, STR_DLNAME, sizeof (STR_DLNAME) - 1) == 0)
	    {
	      error = trim (&dlname, &line[sizeof (STR_DLNAME) - 1]);
	    }

#undef  STR_OLD_LIBRARY
#define STR_OLD_LIBRARY	"old_library="
	  else if (strncmp (line, STR_OLD_LIBRARY,
			    sizeof (STR_OLD_LIBRARY) - 1) == 0)
	    {
	      error = trim (&old_name, &line[sizeof (STR_OLD_LIBRARY) - 1]);
	    }
#undef  STR_LIBDIR
#define STR_LIBDIR	"libdir="
	  else if (strncmp (line, STR_LIBDIR, sizeof (STR_LIBDIR) - 1) == 0)
	    {
	      error = trim (&libdir, &line[sizeof(STR_LIBDIR) - 1]);
	    }

#undef  STR_DL_DEPLIBS
#define STR_DL_DEPLIBS	"dependency_libs="
	  else if (strncmp (line, STR_DL_DEPLIBS,
			    sizeof (STR_DL_DEPLIBS) - 1) == 0)
	    {
	      error = trim (&deplibs, &line[sizeof (STR_DL_DEPLIBS) - 1]);
	    }
	  else if (strcmp (line, "installed=yes\n") == 0)
	    {
	      installed = 1;
	    }
	  else if (strcmp (line, "installed=no\n") == 0)
	    {
	      installed = 0;
	    }

#undef  STR_LIBRARY_NAMES
#define STR_LIBRARY_NAMES "library_names="
	  else if (! dlname && strncmp (line, STR_LIBRARY_NAMES,
					sizeof (STR_LIBRARY_NAMES) - 1) == 0)
	    {
	      char *last_libname;
	      error = trim (&dlname, &line[sizeof (STR_LIBRARY_NAMES) - 1]);
	      if (! error && dlname &&
		  (last_libname = strrchr (dlname, ' ')) != NULL)
		{
		  last_libname = strdup (last_libname + 1);
		  LT_DLMEM_REASSIGN (dlname, last_libname);
		}
	    }

	  if (error)
	    {
	      break;
	    }
	}

      fclose (file);
      LT_DLFREE (line);

      /* allocate the handle */
      handle = (lt_dlhandle) LT_DLMALLOC (struct lt_dlhandle_struct, 1);
      if (!handle || error)
	{
	  LT_DLFREE (handle);
	  if (!error)
	    {
	      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	    }

	  free_vars (dlname, old_name, libdir, deplibs);
	  /* handle is already set to 0 */
	  goto cleanup;
	}

      handle->info.ref_count = 0;
      if (load_deplibs (handle, deplibs) == 0)
	{
	  newhandle = handle;
	  /* find_module may replace newhandle */
	  if (find_module (&newhandle, dir, libdir, dlname, old_name, installed))
	    {
	      unload_deplibs (handle);
	      error = 1;
	    }
	}
      else
	{
	  error = 1;
	}

      free_vars (dlname, old_name, libdir, deplibs);
      if (error)
	{
	  LT_DLFREE (handle);
	  goto cleanup;
	}

      if (handle != newhandle)
	{
	  unload_deplibs (handle);
	}
    }
  else
    {
      /* not a libtool module */
      handle = (lt_dlhandle) LT_DLMALLOC (struct lt_dlhandle_struct, 1);
      if (!handle)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  /* handle is already set to 0 */
	  goto cleanup;
	}
      handle->info.ref_count = 0;
      /* non-libtool modules don't have dependencies */
      handle->depcount    = 0;
      handle->deplibs	  = 0;
      newhandle	    	  = handle;

      /* If the module has no directory name component, try to find it
	 first in user_search_path and then other prescribed paths.
	 Otherwise (or in any case if the module was not yet found) try
	 opening just the module name as passed.  */
      if ((dir || (!find_file (basename, user_search_path, 0, &newhandle)
		   && !find_file (basename, getenv ("LTDL_LIBRARY_PATH"),
				  0, &newhandle)
#ifdef LTDL_SHLIBPATH_VAR
		   && !find_file (basename, getenv (LTDL_SHLIBPATH_VAR),
				  0, &newhandle)
#endif
#ifdef LTDL_SYSSEARCHPATH
		   && !find_file (basename, sys_search_path, 0, &newhandle)
#endif
		   )) && tryall_dlopen (&newhandle, filename))
	{
	  LT_DLFREE (handle);
	  goto cleanup;
	}
    }

 register_handle:
  LT_DLMEM_REASSIGN (handle, newhandle);

  if (handle->info.ref_count == 0)
    {
      handle->info.ref_count	= 1;
      handle->info.name		= name;
      handle->next		= handles;

      MUTEX_LOCK ();
      handles			= handle;
      MUTEX_UNLOCK ();

      name = 0;	/* don't free this during `cleanup' */
    }

  MUTEX_SETERROR (saved_error);

 cleanup:
  LT_DLFREE (dir);
  LT_DLFREE (name);
  LT_DLFREE (canonical);

  return handle;
}

lt_dlhandle
lt_dlopenext (filename)
     const char *filename;
{
  lt_dlhandle handle;
  char	*tmp;
  int	len;
  const char *saved_error;

  MUTEX_GETERROR (saved_error);

  if (!filename)
    {
      return lt_dlopen (filename);
    }

  len = strlen (filename);
  if (!len)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (FILE_NOT_FOUND));
      return 0;
    }

  /* try "filename.la" */
  tmp = LT_DLMALLOC (char, len+4);
  if (!tmp)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
      return 0;
    }
  strcpy (tmp, filename);
  strcat (tmp, ".la");
  handle = lt_dlopen (tmp);
  if (handle)
    {
      MUTEX_SETERROR (saved_error);
      LT_DLFREE (tmp);
      return handle;
    }

#ifdef LTDL_SHLIB_EXT
  /* try "filename.EXT" */
  if (strlen(shlib_ext) > 3)
    {
      LT_DLFREE (tmp);
      tmp = LT_DLMALLOC (char, len + strlen (shlib_ext) + 1);
      if (!tmp)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  return 0;
	}
      strcpy (tmp, filename);
    }
  else
    {
      tmp[len] = '\0';
    }

  strcat(tmp, shlib_ext);
  handle = lt_dlopen (tmp);
  if (handle)
    {
      MUTEX_SETERROR (saved_error);
      LT_DLFREE (tmp);
      return handle;
    }
#endif

  /* try the normal file name */
  handle = lt_dlopen (filename);
  if (handle)
    {
      return handle;
    }

  MUTEX_SETERROR (LT_DLSTRERROR (FILE_NOT_FOUND));
  LT_DLFREE (tmp);
  return 0;
}

int
lt_dlclose (handle)
     lt_dlhandle handle;
{
  lt_dlhandle cur, last;
  int errors = 0;

  MUTEX_LOCK ();

  /* check whether the handle is valid */
  last = cur = handles;
  while (cur && handle != cur)
    {
      last = cur;
      cur = cur->next;
    }

  if (!cur)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
      ++errors;
      goto done;
    }

  handle->info.ref_count--;

  /* Note that even with resident modules, we must track the ref_count
     correctly incase the user decides to reset the residency flag
     later (even though the API makes no provision for that at the
     moment).  */
  if (handle->info.ref_count <= 0 && !LT_DLIS_RESIDENT (handle))
    {
      lt_user_data data = handle->loader->dlloader_data;

      if (handle != handles)
	{
	  last->next = handle->next;
	}
      else
	{
	  handles = handle->next;
	}

      errors += handle->loader->module_close (data, handle->module);
      errors += unload_deplibs(handle);

      LT_DLFREE (handle->info.filename);
      LT_DLFREE (handle->info.name);
      LT_DLFREE (handle);

      goto done;
    }

  if (LT_DLIS_RESIDENT (handle))
    {
      MUTEX_SETERROR (LT_DLSTRERROR (CLOSE_RESIDENT_MODULE));
      ++errors;
    }

 done:
  MUTEX_UNLOCK ();

  return errors;
}

lt_ptr
lt_dlsym (handle, symbol)
     lt_dlhandle handle;
     const char *symbol;
{
  int	lensym;
  char	lsym[LT_SYMBOL_LENGTH];
  char	*sym;
  lt_ptr address;
  lt_user_data data;

  if (!handle)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
      return 0;
    }

  if (!symbol)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (SYMBOL_NOT_FOUND));
      return 0;
    }

  lensym = strlen(symbol);
  if (handle->loader->sym_prefix)
    {
      lensym += strlen(handle->loader->sym_prefix);
    }

  if (handle->info.name)
    {
      lensym += strlen(handle->info.name);
    }

  if (lensym + LT_SYMBOL_OVERHEAD < LT_SYMBOL_LENGTH)
    {
      sym = lsym;
    }
  else
    {
      sym = LT_DLMALLOC (char, lensym + LT_SYMBOL_OVERHEAD + 1);
    }

  if (!sym)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (BUFFER_OVERFLOW));
      return 0;
    }

  data = handle->loader->dlloader_data;
  if (handle->info.name)
    {
      const char *saved_error;

      MUTEX_GETERROR (saved_error);

      /* this is a libtool module */
      if (handle->loader->sym_prefix)
	{
	  strcpy(sym, handle->loader->sym_prefix);
	  strcat(sym, handle->info.name);
	}
      else
	{
	  strcpy(sym, handle->info.name);
	}

      strcat(sym, "_LTX_");
      strcat(sym, symbol);

      /* try "modulename_LTX_symbol" */
      address = handle->loader->find_sym (data, handle->module, sym);
      if (address)
	{
	  if (sym != lsym)
	    {
	      LT_DLFREE (sym);
	    }
	  return address;
	}
      MUTEX_SETERROR (saved_error);
    }

  /* otherwise try "symbol" */
  if (handle->loader->sym_prefix)
    {
      strcpy(sym, handle->loader->sym_prefix);
      strcat(sym, symbol);
    }
  else
    {
      strcpy(sym, symbol);
    }

  address = handle->loader->find_sym (data, handle->module, sym);
  if (sym != lsym)
    {
      LT_DLFREE (sym);
    }

  return address;
}

const char *
lt_dlerror ()
{
  const char *error;

  MUTEX_GETERROR (error);
  MUTEX_SETERROR (0);

  return error;
}

int
lt_dladdsearchdir (search_dir)
     const char *search_dir;
{
  int errors = 0;

  if (!search_dir || !strlen(search_dir))
    {
      return errors;
    }

  MUTEX_LOCK ();
  if (!user_search_path)
    {
      user_search_path = strdup (search_dir);
      if (!user_search_path)
	{
	  last_error = LT_DLSTRERROR (NO_MEMORY);
	  ++errors;
	}
    }
  else
    {
      size_t len = strlen (user_search_path) + 1 + strlen (search_dir);
      char  *new_search_path = LT_DLMALLOC (char, 1+ len);

      if (!new_search_path)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  ++errors;
	}
      else
	{
	  sprintf (new_search_path, "%s%c%s", user_search_path,
		   LT_PATHSEP_CHAR, search_dir);

	  LT_DLMEM_REASSIGN (user_search_path, new_search_path);
	}
    }
  MUTEX_UNLOCK ();

  return errors;
}

int
lt_dlsetsearchpath (search_path)
     const char *search_path;
{
  int errors = 0;

  MUTEX_LOCK ();
  LT_DLFREE (user_search_path);
  MUTEX_UNLOCK ();

  if (!search_path || !strlen (search_path))
    {
      return errors;
    }

  MUTEX_LOCK ();
  user_search_path = strdup (search_path);
  if (!user_search_path)
    {
      ++errors;
    }
  MUTEX_UNLOCK ();

  return errors;
}

const char *
lt_dlgetsearchpath ()
{
  const char *saved_path;

  MUTEX_LOCK ();
  saved_path = user_search_path;
  MUTEX_UNLOCK ();

  return saved_path;
}

int
lt_dlmakeresident (handle)
     lt_dlhandle handle;
{
  int errors = 0;

  if (!handle)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
      ++errors;
    }
  else
    {
      LT_DLSET_FLAG (handle, LT_DLRESIDENT_FLAG);
    }

  return errors;
}

int
lt_dlisresident	(handle)
     lt_dlhandle handle;
{
  if (!handle)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
      return -1;
    }

  return LT_DLIS_RESIDENT (handle);
}




/* --- MODULE INFORMATION --- */

const lt_dlinfo *
lt_dlgetinfo (handle)
     lt_dlhandle handle;
{
  if (!handle)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (INVALID_HANDLE));
      return 0;
    }

  return &(handle->info);
}

lt_dlhandle
lt_dlhandle_next (place)
     lt_dlhandle place;
{
  return place ? place->next : (lt_dlhandle) 0;
}

int
lt_dlforeach (func, data)
     int (*func) LT_PARAMS((lt_dlhandle handle, lt_ptr data));
     lt_ptr data;
{
  int errors = 0;
  lt_dlhandle cur;

  MUTEX_LOCK ();

  cur = handles;
  while (cur)
    {
      lt_dlhandle tmp = cur;

      cur = cur->next;
      if ((*func) (tmp, data))
	{
	  ++errors;
	  break;
	}
    }

  MUTEX_UNLOCK ();

  return errors;
}

lt_dlcaller_id
lt_dlcaller_register ()
{
  static int last_caller_id = -1;
  int result;

  MUTEX_LOCK ();
  result = ++last_caller_id;
  MUTEX_UNLOCK ();

  return result;
}

#define N_ELEMENTS(a)	(sizeof(a) / sizeof(*(a)))

lt_ptr
lt_dlcaller_set_data (key, handle, data)
     lt_dlcaller_id key;
     lt_dlhandle handle;
     lt_ptr data;
{
  int n_elements = 0;
  lt_ptr stale = (lt_ptr) 0;
  int i;

  /* This needs to be locked so that the caller data can be updated
     simultaneously by different threads.  */
  MUTEX_LOCK ();

  if (handle->caller_data)
    n_elements = N_ELEMENTS (handle->caller_data);

  for (i = 0; i < n_elements; ++i)
    {
      if (handle->caller_data[i].key == key)
	{
	  stale = handle->caller_data[i].data;
	  break;
	}
    }

  /* Ensure that there is enough room in this handle's caller_data
     array to accept a new element.  */
  if (i == n_elements)
    {
      lt_caller_data *temp
	= LT_DLREALLOC (lt_caller_data, handle->caller_data, 1+ n_elements);

      if (temp == 0)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
	  stale =  (lt_ptr) 0;
	  goto done;
	}
      else
	{
	  handle->caller_data = temp;
	}

      /* We only need this if we needed to allocate a new caller_data.  */
      handle->caller_data[i].key  = key;
    }

  handle->caller_data[i].data = data;

 done:
  MUTEX_UNLOCK ();

  return stale;
}

lt_ptr
lt_dlcaller_get_data  (key, handle)
     lt_dlcaller_id key;
     lt_dlhandle handle;
{
  lt_ptr result = (lt_ptr) 0;
  int n_elements = 0;

  /* This needs to be locked so that the caller data isn't updated by
     another thread part way through this function.  */
  MUTEX_LOCK ();

  if (handle->caller_data)
    n_elements = N_ELEMENTS (handle->caller_data);

  /* Locate the index of the element with a matching KEY.  */
  {
    int i;
    for (i = 0; i < n_elements; ++i)
      {
	if (handle->caller_data[i].key == key)
	  {
	    result = handle->caller_data[i].data;
	    break;
	  }
      }
  }

  MUTEX_UNLOCK ();

  return result;
}



/* --- USER MODULE LOADER API --- */


int
lt_dlloader_add (place, dlloader, loader_name)
     lt_dlloader *place;
     const struct lt_user_dlloader *dlloader;
     const char *loader_name;
{
  int errors = 0;
  lt_dlloader *node = 0, *ptr = 0;

  if ((dlloader == 0)	/* diagnose null parameters */
      || (dlloader->module_open == 0)
      || (dlloader->module_close == 0)
      || (dlloader->find_sym == 0))
    {
      MUTEX_SETERROR (LT_DLSTRERROR (INVALID_LOADER));
      return 1;
    }

  /* Create a new dlloader node with copies of the user callbacks.  */
  node = LT_DLMALLOC (lt_dlloader, 1);
  if (node == 0)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (NO_MEMORY));
      return 1;
    }

  node->next		= 0;
  node->loader_name	= loader_name;
  node->sym_prefix	= dlloader->sym_prefix;
  node->dlloader_exit	= dlloader->dlloader_exit;
  node->module_open	= dlloader->module_open;
  node->module_close	= dlloader->module_close;
  node->find_sym	= dlloader->find_sym;
  node->dlloader_data	= dlloader->dlloader_data;

  MUTEX_LOCK ();
  if (!loaders)
    {
      /* If there are no loaders, NODE becomes the list! */
      loaders = node;
    }
  else if (!place)
    {
      /* If PLACE is not set, add NODE to the end of the
	 LOADERS list. */
      for (ptr = loaders; ptr->next; ptr = ptr->next)
	{
	  /*NOWORK*/;
	}

      ptr->next = node;
    }
  else if (loaders == place)
    {
      /* If PLACE is the first loader, NODE goes first. */
      node->next = place;
      loaders = node;
    }
  else
    {
      /* Find the node immediately preceding PLACE. */
      for (ptr = loaders; ptr->next != place; ptr = ptr->next)
	{
	  /*NOWORK*/;
	}

      if (ptr->next != place)
	{
	  last_error = LT_DLSTRERROR (INVALID_LOADER);
	  ++errors;
	}
      else
	{
	  /* Insert NODE between PTR and PLACE. */
	  node->next = place;
	  ptr->next  = node;
	}
    }

  MUTEX_UNLOCK ();

  return errors;
}

int
lt_dlloader_remove (loader_name)
     const char *loader_name;
{
  lt_dlloader *place = lt_dlloader_find (loader_name);
  lt_dlhandle handle;
  int errors = 0;

  if (!place)
    {
      MUTEX_SETERROR (LT_DLSTRERROR (INVALID_LOADER));
      return 1;
    }

  MUTEX_LOCK ();

  /* Fail if there are any open modules which use this loader. */
  for  (handle = handles; handle; handle = handle->next)
    {
      if (handle->loader == place)
	{
	  MUTEX_SETERROR (LT_DLSTRERROR (REMOVE_LOADER));
	  ++errors;
	  goto done;
	}
    }

  if (place == loaders)
    {
      /* PLACE is the first loader in the list. */
      loaders = loaders->next;
    }
  else
    {
      /* Find the loader before the one being removed. */
      lt_dlloader *prev;
      for (prev = loaders; prev->next; prev = prev->next)
	{
	  if (!strcmp (prev->next->loader_name, loader_name))
	    {
	      break;
	    }
	}

      place = prev->next;
      prev->next = prev->next->next;
    }

  if (place->dlloader_exit)
    {
      errors = place->dlloader_exit (place->dlloader_data);
    }

  LT_DLFREE (place);

 done:
  MUTEX_UNLOCK ();

  return errors;
}

lt_dlloader *
lt_dlloader_next (place)
     lt_dlloader *place;
{
  lt_dlloader *next;

  MUTEX_LOCK ();
  next = place ? place->next : loaders;
  MUTEX_UNLOCK ();

  return next;
}

const char *
lt_dlloader_name (place)
     lt_dlloader *place;
{
  const char *name = 0;

  if (place)
    {
      MUTEX_LOCK ();
      name = place ? place->loader_name : 0;
      MUTEX_UNLOCK ();
    }
  else
    {
      MUTEX_SETERROR (LT_DLSTRERROR (INVALID_LOADER));
    }

  return name;
}

lt_user_data *
lt_dlloader_data (place)
     lt_dlloader *place;
{
  lt_user_data *data = 0;

  if (place)
    {
      MUTEX_LOCK ();
      data = place ? &(place->dlloader_data) : 0;
      MUTEX_UNLOCK ();
    }
  else
    {
      MUTEX_SETERROR (LT_DLSTRERROR (INVALID_LOADER));
    }

  return data;
}

lt_dlloader *
lt_dlloader_find (loader_name)
     const char *loader_name;
{
  lt_dlloader *place = 0;

  MUTEX_LOCK ();
  for (place = loaders; place; place = place->next)
    {
      if (strcmp (place->loader_name, loader_name) == 0)
	{
	  break;
	}
    }
  MUTEX_UNLOCK ();

  return place;
}
