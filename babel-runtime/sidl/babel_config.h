/* sidl/babel_config.h.  Generated automatically by configure.  */
/* sidl/babel_config.h.in.  Generated from configure.ac by autoheader.  */


#ifndef included_babel_config_h
#define included_babel_config_h


/* If defined, C++ support was disabled at configure time */
/* #undef CXX_DISABLED */

/* Define to dummy `main' function (if any) required to link to the Fortran 77
   libraries. */
/* #undef F77_DUMMY_MAIN */

/* If defined, Fortran support was disabled at configure time */
/* #undef FORTRAN77_DISABLED */

/* Define to 1 if you have the `atexit' function. */
#define HAVE_ATEXIT 1

/* Define to 1 if you have the `bcopy' function. */
/* #undef HAVE_BCOPY */

/* define if the compiler has complex<T> */
#define HAVE_COMPLEX 

/* define if the compiler has complex math functions */
/* #undef HAVE_COMPLEX_MATH1 */

/* define if the compiler has more complex math functions */
/* #undef HAVE_COMPLEX_MATH2 */

/* define if complex math functions are in std:: */
#define HAVE_COMPLEX_MATH_IN_NAMESPACE_STD 

/* Define to 1 if you have the <ctype.h> header file. */
#define HAVE_CTYPE_H 1

/* Define to 1 if you have the <dirent.h> header file, and it defines `DIR'.
   */
#define HAVE_DIRENT_H 1

/* Define if you have the GNU dld library. */
/* #undef HAVE_DLD */

/* Define to 1 if you have the <dld.h> header file. */
/* #undef HAVE_DLD_H */

/* Define to 1 if you have the `dlerror' function. */
#define HAVE_DLERROR 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <dl.h> header file. */
/* #undef HAVE_DL_H */

/* Define to 1 if you have the `getcwd' function. */
#define HAVE_GETCWD 1

/* define if the compiler supports IEEE math library */
#define HAVE_IEEE_MATH 

/* Define to 1 if you have the `index' function. */
/* #undef HAVE_INDEX */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <jni.h> header file. */
#define HAVE_JNI_H 1

/* Define if you have the libdl library or equivalent. */
#define HAVE_LIBDL 1

/* define if long long is a built in type */
/* #undef HAVE_LONG_LONG */

/* Define to 1 if your system has a working `malloc' function. */
#define HAVE_MALLOC 1

/* Define to 1 if you have the <malloc.h> header file. */
#define HAVE_MALLOC_H 1

/* Define to 1 if you have the `memcpy' function. */
#define HAVE_MEMCPY 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the `memset' function. */
#define HAVE_MEMSET 1

/* define if the compiler implements namespaces */
#define HAVE_NAMESPACES 

/* Define to 1 if you have the <ndir.h> header file, and it defines `DIR'. */
/* #undef HAVE_NDIR_H */

/* define if the compiler has numeric_limits<T> */
/* #undef HAVE_NUMERIC_LIMITS */

/* Define if libtool can extract symbol lists from object files. */
#define HAVE_PRELOADED_SYMBOLS 1

/* Define to 1 if the system has the type `ptrdiff_t'. */
#define HAVE_PTRDIFF_T 1

/* Define to 1 if you have the `rindex' function. */
/* #undef HAVE_RINDEX */

/* Define if you have the shl_load function. */
/* #undef HAVE_SHL_LOAD */

/* Define to 1 if `stat' has the bug that it succeeds when given the
   zero-length file name argument. */
/* #undef HAVE_STAT_EMPTY_STRING_BUG */

/* define if the compiler supports ISO C++ standard library */
#define HAVE_STD 

/* Define to 1 if you have the <stddef.h> header file. */
#define HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* define if the compiler supports Standard Template Library */
#define HAVE_STL 

/* Define to 1 if you have the `strchr' function. */
#define HAVE_STRCHR 1

/* Define to 1 if you have the `strcmp' function. */
#define HAVE_STRCMP 1

/* Define to 1 if you have the `strdup' function. */
#define HAVE_STRDUP 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strrchr' function. */
#define HAVE_STRRCHR 1

/* Define to 1 if you have the <sys/dir.h> header file, and it defines `DIR'.
   */
/* #undef HAVE_SYS_DIR_H */

/* Define to 1 if you have the <sys/dl.h> header file. */
/* #undef HAVE_SYS_DL_H */

/* Define to 1 if you have the <sys/ndir.h> header file, and it defines `DIR'.
   */
/* #undef HAVE_SYS_NDIR_H */

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* If defined, Java support was disabled at configure time */
/* #undef JAVA_DISABLED */

/* Define to 1 if `lstat' dereferences a symlink specified with a trailing
   slash. */
#define LSTAT_FOLLOWS_SLASHED_SYMLINK 1

/* Define if the OS needs help to load dependent libraries for dlopen(). */
/* #undef LTDL_DLOPEN_DEPLIBS */

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LTDL_OBJDIR ".libs/"

/* Define to the name of the environment variable that determines the dynamic
   library search path. */
#define LTDL_SHLIBPATH_VAR "LD_LIBRARY_PATH"

/* Define to the extension used for shared libraries, say, ".so". */
#define LTDL_SHLIB_EXT ".so"

/* Define to the system default library search path. */
#define LTDL_SYSSEARCHPATH "/lib:/usr/lib"

/* Define if dlsym() requires a leading underscode in symbol names. */
/* #undef NEED_USCORE */

/* Name of package */
#define PACKAGE "babel-runtime"

/* Define to the address where bug reports for this package should be sent. */
/* #undef PACKAGE_BUGREPORT */

/* Define to the full name of this package. */
/* #undef PACKAGE_NAME */

/* Define to the full name and version of this package. */
/* #undef PACKAGE_STRING */

/* Define to the one symbol short name of this package. */
/* #undef PACKAGE_TARNAME */

/* Define to the version of this package. */
/* #undef PACKAGE_VERSION */

/* If defined, Python support was disabled at configure time */
#define PYTHON_DISABLED 1

/* If defined, server-side Python support was disabled at configure time */
#define PYTHON_SERVER_DISABLED 1

/* Fully qualified string name of the Python shared library */
/* #undef PYTHON_SHARED_LIBRARY */

/* Directory of the Python shared library */
/* #undef PYTHON_SHARED_LIBRARY_DIR */

/* A string indicating the Python version number */
#define PYTHON_VERSION "1.5"

/* define if C++ requires old .h-style header includes */
/* #undef REQUIRE_OLD_CXX_HEADER_SUFFIX */

/* Define SIDL_DYNAMIC_LIBRARY to force dynamic loading of libraries */
/* #undef SIDL_DYNAMIC_LIBRARY */

/* F77 char args are strings */
#define SIDL_F77_CHAR_AS_STRING 

/* F77 logical false value */
#define SIDL_F77_FALSE 0

/* F77 symbols are lower case */
#define SIDL_F77_LOWER_CASE 

/* F77 symbols are mixed case */
/* #undef SIDL_F77_MIXED_CASE */

/* one underscore after F77 symbols */
/* #undef SIDL_F77_ONE_UNDERSCORE */

/* F77 strings lengths immediately follow string */
#define SIDL_F77_STR_LEN_FAR 

/* F77 strings lengths at end */
/* #undef SIDL_F77_STR_LEN_NEAR */

/* Minimum size for out strings */
#define SIDL_F77_STR_MINSIZE 512

/* F77 strings as length-char* structs */
/* #undef SIDL_F77_STR_STRUCT_LEN_STR */

/* F77 strings as char*-length structs */
/* #undef SIDL_F77_STR_STRUCT_STR_LEN */

/* F77 logical true value */
#define SIDL_F77_TRUE 1

/* two underscores after F77 symbols */
#define SIDL_F77_TWO_UNDERSCORE 

/* F77 symbols are upper case */
/* #undef SIDL_F77_UPPER_CASE */

/* no underscores after F77 symbols */
/* #undef SIDL_F77_ZERO_UNDERSCORE */

/* Define SIDL_STATIC_LIBRARY to force static loading of libraries */
#define SIDL_STATIC_LIBRARY 1

/* The size of a `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* The size of a `long', as computed by sizeof. */
#define SIZEOF_LONG 4

/* The size of a `long long', as computed by sizeof. */
#define SIZEOF_LONG_LONG 8

/* The size of a `short', as computed by sizeof. */
#define SIZEOF_SHORT 2

/* The size of a `void *', as computed by sizeof. */
#define SIZEOF_VOID_P 4

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Version number of package */
#define VERSION "0.7.5"

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* used when a compiler does not recognize int32_t */
/* #undef int32_t */

/* used when a compiler does not recognize int64_t */
/* #undef int64_t */

/* Define to equivalent of C99 restrict keyword, or to nothing if this is not
   supported. Do not define if restrict is supported directly. */
#define restrict __restrict__

/* Define to `unsigned' if <sys/types.h> does not define. */
/* #undef size_t */


/*
 * Set flags for dynamic or static loading of implementations in Babel stubs.
 * One and only one of SIDL_STATIC_LIBRARY and SIDL_DYNAMIC_LIBRARY may be set.
 * If neither is set, then SIDL_DYNAMIC_LIBRARY is chosen as the default if
 * PIC is set and SIDL_STATIC_LIBRARY is chosen otherwise.  This behavior is
 * consistent with GNU libtool.  In general, we want to generate dynamic
 * loading with shared libraries (indicated by -DPIC in libtool) and static
 * loading with static libraries.
 */
#if (!defined(SIDL_STATIC_LIBRARY) && !defined(SIDL_DYNAMIC_LIBRARY))
#ifdef PIC
#define SIDL_DYNAMIC_LIBRARY
#else
#define SIDL_STATIC_LIBRARY
#endif
#endif

#if (defined(SIDL_STATIC_LIBRARY) && defined(SIDL_DYNAMIC_LIBRARY))
#error Cannot define both SIDL_STATIC_LIBRARY and SIDL_DYNAMIC_LIBRARY
#endif
#if (!defined(SIDL_STATIC_LIBRARY) && !defined(SIDL_DYNAMIC_LIBRARY))
#error Must define one of SIDL_STATIC_LIBRARY or SIDL_DYNAMIC_LIBRARY
#endif

/*
 * The USE_DL_IMPORT flag is required for proper Python linking under CYGWIN.
 */
#if defined(__CYGWIN__) && !defined(USE_DL_IMPORT)
#define USE_DL_IMPORT
#endif

#endif

