# LLNL_FORTRAN_STRING_TEST
# ------------------------
# Test for the correct passing of string data
#

AC_DEFUN([LLNL_FORTRAN_STRING_TEST_PROLOGUE],[
#ifdef SIDL_$1_ONE_UNDERSCORE
#ifdef SIDL_$1_UPPER_CASE
#define TESTFUNC STR_TST_
#else
#define TESTFUNC str_tst_
#endif
#else
#ifdef SIDL_$1_TWO_UNDERSCORE
#ifdef SIDL_$1_UPPER_CASE
#define TESTFUNC STR_TST__
#else
#define TESTFUNC str_tst__
#endif
#else
#ifdef SIDL_$1_UPPER_CASE
#define TESTFUNC STR_TST
#else
#define TESTFUNC str_tst
#endif
#endif
#endif
#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#else
#include <sys/types.h>
#endif
typedef int SIDL_$1_Bool;
#ifdef __cplusplus
extern "C"
#else
extern
#endif
])

dnl LLNL_FORTRAN_STRING_TEST(longname,shortname)
AC_DEFUN([LLNL_FORTRAN_STRING_TEST],
[AC_REQUIRE([LLNL_$2_NAME_MANGLING])dnl
AC_CACHE_CHECK(dnl
[for $1 ($2) binary string passing convention],llnl_cv_$2_string_passing,
[llnl_cv_$2_string_passing="fail"
AC_LANG_PUSH($1)dnl
AC_COMPILE_IFELSE([
       subroutine str_tst(l,a,b,c)
       implicit none
       logical l
       character*(*) a, b, c
       l = .true.
       if (len ( a ) .ne. 3) l = .false.
       if (len ( b ) .ne. 0) l = .false.
       if (len ( c ) .ne. 7) l = .false.
       if (a .ne. 'yes') l = .false.
       if (b .ne. '') l = .false.
       if (c .ne. 'confirm') l = .false.
       end],[
  mv conftest.$ac_objext cfortran_test.$ac_objext
  AC_LANG_PUSH(C)dnl
  ac_save_LIBS=$LIBS
  LIBS="cfortran_test.$ac_objext $LIBS $3"
  for intsize in int64_t int32_t; do
    if test "$llnl_cv_$2_string_passing" = "fail" ; then
      AC_RUN_IFELSE(
        [AC_LANG_PROGRAM(LLNL_FORTRAN_STRING_TEST_PROLOGUE($2)
void TESTFUNC(SIDL_$2_Bool *l,char *s1, char *s2, char *s3, 
              $intsize l1, $intsize l2, $intsize l3);
,[
  int l;
  char s1[[]] = "yes"; const $intsize l1 = ($intsize)(sizeof(s1)-1);
  char s2[[]] = ""; const $intsize l2 = ($intsize)(sizeof(s2)-1);
  char s3[[]] = "confirm"; const $intsize l3 = ($intsize)(sizeof(s3)-1);
  TESTFUNC(&l,s1,s2,s3,l1,l2,l3);
  return (l == SIDL_$2_TRUE) ? 0 : -1;
])],[AC_RUN_IFELSE(
        [AC_LANG_PROGRAM(LLNL_FORTRAN_STRING_TEST_PROLOGUE($2)
void TESTFUNC(SIDL_$2_Bool *l,char *s1, char *s2, char *s3, 
              $intsize l1, $intsize l2, $intsize l3);
,[
  int l;
  const $intsize upper = ((($intsize)1) << (sizeof($intsize)*4));
  char s1[[]] = "yes";const $intsize l1 = ($intsize)(sizeof(s1)-1) | upper;
  char s2[[]] = "";const $intsize l2 = ($intsize)(sizeof(s2)-1) | upper;
  char s3[[]] = "confirm"; const $intsize l3 = ($intsize)(sizeof(s3)-1) |upper;
  TESTFUNC(&l,s1,s2,s3,l1,l2,l3);
  return (l == SIDL_$2_TRUE) ? 0 : -1;
])],,[dnl Setting upper bits causes runtime failure; hence, $intsize is right.
   llnl_cv_$2_string_passing="far $intsize"])],
    [
        AC_RUN_IFELSE(
          [AC_LANG_PROGRAM([LLNL_FORTRAN_STRING_TEST_PROLOGUE($2)
void TESTFUNC(SIDL_$2_Bool *l, char *s1, $intsize l1, char *s2, $intsize l2, char *s3, 
              $intsize l3);],[
    int l;
    char s1[[]] = "yes"; const $intsize l1 = ($intsize)(sizeof(s1)-1);
    char s2[[]] = ""; const $intsize l2 = ($intsize)(sizeof(s2)-1);
    char s3[[]] = "confirm"; const $intsize l3 = ($intsize)(sizeof(s3)-1);
    TESTFUNC(&l,s1,l1,s2,l2,s3,l3);
    return (l == SIDL_$2_TRUE) ? 0 : -1;
])],[AC_RUN_IFELSE(
        [AC_LANG_PROGRAM(LLNL_FORTRAN_STRING_TEST_PROLOGUE($2)
void TESTFUNC(SIDL_$2_Bool *l,char *s1, $intsize l1, char *s2, 
              $intsize l2, char *s3, $intsize l3);
,[
  int l;
  const $intsize upper = ((($intsize)1) << (sizeof($intsize)*4));
  char s1[[]] = "yes";const $intsize l1 = ($intsize)(sizeof(s1)-1) | upper;
  char s2[[]] = "";const $intsize l2 = ($intsize)(sizeof(s2)-1) | upper;
  char s3[[]] = "confirm";const $intsize l3 = ($intsize)(sizeof(s3)-1) | upper;
  TESTFUNC(&l,s1,l1,s2,l2,s3,l3);
  return (l == SIDL_$2_TRUE) ? 0 : -1;
])],,[dnl Setting upper bits causes runtime failure; hence, $intsize is right.
   llnl_cv_$2_string_passing="near $intsize"])],
      [
          AC_RUN_IFELSE(
            [struct fortran_str_arg; /* forward declaration */
AC_LANG_PROGRAM([LLNL_FORTRAN_STRING_TEST_PROLOGUE($2)
void TESTFUNC(SIDL_$2_Bool *l,struct fortran_str_arg *s1, struct fortran_str_arg *s2, struct fortran_str_arg *s3);
struct fortran_str_arg { char *str; $intsize len; int dummy; };
],[
    int l;
    char s1[[]] = "yes"; char s2[[]] = ""; char s3[[]] = "confirm"; 
    struct fortran_str_arg a1 = { s1, ($intsize)(sizeof(s1) - 1), -1 },
     a2 = { s2 , ($intsize)(sizeof(s2)-1), -1 },
     a3 = { s3 , ($intsize)(sizeof(s3)-1), -1 };
    TESTFUNC(&l, &a1, &a2, &a3);
    return (l == SIDL_$2_TRUE) ? 0 : -1;
])],[AC_RUN_IFELSE(
        [AC_LANG_PROGRAM(struct fortran_str_arg; /* forward declaration */
LLNL_FORTRAN_STRING_TEST_PROLOGUE($2)
void TESTFUNC(SIDL_$2_Bool *l,struct fortran_str_arg *s1, struct fortran_str_arg *s2, struct fortran_str_arg *s3);
struct fortran_str_arg { char *str; $intsize len; int dummy; };
,[
  int l;
  const $intsize upper = ((($intsize)1) << (sizeof($intsize)*4));
  char s1[[]] = "yes"; char s2[[]] = ""; char s3[[]] = "confirm"; 
  struct fortran_str_arg a1 = { s1, ($intsize)(sizeof(s1) - 1)|upper, -1 },
   a2 = { s2 , ($intsize)(sizeof(s2)-1)|upper, -1 },
   a3 = { s3 , ($intsize)(sizeof(s3)-1)|upper, -1 };
  TESTFUNC(&l, &a1, &a2, &a3);
  return (l == SIDL_$2_TRUE) ? 0 : -1;
])],,[dnl Setting upper bits causes runtime failure; hence, $intsize is right.
   llnl_cv_$2_string_passing="struct_str_len $intsize"])],
        [
            AC_RUN_IFELSE(
              [struct fortran_str_arg; /* forward declaration */
AC_LANG_PROGRAM([LLNL_FORTRAN_STRING_TEST_PROLOGUE($2)
void TESTFUNC(SIDL_$2_Bool *l,struct fortran_str_arg *s1, struct fortran_str_arg *s2, struct fortran_str_arg *s3);
struct fortran_str_arg { $intsize len; char *str; };
],[
    int l;
    char s1[[]] = "yes"; char s2[[]] = ""; char s3[[]] = "confirm"; 
    struct fortran_str_arg a1 = { ($intsize)(sizeof(s1) - 1), s1 },
     a2 = { ($intsize)(sizeof(s2)-1), s2 },
     a3 = { ($intsize)(sizeof(s3)-1), s3 };
    TESTFUNC(&l,&a1, &a2, &a3);
    return (l == SIDL_$2_TRUE) ? 0 : -1;
])],[AC_RUN_IFELSE(
        [AC_LANG_PROGRAM(struct fortran_str_arg; /* forward declaration */
LLNL_FORTRAN_STRING_TEST_PROLOGUE($2)
void TESTFUNC(SIDL_$2_Bool *l,struct fortran_str_arg *s1, struct fortran_str_arg *s2, struct fortran_str_arg *s3);
struct fortran_str_arg { $intsize len; char *str; };
,[
  int l;
  const $intsize upper = ((($intsize)1) << (sizeof($intsize)*4));
  char s1[[]] = "yes"; char s2[[]] = ""; char s3[[]] = "confirm"; 
  struct fortran_str_arg a1 = { ($intsize)(sizeof(s1) - 1)|upper, s1 },
   a2 = { ($intsize)(sizeof(s2)-1)|upper, s2 },
   a3 = { ($intsize)(sizeof(s3)-1)|upper, s3 };
  TESTFUNC(&l, &a1, &a2, &a3);
  return (l == SIDL_$2_TRUE) ? 0 : -1;
])],,[dnl Setting upper bits causes runtime failure; hence, $intsize is right.
   llnl_cv_$2_string_passing="struct_len_str $intsize"])])
        ])
      ])
    ])
  fi
  done
  LIBS=$ac_save_LIBS
  AC_LANG_POP(C)dnl
  rm -f  conftest* cfortran_test*
  ],
  [AC_MSG_ERROR([unable to compile $2 subroutine])])
  dnl AC_COMPILE_IFELSE
  AC_LANG_POP($1)dnl
  ])
  case "$llnl_cv_$2_string_passing" in
  near*)
    AC_DEFINE(SIDL_$2_STR_LEN_NEAR,,[$2 string lengths immediately follow string])
    ;;
  far*)
    AC_DEFINE(SIDL_$2_STR_LEN_FAR,,[$2 string lengths at end of argument list])
    ;;
  struct_str_len*)
    AC_DEFINE(SIDL_$2_STR_STRUCT_STR_LEN,,[$2 strings as char*-length structs])
    ;;
  struct_len_str*)
    AC_DEFINE(SIDL_$2_STR_STRUCT_LEN_STR,,[$2 strings as length-char* structs])
    ;;
  *)
    AC_MSG_ERROR([unable to determine $2 binary string passing convention])
    ;;
  esac
  case "$llnl_cv_$2_string_passing" in
  *int32_t)
    AC_DEFINE(SIDL_$2_STR_INT_SIZE,int32_t,[$2 string length integer size])
    ;;
  *int64_t)
    AC_DEFINE(SIDL_$2_STR_INT_SIZE,int64_t,[$2 string length integer size])
    ;;
  *)
    AC_MSG_WARN([Guessing $2 string length integer size as int.])
    AC_DEFINE(SIDL_$2_STR_INT_SIZE,int,[$2 string length integer size])
    ;;
  esac
])