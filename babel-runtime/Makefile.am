##
## File:        Makefile.am 
## Package:     Babel makefiles
## Revision:    $Revision$
## Modified:    $Date$
## Description: automake makefile to build the runtime library
##
## Copyright (c) 2000-2002, The Regents of the University of Calfornia.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the Components Team <components@llnl.gov>
## UCRL-CODE-2002-054
## All rights reserved.
## 
## This file is part of Babel. For more information, see
## http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
## for Our Notice and the LICENSE file for the GNU Lesser General Public
## License.
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License (as published by
## the Free Software Foundation) version 2.1 dated February 1999.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
## conditions of the GNU Lesser General Public License for more details.
## 
## You should have recieved a copy of the GNU Lesser General Public License
## along with this program; if not, write to the Free Software Foundation,
## Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

if WITH_SIDLX
SIDLX_SUBDIR=sidlx
else
SIDLX_SUBDIR=
endif
if SUPPORT_JAVA
JAVA_SUBDIR=java
else
JAVA_SUBDIR=
endif

SUBDIRS = m4 bin config sidl $(JAVA_SUBDIR) $(SIDLX_SUBDIR)
DISTCLEANFILES=babel-libtool
#
# Unfortuntately, Python uses its own makefile system so we have to 
# deal with it explicitly here in the parent directory instead of
# integrating it inside automake.
#

if SUPPORT_CYGWIN
SO = dll
else
SO = so
endif

PYTHONLIB = @PYTHONLIB@
PYTHONISHFILES = python/sidlObjA.h python/sidlObjA.c			\
		 python/sidlPyArrays.h python/sidlPyArrays.c		\
		 python/sidlsetup.py python/__init__.py			\
		 python/sidlBaseException.py

# USE runtime/python/genmakefile.py to GENERATE THIS LIST
# ./configure ; make ; cd runtime/python ; python genmakefile.py
RUNTIME_ONLY_FILES=\
		python/setup.py \
		python/sidl/Resolve.py \
		python/sidl/Scope.py \
		python/sidl/__init__.py \
		python/sidl/io/__init__.py \
		python/sidl/io/sidl_io_Deserializer_Module.c \
		python/sidl/io/sidl_io_IOException_Module.c \
		python/sidl/io/sidl_io_Serializable_Module.c \
		python/sidl/io/sidl_io_Serializer_Module.c \
		python/sidl/rmi/__init__.py \
		python/sidl/rmi/sidl_rmi_BindException_Module.c \
		python/sidl/rmi/sidl_rmi_Call_Module.c \
		python/sidl/rmi/sidl_rmi_ConnectException_Module.c \
		python/sidl/rmi/sidl_rmi_ConnectRegistry_Module.c \
		python/sidl/rmi/sidl_rmi_InstanceHandle_Module.c \
		python/sidl/rmi/sidl_rmi_InstanceRegistry_Module.c \
		python/sidl/rmi/sidl_rmi_Invocation_Module.c \
		python/sidl/rmi/sidl_rmi_MalformedURLException_Module.c \
		python/sidl/rmi/sidl_rmi_NetworkException_Module.c \
		python/sidl/rmi/sidl_rmi_NoRouteToHostException_Module.c \
		python/sidl/rmi/sidl_rmi_NoServerException_Module.c \
		python/sidl/rmi/sidl_rmi_ObjectDoesNotExistException_Module.c \
		python/sidl/rmi/sidl_rmi_ProtocolException_Module.c \
		python/sidl/rmi/sidl_rmi_ProtocolFactory_Module.c \
		python/sidl/rmi/sidl_rmi_Response_Module.c \
		python/sidl/rmi/sidl_rmi_Return_Module.c \
		python/sidl/rmi/sidl_rmi_ServerInfo_Module.c \
		python/sidl/rmi/sidl_rmi_ServerRegistry_Module.c \
		python/sidl/rmi/sidl_rmi_TicketBook_Module.c \
		python/sidl/rmi/sidl_rmi_Ticket_Module.c \
		python/sidl/rmi/sidl_rmi_TimeOutException_Module.c \
		python/sidl/rmi/sidl_rmi_UnexpectedCloseException_Module.c \
		python/sidl/rmi/sidl_rmi_UnknownHostException_Module.c \
		python/sidl/sidl_BaseClass_Module.c \
		python/sidl/sidl_BaseException_Module.c \
		python/sidl/sidl_BaseInterface_Module.c \
		python/sidl/sidl_CastException_Module.c \
		python/sidl/sidl_ClassInfoI_Module.c \
		python/sidl/sidl_ClassInfo_Module.c \
		python/sidl/sidl_DFinder_Module.c \
		python/sidl/sidl_DLL_Module.c \
		python/sidl/sidl_Finder_Module.c \
		python/sidl/sidl_InvViolation_Module.c \
		python/sidl/sidl_LangSpecificException_Module.c \
		python/sidl/sidl_Loader_Module.c \
		python/sidl/sidl_MemoryAllocationException_Module.c \
		python/sidl/sidl_NotImplementedException_Module.c \
		python/sidl/sidl_PostViolation_Module.c \
		python/sidl/sidl_PreViolation_Module.c \
		python/sidl/sidl_RuntimeException_Module.c \
		python/sidl/sidl_SIDLException_Module.c \
		python/sidl_BaseClass_IOR.h \
		python/sidl_BaseClass_Module.h \
		python/sidl_BaseException_IOR.h \
		python/sidl_BaseException_Module.h \
		python/sidl_BaseInterface_IOR.h \
		python/sidl_BaseInterface_Module.h \
		python/sidl_CastException_IOR.h \
		python/sidl_CastException_Module.h \
		python/sidl_ClassInfoI_IOR.h \
		python/sidl_ClassInfoI_Module.h \
		python/sidl_ClassInfo_IOR.h \
		python/sidl_ClassInfo_Module.h \
		python/sidl_DFinder_IOR.h \
		python/sidl_DFinder_Module.h \
		python/sidl_DLL_IOR.h \
		python/sidl_DLL_Module.h \
		python/sidl_Finder_IOR.h \
		python/sidl_Finder_Module.h \
		python/sidl_IOR.h \
		python/sidl_InvViolation_IOR.h \
		python/sidl_InvViolation_Module.h \
		python/sidl_LangSpecificException_IOR.h \
		python/sidl_LangSpecificException_Module.h \
		python/sidl_Loader_IOR.h \
		python/sidl_Loader_Module.h \
		python/sidl_MemoryAllocationException_IOR.h \
		python/sidl_MemoryAllocationException_Module.h \
		python/sidl_NotImplementedException_IOR.h \
		python/sidl_NotImplementedException_Module.h \
		python/sidl_PostViolation_IOR.h \
		python/sidl_PostViolation_Module.h \
		python/sidl_PreViolation_IOR.h \
		python/sidl_PreViolation_Module.h \
		python/sidl_Resolve_IOR.h \
		python/sidl_RuntimeException_IOR.h \
		python/sidl_RuntimeException_Module.h \
		python/sidl_SIDLException_IOR.h \
		python/sidl_SIDLException_Module.h \
		python/sidl_Scope_IOR.h \
		python/sidl_io_Deserializer_IOR.h \
		python/sidl_io_Deserializer_Module.h \
		python/sidl_io_IOException_IOR.h \
		python/sidl_io_IOException_Module.h \
		python/sidl_io_IOR.h \
		python/sidl_io_Serializable_IOR.h \
		python/sidl_io_Serializable_Module.h \
		python/sidl_io_Serializer_IOR.h \
		python/sidl_io_Serializer_Module.h \
		python/sidl_rmi_BindException_IOR.h \
		python/sidl_rmi_BindException_Module.h \
		python/sidl_rmi_Call_IOR.h \
		python/sidl_rmi_Call_Module.h \
		python/sidl_rmi_ConnectException_IOR.h \
		python/sidl_rmi_ConnectException_Module.h \
		python/sidl_rmi_ConnectRegistry_IOR.h \
		python/sidl_rmi_ConnectRegistry_Module.h \
		python/sidl_rmi_IOR.h \
		python/sidl_rmi_InstanceHandle_IOR.h \
		python/sidl_rmi_InstanceHandle_Module.h \
		python/sidl_rmi_InstanceRegistry_IOR.h \
		python/sidl_rmi_InstanceRegistry_Module.h \
		python/sidl_rmi_Invocation_IOR.h \
		python/sidl_rmi_Invocation_Module.h \
		python/sidl_rmi_MalformedURLException_IOR.h \
		python/sidl_rmi_MalformedURLException_Module.h \
		python/sidl_rmi_NetworkException_IOR.h \
		python/sidl_rmi_NetworkException_Module.h \
		python/sidl_rmi_NoRouteToHostException_IOR.h \
		python/sidl_rmi_NoRouteToHostException_Module.h \
		python/sidl_rmi_NoServerException_IOR.h \
		python/sidl_rmi_NoServerException_Module.h \
		python/sidl_rmi_ObjectDoesNotExistException_IOR.h \
		python/sidl_rmi_ObjectDoesNotExistException_Module.h \
		python/sidl_rmi_ProtocolException_IOR.h \
		python/sidl_rmi_ProtocolException_Module.h \
		python/sidl_rmi_ProtocolFactory_IOR.h \
		python/sidl_rmi_ProtocolFactory_Module.h \
		python/sidl_rmi_Response_IOR.h \
		python/sidl_rmi_Response_Module.h \
		python/sidl_rmi_Return_IOR.h \
		python/sidl_rmi_Return_Module.h \
		python/sidl_rmi_ServerInfo_IOR.h \
		python/sidl_rmi_ServerInfo_Module.h \
		python/sidl_rmi_ServerRegistry_IOR.h \
		python/sidl_rmi_ServerRegistry_Module.h \
		python/sidl_rmi_TicketBook_IOR.h \
		python/sidl_rmi_TicketBook_Module.h \
		python/sidl_rmi_Ticket_IOR.h \
		python/sidl_rmi_Ticket_Module.h \
		python/sidl_rmi_TimeOutException_IOR.h \
		python/sidl_rmi_TimeOutException_Module.h \
		python/sidl_rmi_UnexpectedCloseException_IOR.h \
		python/sidl_rmi_UnexpectedCloseException_Module.h \
		python/sidl_rmi_UnknownHostException_IOR.h \
		python/sidl_rmi_UnknownHostException_Module.h \
		python/sidlx/__init__.py \
		python/sidlx/rmi/CallType.py \
		python/sidlx/rmi/__init__.py \
		python/sidlx/rmi/sidlx_rmi_ChildSocket_Module.c \
		python/sidlx/rmi/sidlx_rmi_ClientSocket_Module.c \
		python/sidlx/rmi/sidlx_rmi_Common_Module.c \
		python/sidlx/rmi/sidlx_rmi_GenNetworkException_Module.c \
		python/sidlx/rmi/sidlx_rmi_IPv4Socket_Module.c \
		python/sidlx/rmi/sidlx_rmi_JimEchoServer_Module.c \
		python/sidlx/rmi/sidlx_rmi_NoServerException_Module.c \
		python/sidlx/rmi/sidlx_rmi_ServerSocket_Module.c \
		python/sidlx/rmi/sidlx_rmi_SimCall_Module.c \
		python/sidlx/rmi/sidlx_rmi_SimHandle_Module.c \
		python/sidlx/rmi/sidlx_rmi_SimReturn_Module.c \
		python/sidlx/rmi/sidlx_rmi_SimpleOrb_Module.c \
		python/sidlx/rmi/sidlx_rmi_SimpleServer_Module.c \
		python/sidlx/rmi/sidlx_rmi_SimpleTicketBook_Module.c \
		python/sidlx/rmi/sidlx_rmi_SimpleTicket_Module.c \
		python/sidlx/rmi/sidlx_rmi_Simsponse_Module.c \
		python/sidlx/rmi/sidlx_rmi_Simvocation_Module.c \
		python/sidlx/rmi/sidlx_rmi_Socket_Module.c \
		python/sidlx_IOR.h \
		python/sidlx_rmi_CallType_IOR.h \
		python/sidlx_rmi_ChildSocket_IOR.h \
		python/sidlx_rmi_ChildSocket_Module.h \
		python/sidlx_rmi_ClientSocket_IOR.h \
		python/sidlx_rmi_ClientSocket_Module.h \
		python/sidlx_rmi_Common_IOR.h \
		python/sidlx_rmi_Common_Module.h \
		python/sidlx_rmi_GenNetworkException_IOR.h \
		python/sidlx_rmi_GenNetworkException_Module.h \
		python/sidlx_rmi_IOR.h \
		python/sidlx_rmi_IPv4Socket_IOR.h \
		python/sidlx_rmi_IPv4Socket_Module.h \
		python/sidlx_rmi_JimEchoServer_IOR.h \
		python/sidlx_rmi_JimEchoServer_Module.h \
		python/sidlx_rmi_NoServerException_IOR.h \
		python/sidlx_rmi_NoServerException_Module.h \
		python/sidlx_rmi_ServerSocket_IOR.h \
		python/sidlx_rmi_ServerSocket_Module.h \
		python/sidlx_rmi_SimCall_IOR.h \
		python/sidlx_rmi_SimCall_Module.h \
		python/sidlx_rmi_SimHandle_IOR.h \
		python/sidlx_rmi_SimHandle_Module.h \
		python/sidlx_rmi_SimReturn_IOR.h \
		python/sidlx_rmi_SimReturn_Module.h \
		python/sidlx_rmi_SimpleOrb_IOR.h \
		python/sidlx_rmi_SimpleOrb_Module.h \
		python/sidlx_rmi_SimpleServer_IOR.h \
		python/sidlx_rmi_SimpleServer_Module.h \
		python/sidlx_rmi_SimpleTicketBook_IOR.h \
		python/sidlx_rmi_SimpleTicketBook_Module.h \
		python/sidlx_rmi_SimpleTicket_IOR.h \
		python/sidlx_rmi_SimpleTicket_Module.h \
		python/sidlx_rmi_Simsponse_IOR.h \
		python/sidlx_rmi_Simsponse_Module.h \
		python/sidlx_rmi_Simvocation_IOR.h \
		python/sidlx_rmi_Simvocation_Module.h \
		python/sidlx_rmi_Socket_IOR.h \
		python/sidlx_rmi_Socket_Module.h

if BABEL_RUNTIME_ONLY
babelpythonsrc=$(srcdir)
all-local-python: all-local-python2
clean-local-python: clean-local-python2
babel-libtool:
	cp libtool babel-libtool
bin_SCRIPTS=babel-libtool

else
babelpythonsrc=.
all-local-python: all-local-python1
clean-local-python: clean-local-python1
endif

dist-hook: sidl/libsidl.la dist-hook-python dist-hook-patch

if SUPPORT_PYTHON
all-local: all-local-python
clean-local: clean-local-python
runtime_install_data=install-data-am-patch install-data-am-python
runtime_uninstall_data=uninstall-data-am-patch uninstall-data-am-python
else
all-local clean-local:
	@echo "Python not supported"
runtime_install_data=install-data-am-patch
runtime_uninstall_data=uninstall-data-am-patch
endif
install-data-am: $(runtime_install_data)
uninstall-am: $(runtime_uninstall_data)

sidl/libsidl.la:
	@-cd sidl && $(MAKE) libsidl.la

all-local-python1: python/babel-stamp
python/babel-stamp: $(PYTHONISHFILES)
	@-if ! test -d python; then mkdir -p python || exit 1; fi
	@-if ! test -d python/sidl; then mkdir -p python/sidl || exit 1; fi
	@-if test "X$(srcdir)" != "X."; then				\
	  for file in $(PYTHONISHFILES); do				\
	    d=`dirname $$file`;						\
	    test -f $$d							\
	    || mkdir -p $$d						\
	    || exit 1;							\
	    echo cp -p $(srcdir)/$$file $$file;				\
	    cp -p $(srcdir)/$$file $$file || exit 1;			\
	  done;								\
	fi
	@if test -d python; then					\
	  abspath=`cd $(srcdir) && pwd`;				\
	  cd python;							\
	  echo $(SHELL) ../../bin/babel -!		\
	    --generate-sidl-stdlib -cpython				\
	    $$abspath/sidl/sidl.sidl $$abspath/sidlx/sidlx.sidl;	\
	  $(SHELL) ../../bin/babel -!		\
	    --generate-sidl-stdlib -cpython  				\
	    $$abspath/sidl/sidl.sidl $$abspath/sidlx/sidlx.sidl;	\
	  $(PYTHON) sidlsetup.py --library-dirs=../sidl/.libs \
	   --include-dirs=../sidl --include-dirs=../config  $(PYTHON_SETUP_ARGS) build_ext   \
	   --inplace --rpath=$(libdir);					\
	  $(PYTHON) setup.py --library-dirs=../sidl/.libs \
	   --include-dirs=../sidl --include-dirs=../config $(PYTHON_SETUP_ARGS) build_ext 	\
	   --inplace --rpath=$(libdir);							\
	fi
	touch python/babel-stamp

all-local-python2: python/babel-stamp2
python/babel-stamp2:
	@-if ! test -d python; then mkdir -p python || exit 1; fi
	@-if ! test -d python/sidl; then mkdir -p python/sidl || exit 1; fi
	@-if test "X$(srcdir)" != "X."; then				\
	  for file in $(PYTHONISHFILES) $(RUNTIME_ONLY_FILES); do	\
	    d=`dirname $$file`;						\
	    test -f $$d							\
	    || mkdir -p $$d						\
	    || exit 1;							\
	    echo cp -p $(srcdir)/$$file $$file;				\
	    cp -p $(srcdir)/$$file $$file || exit 1;			\
	  done;								\
	fi
	@if test -d python; then					\
	  cd python;							\
	  $(PYTHON) sidlsetup.py --library-dirs=../sidl/.libs \
	   --include-dirs=$(srcdir)/sidl --include-dirs=../sidl \
	   --include-dirs=../config $(PYTHON_SETUP_ARGS)  build_ext   \
	   --inplace --rpath=$(libdir);					\
	  $(PYTHON) setup.py --library-dirs=../sidl/.libs \
	   --include-dirs=$(srcdir)/sidl --include-dirs=../sidl \
	   --include-dirs=../config $(PYTHON_SETUP_ARGS)  build_ext 	\
	   --inplace --rpath=$(libdir);					\
	fi
	touch python/babel-stamp2

clean-local-python1:
	@if test -d python; then					\
          cd python;                                                    \
          echo rm -f sedscript Makefile Makefile.pre Makefile.pre.in    \
            Setup config.c *setup.installed;                            \
          rm -f sedscript Makefile Makefile.pre Makefile.pre.in Setup   \
             config.c *setup.installed;         			\
          echo rm -f babel.make.package babel-stamp *.so *.dll *.a *.o *~; \
	  rm -f babel.make.package babel-stamp *.so *.dll *.a *.o *~; \
	  echo rm -f sidl_*IOR.* sidl_*_Module.h setup.py babel.make ;	\
	  rm -f sidl_*IOR.* sidl_*_Module.h setup.py babel.make ;	\
	  if test -d sidl; then						\
	    cd sidl;							\
	    echo rm -f config.c *.so *.dll *.a *.o *~ *.c *.h __init__.*;\
	    rm -f config.c *.so *.dll *.a *.o *~ *.c *.h __init__.*;	\
	    echo rm -f babel.make ; 					\
	    rm -f babel.make ; 						\
	    if test -d io; then						\
	      cd io;							\
	      rm -f *.so *.dll *.a *.o;					\
	      cd ..;							\
	    fi;								\
	    if test -d rmi; then					\
	      cd rmi;							\
	      rm -f *.so *.dll *.a *.o;					\
	      cd ..;							\
	    fi;								\
	  fi;								\
	fi
	@-if test "X$(srcdir)" != "X."; then				\
	  echo rm -rf python;					\
	  rm -rf python;						\
	else \
	  echo rm -rf python/build; \
	  rm -rf python/build; \
	fi

clean-local-python2:
	@if test -d python; then					\
          cd python;                                                    \
          echo rm -f sedscript Makefile Makefile.pre Makefile.pre.in    \
            Setup config.c *setup.installed;                            \
          rm -f sedscript Makefile Makefile.pre Makefile.pre.in Setup   \
             config.c *setup.installed;         			\
          echo rm -f babel.make.package babel-stamp *.so *.dll *.a *.o *~; \
	  rm -f babel.make.package babel-stamp *.so *.dll *.a *.o *~; \
	  if test -d sidl; then						\
	    cd sidl;							\
	    echo rm -f config.c *.so *.dll *.a *.o *~ babel.make ;	\
	    rm -f config.c *.so *.dll *.a *.o *~ babel.make ;		\
	    if test -d io; then						\
	      cd io;							\
	      rm -f *.so *.dll *.a *.o;					\
	      cd ..;							\
	    fi;								\
	    if test -d rmi; then					\
	      cd rmi;							\
	      rm -f *.so *.dll *.a *.o;					\
	      cd ..;							\
	    fi;								\
	  fi;								\
	fi
	@-if test "X$(srcdir)" != "X."; then				\
	  echo rm -rf python;					\
	  rm -rf python;						\
	else \
	  echo rm -rf python/build; \
	  rm -rf python/build; \
	fi

dist-hook-patch:
	mkdir -p $(DESTDIR)$(distdir)/patches
	@if test -d $(srcdir)/patches; then \
	    cp -pf $(srcdir)/patches/README $(DESTDIR)$(distdir)/patches/; \
	    if test "`echo "$(srcdir)"/patches/*.txt`" != "$(srcdir)/patches/*.txt"; then \
	    	for file in $(srcdir)/patches/*.txt; do \
		    echo cp -p $$file $(DESTDIR)$(distdir)/patches/`basename $$file`; \
		    cp -p $$file $(DESTDIR)$(distdir)/patches/`basename $$file`; \
	   	done; \
	    fi; \
	fi

dist-hook-python: all-local-python
	@for file in $(PYTHONISHFILES); do				\
	  d=`dirname $(distdir)/$$file`;				\
	  test -f $$d							\
	  || mkdir -p $$d						\
	  || exit 1;							\
	  echo cp -p $(srcdir)/$$file $(DESTDIR)$(distdir)/$$file;	\
	  cp -p $(srcdir)/$$file $(DESTDIR)$(distdir)/$$file || exit 1;	\
	done
	@for file in $(RUNTIME_ONLY_FILES); do				\
	  d=`dirname $(distdir)/$$file`;				\
	  test -f $$d							\
	  || mkdir -p $$d						\
	  || exit 1;							\
	  echo cp -p $$file $(DESTDIR)$(distdir)/$$file;		\
	  cp -p $$file $(DESTDIR)$(distdir)/$$file || exit 1;		\
	done

install-data-am-patch:
	@$(NORMAL_INSTALL)
	$(mkinstalldirs) $(DESTDIR)$(datadir)/$(PACKAGE)-$(VERSION)/patches
	@if test "`echo "$(srcdir)"/patches/*.txt`" != "$(srcdir)/patches/*.txt"; then \
	    for f in $(srcdir)/patches/*.txt; do \
		echo $(INSTALL_DATA) $$f $(DESTDIR)$(datadir)/$(PACKAGE)-$(VERSION)/patches/`basename $$f`; \
		$(INSTALL_DATA) $$f $(DESTDIR)$(datadir)/$(PACKAGE)-$(VERSION)/patches/`basename $$f`; \
	    done; \
	fi

install-data-am-python: install-data-am-patch
	test -d python || exit 1
	cd python ;							\
	$(PYTHON) sidlsetup.py --library-dirs=../sidl/.libs \
	 --include-dirs=../$(babelpythonsrc)/sidl --include-dirs=../config      \
	 $(PYTHON_SETUP_ARGS) build_ext --rpath=$(libdir);		\
	$(PYTHON) setup.py --library-dirs=../sidl/.libs \
	 --include-dirs=../$(babelpythonsrc)/sidl --include-dirs=../config 	\
	 $(PYTHON_SETUP_ARGS) build_ext --rpath=$(libdir);		\
	$(PYTHON) sidlsetup.py --library-dirs=../sidl/.libs \
	   --include-dirs=../sidl --include-dirs=../config install 	\
	   $(PYTHON_SETUP_ARGS) 					\
	   --prefix=$(DESTDIR)$(prefix) 				\
	   --exec-prefix=$(DESTDIR)$(exec_prefix)			\
	   --record=sidlsetup.installed;		\
	$(PYTHON) setup.py --library-dirs=../sidl/.libs \
	   --include-dirs=../sidl  --include-dirs=../config install  	\
	   $(PYTHON_SETUP_ARGS) 					\
	   --prefix=$(DESTDIR)$(prefix) 				\
	   --exec-prefix=$(DESTDIR)$(exec_prefix)			\
	   --record=setup.installed

uninstall-data-am-patch:
	@$(NORMAL_UNINSTALL)
	rm -rf $(DESTDIR)$(datadir)/$(PACKAGE)-$(VERSION)/patches

uninstall-data-am-python: uninstall-data-am-patch
	@test -d python && test -f python/setup.installed &&		\
	 rm -f `cat python/setup.installed`
	@test -d python && test -f python/sidlsetup.installed &&	\
	 rm -f `cat python/sidlsetup.installed`
