#! /usr/bin/env python
# Build file for Python modules
import sys
from re import compile
from distutils.core import setup, Extension

inc_re = compile('^--include-dirs=(.*)$')
lib_re = compile('^--library-dirs=(.*)$')
exlib_re = compile('^--extra-library=(.*)$')
old_argv = sys.argv
sys.argv = []
inc_dirs = ['.']
lib_dirs = []
libs = ['sidl']

for i in old_argv:
  m = inc_re.match(i)
  if (m):
    if (len(m.group(1))): inc_dirs.append(m.group(1))
  else:
    m = lib_re.match(i)
    if (m):
      if (len(m.group(1))): lib_dirs.append(m.group(1))
    else:
      m = exlib_re.match(i)
      if (m):
        if (len(m.group(1))): libs.append(m.group(1))
      else:
        sys.argv.append(i)
setup(name='babel',
  include_dirs=inc_dirs,
  headers = [
    'sidl_BaseClass_Module.h',
    'sidl_BaseException_Module.h',
    'sidl_BaseInterface_Module.h',
    'sidl_ClassInfoI_Module.h',
    'sidl_ClassInfo_Module.h',
    'sidl_DFinder_Module.h',
    'sidl_DLL_Module.h',
    'sidl_Finder_Module.h',
    'sidl_InvViolation_Module.h',
    'sidl_Loader_Module.h',
    'sidl_PostViolation_Module.h',
    'sidl_PreViolation_Module.h',
    'sidl_SIDLException_Module.h',
    'sidl_io_Deserializer_Module.h',
    'sidl_io_IOException_Module.h',
    'sidl_io_Serializeable_Module.h',
    'sidl_io_Serializer_Module.h',
    'sidl_rmi_ConnectRegistry_Module.h',
    'sidl_rmi_InstanceHandle_Module.h',
    'sidl_rmi_InstanceRegistry_Module.h',
    'sidl_rmi_Invocation_Module.h',
    'sidl_rmi_NetworkException_Module.h',
    'sidl_rmi_ProtocolFactory_Module.h',
    'sidl_rmi_Response_Module.h'
  ],
  packages = [
    'sidl',
    'sidl.io',
    'sidl.rmi'
  ],
  ext_modules = [
    Extension('sidl.DFinder',
      ["sidl/sidl_DFinder_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.ClassInfoI',
      ["sidl/sidl_ClassInfoI_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.SIDLException',
      ["sidl/sidl_SIDLException_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.Finder',
      ["sidl/sidl_Finder_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.rmi.InstanceRegistry',
      ["sidl/rmi/sidl_rmi_InstanceRegistry_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.InvViolation',
      ["sidl/sidl_InvViolation_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.rmi.ConnectRegistry',
      ["sidl/rmi/sidl_rmi_ConnectRegistry_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.io.Deserializer',
      ["sidl/io/sidl_io_Deserializer_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.rmi.Invocation',
      ["sidl/rmi/sidl_rmi_Invocation_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.io.Serializeable',
      ["sidl/io/sidl_io_Serializeable_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.rmi.NetworkException',
      ["sidl/rmi/sidl_rmi_NetworkException_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.PreViolation',
      ["sidl/sidl_PreViolation_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.Loader',
      ["sidl/sidl_Loader_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.io.Serializer',
      ["sidl/io/sidl_io_Serializer_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.BaseException',
      ["sidl/sidl_BaseException_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.BaseClass',
      ["sidl/sidl_BaseClass_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.rmi.ProtocolFactory',
      ["sidl/rmi/sidl_rmi_ProtocolFactory_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.rmi.InstanceHandle',
      ["sidl/rmi/sidl_rmi_InstanceHandle_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.rmi.Response',
      ["sidl/rmi/sidl_rmi_Response_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.PostViolation',
      ["sidl/sidl_PostViolation_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.DLL',
      ["sidl/sidl_DLL_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.ClassInfo',
      ["sidl/sidl_ClassInfo_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.io.IOException',
      ["sidl/io/sidl_io_IOException_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs),
    Extension('sidl.BaseInterface',
      ["sidl/sidl_BaseInterface_Module.c"
      ],
      library_dirs=lib_dirs,
      libraries=libs)
  ])
