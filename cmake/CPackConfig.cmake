# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BUILD_SOURCE_DIRS "/home/javeda2/mrprogs/gadgetron_fork;/home/javeda2/mrprogs/gadgetron_fork/cmake")
set(CPACK_CMAKE_GENERATOR "Ninja")
set(CPACK_COMPONENTS_ALL "Unspecified;main;scripts")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "build-essential, ismrmrd, libfftw3-dev, python-dev, python-numpy, python-psutil, liblapack-dev, libxml2-dev, libxslt-dev, libarmadillo-dev, libace-dev, python-matplotlib, python-libxml2, python-h5py, libboost-all-dev, libhdf5-serial-dev, h5utils, hdf5-tools, libgtest-dev")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS "ON")
set(CPACK_DEB_COMPONENT_INSTALL "OFF")
set(CPACK_DEB_PACKAGE_COMPONENT "OFF")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "/home/javeda2/anaconda3/envs/gadgetron/share/cmake-3.21/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "GADGETRON built using CMake")
set(CPACK_GENERATOR "DEB")
set(CPACK_INSTALL_CMAKE_PROJECTS "/home/javeda2/mrprogs/gadgetron_fork/cmake;GADGETRON;ALL;/")
set(CPACK_INSTALL_PREFIX "/home/javeda2/anaconda3/envs/gadgetron")
set(CPACK_MODULE_PATH "/home/javeda2/mrprogs/gadgetron_fork/cmake")
set(CPACK_NSIS_DISPLAY_NAME "GADGETRON 0.1.1")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
set(CPACK_NSIS_PACKAGE_NAME "GADGETRON 0.1.1")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OUTPUT_CONFIG_FILE "/home/javeda2/mrprogs/gadgetron_fork/cmake/CPackConfig.cmake")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "/home/javeda2/anaconda3/envs/gadgetron/share/cmake-3.21/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GADGETRON built using CMake")
set(CPACK_PACKAGE_FILE_NAME "GADGETRON-0.1.1-Linux")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "GADGETRON 0.1.1")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "GADGETRON 0.1.1")
set(CPACK_PACKAGE_NAME "GADGETRON")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "Humanity")
set(CPACK_PACKAGE_VERSION "0.1.1")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "1")
set(CPACK_PACKAGING_INSTALL_PREFIX "/home/javeda2/anaconda3/envs/gadgetron")
set(CPACK_PROJECT_CONFIG_FILE "/home/javeda2/mrprogs/gadgetron_fork/cmake/cpack_options.cmake")
set(CPACK_RESOURCE_FILE_LICENSE "/home/javeda2/anaconda3/envs/gadgetron/share/cmake-3.21/Templates/CPack.GenericLicense.txt")
set(CPACK_RESOURCE_FILE_README "/home/javeda2/anaconda3/envs/gadgetron/share/cmake-3.21/Templates/CPack.GenericDescription.txt")
set(CPACK_RESOURCE_FILE_WELCOME "/home/javeda2/anaconda3/envs/gadgetron/share/cmake-3.21/Templates/CPack.GenericWelcome.txt")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES ";.git;.gitignore;todo.txt;_clang-format;build/")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/home/javeda2/mrprogs/gadgetron_fork/cmake/CPackSourceConfig.cmake")
set(CPACK_SYSTEM_NAME "Linux")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "Linux")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "/home/javeda2/mrprogs/gadgetron_fork/cmake/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
