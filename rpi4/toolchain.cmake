# the name of the target operating system
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm64)
set(TRIPLE aarch64-linux-gnu)

# which C and C++ compiler to use
set(CMAKE_C_COMPILER ${TRIPLE}-gcc)
set(CMAKE_CXX_COMPILER ${TRIPLE}-g++)

# location of the target environment
set(CMAKE_SYSROOT /var/lib/schroot/chroots/rpi-bullseye-arm64)

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${CMAKE_SYSROOT}/usr/lib/pkgconfig:${CMAKE_SYSROOT}/usr/lib/${TRIPLE}/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig")
set(ENV{PKG_CONFIG_SYSROOT_DIR} ${CMAKE_SYSROOT})

# adjust the default behavior of the FIND_XXX() commands:
# search for headers and libraries in the target environment,
# search for programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)