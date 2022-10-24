# https://github.com/jocover/jetson-ffmpeg/blob/master/CMakeLists.txt

find_library(LIB_NVJPEG nvjpeg PATHS /usr/lib/aarch64-linux-gnu/tegra)

add_library(nvmpi
  /usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvJpegDecoder.cpp
)

target_link_libraries(nvmpi PRIVATE ${LIB_NVJPEG})
target_include_directories(nvmpi PUBLIC 
  /usr/src/jetson_multimedia_api/include
  /usr/src/jetson_multimedia_api/include/libjpeg-8b
)
