# Modified from https://github.com/jocover/jetson-ffmpeg/blob/master/CMakeLists.txt

# nvjpeg library
find_library(LIB_NVJPEG nvjpeg PATHS /usr/lib/aarch64-linux-gnu/tegra)

# Compile NvJpegDecoder & NvJpegEncoder
add_library(nvmpi
  /usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvJpegDecoder.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvJpegEncoder.cpp
)

# create a library
target_link_libraries(nvmpi PRIVATE ${LIB_NVJPEG})

# plubic the necessary include
target_include_directories(nvmpi PUBLIC 
  /usr/src/jetson_multimedia_api/include
  /usr/src/jetson_multimedia_api/include/libjpeg-8b
)
