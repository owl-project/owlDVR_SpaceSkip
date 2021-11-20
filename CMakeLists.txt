# ======================================================================== #
# Copyright 2019-2020 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
if (POLICY CMP0071)
  cmake_policy(SET CMP0071 NEW)
endif()
project(Optix-OWL VERSION 1.0.4 LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 2.8)

option(DUMP_FRAMES "DUMP_FRAMES" OFF)

set(owl_dir ${PROJECT_SOURCE_DIR}/submodules/owl)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/cmake/")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add_definitions(${OWL_CXX_FLAGS})

include_directories(${OWL_INCLUDES})

add_subdirectory(submodules/cuteeOWL EXCLUDE_FROM_ALL)

if (DUMP_FRAMES)
  add_definitions(-DDUMP_FRAMES=1)
endif()

include_directories(${QT_OWL_INCLUDES})

# cuda_compile_and_embed(deviceCode_ptx deviceCode.cu)
embed_ptx(
  OUTPUT_TARGET
    deviceCode_ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.cu
)
set_property(SOURCE ${deviceCode_ptx} PROPERTY SKIP_AUTOMOC ON)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${OWL_CXX_FLAGS}")
find_package(Qt5Widgets REQUIRED)
set(CMAKE_AUTOMOC ON)
add_executable(owlDVRViewer
  viewer.cpp
  Model.cpp
  Renderer.cpp
  SpaceSkipper.cu
  )

target_link_libraries(owlDVRViewer
    PRIVATE 
    owl::owl 
    deviceCode_ptx
    qtOWL
  )
#QT_OWL_LINK(owlDVRViewer)

# not needed for public release
#add_executable(dvrSubSampleTool
#  dvrSubSample.cpp
#  )
