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

cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

project(Optix-OWL VERSION 1.0.4 LANGUAGES C CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(submodules/owl EXCLUDE_FROM_ALL)
add_subdirectory(submodules/cuteeOWL EXCLUDE_FROM_ALL)

include(embed_ptx)

option(DUMP_FRAMES "DUMP_FRAMES" OFF)
if (DUMP_FRAMES)
  add_definitions(-DDUMP_FRAMES=1)
endif()

embed_ptx(
  OUTPUT_TARGET
    deviceCode_ptx
  EMBEDDED_SYMBOL_NAMES
    deviceCode_ptx
  PTX_LINK_LIBRARIES
    qtOWL
  SOURCES
    deviceCode.cu
)

add_executable(owlDVRViewer
  viewer.cpp
  Model.cpp
  Renderer.cpp
  SpaceSkipper.cu
)

target_link_libraries(owlDVRViewer PRIVATE qtOWL deviceCode_ptx)

# not needed for public release
#add_executable(dvrSubSampleTool
#  dvrSubSample.cpp
#  )
