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
if (DUMP_FRAMES)
  add_definitions(-DDUMP_FRAMES=1)
endif()
set_property(SOURCE ${deviceCode_ptx} PROPERTY SKIP_AUTOMOC ON)
#target_include_directories(deviceCode_ptx PUBLIC ${PROJECT_SOURCE_DIR}/submodules/cuteeOWL/)



set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/cmake/")

#set(owl_dir ${PROJECT_SOURCE_DIR}/submodules/owl)
add_subdirectory(${PROJECT_SOURCE_DIR}/submodules/owl EXCLUDE_FROM_ALL)

embed_ptx(
  OUTPUT_TARGET
    deviceCode_ptx
  PTX_LINK_LIBRARIES
    owl::owl
#    qtOWL
  SOURCES
    deviceCode.cu
)

set(CMAKE_AUTOMOC ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Qt5Widgets REQUIRED)
get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
string(REPLACE "-fPIC" "" new_core_options ${core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)



add_subdirectory(submodules/cuteeOWL EXCLUDE_FROM_ALL)

include_directories(${QT_OWL_INCLUDES})
include_directories(${PROJECT_SOURCE_DIR}/submodules/cuteeOWL/)

add_executable(owlDVRViewer
  viewer.cpp
  Model.cpp
  Renderer.cpp
  SpaceSkipper.cu
  )
set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
set_property(TARGET owlDVRViewer PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)

target_link_libraries(owlDVRViewer
    PRIVATE 
    owl::owl 
    deviceCode_ptx
    qtOWL
  )
