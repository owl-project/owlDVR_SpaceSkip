// ======================================================================== //
// Copyright 2020-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "owl/owl.h"
#include "Renderer.h"
#include <cuda_runtime.h>

#define SS_SINGLE_RAY 1

namespace dvr {
  
  struct RayGen {
    /* we don't need any special parameters to raygen program; we pass
       everythign through lauch params, whcih is faster/more
       efficient */
  };

  /*! space-skipping geometry - we don't actually need anything, all
      we need is the ray distance in the CH program */
  struct SSGeom
  {
#if USER_GEOM_METHOD
    box3f *activeBoxes;
#else
    vec3f *vertices;
    vec3i *indices;
#endif
  };
  struct SSPRD {
    int primID; // only for debugging/visualizing
    vec3f Ng; // only for debugging/visualizing
#if USER_GEOM_METHOD
    float t0, t1;
# else    
    float t;
# if SS_SINGLE_RAY
    bool backFace;
# endif
#endif
  };

  struct LaunchParams
  {
    uint32_t *fbPointer;
    float4   *accumBuffer;
    int       accumID;
#ifdef DUMP_FRAMES
    // to allow dumping rgba and depth for some unrelated compositing work....
    float    *fbDepth;
#endif
    struct {
      vec3f org;
      vec3f dir_00;
      vec3f dir_du;
      vec3f dir_dv;
    } camera;
    struct {
      vec3i               dims;
      cudaTextureObject_t texture;
      box3f               domain;
    } volume;
    struct {
      cudaTextureObject_t texture;
      range1f             domain;
      float               opacityScale;
    } transferFunc;
    struct {
      float dt;
      vec3f gradientDelta;
      
      /*! space skip geometry (actually a group, not a geom, but oh
       well...) */
      OptixTraversableHandle ssGeom;
      int   showBoxes;
      int   heatMapEnabled;
      float heatMapScale;
      int   spp;
      int   spaceSkipMode;
    } render;
  };
  
}
