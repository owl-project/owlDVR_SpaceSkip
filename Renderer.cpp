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

#include "Renderer.h"
#include "qtOWL/ColorMaps.h"
#include "deviceCode.h"
#include "owl/helper/cuda.h"

namespace dvr {
  extern "C" char deviceCode_ptx[];

  bool  Renderer::heatMapEnabled = false;
  float Renderer::heatMapScale = 1e-5f;
  int   Renderer::spp = 1;
  
  OWLVarDecl rayGenVars[]
  = {
     { nullptr /* sentinel to mark end of list */ }
  };

  OWLVarDecl launchParamsVars[]
  = {
     { "fbPointer",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,fbPointer) },
     { "accumBuffer",   OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,accumBuffer) },
     { "accumID",   OWL_INT, OWL_OFFSETOF(LaunchParams,accumID) },
#ifdef DUMP_FRAMES
    // to allow dumping rgba and depth for some unrelated compositing work....
     { "fbDepth",     OWL_BUFPTR,      OWL_OFFSETOF(LaunchParams,fbDepth) },
#endif
     // volume data
     { "volume.dims", OWL_INT3, OWL_OFFSETOF(LaunchParams,volume.dims) },
     { "volume.domain.lower", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,volume.domain.lower) },
     { "volume.domain.upper", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,volume.domain.upper) },
     { "volume.texture", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams,volume.texture) },
     // xf data
     { "transferFunc.domain",OWL_FLOAT2, OWL_OFFSETOF(LaunchParams,transferFunc.domain) },
     { "transferFunc.texture",   OWL_USER_TYPE(cudaTextureObject_t),OWL_OFFSETOF(LaunchParams,transferFunc.texture) },
     { "transferFunc.opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,transferFunc.opacityScale) },
     // render settings
     { "render.dt",            OWL_FLOAT, OWL_OFFSETOF(LaunchParams,render.dt) },
     { "render.spp",           OWL_INT,   OWL_OFFSETOF(LaunchParams,render.spp) },
     { "render.showBoxes",     OWL_INT,   OWL_OFFSETOF(LaunchParams,render.showBoxes) },
     { "render.gradientDelta", OWL_FLOAT3,OWL_OFFSETOF(LaunchParams,render.gradientDelta) },
     { "render.ssGeom", OWL_GROUP, OWL_OFFSETOF(LaunchParams,render.ssGeom) },
     { "render.heatMapEnabled", OWL_INT, OWL_OFFSETOF(LaunchParams,render.heatMapEnabled) },
     { "render.heatMapScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,render.heatMapScale) },
     { "render.spaceSkipMode", OWL_INT, OWL_OFFSETOF(LaunchParams,render.spaceSkipMode) },
     // camera settings
     { "camera.org",    OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.org) },
     { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_00) },
     { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_du) },
     { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_dv) },
     { nullptr /* sentinel to mark end of list */ }
  };
  
  Renderer::Renderer(Model::SP model)
    : model(model),
      xfDomain(model->valueRange)
  {
    owl = owlContextCreate(nullptr,1);
    module = owlModuleCreate(owl,deviceCode_ptx);
    rayGen = owlRayGenCreate(owl,module,"renderFrame",
                             sizeof(RayGen),rayGenVars,-1);
    lp = owlParamsCreate(owl,sizeof(LaunchParams),launchParamsVars,-1);

    owlParamsSet3i(lp,"volume.dims",
                   model->volumeDims.x,
                   model->volumeDims.y,
                   model->volumeDims.z);
    owlParamsSet3f(lp,"render.gradientDelta",
                   1.f/model->volumeDims.x,
                   1.f/model->volumeDims.y,
                   1.f/model->volumeDims.z);

    owlParamsSet3f(lp,"volume.domain.lower",
                   model->domain.lower.x,
                   model->domain.lower.y,
                   model->domain.lower.z);
    owlParamsSet3f(lp,"volume.domain.upper",
                   model->domain.upper.x,
                   model->domain.upper.y,
                   model->domain.upper.z);
    // ------------------------------------------------------------------
    // create *volume* texture:
    // ------------------------------------------------------------------
    cudaTextureObject_t volumeTexture;
    cudaResourceDesc res_desc = {};
    cudaChannelFormatDesc channel_desc
      = model->elementType == OWL_FLOAT
      ? cudaCreateChannelDesc<float>()
      : cudaCreateChannelDesc<uint8_t>();

    if (channel_desc.f ==  cudaChannelFormatKindFloat)
      std::cout << "IS A FLOAT TEXTURE" << std::endl;
    PRINT(model->volumeDims);
    // texture<float, 3, cudaReadModeElementType> 
    cudaArray_t   voxelArray;
    CUDA_CALL(Malloc3DArray(&voxelArray,
                            &channel_desc,
                            make_cudaExtent(model->volumeDims.x,
                                            model->volumeDims.y,
                                            model->volumeDims.z)));
    
    cudaMemcpy3DParms copyParams = {0};
    cudaExtent volumeSize = make_cudaExtent(model->volumeDims.x,
                                            model->volumeDims.y,
                                            model->volumeDims.z);
    copyParams.srcPtr
      = make_cudaPitchedPtr((void *)model->volumeData,
                            volumeSize.width
                            * (model->elementType == OWL_FLOAT
                               ? sizeof(float)
                               : sizeof(uint8_t)),
                            volumeSize.width,
                            volumeSize.height);
    copyParams.dstArray = voxelArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    CUDA_CALL(Memcpy3D(&copyParams));
    
    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));
    
    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = voxelArray;
    
    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));
    
    texDescr.normalizedCoords = true; // access with normalized texture coordinates
    texDescr.filterMode       = cudaFilterModeLinear; // linear interpolation
    // wrap texture coordinates
    texDescr.addressMode[0] = cudaAddressModeClamp;//Wrap;
    texDescr.addressMode[1] = cudaAddressModeClamp;//Wrap;
    texDescr.addressMode[2] = cudaAddressModeClamp;//Wrap;
    texDescr.sRGB                = 0;

    // texDescr.addressMode[0]      = cudaAddressModeBorder;
    // texDescr.addressMode[1]      = cudaAddressModeBorder;
    texDescr.filterMode          = cudaFilterModeLinear;
    texDescr.normalizedCoords    = 1;
    texDescr.maxAnisotropy       = 1;
    texDescr.maxMipmapLevelClamp = 0;
    texDescr.minMipmapLevelClamp = 0;
    texDescr.mipmapFilterMode    = cudaFilterModePoint;
    texDescr.borderColor[0]      = 0.0f;
    texDescr.borderColor[1]      = 0.0f;
    texDescr.borderColor[2]      = 0.0f;
    texDescr.borderColor[3]      = 0.0f;
    texDescr.sRGB                = 0;
    
    texDescr.readMode 
      = model->elementType == OWL_FLOAT
      ? cudaReadModeElementType
      : cudaReadModeNormalizedFloat;
    if (model->elementType == OWL_FLOAT) {
      texDescr.readMode = cudaReadModeElementType;
      std::cout << "USING READ MODE ELEMENT TYPE" << std::endl;
    }
    
    CUDA_CALL(CreateTextureObject(&volumeTexture, &texRes, &texDescr, NULL));
    owlParamsSetRaw(lp,"volume.texture",&volumeTexture);

    // ------------------------------------------------------------------
    // compute macrocellss - do this before setting color map, since
    // color map needs rebuiding accel
    // ------------------------------------------------------------------
    if (model->elementType == OWL_FLOAT)
      ss.computeMCs(owl,module,(const float *)model->volumeData,
                    model->volumeDims);
    else if (model->elementType == OWL_UCHAR)
      ss.computeMCs(owl,module,(const uint8_t *)model->volumeData,
                    model->volumeDims);
    else throw std::runtime_error
           ("unsupported volume type in macrocell generation");
    
    
    // ------------------------------------------------------------------
    // transfer function
    // ------------------------------------------------------------------

    colorMap = qtOWL::ColorMapLibrary().getMap(0);
    setColorMap(colorMap);
    setOpacityScale(1.f);
    owlParamsSet2f(lp,"transferFunc.domain",
                   model->valueRange.lower,
                   model->valueRange.upper);
    

#ifdef DUMP_FRAMES
    fbDepth = owlDeviceBufferCreate(owl,OWL_FLOAT,1,nullptr);
    fbSize  = vec2i(1);
    owlParamsSetBuffer(lp,"fbDepth",fbDepth);
#endif

    setShowBoxesMode(false);
    
    owlBuildPrograms(owl);
    owlBuildPipeline(owl);
    owlBuildSBT(owl);
  }

  void Renderer::setShowBoxesMode(bool flag)
  {
    showBoxes = flag;
    PRINT(showBoxes);
    owlParamsSet1i(lp,"render.showBoxes",flag);
  }
  
  void Renderer::setColorMap(const std::vector<vec4f> &newCM)
  {
    static ProfilePrinter fullRebuild("full rebuild");
    fullRebuild.enter();
      
    this->colorMap = newCM;
    if (!colorMapBuffer)
      colorMapBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT4,
                                             newCM.size(),nullptr);
    owlBufferUpload(colorMapBuffer,newCM.data());
    
    if (colorMapTexture != 0) {
      CUDA_CALL(DestroyTextureObject(colorMapTexture));
      colorMapTexture = 0;
    }

    cudaResourceDesc res_desc = {};
    cudaChannelFormatDesc channel_desc
      = cudaCreateChannelDesc<float4>();
    
    // cudaArray_t   voxelArray;
    if (colorMapArray == 0) {
      CUDA_CALL(MallocArray(&colorMapArray,
                            &channel_desc,
                            newCM.size(),1));
    }
    
    int pitch = newCM.size()*sizeof(newCM[0]);
    CUDA_CALL(Memcpy2DToArray(colorMapArray,
                              /* offset */0,0,
                              newCM.data(),
                              pitch,pitch,1,
                              cudaMemcpyHostToDevice));
    
    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = colorMapArray;
    
    cudaTextureDesc tex_desc     = {};
    tex_desc.addressMode[0]      = cudaAddressModeBorder;
    tex_desc.addressMode[1]      = cudaAddressModeBorder;
    tex_desc.filterMode          = cudaFilterModeLinear;
    tex_desc.normalizedCoords    = 1;
    tex_desc.maxAnisotropy       = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode    = cudaFilterModePoint;
    tex_desc.borderColor[0]      = 0.0f;
    tex_desc.borderColor[1]      = 0.0f;
    tex_desc.borderColor[2]      = 0.0f;
    tex_desc.borderColor[3]      = 0.0f;
    tex_desc.sRGB                = 0;
    CUDA_CALL(CreateTextureObject(&colorMapTexture, &res_desc, &tex_desc,
                                  nullptr));

    rebuildSpaceSkipAccel();
    
    fullRebuild.leave();
  }
  
  void Renderer::rebuildSpaceSkipAccel()
  {
    OWLGroup accel =
      ss.computeAccel((const float4*)owlBufferGetPointer(colorMapBuffer,0),
                      colorMap.size(),
                      xfDomain);
    owlParamsSetGroup(lp,"render.ssGeom",accel);
    // OWLTexture xfTexture
    //   = owlTexture2DCreate(owl,OWL_TEXEL_FORMAT_RGBA32F,
    //                        colorMap.size(),1,
    //                        colorMap.data());
    owlParamsSetRaw(lp,"transferFunc.texture",&colorMapTexture);
#if USER_GEOM_METHOD
    owlBuildSBT(owl);
#endif
  }

  void Renderer::setRange(interval<float> xfDomain)
  {
    this->xfDomain = xfDomain;
    PING; PRINT(xfDomain);
    owlParamsSet2f(lp,"transferFunc.domain",xfDomain.lo,xfDomain.hi);
    rebuildSpaceSkipAccel();
  }

  void Renderer::setOpacityScale(float scale)
  {
    owlParamsSet1f(lp,"transferFunc.opacityScale",scale);
  }
  


  void Renderer::set_dt(float dt)
  {
    PRINT(dt);
    owlParamsSet1f(lp,"render.dt",dt);
  }
  
  void Renderer::setCamera(const vec3f &org,
                           const vec3f &dir_00,
                           const vec3f &dir_du,
                           const vec3f &dir_dv)
  {
    owlParamsSet3f(lp,"camera.org",   org.x,org.y,org.z);
    owlParamsSet3f(lp,"camera.dir_00",dir_00.x,dir_00.y,dir_00.z);
    owlParamsSet3f(lp,"camera.dir_du",dir_du.x,dir_du.y,dir_du.z);
    owlParamsSet3f(lp,"camera.dir_dv",dir_dv.x,dir_dv.y,dir_dv.z);
  }

  void Renderer::render(const vec2i &fbSize,
                        uint32_t *fbPointer)
  {
    if (fbSize != this->fbSize) {
#ifdef DUMP_FRAMES
      owlBufferResize(fbDepth,fbSize.x*fbSize.y);
#endif
      if (!accumBuffer)
        accumBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT4,1,nullptr);
      owlBufferResize(accumBuffer,fbSize.x*fbSize.y);
      owlParamsSetBuffer(lp,"accumBuffer",accumBuffer);
      this->fbSize = fbSize;
    }
    owlParamsSetPointer(lp,"fbPointer",fbPointer);

    owlParamsSet1i(lp,"accumID",accumID);
    accumID++;
    owlParamsSet1i(lp,"render.spp",max(spp,1));
    owlParamsSet1i(lp,"render.heatMapEnabled",heatMapEnabled);
    owlParamsSet1f(lp,"render.heatMapScale",heatMapScale);
    owlParamsSet1i(lp,"render.spaceSkipMode",spaceSkipMode);

    owlLaunch2D(rayGen,fbSize.x,fbSize.y,lp);
  }
                             
}
