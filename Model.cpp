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

#include "Model.h"
#include <fstream>

namespace dvr {

  Model::~Model()
  {
    free(volumeData);
  }

  void createTestCase(Model::SP model, const vec3i &N)
  {
    model->volumeDims = N;
    const vec3f center = .5f*vec3f(model->volumeDims);
    float *values = (float*)malloc(owl::volume(model->volumeDims)*sizeof(float));
    owl::parallel_for(model->volumeDims.z,[&](int iz){
        for (int iy=0;iy<model->volumeDims.y;iy++)
          for (int ix=0;ix<model->volumeDims.x;ix++) {
            int ii
              = ix 
              + iy * model->volumeDims.x
              + iz * model->volumeDims.x * model->volumeDims.y;
            float unitDist
              = length(vec3f(ix,iy,iz)+vec3f(.5f) - center)
              / (.5f*length(vec3f(model->volumeDims)));
            
            float value = cosf(35.f*unitDist);
            // float value = cosf(53.f*sqrtf(unitDist));
            values[ii] = .5f*(1.f+value);
            model->valueRange.extend(value);
          }
      });
    model->normalizedValueRange = model->valueRange;
    model->volumeData  = values;
    model->elementType = OWL_FLOAT;
  }

  void loadRAW_float(Model::SP model,
                     const std::string &fileName,
                     const vec3i &dims)
  {
    size_t N = dims.x*size_t(dims.y)*dims.z;
    
    float *voxels = new float[N];
    std::ifstream in(fileName,std::ios::binary);
    in.read((char *)voxels,N*sizeof(float));
    for (size_t i=0;i<N;i++) {

      // voxels[i] = clamp(voxels[i],0.f,1.f);
      
      model->valueRange.extend(voxels[i]);
    }

#if 0
    std::cout << "MANUALLY NORMALIZING THE MODEL TO [0,1] VALUE RANGE!" << std::endl;
    for (size_t i=0;i<N;i++) 
      voxels[i]
        = (voxels[i]-model->valueRange.lower)
        / (model->valueRange.upper - model->valueRange.lower);
    model->valueRange = { 0.f, 1.f };
#endif
    
    model->domain.lower = 0.5f;
    model->domain.upper = vec3f(dims)-0.5f;
    model->volumeDims = dims;
    model->volumeData = voxels;
    model->elementType = OWL_FLOAT;
    model->normalizedValueRange = model->valueRange;
  }

  void loadRAW_uchar(Model::SP model,
                     const std::string &fileName,
                     const vec3i &dims)
  {
    size_t N = dims.x*size_t(dims.y)*dims.z;
    uint8_t *input = new uint8_t[N];
    std::ifstream in(fileName,std::ios::binary);
    in.read((char *)input,N*sizeof(*input));
    
    for (size_t i=0;i<N;i++) {
      model->valueRange.extend(input[i]/255.f);
    }
    
    model->domain.lower = 0.5f;
    model->domain.upper = vec3f(dims)-0.5f;
    model->volumeDims = dims;
    model->volumeData = input;
    model->elementType = OWL_UCHAR; 
    model->normalizedValueRange = model->valueRange;
    // model->normalizedValueRange.lower *= 1.f/255.f;
    // model->normalizedValueRange.upper *= 1.f/255.f;
  }
  
  void loadSubBrick_float(Model::SP model,
                          const std::string &fileName,
                          const vec3i &fullSize,
                          const vec3i &subBrickID,
                          int subBrickSize)
  {
    vec3i idx_begin = subBrickID*subBrickSize;
    vec3i idx_end = min(idx_begin + subBrickSize + 1, fullSize);

    vec3i ourSize = idx_end - idx_begin;
    size_t N = ourSize.x*size_t(ourSize.y)*ourSize.z;
    
    float *voxels = new float[N];
    std::ifstream in(fileName,std::ios::binary);
    float *readPtr = voxels;
    for (int iz=idx_begin.z;iz<idx_end.z;iz++)
      for (int iy=idx_begin.y;iy<idx_end.y;iy++) {
        size_t ofs = sizeof(float) * (size_t(iz)*fullSize.x*fullSize.y+
                                      size_t(iy)*fullSize.x);
        in.seekg(ofs);
        int Nx = (idx_end.x-idx_begin.x);
        in.read((char *)readPtr,Nx*sizeof(float));

        for (size_t i=0;i<Nx;i++) 
          model->valueRange.extend(readPtr[i]);
        
        readPtr += Nx;
      }

    model->domain.lower  = 0.5f + vec3f(idx_begin);
    model->domain.upper  = vec3f(idx_end)-0.5f;
    model->volumeDims    = ourSize;
    model->volumeData    = voxels;
    model->elementType   = OWL_FLOAT;
    model->normalizedValueRange = model->valueRange;
    // model->normalizedValueRange.lower *= 1.f/255.f;
    // model->normalizedValueRange.upper *= 1.f/255.f;
  }
  

  void loadSubBrick_uchar(Model::SP model,
                          const std::string &fileName,
                          const vec3i &fullSize,
                          const vec3i &subBrickID,
                          int subBrickSize)
  {
    vec3i idx_begin = subBrickID*subBrickSize;
    vec3i idx_end   = min(idx_begin + subBrickSize + 1, fullSize);

    vec3i ourSize = idx_end - idx_begin;
    size_t N = ourSize.x*size_t(ourSize.y)*ourSize.z;
    
    float *voxels = new float[N];
    std::ifstream in(fileName,std::ios::binary);
    int Nx = (idx_end.x-idx_begin.x);
    std::vector<uint8_t> readBuf(Nx);
    
    float *readPtr = voxels;
    for (int iz=idx_begin.z;iz<idx_end.z;iz++)
      for (int iy=idx_begin.y;iy<idx_end.y;iy++) {
        size_t ofs = sizeof(uint8_t) * (size_t(iz)*fullSize.x*fullSize.y+
                                        size_t(iy)*fullSize.x);
        in.seekg(ofs);
        in.read((char *)readBuf.data(),Nx*sizeof(uint8_t));
        
        for (size_t i=0;i<Nx;i++) {
          readPtr[i] = readBuf[i] / 255.f;
          model->valueRange.extend(readPtr[i]);
        }
        
        readPtr += Nx;
      }
    
    model->domain.lower  = 0.5f + vec3f(idx_begin);
    model->domain.upper  = vec3f(idx_end)-0.5f;
    model->volumeDims    = ourSize;
    model->volumeData    = voxels;
    model->elementType   = OWL_FLOAT;
    model->normalizedValueRange = model->valueRange;
    model->normalizedValueRange.lower *= 1.f/255.f;
    model->normalizedValueRange.upper *= 1.f/255.f;
  }
  

  Model::SP Model::load(const std::string &fileName,
                        const vec3i &dims,
                        const std::string &formatString,
                        const vec3i &subBrickID,
                        int subBrickSize)
  {
    Model::SP model = std::make_shared<Model>();
    if (fileName == "test")
      createTestCase(model,dims);
    else if (subBrickSize > 0) {
      if (formatString == "float")
        loadSubBrick_float(model,fileName,dims,
                           subBrickID,
                           subBrickSize);
      else 
        loadSubBrick_uchar(model,fileName,dims,
                           subBrickID,
                           subBrickSize);
    } else if (formatString == "float") {
      loadRAW_float(model,fileName,dims);
    } else if (formatString == "byte" || formatString == "uchar") {
      loadRAW_uchar(model,fileName,dims);
    } else
      throw std::runtime_error("unknown voxel format '"+formatString+"'");
    return model;
  }

}
