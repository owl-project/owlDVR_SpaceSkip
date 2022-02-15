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

#pragma once

#include "owl/owl.h"
#include "owl/common/math/box.h"
#include "owl/common/parallel/parallel_for.h"
#include <vector>
#include <map>
#include <stdexcept>
#include <string>
#include <iostream>
#include <memory>

namespace dvr {

#define CUDA_CALL OWL_CUDA_CALL
#define CUDA_SYNC_CHECK OWL_CUDA_SYNC_CHECK
  
  using namespace owl;
  using namespace owl::common;
  
  typedef owl::interval<float> range1f;
  
  struct Model {
    typedef std::shared_ptr<Model> SP;
    
    virtual ~Model();
    
    static Model::SP load(const std::string &fileName,
                          const vec3i &dims,
                          const std::string &formatString,
                          const vec3i &subBrickID,
                          int subBrickSize);
    
    box3f getBounds() const
    {
      return domain;
    }

    /*! the domain over which this volume is defined - for a regular
     *  (ie, non-sub-brick) volume,this is usually { 0.5f .. dims-.5f
     *  }, for sub-bricks is specifies the world-space coodinates in
     *  the [0..fullModelDims] space that this subset is defined
     *  for */
    box3f        domain;
    OWLDataType  elementType;
    vec3i        volumeDims;
    void        *volumeData;
    range1f      valueRange;
    /*! valuerange when using normalized flaot values (for uchar) or floats */
    range1f      normalizedValueRange;
  };
  
}
