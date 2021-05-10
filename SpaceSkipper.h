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

#include "Model.h"
#include <cuda_runtime.h>
#include <chrono>

/*! for reference comparison to Ganter et al; this uses user geometry
    boxes for active 'blocks', with these active-blocks being computed
    on the host */
#define USER_GEOM_METHOD 1

namespace dvr {

  /*! used to measure the average of N different runs (bracketed with
      enter() and leave() calls); of these the first NUM_IGNORE will
      get ignored, the rest will go into an averagge and get
      printed */
  struct ProfilePrinter
  {
    ProfilePrinter(const std::string &name,
                   int NUM_IGNORE=10)
      : name(name),
        NUM_IGNORE(NUM_IGNORE)
    {}

    inline void enter()
    {
      t_last_in = std::chrono::high_resolution_clock::now();
    }
    inline void leave()
    {
      using std::chrono::high_resolution_clock;
      using std::chrono::duration;
      using std::chrono::duration_cast;
      
      high_resolution_clock::time_point t_now
        = high_resolution_clock::now();
      duration<double> timeThisRun
        = duration_cast<duration<double>>(t_now - t_last_in);
      numCalled++;
      if (numCalled > NUM_IGNORE) {
        timeInside += timeThisRun.count();
        int numNotIgnored = numCalled - NUM_IGNORE;
        std::cout << "#" << name << "\t: avg time in (across "
                  << numNotIgnored << " runs) : "
                  << prettyDouble(timeInside/numNotIgnored)
                  << std::endl;
      }
    }
    int numCalled = 0;
    std::chrono::high_resolution_clock::time_point t_last_in;
    double timeInside = 0.f;
    const std::string name;
    const int NUM_IGNORE;
  };
  
  struct SpaceSkipper {

    template<typename voxel_t>
    void computeMCs(OWLContext owl,
                    OWLModule module,
                    const voxel_t *voxels,
                    /*! number of voxels in the volume */
                    const vec3i &numVoxels,
                    /*! number of volume cells per macros cell side*/
                    int mcSize=32);
    
    OWLGroup computeAccel(const float4 *colorMap,
                          int colorMapSize,
                          const range1f &colorMapRange);

  private:
    /*! host side code to compute intiail macro cells */
    template<typename voxel_t>
    std::vector<range1f> computeMCRanges(const voxel_t *voxels);
    void genTriangles();
    
    OWLBuffer indexBuffer = 0;
    OWLBuffer vertexBuffer = 0;
    /*! NxNxN array of bools, saying which of the (macro) cells is active */
    bool     *d_mcActive = 0;
    range1f  *d_mcRanges = 0;
    
    /*! used as an atomic by triangle 'isosurface' extractor, to
        decide where in the buffer to write the next triangle */
    int      *d_numTriangles = 0;
    int       numTriangles;
    int       numVertices;
    int       mcSize = 0;
    vec3i     numVoxels = 0;

    OWLContext owl;
    OWLModule  module;
    OWLGeom    ssGeom = 0;
    OWLGroup   blas;
    OWLGroup   ias = 0;

#if USER_GEOM_METHOD
    struct KDTreeNode {
      box3i   region;
      int     child;
    };
    std::vector<KDTreeNode> kdTreeNodes;
    
    void buildKDTree(const std::vector<range1f> &mcRanges);
    void buildKDTree(int nodeID,
                     const std::vector<range1f> &mcRanges,
                     const box3i &region);
    void rebuildActiveBoxes();
    bool extractActiveBoxes(const bool *h_mcActive,
                            int nodeID);
    void addActiveBox(const box3i &mcRange);
    std::vector<box3f> activeBoxes;
    OWLBuffer activeBoxesBuffer = 0;
#else
#endif
    
    vec3i numCells = 0;
    vec3i numMCs = 0;
  };
    
}
