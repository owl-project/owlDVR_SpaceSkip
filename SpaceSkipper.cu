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
#include "owl/helper/cuda.h"
#include "owl/common/arrayND/array3D.h"
#include "deviceCode.h"

namespace dvr {

  __global__ void computeVertices(vec3f *vertexArray,
                                  vec3i numMCVertices,
                                  vec3i numVoxels,
                                  int   mcSize)
  {
    const vec3i vtxIdx
      = vec3i(threadIdx)
      + vec3i(blockIdx)*vec3i(blockDim.x,blockDim.y,blockDim.z);
    if (vtxIdx.x >= numMCVertices.x) return;
    if (vtxIdx.y >= numMCVertices.y) return;
    if (vtxIdx.z >= numMCVertices.z) return;

    const vec3f vtxPos
      = vec3f(min(vtxIdx*mcSize,numVoxels));
    const int vtxID
      = vtxIdx.x
      + vtxIdx.y * numMCVertices.x
      + vtxIdx.z * numMCVertices.x * numMCVertices.y;
    vertexArray[vtxID] = vtxPos;
  }

  inline dim3 to_dims(const vec3i v)
  { return { (unsigned)v.x,(unsigned)v.y,(unsigned)v.z }; }

  template<typename T> __both__ float normalizedTextureValue(T t);
  template<> __both__ float normalizedTextureValue(float f) { return f; }
  template<> __both__ float normalizedTextureValue(uint8_t ui) { return ui/255.f; }
  
    /*! host side code to compute intiail macro cells */
  template<typename voxel_t>
  std::vector<range1f> SpaceSkipper::computeMCRanges(const voxel_t *voxels)
  {
    std::vector<range1f> mcRanges(volume(numMCs));
    owl::array3D::parallel_for
      (numMCs,
       [&](const vec3i mcIdx){

         vec3i begin = mcIdx*mcSize;
         vec3i end = min(begin+mcSize+1,numVoxels);

         range1f range;
         owl::array3D::for_each
           (begin,end,
            [&](const vec3i voxelIdx){
              voxel_t v = voxels[owl::array3D::linear(voxelIdx,numVoxels)];
              float f = normalizedTextureValue(v);
              
              range.extend(f);
            });
         mcRanges[owl::array3D::linear(mcIdx,numMCs)] = range;
       });
    return mcRanges;
  }
  
  template<typename voxel_t>
  void SpaceSkipper::computeMCs(OWLContext owl,
                                OWLModule module,
                                const voxel_t *voxels,
                                /*! number of voxels in the volume */
                                const vec3i &numVoxels,
                                /*! number of volume cells per macros cell side*/
                                int mcSize)
  {
    this->owl = owl;
    this->module = module;
    this->mcSize = mcSize;
    this->numVoxels = numVoxels;
    
    numCells = numVoxels - 1;
    numMCs = divRoundUp(numCells,vec3i(mcSize));
    PRINT(numMCs);
    PRINT(volume(numMCs));
    
    vec3i numMCVertices = numMCs+1;
    
    // ------------------------------------------------------------------
    // allocate - and generate vertices
    // ------------------------------------------------------------------
    int maxNumVertices = volume(numMCVertices);
    vertexBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT3,maxNumVertices,0);
    PRINT(maxNumVertices);
    this->numVertices = maxNumVertices;
    
    const vec3i blockSize = vec3i(4);
    computeVertices
      <<<to_dims(divRoundUp(numMCVertices,blockSize)),to_dims(blockSize)>>>
      ((vec3f*)owlBufferGetPointer(vertexBuffer,0),
       numMCVertices,numVoxels,mcSize);

    // ------------------------------------------------------------------
    // allocate buffer to hold triangles
    // ------------------------------------------------------------------
    vec3i cellFacesPerSlice(numMCs.y*numMCs.z,
                            numMCs.x*numMCs.z,
                            numMCs.x*numMCs.y);
    PRINT(cellFacesPerSlice);
    vec3i totalFacesPerDim
      = cellFacesPerSlice * (numMCs+1);
    int maxNumTriangles
      = 2*totalFacesPerDim.x
      + 2*totalFacesPerDim.y
      + 2*totalFacesPerDim.z;
    PRINT(maxNumTriangles);
    indexBuffer = owlDeviceBufferCreate(owl,OWL_INT3,maxNumTriangles,0);

    // ------------------------------------------------------------------
    // allocate array(s) of macro cells
    // ------------------------------------------------------------------
    const std::vector<range1f> mcRanges
      = computeMCRanges(voxels);
    CUDA_CALL(Malloc(&d_mcActive,volume(numMCs)*sizeof(bool)));
    CUDA_CALL(Malloc(&d_mcRanges,volume(numMCs)*sizeof(range1f)));
    CUDA_CALL(Memcpy(d_mcRanges,mcRanges.data(),
                     mcRanges.size()*sizeof(mcRanges[0]),cudaMemcpyDefault));
    
    // ------------------------------------------------------------------
    // allocate atomic to hold counter
    // ------------------------------------------------------------------
    CUDA_CALL(Malloc(&d_numTriangles,sizeof(int)));

#if USER_GEOM_METHOD
    buildKDTree(mcRanges);
    std::cout << "kd-tree built, have " << kdTreeNodes.size() << " nodes" << std::endl;
#else
#endif
  }

#if USER_GEOM_METHOD
  void SpaceSkipper::buildKDTree(const std::vector<range1f> &mcRanges)
  {
    kdTreeNodes.clear();
    kdTreeNodes.push_back({});
    buildKDTree(0,mcRanges,box3i(vec3i(0),numMCs));
  }

  void SpaceSkipper::buildKDTree(int nodeID,
                                 const std::vector<range1f> &mcRanges,
                                 const box3i &region)
  {
    vec3i size = region.size();
    if (size == vec3i(1)) {
      auto &node = kdTreeNodes[nodeID];
      node.region = region;
      int cellID
        = region.lower.x
        + region.lower.y * numMCs.x
        + region.lower.z * numMCs.x * numMCs.y;
      node.child = -1;
    } else {
      int dim = arg_max(size);
      box3i lRegion = region;
      box3i rRegion = region;
      lRegion.upper[dim] = rRegion.lower[dim]
        = (region.lower[dim]+region.upper[dim])/2;
      int child = kdTreeNodes.size();
      kdTreeNodes.push_back({});
      kdTreeNodes.push_back({});
      buildKDTree(child+0,mcRanges,lRegion);
      buildKDTree(child+1,mcRanges,rRegion);
      auto &node  = kdTreeNodes[nodeID];
      node.region = region;
      node.child  = child;
    }
  }
#else
#endif
  

  inline __device__
  float remap(const float f, const range1f &range)
  {
    return (f - range.lower) / (range.upper - range.lower);
  }

  __global__ void updateMCs(bool *mcActive,
                            const range1f *mcRanges,
                            const vec3i numMCs,
                            const float4 *colorMap,
                            int colorMapSize,
                            const range1f xfDomain,
                            int *d_numTriangles)
  {
    const vec3i mcIdx
      = vec3i(threadIdx)
      + vec3i(blockIdx)*vec3i(blockDim.x,blockDim.y,blockDim.z);

    if (!array3D::validIndex(mcIdx,numMCs)) return;

    range1f valueRange = mcRanges[array3D::linear(mcIdx,numMCs)];
    bool isActive = false;
    if (xfDomain.lower != xfDomain.upper) {
    
      valueRange.lower = remap(valueRange.lower,xfDomain);
      valueRange.upper = remap(valueRange.upper,xfDomain);

      if (valueRange.upper >= 0.f && valueRange.lower <= 1.f) {
        int numCMIntervals = colorMapSize-1;
        int idx_lo = clamp(int(valueRange.lower*numCMIntervals),0,numCMIntervals);
        int idx_hi = clamp(int(valueRange.upper*numCMIntervals),0,numCMIntervals);
        
        for (int i=idx_lo;i<=idx_hi;i++) {
          if (colorMap[i].w > 1e-3f) {
            isActive = true;
          }
        }
      }
    }
    
    mcActive[array3D::linear(mcIdx,numMCs)] = isActive;
    if (isActive) atomicAdd(d_numTriangles,1);
  }

  template<int DIM> inline __device__ vec3i lower_neighbor(const vec3i idx);
  template<int DIM> inline __device__ vec3i vertexIdx_du();
  template<int DIM> inline __device__ vec3i vertexIdx_dv();

  template<> inline __device__
  vec3i lower_neighbor<0>(const vec3i idx) { return idx-vec3i(1,0,0); }
  template<> inline __device__
  vec3i lower_neighbor<1>(const vec3i idx) { return idx-vec3i(0,1,0); }
  template<> inline __device__
  vec3i lower_neighbor<2>(const vec3i idx) { return idx-vec3i(0,0,1); }

  template<> inline __device__
  vec3i vertexIdx_du<0>() { return { 0,1,0 }; };
  template<> inline __device__
  vec3i vertexIdx_dv<0>() { return { 0,0,1 }; };

  template<> inline __device__
  vec3i vertexIdx_du<1>() { return { 0,0,1 }; };
  template<> inline __device__
  vec3i vertexIdx_dv<1>() { return { 1,0,0 }; };

  template<> inline __device__
  vec3i vertexIdx_du<2>() { return { 1,0,0 }; };
  template<> inline __device__
  vec3i vertexIdx_dv<2>() { return { 0,1,0 }; };

  template<int DIM>
  __global__ void generateTrianglesForCellFace(int *d_numTriangles,
                                               int3 *triangles,
                                               const bool *mcActive,
                                               const vec3i numMCs)
  {
    const vec3i mcIdx
      = vec3i(threadIdx)
      + vec3i(blockIdx)*vec3i(blockDim.x,blockDim.y,blockDim.z);
    if (!array3D::validIndex(mcIdx,numMCs+1)) return;

    // OUR index:
    const vec3i idx1 = mcIdx;
    // index of our lower neighbor, in given DIM
    const vec3i idx0 = lower_neighbor<DIM>(idx1);

    const bool active0
      =  array3D::validIndex(idx0,numMCs)
      && mcActive[array3D::linear(idx0,numMCs)];
    const bool active1
      =  array3D::validIndex(idx1,numMCs)
      && mcActive[array3D::linear(idx1,numMCs)];

    if (active0 == active1) {
      /* both active or both inacive - not a boundary */
      return;
    }

    const vec3i du = vertexIdx_du<DIM>();
    const vec3i dv = vertexIdx_dv<DIM>();
    
    const int vtx00 = array3D::linear(idx1,numMCs+1);
    const int vtx01 = array3D::linear(idx1+du,numMCs+1);
    const int vtx10 = array3D::linear(idx1+dv,numMCs+1);
    const int vtx11 = array3D::linear(idx1+du+dv,numMCs+1);

    int triangleID = atomicAdd(d_numTriangles,2);
    if (active0 && !active1) {
      // active boundary, facing from idx0 to idx1
      triangles[triangleID+0] = { vtx00,vtx01,vtx11 };
      triangles[triangleID+1] = { vtx00,vtx11,vtx10 };
    } else {
      // active boundary, facing from idx1 to idx1
      triangles[triangleID+0] = { vtx01,vtx00,vtx11 };
      triangles[triangleID+1] = { vtx11,vtx00,vtx10 };
    }
  }
  
                               
  void SpaceSkipper::genTriangles()
  {
    CUDA_CALL(Memset(d_numTriangles,0,sizeof(int)));
    
    // +1 because we always do the lower side of the box, so have to
    // include the first box that's outside...
    const vec3i numJobs = numMCs+1;
    const vec3i blockSize(4);
    generateTrianglesForCellFace<0>
      <<<to_dims(divRoundUp(numJobs,blockSize)),to_dims(blockSize)>>>
      (d_numTriangles,
       (int3*)owlBufferGetPointer(indexBuffer,0),
       d_mcActive,numMCs);
    generateTrianglesForCellFace<1>
      <<<to_dims(divRoundUp(numJobs,blockSize)),to_dims(blockSize)>>>
      (d_numTriangles,
       (int3*)owlBufferGetPointer(indexBuffer,0),
       d_mcActive,numMCs);
    generateTrianglesForCellFace<2>
      <<<to_dims(divRoundUp(numJobs,blockSize)),to_dims(blockSize)>>>
      (d_numTriangles,
       (int3*)owlBufferGetPointer(indexBuffer,0),
       d_mcActive,numMCs);
    CUDA_SYNC_CHECK();
    CUDA_CALL(Memcpy(&numTriangles,d_numTriangles,sizeof(int),cudaMemcpyDefault));
    PRINT(numTriangles);
  }
#if USER_GEOM_METHOD
  bool SpaceSkipper::extractActiveBoxes(const bool *h_mcActive,
                                        int nodeID)
  {
#if 1
    activeBoxes.clear();
    for (int iz=0;iz<numMCs.z;iz++)
      for (int iy=0;iy<numMCs.y;iy++)
        for (int ix=0;ix<numMCs.x;ix++) {
          vec3i mcIdx(ix,iy,iz);
          int mcID = array3D::linear(mcIdx,numMCs);
          if (h_mcActive[mcID])
            addActiveBox({mcIdx,mcIdx+1});
        }
    return false;
#endif
    // if (nodeID ==1 || nodeID == 2) return true;
    auto &node = kdTreeNodes[nodeID];
    if (node.child == -1) {
      // this is a leaf
      return h_mcActive[array3D::linear(node.region.lower,numMCs)];
    } else {
      bool lActive = extractActiveBoxes(h_mcActive,node.child+0);
      bool rActive = extractActiveBoxes(h_mcActive,node.child+1);

      if (lActive && rActive)
        // just merge
        return true;

      auto &lChild = kdTreeNodes[node.child+0];
      auto &rChild = kdTreeNodes[node.child+1];
      if (lActive)
        addActiveBox(lChild.region);
      if (rActive)
        addActiveBox(rChild.region);
      return 0;
    }
  }

  void SpaceSkipper::addActiveBox(const box3i &mcRange)
  {
    vec3f lower = vec3f(min(mcRange.lower*mcSize,numVoxels));
    vec3f upper = vec3f(min(mcRange.upper*mcSize,numVoxels));
    activeBoxes.push_back({lower,upper});
  }
  
#endif
  
  OWLGroup SpaceSkipper::computeAccel(const float4 *colorMap,
                                      int colorMapSize,
                                      const range1f &xfDomain)
  {
    static ProfilePrinter prof_updateMCs("updateMCs");
#if USER_GEOM_METHOD
    static ProfilePrinter prof_extractBoxes("extractBoxes");
    static ProfilePrinter prof_buildGeom("buildGeom");
#endif
    // ------------------------------------------------------------------
    // update macrocells on/off grid
    // ------------------------------------------------------------------
    const vec3i blockSize = vec3i(4);
    prof_updateMCs.enter();
    CUDA_CALL(Memset(d_numTriangles,0,sizeof(int)));
    updateMCs
      <<<to_dims(divRoundUp(numMCs,blockSize)),to_dims(blockSize)>>>
      (d_mcActive,d_mcRanges,numMCs,
       colorMap,colorMapSize,xfDomain,d_numTriangles);
    cudaDeviceSynchronize();
    prof_updateMCs.leave();
    int numActiveCells;
    CUDA_CALL(Memcpy(&numActiveCells,d_numTriangles,sizeof(int),cudaMemcpyDefault));
    PRINT(numActiveCells);
    
#if USER_GEOM_METHOD
    if (numActiveCells == 0)
      return 0;

    
    prof_extractBoxes.enter();    
    bool *h_mcActive = new bool[volume(numMCs)];
    CUDA_CALL(Memcpy(h_mcActive,
                     d_mcActive,
                     volume(numMCs)*sizeof(bool),
                     cudaMemcpyDefault));
    activeBoxes.clear();
    bool allActive = extractActiveBoxes(h_mcActive,0);
    if (allActive)
      addActiveBox(kdTreeNodes[0].region);
    delete[] h_mcActive;
    prof_extractBoxes.leave();    
    
    std::cout << "generated active boxes, found " << activeBoxes.size() << " boxes..." << std::endl;

    prof_buildGeom.enter();    
    if (activeBoxesBuffer) owlBufferDestroy(activeBoxesBuffer);
    activeBoxesBuffer = owlDeviceBufferCreate(owl,OWL_USER_TYPE(box3f),
                                              activeBoxes.size(),
                                              activeBoxes.data());

    if (ssGeom == 0) {
      OWLVarDecl ssGeomVars[]
        = {
           { "activeBoxes", OWL_BUFPTR, OWL_OFFSETOF(SSGeom,activeBoxes) },
           { nullptr /* sentinel */ }
      };
      OWLGeomType ssGT = owlGeomTypeCreate(owl,OWL_GEOM_USER,
                                           sizeof(SSGeom),ssGeomVars,-1);
      owlGeomTypeSetIntersectProg(ssGT,0,module,"SpaceSkipperIsec");
      owlGeomTypeSetBoundsProg(ssGT,module,"SpaceSkipperBounds");
      owlGeomTypeSetClosestHit(ssGT,0,module,"SpaceSkipperCH");
      owlBuildPrograms(owl);
      ssGeom = owlGeomCreate(owl,ssGT);
    }

    owlGeomSetBuffer(ssGeom,"activeBoxes",activeBoxesBuffer);
    owlGeomSetPrimCount(ssGeom,activeBoxes.size());

    if (ias == 0) {
      blas = owlUserGeomGroupCreate(owl,1,&ssGeom);
      owlGroupBuildAccel(blas);
      
      ias = owlInstanceGroupCreate(owl,1,&blas);
      owlGroupBuildAccel(ias);
    } else {
      owlGroupBuildAccel(blas);
      owlGroupBuildAccel(ias);
    }
    prof_buildGeom.leave();
    return ias;
#else
    // todo - turn off triangle generation
    static ProfilePrinter prof_genTriangles("gen triangles");
    static ProfilePrinter prof_buildBVHes("build BVHes");

    if (d_numTriangles == 0)
      // not yet allocated....
      return 0;

    
    // ------------------------------------------------------------------
    // compute the actual triangles
    // ------------------------------------------------------------------
    prof_genTriangles.enter();
    genTriangles();
    prof_genTriangles.leave();
    if (numTriangles == 0)
      return 0;
    
    if (ias == 0) {
      OWLVarDecl ssGeomVars[]
        = {
           { "vertices", OWL_BUFPTR, OWL_OFFSETOF(SSGeom,vertices) },
           { "indices", OWL_BUFPTR, OWL_OFFSETOF(SSGeom,indices) },
           { nullptr /* sentinel */ }
      };
      OWLGeomType ssGT = owlGeomTypeCreate(owl,OWL_TRIANGLES,
                                           sizeof(SSGeom),ssGeomVars,-1);
      owlGeomTypeSetClosestHit(ssGT,0,module,"SpaceSkipperCH");
      
      ssGeom = owlGeomCreate(owl,ssGT);
      owlGeomSetBuffer(ssGeom,"vertices",vertexBuffer);
      owlGeomSetBuffer(ssGeom,"indices",indexBuffer);
      
      owlTrianglesSetVertices(ssGeom,vertexBuffer,
                                  numVertices,sizeof(vec3f),0);
      owlTrianglesSetIndices(ssGeom,indexBuffer,
                             numTriangles,sizeof(vec3i),0);
      blas = owlTrianglesGeomGroupCreate(owl,1,&ssGeom);
      owlGroupBuildAccel(blas);
      
      ias = owlInstanceGroupCreate(owl,1,&blas);
      owlGroupBuildAccel(ias);
    }
    else {
      prof_buildBVHes.enter();
      owlTrianglesSetIndices(ssGeom,indexBuffer,
                             numTriangles,sizeof(vec3i),0);
      owlGroupBuildAccel(blas);
      owlGroupRefitAccel(ias);
      prof_buildBVHes.leave();
    }

    if (numTriangles == 0)
      return 0;
    else
      return ias;
#endif
  }
    
  template void SpaceSkipper::computeMCs(OWLContext owl,
                                         OWLModule module,
                                         const float *voxel,
                                         const vec3i &numVoxels,
                                         int mcSize);
  template void SpaceSkipper::computeMCs(OWLContext owl,
                                         OWLModule module,
                                         const uint8_t *voxel,
                                         const vec3i &numVoxels,
                                         int mcSize);

}
