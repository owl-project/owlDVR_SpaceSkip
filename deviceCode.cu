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

#include "deviceCode.h"
#include "owl/common/math/random.h"

namespace dvr {

  extern "C" __constant__ LaunchParams optixLaunchParams;

  typedef owl::common::LCG<4> Random;
  
  inline __device__
  vec3f backGroundColor()
  {
    const vec2i pixelID = owl::getLaunchIndex();
    const float t = pixelID.y / (float)optixGetLaunchDimensions().y;
    const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
    return c;
  }

  inline __device__ vec4f over(const vec4f &A, const vec4f &B)
  {
    return A + (1.f-A.w)*B;
  }
  

  inline __device__
  bool intersect(const Ray &ray,
                 const box3f &box,
                 float &t0,
                 float &t1)
  {
    vec3f lo = (box.lower - ray.origin) / ray.direction;
    vec3f hi = (box.upper - ray.origin) / ray.direction;
    
    vec3f nr = min(lo,hi);
    vec3f fr = max(lo,hi);

    t0 = max(ray.tmin,reduce_max(nr));
    t1 = min(ray.tmax,reduce_min(fr));

    return t0 < t1;
  }

  inline __device__
  vec4f sampleVolume(const vec3f &pos,
                     vec3f &gradient)
  {
    auto &lp = optixLaunchParams;
    const box3f bounds = lp.volume.domain;
    
    vec3f tc
      = (pos - bounds.lower)
      / bounds.size();
    // if (tc.x < 0.f ||
    //     tc.y < 0.f ||
    //     tc.z < 0.f ||
    //     tc.x > 1.f ||
    //     tc.y > 1.f ||
    //     tc.z > 1.f)
    //   return vec4f(0.f);

    // load raw scalar field value from texture
    float value = tex3D<float>(lp.volume.texture,tc.x,tc.y,tc.z);

    // re-map to [0,1] in specified transfer function domain
    const range1f xfDomain = lp.transferFunc.domain;
    if (value < xfDomain.lower ||
        value > xfDomain.upper) {
      gradient = 0.f;
      return vec4f(1.f,1.f,1.f,0.f);
    }
    float remapped
      = (value - xfDomain.lower) / (xfDomain.upper - xfDomain.lower);
    
    float4 xf = tex2D<float4>(lp.transferFunc.texture,remapped,0.5f);
    xf.w *= lp.transferFunc.opacityScale;


    if (xf.w > 1e-4f) {
      const vec3f delta = lp.render.gradientDelta;
      gradient = vec3f(+tex3D<float>(lp.volume.texture,tc.x+delta.x,tc.y,tc.z)
                       -tex3D<float>(lp.volume.texture,tc.x-delta.x,tc.y,tc.z),
                       +tex3D<float>(lp.volume.texture,tc.x,tc.y+delta.y,tc.z)
                       -tex3D<float>(lp.volume.texture,tc.x,tc.y-delta.y,tc.z),
                       +tex3D<float>(lp.volume.texture,tc.x,tc.y,tc.z+delta.z)
                       -tex3D<float>(lp.volume.texture,tc.x,tc.y,tc.z-delta.z));
      gradient = gradient / (length(gradient + 1e-4f));
    } else gradient = 0.f;

    return xf;
  }

  inline __device__
  float firstSampleT(const range1f &rayInterval,
                     const float dt,
                     const float ils_t0)
  {
    float numSegsf = floor((rayInterval.lower - dt*ils_t0)/dt);
    float t = dt * (ils_t0 + numSegsf);
    if (t < rayInterval.lower) t += dt;
    return t;
  }
               
  inline __device__
  vec4f integrateRay(const Ray &ray,
                     const float ils_t0,
                     float &z)
  {
    auto &lp = optixLaunchParams;

    const box3f bounds = lp.volume.domain;
    float t0, t1;
    vec3f color = 0.f;
    float alpha = 0.f;
    if (!intersect(ray,bounds,t0,t1)) {
      z = 1e10f;
      return vec4f(0.f);
    }

    
    const float dt = lp.render.dt;
    for (float t = firstSampleT({t0,t1},dt,ils_t0); t < t1 && alpha < .99f; t += dt) {
      vec3f gradient;
      vec4f sample = sampleVolume(ray.origin+t*ray.direction,gradient);
      sample.w *= 1.f;
      color += dt * (1.f-alpha) * sample.w * vec3f(sample)
        * (.1f+.9f*fabsf(dot(gradient,ray.direction)));
      alpha += dt * (1.f-alpha)*sample.w;
    }

    z = t0;
    return vec4f(color,alpha);
  }
  
  inline __device__ Ray generateRay(const vec2f screen)
  {
    auto &lp = optixLaunchParams;
    vec3f org = lp.camera.org;
    vec3f dir
      = lp.camera.dir_00
      + screen.u * lp.camera.dir_du
      + screen.v * lp.camera.dir_dv;
    dir = normalize(dir);
    if (fabs(dir.x) < 1e-5f) dir.x = 1e-5f;
    if (fabs(dir.y) < 1e-5f) dir.y = 1e-5f;
    if (fabs(dir.z) < 1e-5f) dir.z = 1e-5f;
    return Ray(org,dir,0.f,1e10f);
  }

#if USER_GEOM_METHOD
  inline __device__
  void intersectBox(const box3f &box,
                    const vec3f &org,
                    const vec3f &dir,
                    float &t0, float &t1)
  {
    vec3f lower = (box.lower - org) / dir;
    vec3f upper = (box.upper - org) / dir;
    vec3f nr = min(lower,upper);
    vec3f fr = max(lower,upper);
    t0 = max(t0,reduce_max(nr));
    t1 = min(t1,reduce_min(fr));
  }
  
  OPTIX_INTERSECT_PROGRAM(SpaceSkipperIsec)()
  {
    const SSGeom &self = owl::getProgramData<SSGeom>();
    SSPRD &prd = owl::getPRD<SSPRD>();
    int primID = optixGetPrimitiveIndex();
    box3f box = self.activeBoxes[primID];
    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();
    const vec3f org = optixGetWorldRayOrigin();
    const vec3f dir = optixGetWorldRayDirection();
    intersectBox(box,org,dir,t0,t1);

    if (t1 <= t0) 
      return;
    
    prd.t0 = t0;
    prd.t1 = t1;
    prd.primID = primID;
    optixReportIntersection(t0,primID);
  }
  OPTIX_CLOSEST_HIT_PROGRAM(SpaceSkipperCH)()
  {
    // compute a semi-sueful normal, mostly for debugging
    const SSGeom &self = owl::getProgramData<SSGeom>();
    SSPRD &prd = owl::getPRD<SSPRD>();
    box3f box = self.activeBoxes[prd.primID];
    const vec3f org = optixGetWorldRayOrigin();
    const vec3f dir = optixGetWorldRayDirection();
    vec3f enterPos = org + prd.t0 * dir;
    enterPos = (enterPos - box.lower) / box.size();
    enterPos = 2.f*enterPos - 1.f;
    prd.Ng = 0;
    prd.Ng[arg_max(abs(enterPos))] = 1;
  }
  OPTIX_BOUNDS_PROGRAM(SpaceSkipperBounds)(const void *geomData,
                                           box3f &primBounds,
                                           const int primID)
  {
    const SSGeom &self = *(const SSGeom *)geomData;
    primBounds = self.activeBoxes[primID];
  }
#else
  OPTIX_CLOSEST_HIT_PROGRAM(SpaceSkipperCH)()
  {
    SSPRD &prd = owl::getPRD<SSPRD>();
    prd.t = optixGetRayTmax();
    prd.primID = optixGetPrimitiveIndex();
    const SSGeom &self = owl::getProgramData<SSGeom>();
    vec3i index = self.indices[prd.primID];
    vec3f A = self.vertices[index.x];
    vec3f B = self.vertices[index.y];
    vec3f C = self.vertices[index.z];
    prd.Ng = normalize(cross(B-A,C-A));
#if SS_SINGLE_RAY
    prd.backFace = (optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE);
#endif
  }
#endif

  inline __device__ vec3f hue_to_rgb(float hue)
  {
    float s = saturate( hue ) * 6.0f;
    float r = saturate( fabsf(s - 3.f) - 1.0f );
    float g = saturate( 2.0f - fabsf(s - 2.0f) );
    float b = saturate( 2.0f - fabsf(s - 4.0f) );
    return vec3f(r, g, b); 
  }
    
  inline __device__ vec3f temperature_to_rgb(float t)
  {
    float K = 4.0f / 6.0f;
    float h = K - K * t;
    float v = .5f + 0.5f * t;
    return v * hue_to_rgb(h);
  }
    
                                    
  inline __device__
  vec3f heatMap(float t)
  {
    return temperature_to_rgb(t);
  }
  
  OPTIX_RAYGEN_PROGRAM(renderFrame)()
  {
    auto &lp = optixLaunchParams;
    const int spp = lp.render.spp; 
    const vec2i threadIdx = owl::getLaunchIndex();
    Ray ray = generateRay(vec2f(threadIdx)+vec2f(.5f));

    vec4f bgColor = vec4f(backGroundColor(),1.f);
    Random random(threadIdx.x,threadIdx.y);

    bool useSpaceSkipping
      =  ((lp.render.spaceSkipMode%3) == 0)
      || ((lp.render.spaceSkipMode%3) == 1 && threadIdx.x >= owl::getLaunchDims().x/2);
    uint64_t clock_begin = clock();

    vec4f accumColor = 0.f;
    float z;
    for (int sampleID=0;sampleID<spp;sampleID++) {
      const float ils_t0 = random();

      vec4f color;
      if (lp.render.showBoxes) {
        color = integrateRay(ray,ils_t0,z);
        color = over(color,bgColor);
        
        SSPRD prd = { -1, -1 };
        owl::traceRay(lp.render.ssGeom,ray,prd,
                      OPTIX_RAY_FLAG_DISABLE_ANYHIT|
#if 1
                      // show only front side:
                      OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES
#else
                      // show only back side:
                      OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES
#endif
                      );
        vec4f box_color
          = prd.primID == -1
          ? vec4f(1,1,1,1)
          : vec4f(randomColor(prd.primID/2)
                  //vec3f(.1,.7f,.1f)
                  *vec3f(.2f+.6f*fabsf(dot(ray.direction,prd.Ng)))
                  ,.6);
        color = over(box_color,color);
      }
      else {
        if (useSpaceSkipping) {
          color = vec4f(0.f);
          float last_t_end = 0.f;
          while (1) {
#if USER_GEOM_METHOD
            SSPRD prd;
            prd.primID = -1;
            ray.tmin = last_t_end;
            ray.tmax = 1e10f;
            owl::traceRay(lp.render.ssGeom,ray,prd,
                          OPTIX_RAY_FLAG_DISABLE_ANYHIT);
            if (prd.primID < 0)
              // no hit at all
              break;
            ray.tmin = prd.t0;
            ray.tmax = prd.t1;
            vec4f thisSeg_color = integrateRay(ray,ils_t0,z);
            color = over(color,thisSeg_color);
            if (color.w >= .98f)
              break;

            last_t_end = prd.t1*(1.f+1e-6f);
#elif SS_SINGLE_RAY
            SSPRD prd = { -1, -1 };
            ray.tmin = last_t_end;
            ray.tmax = 1e10f;
            float thisSeg_t0=last_t_end;
            owl::traceRay(lp.render.ssGeom,ray,prd,
                          OPTIX_RAY_FLAG_DISABLE_ANYHIT);
            if (prd.primID < 0)
              // no hit at all
              break;
            float thisSeg_t1=prd.t;
            if (prd.backFace) {
              // this is a EXIT hit, so we must be inside
            }  else {
              // this is a ENTRY hit, so work _starts_ there
              thisSeg_t0 = prd.t;
              ray.tmin = prd.t*(1.f+1e-6f);
              ray.tmax = 1e10f;
              prd.primID = -1;
              owl::traceRay(lp.render.ssGeom,ray,prd,
                            OPTIX_RAY_FLAG_DISABLE_ANYHIT);
              if (prd.primID < 0)
                // could not find exit face!? error ...
                break;
              thisSeg_t1 = prd.t;
            }
            ray.tmin = thisSeg_t0;
            ray.tmax = thisSeg_t1;
            vec4f thisSeg_color = integrateRay(ray,ils_t0,z);
            color = over(color,thisSeg_color);
            if (color.w >= .98f)
              break;

            last_t_end = thisSeg_t1*(1.f+1e-6f);
#else
            SSPRD prd = { -1, -1 };
            ray.tmin = last_t_end;
            ray.tmax = 1e10f;
            owl::traceRay(lp.render.ssGeom,ray,prd,
                          OPTIX_RAY_FLAG_DISABLE_ANYHIT|
                          // show only back side:
                          OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES
                          );
            if (prd.primID < 0) break;
            const float thisSeg_t1 = prd.t;

            prd = { -1, -1 };
            Ray backRay = Ray(ray.origin+thisSeg_t1*ray.direction,
                              -ray.direction,0.f,thisSeg_t1-last_t_end);
            owl::traceRay(lp.render.ssGeom,backRay,prd,
                          OPTIX_RAY_FLAG_DISABLE_ANYHIT|
                          // show only back side:
                          OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES
                          );
            float thisSeg_t0
              = (prd.primID < 0)
              ? last_t_end
              : (thisSeg_t1-prd.t);
            ray.tmin = thisSeg_t0;
            ray.tmax = thisSeg_t1;
            // careful - this overwrites z if there's more than one segment
            vec4f thisSeg_color = integrateRay(ray,ils_t0,z);
            color = over(color,thisSeg_color);
            if (color.w >= .98f)
              break;

            last_t_end = thisSeg_t1*(1.f+1e-6f);
#endif
          }
        } else {
          // no space skipping - just integrate the whole
          // ray-worldbounds overlap segment
          float z;
          color = integrateRay(ray,ils_t0,z);
        }
        color = over(color,bgColor);
      }
      accumColor += color;
    }

    uint64_t clock_end = clock();
    if (lp.render.heatMapEnabled > 0.f) {
      float t = (clock_end-clock_begin)*(lp.render.heatMapScale/spp);
      accumColor = over(vec4f(heatMap(t),.5f),accumColor);
    }

    int pixelID = threadIdx.x + owl::getLaunchDims().x*threadIdx.y;
    if (lp.accumID > 0)
      accumColor += vec4f(lp.accumBuffer[pixelID]);
    lp.accumBuffer[pixelID] = accumColor;
    accumColor *= (1.f/(lp.accumID+1));
    
    bool crossHairs = (owl::getLaunchIndex().x == owl::getLaunchDims().x/2
                       ||
                       owl::getLaunchIndex().y == owl::getLaunchDims().y/2
                       );
    if (crossHairs) accumColor = vec4f(1.f) - accumColor;
    
    lp.fbPointer[pixelID] = make_rgba(vec3f(accumColor*(1.f/spp)));
#if DUMP_FRAMES
    lp.fbDepth[pixelID]   = z;
#endif
  }
  
}
