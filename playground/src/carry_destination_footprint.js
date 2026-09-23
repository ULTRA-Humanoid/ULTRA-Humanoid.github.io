/** Isolated geometric preflight; never changes a goal, reference or physics. */
import {quatRotateOne} from './math.js';
const EPS=1e-9;
const finite=(v,n)=>{if(v?.length!==n)return false;for(let i=0;i<n;i++)if(!Number.isFinite(v[i]))return false;return true;};
function unitQuaternion(q){
  if(!finite(q,4))throw new Error('A finite XYZW quaternion is required');
  const norm=Math.hypot(...q);
  if(norm<1e-8||Math.abs(norm-1)>1e-3)throw new Error('A unit collision-pose quaternion is required');
  return Array.from(q,v=>v/norm);
}
function convexHull(points){
  const sorted=points.map(p=>[p[0],p[1]]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  const unique=sorted.filter((p,i)=>!i||p[0]!==sorted[i-1][0]||p[1]!==sorted[i-1][1]);
  const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const half=rows=>{const result=[];for(const p of rows){while(result.length>1&&cross(result.at(-2),result.at(-1),p)<=0)result.pop();result.push(p);}return result;};
  const a=half(unique),b=half([...unique].reverse());a.pop();b.pop();
  const hull=[...a,...b];if(hull.length<3)throw new Error('A nondegenerate collision footprint is required');
  return hull;
}
function bounds(points){
  let minX=Infinity,minY=Infinity,minZ=Infinity,maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;
  for(const p of points){
    if(p[0]<minX)minX=p[0];if(p[0]>maxX)maxX=p[0];
    if(p[1]<minY)minY=p[1];if(p[1]>maxY)maxY=p[1];
    if(p[2]<minZ)minZ=p[2];if(p[2]>maxZ)maxZ=p[2];
  }
  return{minX,maxX,minY,maxY,minZ,maxZ};
}
function bodyName(model,id){
  let result='',i=model.name_bodyadr[id];while(model.names[i])result+=String.fromCharCode(model.names[i++]);return result;
}

// Reduction of a compiled collision mesh to a SUPERSET of its 3D convex hull
// vertices, exact to HULL_FILTER_TOLERANCE (1e-9 m). The XY convex hull and the
// axis bounds of a rotated point set are those of its hull vertices, so
// projecting only this subset returns the same polygon and the same min/max: a
// point strictly inside, or within the tolerance of a facet of, an inner
// polytope P = conv(E) with E ⊆ V is not a hull vertex of V (a vertex that
// protrudes less than 1e-9 m past P could be dropped, moving a hull edge by at
// most that much; planner tolerances are centimetres). E = the vertices extreme
// in a fixed set of directions; P's facets are found by brute force over
// triples of E (|E| ≤ ~80). Points of E are always kept. v16 projected every
// one of the ~13k–38k vertices of each OBJ on every plan check (~57% of a
// click's planning time); on the four shipped meshes 600 random poses gave
// identical hulls and bounds.
const HULL_FILTER_TOLERANCE=1e-9;
export function hullVertexSuperset(vertices){
  const n=vertices.length;if(n<=64)return vertices;
  const directions=[];
  for(const x of[-1,0,1])for(const y of[-1,0,1])for(const z of[-1,0,1]){
    if(!x&&!y&&!z)continue;const l=Math.hypot(x,y,z);directions.push([x/l,y/l,z/l]);
  }
  const spiral=48,golden=Math.PI*(3-Math.sqrt(5));
  for(let i=0;i<spiral;i++){
    const z=1-2*(i+.5)/spiral,r=Math.sqrt(Math.max(0,1-z*z)),a=golden*i;
    directions.push([r*Math.cos(a),r*Math.sin(a),z]);
  }
  const extremeIndex=new Set();
  for(const d of directions){
    let best=-Infinity,at=0;
    for(let i=0;i<n;i++){const v=vertices[i],dot=v[0]*d[0]+v[1]*d[1]+v[2]*d[2];if(dot>best){best=dot;at=i;}}
    extremeIndex.add(at);
  }
  const E=[...extremeIndex].map(i=>vertices[i]);
  const facets=[];
  for(let a=0;a<E.length;a++)for(let b=a+1;b<E.length;b++)for(let c=b+1;c<E.length;c++){
    const A=E[a],B=E[b],C=E[c];
    const ux=B[0]-A[0],uy=B[1]-A[1],uz=B[2]-A[2],vx=C[0]-A[0],vy=C[1]-A[1],vz=C[2]-A[2];
    let nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
    const l=Math.hypot(nx,ny,nz);if(l<1e-12)continue;
    nx/=l;ny/=l;nz/=l;
    const d=nx*A[0]+ny*A[1]+nz*A[2];
    let above=false,below=false;
    for(const P of E){const s=nx*P[0]+ny*P[1]+nz*P[2]-d;if(s>HULL_FILTER_TOLERANCE)above=true;if(s<-HULL_FILTER_TOLERANCE)below=true;if(above&&below)break;}
    if(above&&below)continue;
    if(!above)facets.push([nx,ny,nz,d]);
    if(!below)facets.push([-nx,-ny,-nz,-d]);
  }
  if(!facets.length)return vertices;
  const kept=[];
  for(let i=0;i<n;i++){
    const v=vertices[i];
    if(extremeIndex.has(i)){kept.push(v);continue;}
    let outside=false;
    for(const f of facets){if(f[0]*v[0]+f[1]*v[1]+f[2]*v[2]-f[3]>HULL_FILTER_TOLERANCE){outside=true;break;}}
    if(outside)kept.push(v);
  }
  return kept.length>=4?kept:vertices;
}

// Cache only a body's current exact pose and exact compiled local geometry.
// Compare values rather than quantizing: even a tiny pose/vertex edit invalidates
// the entry. Returned projections are detached so callers may edit their copy.
function sameShape(object,snapshot){
  return object.bodyId===snapshot.bodyId&&object.name===snapshot.name
    &&object.meshes?.length===snapshot.meshes.length&&object.meshes.every((mesh,i)=>{
      const saved=snapshot.meshes[i];
      // The constructor deep-freezes verticesLocal (and its reduced hull set), so
      // the same frozen array object is the same geometry. A foreign, unfrozen
      // mesh still gets the full value comparison.
      if(mesh.geomId!==saved.geomId||mesh.verticesLocal?.length!==saved.verticesLocal.length)return false;
      if(Object.isFrozen(mesh.verticesLocal)&&mesh.verticesLocal===saved.verticesLocal
        &&(mesh.hullVerticesLocal??null)===(saved.hullVerticesLocal??null))return true;
      return mesh.verticesLocal.every((v,j)=>v?.length===3&&v.every((x,k)=>Object.is(x,saved.verticesLocal[j][k])));
    });
}

/** Cache compiled mesh vertices in their own object-body frame. MuJoCo's
 * compiled geom pose includes mesh recentering/rotation; original OBJ extents
 * or policy surface point clouds do not provide this collision geometry.
 * Current box assets have collision geoms directly on their rigid body. Fail
 * explicitly for descendant collision bodies, rather than omit geometry. */
export class ObjectCollisionMeshes {
  #projectionCache=new WeakMap();
  #projectionCacheStats={hits:0,misses:0};
  constructor(model,bodyIds){
    if(!Array.isArray(bodyIds)||!bodyIds.length||new Set(bodyIds).size!==bodyIds.length
      ||bodyIds.some(id=>!Number.isInteger(id)||id<=0||id>=model.nbody))throw new Error('Valid distinct object body IDs are required');
    this.objects=bodyIds.map(bodyId=>{
      const meshes=[];
      for(let geomId=0;geomId<model.ngeom;geomId++){
        if(model.geom_contype[geomId]===0&&model.geom_conaffinity[geomId]===0)continue;
        let ancestor=model.geom_bodyid[geomId];
        while(ancestor>0&&ancestor!==bodyId)ancestor=model.body_parentid[ancestor];
        if(ancestor!==bodyId)continue;
        if(model.geom_bodyid[geomId]!==bodyId)throw new Error('Descendant collision bodies require explicit rigid-shape support');
        if(model.geom_type[geomId]!==7)throw new Error('Object collision preflight currently requires mesh geoms');
        const meshId=model.geom_dataid[geomId],first=model.mesh_vertadr[meshId]*3,count=model.mesh_vertnum[meshId]*3;
        const p=Array.from(model.geom_pos.slice(geomId*3,geomId*3+3)),wxyz=model.geom_quat.slice(geomId*4,geomId*4+4);
        const q=unitQuaternion([wxyz[1],wxyz[2],wxyz[3],wxyz[0]]),verticesLocal=[];
        if(!finite(p,3)||!Number.isInteger(count)||count<12)throw new Error('Complete compiled mesh geometry is required');
        for(let index=first;index<first+count;index+=3){
          const v=Array.from(model.mesh_vert.slice(index,index+3));
          if(!finite(v,3))throw new Error('Finite compiled mesh vertices are required');
          verticesLocal.push(quatRotateOne(q,v).map((x,i)=>x+p[i]));
        }
        for(const v of verticesLocal)Object.freeze(v);
        Object.freeze(verticesLocal);
        const hullVerticesLocal=Object.freeze(hullVertexSuperset(verticesLocal));
        meshes.push(Object.freeze({geomId,verticesLocal,hullVerticesLocal}));
      }
      if(!meshes.length)throw new Error('Object has no supported collision mesh');
      return {bodyId,name:bodyName(model,bodyId),meshes};
    });
  }
  read(data){
    return this.objects.map(object=>{
      const p=Array.from(data.xpos.slice(object.bodyId*3,object.bodyId*3+3));
      const q=data.xquat.slice(object.bodyId*4,object.bodyId*4+4);
      const orientation=[q[1],q[2],q[3],q[0]],pose=[...p,...orientation];
      const prior=this.#projectionCache.get(object);
      if(prior&&pose.every((v,i)=>Object.is(v,prior.pose[i]))&&sameShape(object,prior.shape)){
        this.#projectionCacheStats.hits++;
        return structuredClone(prior.projection);
      }
      const projection=projectObjectCollisionMeshes(object,p,orientation);
      const shape={bodyId:object.bodyId,name:object.name,meshes:object.meshes.map(mesh=>({geomId:mesh.geomId,
        verticesLocal:Object.isFrozen(mesh.verticesLocal)?mesh.verticesLocal:mesh.verticesLocal.map(vertex=>Array.from(vertex)),
        hullVerticesLocal:mesh.hullVerticesLocal??null}))};
      this.#projectionCache.set(object,{pose,shape,projection:structuredClone(projection)});
      this.#projectionCacheStats.misses++;
      return projection;
    });
  }
  get projectionCacheStats(){return{...this.#projectionCacheStats};}
}

export function projectObjectCollisionMeshes(object,positionWorld,quaternionXyzwWorld){
  if(!object||!Number.isInteger(object.bodyId)||object.bodyId<=0||typeof object.name!=='string'
      ||!Array.isArray(object.meshes)||!object.meshes.length||!finite(positionWorld,3))throw new Error('A complete object shape and finite pose are required');
  const q=unitQuaternion(quaternionXyzwWorld),all=[];
  const meshes=object.meshes.map(mesh=>{
    // Frozen meshes built by ObjectCollisionMeshes were validated once; their
    // hullVerticesLocal is an exact superset of the hull vertices (see above).
    const reduced=Object.isFrozen(mesh.verticesLocal)&&Array.isArray(mesh.hullVerticesLocal)&&mesh.hullVerticesLocal.length>=4;
    if(!Number.isInteger(mesh.geomId)||!Array.isArray(mesh.verticesLocal)||mesh.verticesLocal.length<4
        ||(!reduced&&!mesh.verticesLocal.every(p=>finite(p,3))))throw new Error('Complete finite local collision vertices are required');
    const source=reduced?mesh.hullVerticesLocal:mesh.verticesLocal;
    const world=new Array(source.length);
    for(let i=0;i<source.length;i++){const r=quatRotateOne(q,source[i]);world[i]=[r[0]+positionWorld[0],r[1]+positionWorld[1],r[2]+positionWorld[2]];}
    for(const w of world)all.push(w);return{geomId:mesh.geomId,hull:convexHull(world),...bounds(world)};
  });
  return{bodyId:object.bodyId,name:object.name,positionWorld:Array.from(positionWorld),quaternionXyzwWorld:q,meshes,...bounds(all)};
}

function validProjected(object){
  return object&&Number.isInteger(object.bodyId)&&object.bodyId>0&&typeof object.name==='string'
    &&Array.isArray(object.meshes)&&object.meshes.length>0&&object.meshes.every(mesh=>
      Number.isInteger(mesh.geomId)&&Array.isArray(mesh.hull)&&mesh.hull.length>=3
      &&validConvexHull(mesh.hull)&&Number.isFinite(mesh.minZ)&&Number.isFinite(mesh.maxZ)
      &&mesh.minZ<mesh.maxZ);
}
function validConvexHull(hull){
  if(!hull.every(p=>finite(p,2)))return false;
  let sign=0;
  for(let i=0;i<hull.length;i++){
    const a=hull[i],b=hull[(i+1)%hull.length],c=hull[(i+2)%hull.length];
    const cross=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]);
    if(Math.abs(cross)<=1e-12)continue;
    if(sign&&Math.sign(cross)!==sign)return false;sign=Math.sign(cross);
  }
  return sign!==0&&hull.every((a,i)=>{
    const b=hull[(i+1)%hull.length];return hull.every(p=>sign*((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]))>=-1e-10);
  });
}
function separated(a,b,reserve){
  if(a.maxZ<b.minZ-reserve-EPS||b.maxZ<a.minZ-reserve-EPS)return true;
  // Convex projected mesh prisms. Their overlap is conservative for a tilted
  // 3D mesh; their separation proves the corresponding collision meshes clear.
  for(const hull of [a.hull,b.hull])for(let i=0;i<hull.length;i++){
    const p=hull[i],n=hull[(i+1)%hull.length],length=Math.hypot(n[0]-p[0],n[1]-p[1]);
    if(length<EPS)continue;
    const axis=[(p[1]-n[1])/length,(n[0]-p[0])/length];
    const aa=a.hull.map(v=>v[0]*axis[0]+v[1]*axis[1]),bb=b.hull.map(v=>v[0]*axis[0]+v[1]*axis[1]);
    if(Math.max(...aa)<Math.min(...bb)-reserve-EPS||Math.max(...bb)<Math.min(...aa)-reserve-EPS)return true;
  }
  return false;
}

/** Check the exact requested XY with the complete reference's final object
 * orientation and height, and also its separately retained planned endpoint.
 * Checking both avoids silently accepting bounded warp residuals. The caller
 * must still validate the full carried/humanoid path and actual transitions.
 * The reserve is explicit and unchanged by this helper; zero detects direct
 * geometric overlap only. Touching any boundary is refused. */
export function checkCarryDestinationFootprint({object,requestedGoalWorld,finalReferenceFrame,obstacles,trackingReserve=0}={}){
  const invalid=()=>({supported:false,reason:'invalid_carry_destination_geometry'});
  if(!finite(requestedGoalWorld,3)||!finite(finalReferenceFrame,747)||!Array.isArray(obstacles)
    ||!obstacles.every(validProjected)||new Set(obstacles.map(x=>x.bodyId)).size!==obstacles.length
    ||!Number.isFinite(trackingReserve)||trackingReserve<0)return invalid();
  try{
    const referenceObjectPositionWorld=Array.from(finalReferenceFrame.slice(71,74));
    const orientation=Array.from(finalReferenceFrame.slice(74,78));
    const positions=[{kind:'requested_destination',position:[requestedGoalWorld[0],requestedGoalWorld[1],referenceObjectPositionWorld[2]]},
      {kind:'reference_endpoint',position:referenceObjectPositionWorld}];
    const placements=positions.map(({kind,position})=>({kind,...projectObjectCollisionMeshes(object,position,orientation)}));
    const others=obstacles.filter(x=>x.bodyId!==object.bodyId),collisions=[];
    for(const placement of placements)for(const obstacle of others)for(const a of placement.meshes)for(const b of obstacle.meshes){
      if(separated(a,b,trackingReserve))continue;
      collisions.push({placement:placement.kind,obstacleBodyId:obstacle.bodyId,obstacleName:obstacle.name,
        carriedGeomId:a.geomId,obstacleGeomId:b.geomId});
    }
    return{supported:collisions.length===0,reason:collisions.length?'occupied_carry_destination':null,
      requestedGoalWorld:Array.from(requestedGoalWorld),referenceObjectPositionWorld,
      referenceGoalResidualM:Math.hypot(requestedGoalWorld[0]-referenceObjectPositionWorld[0],requestedGoalWorld[1]-referenceObjectPositionWorld[1]),
      trackingReserve,checkedObstacleCount:others.length,placements,collisions};
  }catch{return invalid();}
}
