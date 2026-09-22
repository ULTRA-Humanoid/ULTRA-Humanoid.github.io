/** Isolated geometric preflight; never changes a goal, reference or physics. */
import {quatRotateOne} from './math.js';
const EPS=1e-9;
const finite=(v,n)=>v?.length===n&&Array.from(v).every(Number.isFinite);
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
  return Object.fromEntries(['X','Y','Z'].flatMap((s,i)=>[
    ['min'+s,Math.min(...points.map(p=>p[i]))],['max'+s,Math.max(...points.map(p=>p[i]))]]));
}
function bodyName(model,id){
  let result='',i=model.name_bodyadr[id];while(model.names[i])result+=String.fromCharCode(model.names[i++]);return result;
}

// Cache only a body's current exact pose and exact compiled local geometry.
// Compare values rather than quantizing: even a tiny pose/vertex edit invalidates
// the entry. Returned projections are detached so callers may edit their copy.
function sameShape(object,snapshot){
  return object.bodyId===snapshot.bodyId&&object.name===snapshot.name
    &&object.meshes?.length===snapshot.meshes.length&&object.meshes.every((mesh,i)=>{
      const saved=snapshot.meshes[i];
      return mesh.geomId===saved.geomId&&mesh.verticesLocal?.length===saved.verticesLocal.length
        &&mesh.verticesLocal.every((v,j)=>v?.length===3&&v.every((x,k)=>Object.is(x,saved.verticesLocal[j][k])));
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
        meshes.push({geomId,verticesLocal});
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
        verticesLocal:mesh.verticesLocal.map(vertex=>Array.from(vertex))}))};
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
    if(!Number.isInteger(mesh.geomId)||!Array.isArray(mesh.verticesLocal)||mesh.verticesLocal.length<4
        ||!mesh.verticesLocal.every(p=>finite(p,3)))throw new Error('Complete finite local collision vertices are required');
    const world=mesh.verticesLocal.map(p=>quatRotateOne(q,p).map((x,i)=>x+positionWorld[i]));
    all.push(...world);return{geomId:mesh.geomId,hull:convexHull(world),...bounds(world)};
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
