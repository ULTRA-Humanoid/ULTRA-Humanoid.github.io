/** Propose supported final walking endpoints inside the existing pickup arrival
 * region. No source/action execution, relaxed geometry, or changed box goal. */
import {planPickupFacingApproach} from './pickup_facing_approach.js';

export const PICKUP_ENTRY_REGION_OFFSETS_M=Object.freeze([
 [0,0],[.08,0],[.16,0],[0,-.08],[0,.08],
 [.08,-.08],[.08,.08],[.16,-.08],[.16,.08],
].map(p=>Object.freeze(p)));
export const PICKUP_ENTRY_REGION_POSITION_MARGIN_M=.06;

const compact=p=>({supported:p.supported,reason:p.reason,
 selectedSourceFrames:p.selected?.sourceFrames??null,
 candidates:p.candidates.map(c=>({sourceFrames:c.sourceFrames,reason:c.reason,supported:c.supported,
  terminalPositionErrorM:c.expectedTerminalPositionErrorM,terminalFacingErrorRad:c.expectedTerminalFacingErrorRad,
  requiredEntryPose:Array.from(c.requiredEntryPose),referenceTerminalPose:Array.from(c.referenceTerminalPose),
  wholeGeometry:c.wholeGeometry.supported,detailedGeometry:c.detailedGeometry.supported,
  terminalGeometry:c.terminalStandingGeometry.supported,stagingRouteExact:c.transit.exactEntry}))});

export function planPickupFacingEntryRegion(options){
 const original=planPickupFacingApproach(options);
 const attempts=[{offsetM:[0,0],...compact(original)}];
 // Preserve an already supported complete programme and all its transforms.
 if(original.supported)return Object.freeze({...original,entryRegionAttempts:Object.freeze(attempts),
  entryRegionChanged:false});
 if(original.reason!=='no_supported_pickup_facing_final_source')return original;
 const parent=options.parent,goal=parent.approachGoalWorld,box=options.live.objPosWorld;
 const dx=goal[0]-box[0],dy=goal[1]-box[1],length=Math.hypot(dx,dy);
 if(!(length>0))return original;
 const ux=dx/length,uy=dy/length,proposals=[];
 for(const [radial,tangent] of PICKUP_ENTRY_REGION_OFFSETS_M.slice(1)){
  if(Math.hypot(radial,tangent)>parent.arrivalRadius-PICKUP_ENTRY_REGION_POSITION_MARGIN_M)continue;
  const terminalGoalWorld=[goal[0]+radial*ux-tangent*uy,goal[1]+radial*uy+tangent*ux,goal[2]];
  const p=planPickupFacingApproach({...options,terminalGoalWorld});
  attempts.push({offsetM:[radial,tangent],terminalGoalWorld,...compact(p)});
  // The only executed final walking record remains the existing254-control
  // source. Other lengths are recorded, not silently admitted by this trial.
  if(p.supported&&p.selected.sourceFrames===254)proposals.push({p,offsetM:[radial,tangent],
   displacementM:Math.hypot(radial,tangent)});
 }
 proposals.sort((a,b)=>a.displacementM-b.displacementM
  ||b.p.selected.entryPointClearanceBeyondTransitM-a.p.selected.entryPointClearanceBeyondTransitM
  ||a.p.selected.transit.lengthM-b.p.selected.transit.lengthM);
 const best=proposals[0];
 return Object.freeze({...best?.p??original,entryRegionAttempts:Object.freeze(attempts),
  entryRegionChanged:Boolean(best),entryRegionSelectedOffsetM:best?Object.freeze(best.offsetM):null,
  entryRegionRemainingNominalMarginM:best?parent.arrivalRadius-best.displacementM:null});
}
