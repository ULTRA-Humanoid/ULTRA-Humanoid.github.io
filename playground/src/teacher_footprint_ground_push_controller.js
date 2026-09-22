// Private source092 trial. Only a predeclared footprint branch changes the
// reference alignment; existing complete-source/task/exit ownership remains.
import { TeacherGroundPushController } from './teacher_ground_push_controller.js';
import { GroundPushGoalSequenceController } from './ground_push_sequence.js';
import { planAlignedGroundPush } from './ground_push_mapping.js';

export function groundPushBranch(skill) {
  const branch=skill?.pushFootprintAlignment;
  if(branch?.kind!=='compiled_collision_footprint_branch'
      ||branch.sourceFrames!==skill.sourceFrames
      ||!Number.isFinite(branch.objectHeadingOffsetRadians)
      ||!Array.isArray(branch.sourceInitialObjectQuaternionXyzw)
      ||branch.sourceInitialObjectQuaternionXyzw.length!==4
      ||branch.sourceInitialObjectQuaternionXyzw.some((v,i)=>Math.fround(v)!==skill.frames[0][74+i])) {
    throw new Error('The exact complete source must bind a predeclared collision-footprint branch');
  }
  return Object.freeze({...branch,sourceInitialObjectQuaternionXyzw:Object.freeze(Array.from(branch.sourceInitialObjectQuaternionXyzw))});
}

export class FootprintGroundPushController extends TeacherGroundPushController {
  constructor(skill,goalWorld,options={}) {
    const branch=groundPushBranch(skill);super(skill,goalWorld,options);
    this.pushFootprintBranch=branch;this.alignmentMode='compiled-collision-footprint-branch';
  }
  _referencePlan(proprio) {
    return planAlignedGroundPush(this.skill.frames,this.skill.sourceFrames,proprio.objPosWorld,proprio.objQuatXyzwWorld,this.requestedGoalWorld,
      {objectHeadingOffsetRadians:this.pushFootprintBranch.objectHeadingOffsetRadians,
        startFrame:this.warpStartFrame,endFrame:this.warpEndFrame,maxCorrection:this.maxCorrection});
  }
}

export class FootprintGroundPushSequenceController extends GroundPushGoalSequenceController {
  constructor(skill,goalWorld,options={}) {
    groundPushBranch(skill);super(skill,goalWorld,options);
  }
  _startSegment(proprio) {
    if(!this._liveDistanceSupported(proprio)){this._complete('unsupported_live_distance');return;}
    const target=this.plan.goals[this.segmentIndex],entry=this.references[this.activeReferenceIndex];
    this.triedReferences=new Set([this.activeReferenceIndex]);
    this.child=new FootprintGroundPushController(entry.skill,target,this._segmentOptions(entry));
    this.child.start(proprio);this.phase='approach';this.settlingCount=0;
  }
}
