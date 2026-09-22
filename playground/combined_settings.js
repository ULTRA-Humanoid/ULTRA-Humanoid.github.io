// Classic script: applies defaults before either static module evaluates.
(function(){
  'use strict';
  const defaults=Object.freeze({"debug":"1","restrictedControl":"1","restrictedBackward":"1","restrictedKeyTerminal":"1","restrictedStartupStanding":"1","restrictedApproachRouting":"1","restrictedApproachRecovery":"1","restrictedApproachTerminal":"0","restrictedStudentApproach":"1","restrictedStudentTransport":"1","restrictedLongCarry":"1","restrictedCarryLibrary":"1","taskCoverageCapture":"1","review":"recovery","terminalRefusalRecovery":"1","recoveredFacingTurn":"1","firstPickupStudentRecovery":"1","firstPickupStandingRecovery":"1","firstPickupStandingReanchor":"1","combinedPickupRegion":"1"});
  // Release candidate profile: opt-in via profile=release only. The bare entry stays default-OFF for these three.
  // v4: quietEndings removed from the profile (T20 H001 baseline loss traced to the shortened task-1 ending); default-OFF applies.
  // v5: longClipLibrary removed too (P100 H083 baseline loss traced to the 2-segment long-clip plan); placementCorrection retained.
  // v9 (Phase B · B2): quietExitSettling=onRequest — the FINAL exit's settling ends early only for a click that arrives during it; frozen P100/T20 streams unchanged.
  // v13a (Phase B B9 on B2 v9b; release union A + B2 on v5): skillArbiter added to the release profile only; code default stays OFF (v5 carry-only routing); the 20 defaults are untouched.
  // Phase C.1 focused integration: v15 release flags plus the certified v6j
  // Suitcase router. No Plasticbox, long-carry, or Kick behavior flag is set.
  const releaseProfile=Object.freeze({"placementCorrection":"1","skillArbiter":"1","quietExitSettling":"onRequest","objectClassRouting":"1"});
  const url=new URL(location.href),original=new URLSearchParams(url.search),applied=[],appliedProfile=[];
  const profile=original.get('profile');
  for(const[key,value]of Object.entries(defaults))if(!url.searchParams.has(key)){url.searchParams.set(key,value);applied.push(key);}
  if(profile==='release')for(const[key,value]of Object.entries(releaseProfile))if(!url.searchParams.has(key)){url.searchParams.set(key,value);applied.push(key);appliedProfile.push(key);}
  const differences=Object.entries(defaults).filter(([key,value])=>url.searchParams.get(key)!==value).map(([key,expected])=>({key,expected,actual:url.searchParams.get(key)}));
  const extraParameters=[...original.keys()].filter(key=>!(key in defaults)&&key!=='paused'&&key!=='profile');
  if(applied.length)history.replaceState(history.state,'',url.pathname+url.search+url.hash);
  window.__combinedEntryConfiguration=Object.freeze({version:"combined16_frozen_20260914",label:'Combined carry preview · experimental',
    defaults,appliedDefaults:applied,explicitDifferences:differences,extraParameters,customSettings:differences.length>0||extraParameters.length>0,
    profile:profile??null,releaseProfile,appliedProfileDefaults:appliedProfile,
    query:Object.fromEntries(url.searchParams),ordinaryAnimationDefault:true,explicitPaused:original.get('paused'),
    sourceGraph:'src',baselineEntryUnchanged:true,browserOrtWebParityVerified:false,newTaskCoverageClaim:false});
})();
