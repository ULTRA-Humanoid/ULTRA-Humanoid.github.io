/**
 * Narrow post-task geometry admission for the movable object that was just
 * placed successfully. Physics contacts remain enabled; this only removes the
 * exact retired target from ordinary W/Q reference-clearance inputs.
 */
const COMMAND_KEYS = ['forward', 'backward', 'left', 'right', 'turnLeft', 'turnRight'];
const clock = value => Number.isSafeInteger(value) && value >= 0;

function exactCommand(intent) {
  if (intent?.type !== 'keys' || !clock(intent.revision) || intent.revision === 0
      || !intent.keys || COMMAND_KEYS.some(key => typeof intent.keys[key] !== 'boolean')) return null;
  const held = COMMAND_KEYS.filter(key => intent.keys[key]);
  if (held.length !== 1) return null;
  return held[0] === 'forward' ? 'W' : held[0] === 'turnLeft' ? 'Q' : null;
}

const summary = token => token ? {
  episodeVersion: token.episodeVersion, requestId: token.requestId,
  targetBodyId: token.targetBodyId, targetBodyName: token.targetBodyName,
  issuedAtControl: token.issuedAtControl,
} : null;

export class PostTaskTargetContactAdmission {
  constructor() { this.serial = 0; this.token = null; this.events = []; }

  _event(event, detail = {}) {
    this.events.push({ event, ...detail });
    if (this.events.length > 32) this.events.shift();
  }

  revoke(reason) {
    if (this.token) this._event('revoked', { reason, ...summary(this.token) });
    this.token = null;
  }

  issue(candidate) {
    const retirement = candidate?.retirement;
    const valid = candidate?.task === 'carry'
      && candidate.taskCompletionReason === 'finished'
      && candidate.exitCompletionReason === 'finished'
      && candidate.placementGoalReached === true
      && candidate.requestDisposition === 'outcome'
      && candidate.requestOutcomeReason === 'finished'
      && retirement?.reason === 'box_exit_complete'
      && retirement.requestId === candidate.requestId
      && retirement.task === candidate.task
      && retirement.episodeVersion === candidate.episodeVersion
      && candidate.latestRequestId === candidate.requestId
      && candidate.queuedRequestId === null
      && clock(candidate.episodeVersion) && clock(candidate.requestId) && candidate.requestId > 0
      && clock(candidate.issuedAtControl)
      && Number.isSafeInteger(candidate.targetBodyId) && candidate.targetBodyId > 0
      && typeof candidate.targetBodyName === 'string' && candidate.targetBodyName.length > 0;
    if (!valid) { this.revoke('retirement_not_qualified'); return null; }
    if (this.token && this.token.episodeVersion === candidate.episodeVersion
        && this.token.requestId === candidate.requestId
        && this.token.targetBodyId === candidate.targetBodyId) return summary(this.token);
    this.revoke('new_successful_retirement');
    this.token = Object.freeze({ serial: ++this.serial,
      episodeVersion: candidate.episodeVersion, requestId: candidate.requestId,
      targetBodyId: candidate.targetBodyId, targetBodyName: candidate.targetBodyName,
      issuedAtControl: candidate.issuedAtControl });
    this._event('granted', summary(this.token));
    return summary(this.token);
  }

  selectObstacles(obstacles, context) {
    if (!Array.isArray(obstacles)) throw new Error('Obstacle bounds must be an array');
    const command = exactCommand(context?.intent), token = this.token;
    const current = token && command
      && context.episodeVersion === token.episodeVersion
      && context.activeBoxTask === null && context.boxTaskBusy === false
      && context.skillLoading === false;
    if (!current) return { obstacles, admitted: false, admission: null };
    const selected = obstacles.filter(obstacle => obstacle?.bodyId !== token.targetBodyId);
    if (selected.length === obstacles.length) return { obstacles, admitted: false, admission: null };
    return { obstacles: selected, admitted: true, admission: {
      ...summary(token), command, commandRevision: context.intent.revision,
      ignoredTargetObstacleCount: obstacles.length - selected.length,
    } };
  }

  review() { return { active: summary(this.token), events: structuredClone(this.events) }; }
}
