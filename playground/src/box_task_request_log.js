/** Read-only request lifecycle diagnostics. No controller reads this log.
 * Outcome means the box task result; its separately tracked exit may still run.
 * Terminal records survive reset and stale asynchronous loader callbacks.
 */
const terminal = new Set(['refused', 'superseded', 'reset', 'outcome']);

export class BoxTaskRequestLog {
  constructor() { this.records = new Map(); }

  begin(requestId, task, goalWorld, clock, semanticGoalType = null) {
    this.records.set(requestId, { requestId, episodeVersion: clock.episodeVersion,
      task, goalWorld: goalWorld ? Array.from(goalWorld) : null,
      semanticGoalType: semanticGoalType ?? (goalWorld ? 'carry_destination' : `${task}_task`), events: [] });
    this.transition(requestId, 'received', 'user_request', clock);
  }

  transition(requestId, disposition, reason, clock, detail = {}) {
    const record = this.records.get(requestId);
    if (!record || terminal.has(record.disposition)) return;
    // A stale callback cannot attach a new episode's result to an old request.
    if (clock.episodeVersion !== record.episodeVersion) return;
    const event = { ...clock, disposition, reason, ...structuredClone(detail) };
    record.events.push(event); record.disposition = disposition; record.reason = reason;
  }

  closeUnstarted(disposition, reason, clock, { exceptRequestId = null, statuses = ['received', 'queued', 'loading'] } = {}) {
    for (const record of this.records.values())
      if (record.requestId !== exceptRequestId && statuses.includes(record.disposition))
        this.transition(record.requestId, disposition, reason, clock);
  }

  reset(clock) {
    for (const record of this.records.values())
      this.transition(record.requestId, 'reset', 'episode_reset', clock);
  }

  snapshot() { return structuredClone([...this.records.values()]); }
}
