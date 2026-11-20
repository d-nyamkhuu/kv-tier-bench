// Continuous batching on an elastic pool. A group can skip directly to its
// next completion until a newly ready request needs an earlier step boundary.
export function decodePool(events, options, profile) {
  const groups = [];
  let occupied = 0, peakGroups = 0;

  function dispatch(group, time) {
    group.active.push(...group.pending);
    group.pending.length = 0;
    const batch = group.active.length;
    const context = group.active.reduce((sum, task) => sum + task.ctx, 0) / batch;
    group.rate = profile(batch, context);
    group.start = time;
    let steps = Infinity;
    for (const task of group.active) {
      task.first ??= time + group.rate.step;
      steps = Math.min(steps, Math.ceil(task.left / group.rate.tokens));
    }
    planBoundary(group, steps);
  }

  function planBoundary(group, steps) {
    const generation = ++group.generation;
    const { step, tokens, plain } = group.rate;
    group.end = group.start + steps * step;
    events.at(group.end, 0, time => {
      if (group.generation !== generation) return;
      const batch = group.active.length;
      for (const task of group.active) {
        task.left -= steps * tokens;
        task.steps += steps;
        task.decodeSeconds += steps * step;
        task.batchSteps += steps * batch;
        task.gpuSeconds += steps * step / batch;
        task.accepted += steps * tokens;
        task.plain += steps * plain;
        if (task.left <= 1e-8) task.done(time, task);
      }
      group.active = group.active.filter(task => task.left > 1e-8);
      group.rate = null;
      if (!group.active.length && !group.pending.length) occupied--;
      else events.at(time, 3, now => dispatch(group, now));
    });
  }

  function submit(time, task) {
    const allows = group => {
      const batch = group.active.length + group.pending.length + 1;
      if (batch === 1) return true;
      if (batch > options.batch) return false;
      // Reserve for the longest sequence at its peak context. This conservative
      // bound also applies to requests waiting for the next step.
      const peak = Math.max(task.peak, ...group.active.map(t => t.peak), ...group.pending.map(t => t.peak));
      if (options.batchCap && batch > options.batchCap(peak)) return false;
      return !options.speed || profile(batch, peak).perToken <= 1 / options.speed;
    };
    let group = groups.find(allows);
    if (!group) {
      group = { active: [], pending: [], generation: 0, rate: null };
      groups.push(group);
    }
    Object.assign(task, { steps: 0, decodeSeconds: 0, batchSteps: 0, gpuSeconds: 0, accepted: 0, plain: 0 });
    if (!group.active.length && !group.pending.length) {
      occupied++;
      events.at(time, 3, now => dispatch(group, now));
    } else if (group.rate) {
      const steps = Math.max(1, Math.ceil((time - group.start) / group.rate.step));
      if (group.start + steps * group.rate.step < group.end) planBoundary(group, steps);
    }
    if (task.measured) peakGroups = Math.max(peakGroups, occupied);
    group.pending.push(task);
  }

  return { submit, get peakGroups() { return peakGroups; } };
}
