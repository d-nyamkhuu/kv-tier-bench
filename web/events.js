// A stable min-heap. Completions at a timestamp precede new lookups there.
export class Events {
  heap = [];
  serial = 0;

  before(a, b) {
    return a.t < b.t || (a.t === b.t && (a.priority < b.priority ||
      (a.priority === b.priority && a.serial < b.serial)));
  }

  at(t, priority, run) {
    if (!Number.isFinite(t)) throw new RangeError("Simulation time must be finite. Check the configured bandwidth and GPU rates.");
    const event = { t, priority, run, serial: this.serial++ };
    const heap = this.heap;
    let i = heap.length;
    while (i) {
      const parent = (i - 1) >> 1;
      if (!this.before(event, heap[parent])) break;
      heap[i] = heap[parent];
      i = parent;
    }
    heap[i] = event;
  }

  drain(advance) {
    const heap = this.heap;
    while (heap.length) {
      const first = heap[0], last = heap.pop();
      if (heap.length) {
        let i = 0;
        while (2 * i + 1 < heap.length) {
          let child = 2 * i + 1;
          if (child + 1 < heap.length && this.before(heap[child + 1], heap[child])) child++;
          if (!this.before(heap[child], last)) break;
          heap[i] = heap[child];
          i = child;
        }
        heap[i] = last;
      }
      advance(first.t);
      first.run(first.t);
    }
  }
}
