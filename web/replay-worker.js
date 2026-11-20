import { replay } from "./replay.js";
import { compareTiers } from "./tier-comparison.js";

self.onmessage = ({ data }) => {
  const { id, kind, config, budgetFraction } = data;
  try {
    // Progress goes out once per whole percent, so the page's bar moves without flooding the channel.
    let last = -1;
    const onProgress = fraction => {
      const percent = Math.floor(fraction * 100);
      if (percent !== last) { last = percent; self.postMessage({ id, kind: "progress", fraction }); }
    };
    const result = kind === "comparison" ? compareTiers(config, data.layout, budgetFraction) : replay(config, budgetFraction, { onProgress });
    self.postMessage({ id, kind, result });
  } catch (error) {
    self.postMessage({ id, kind, error: error.message });
  }
};
