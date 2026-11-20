import { replay } from "./replay.js";
import { compareTiers } from "./tier-comparison.js";

self.onmessage = ({ data }) => {
  const { id, kind, config, budgetFraction } = data;
  try {
    const result = kind === "comparison" ? compareTiers(config, data.layout, budgetFraction) : replay(config, budgetFraction);
    self.postMessage({ id, kind, result });
  } catch (error) {
    self.postMessage({ id, kind, error: error.message });
  }
};
