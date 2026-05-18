import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BONSAI_NODE_TYPES = new Set([
  "BonsaiChatNode",
  "BonsaiCsvTagSelectorNode",
  "BonsaiDirectTagGeneratorNode",
]);

let startPromise = null;

function isBonsaiNode(node) {
  return BONSAI_NODE_TYPES.has(node?.comfyClass);
}

function startBonsaiServer() {
  if (startPromise !== null) {
    return startPromise;
  }

  startPromise = api
    .fetchApi("/bonsai/start", { method: "POST" })
    .catch((error) => {
      startPromise = null;
      console.error("[bonsai_node] Bonsai server start failed", error);
    });

  return startPromise;
}

app.registerExtension({
  name: "bonsai_node.start_on_node_added",
  async nodeCreated(node) {
    if (!isBonsaiNode(node)) {
      return;
    }
    await startBonsaiServer();
  },
});
