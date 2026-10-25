const form = document.querySelector("#query-form");
const query = document.querySelector("#query");
const submit = document.querySelector("#submit");
const panel = document.querySelector("#answer-panel");
const answer = document.querySelector("#answer");
const sources = document.querySelector("#sources");
const meta = document.querySelector("#answer-meta");
const status = document.querySelector("#request-status");

document.querySelectorAll(".suggestions button").forEach((button) => {
  button.addEventListener("click", () => {
    query.value = button.textContent;
    query.focus();
  });
});

function setBusy(busy) {
  submit.disabled = busy;
  status.textContent = busy ? "Thinking" : "Ready";
  answer.classList.toggle("cursor", busy);
}

function parseEvent(block) {
  const lines = block.split("\n");
  const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
  const data = lines.filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim()).join("\n");
  return event && data ? { event, data: JSON.parse(data) } : null;
}

function renderComplete(result) {
  const stages = result.stage_latency_ms || {};
  const badges = [
    `${Math.round(result.latency_ms)} ms total`,
    result.cache === "miss" ? "fresh answer" : `${result.cache} cache`,
    result.citation_valid === true ? "citations valid" : null,
  ].filter(Boolean);
  meta.textContent = badges.join(" · ");
  sources.replaceChildren();
  result.sources.forEach((source) => {
    const card = document.createElement(source.source_url ? "a" : "div");
    card.className = "source";
    if (source.source_url) {
      card.href = source.source_url;
      card.target = "_blank";
      card.rel = "noreferrer";
    }
    const id = document.createElement("span");
    id.className = "source-id";
    id.textContent = source.source_id;
    const body = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = source.title;
    const domain = document.createElement("small");
    domain.textContent = source.domain;
    body.append(title, domain);
    const arrow = document.createElement("span");
    arrow.className = "source-arrow";
    arrow.textContent = source.source_url ? "↗" : "";
    card.append(id, body, arrow);
    sources.append(card);
  });
  status.textContent = `Complete · ${Object.keys(stages).length} stages measured`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = query.value.trim();
  if (!question) return;
  panel.hidden = false;
  answer.textContent = "";
  sources.replaceChildren();
  meta.textContent = "streaming";
  setBusy(true);

  try {
    const response = await fetch("/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: question,
        use_hyde: document.querySelector("#hyde").checked,
        use_reranking: document.querySelector("#reranking").checked,
        use_expansion: document.querySelector("#expansion").checked,
        use_cache: true,
      }),
    });
    if (!response.ok || !response.body) throw new Error("The server did not start a stream.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        const message = parseEvent(block);
        if (!message) continue;
        if (message.event === "token") answer.textContent += message.data.text;
        if (message.event === "complete") renderComplete(message.data);
        if (message.event === "error") throw new Error(message.data.message);
      }
      if (done) break;
    }
  } catch (error) {
    answer.textContent = error.message || "Something went wrong. Please try again.";
    meta.textContent = "request failed";
    status.textContent = "Error";
  } finally {
    setBusy(false);
  }
});

async function hydrateStats() {
  const [stats, metrics] = await Promise.all([
    fetch("/stats").then((response) => response.ok ? response.json() : null),
    fetch("/metrics").then((response) => response.ok ? response.json() : null),
  ]);
  if (stats) {
    document.querySelector("#document-count").textContent = stats.total_documents.toLocaleString();
    const counts = document.querySelector("#domain-counts");
    Object.entries(stats.by_domain).forEach(([domain, count]) => {
      const chip = document.createElement("span");
      chip.textContent = `${domain} ${count}`;
      counts.append(chip);
    });
  }
  const total = metrics?.latency_ms?.total;
  if (total?.samples) {
    document.querySelector("#p50").textContent = `${Math.round(total.p50)}ms`;
    document.querySelector("#p95").textContent = `${Math.round(total.p95)}ms`;
    document.querySelector("#sample-size").textContent = `${total.samples} completed queries`;
  }
}

hydrateStats().catch(() => {});
