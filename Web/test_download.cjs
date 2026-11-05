const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const vm = require("node:vm");

const script = readFileSync(`${__dirname}/index.html`, "utf8").match(/<script>([\s\S]*?)<\/script>/)[1];
const result = {
  strokes: [[[110, 100], [110, 110]]], finished: true,
  gcode: "G21\nG90\nM5\n", warnings: [],
};

function page() {
  const elements = Object.fromEntries(
    ["connect", "clear", "download", "status", "completion-status", "gcode-status", "canvas"]
      .map(id => [id, { textContent: "", disabled: id === "download" }]),
  );
  const drawn = [];
  elements.canvas.width = 640;
  elements.canvas.height = 480;
  elements.canvas.getContext = () => ({
    beginPath() {}, moveTo() {}, stroke() {}, arc() {}, fill() {},
    lineTo(...point) { drawn.push(point); }, clearRect() { drawn.length = 0; },
  });
  const requests = [], downloads = [], revoked = [];
  let socket, blob;
  const context = vm.createContext({
    document: {
      getElementById: id => elements[id], body: { appendChild() {} },
      createElement: () => ({ click() { downloads.push({ name: this.download, blob }); }, remove() {} }),
    },
    WebSocket: class { constructor() { socket = this; } },
    fetch: (url, options) => new Promise(resolve => requests.push({
      body: JSON.parse(options.body),
      respond: (data = result, ok = true) => resolve({ ok, json: async () => data }),
    })),
    Blob, URL: {
      createObjectURL(value) { blob = value; return "blob:test"; },
      revokeObjectURL(url) { revoked.push(url); },
    },
    setTimeout: fn => fn(),
  });
  vm.runInContext(script, context);
  elements.connect.onclick();
  const send = data => socket.onmessage({ data: JSON.stringify(data) });
  send({ pen_down: true, x: 90, y: 100, width: 640, height: 480 });
  send({ pen_down: true, x: 100, y: 100, width: 640, height: 480 });
  return {
    elements, requests, downloads, revoked, drawn, send,
    async start() { send({ type: "COMPLETE" }); await Promise.resolve(); },
    done: () => vm.runInContext("completion", context),
  };
}

test("completion sends canvas dimensions and downloads the returned text", async () => {
  const p = page();
  await p.start();
  assert.deepEqual(p.requests[0].body, {
    strokes: [[[90, 100], [100, 100]]], width: 640, height: 480,
  });
  p.requests[0].respond();
  await p.done();
  assert.equal(p.elements.download.disabled, false);
  p.elements.download.onclick();
  assert.equal(p.downloads[0].name, "penpal.gcode");
  assert.equal(await p.downloads[0].blob.text(), result.gcode);
  assert.deepEqual(p.revoked, ["blob:test"]);

  await p.start();
  assert.equal(p.elements.download.disabled, true);
  p.requests[1].respond({ strokes: [], finished: true, gcode: "", warnings: [] });
  await p.done();
  assert.equal(p.elements.download.disabled, true);
  assert.equal(p.elements["gcode-status"].textContent, "No new lines to export.");
});

test("clear discards both active and queued completions", async () => {
  const p = page();
  await p.start();
  p.send({ type: "COMPLETE" });
  p.elements.clear.onclick();
  p.requests[0].respond();
  await p.done();
  assert.equal(p.requests.length, 1);
  assert.equal(p.drawn.length, 0);
  assert.equal(p.elements.download.disabled, true);
  assert.equal(p.elements["completion-status"].textContent, "");
  assert.equal(p.elements["gcode-status"].textContent, "");
});

test("camera resize discards pending output", async () => {
  const p = page();
  await p.start();
  p.send({ pen_down: true, x: 10, y: 10, width: 1280, height: 720 });
  p.requests[0].respond();
  await p.done();
  assert.equal(p.drawn.length, 0);
  assert.equal(p.elements.download.disabled, true);
});

test("warnings and request failures are shown without leaving an old download", async () => {
  const p = page();
  await p.start();
  p.requests[0].respond({ ...result, warnings: ["Example simulator warning"] });
  await p.done();
  assert.equal(p.elements["gcode-status"].textContent, "Example simulator warning");
  await p.start();
  p.requests[1].respond({ error: "Completion failed" }, false);
  await p.done();
  assert.equal(p.elements.download.disabled, true);
  assert.equal(p.elements["completion-status"].textContent, "Completion failed");
});
