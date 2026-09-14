import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  decideEngineRestart,
  isEngineWorking,
  parseHeartbeat,
  restartDelayMs,
  WATCHDOG_FAILURE_THRESHOLD,
} from "./supervisor.mjs";

describe("restartDelayMs", () => {
  it("按指数退避增长", () => {
    assert.equal(restartDelayMs(0), 1000, "首次重启等待 1 秒");
    assert.equal(restartDelayMs(1), 2000, "第二次等待 2 秒");
    assert.equal(restartDelayMs(2), 4000, "第三次等待 4 秒");
  });

  it("延时封顶在 30 秒，不再无限增长", () => {
    assert.equal(restartDelayMs(10), 30000, "高次数时封顶 30 秒");
    assert.equal(restartDelayMs(100), 30000, "极端次数仍封顶 30 秒");
  });
});

describe("parseHeartbeat", () => {
  it("读出在跑任务数与进展时间", () => {
    const text = JSON.stringify({ pid: 1, alive_at: 1, active_tasks: 2, progress_at: 100 });
    const beat = parseHeartbeat(text, 100_500, 101_000);
    assert.equal(beat.activeTasks, 2);
    assert.equal(beat.ageMs, 500);
    assert.equal(beat.progressAgeMs, 1000);
  });

  it("坏 JSON 或字段缺失一律当成没有心跳", () => {
    assert.equal(parseHeartbeat("{", 1, 2), null);
    assert.equal(parseHeartbeat("[]", 1, 2), null);
    const beat = parseHeartbeat(JSON.stringify({ active_tasks: 1 }), 0, 5_000);
    assert.equal(beat.progressAgeMs, null);
  });
});

describe("isEngineWorking", () => {
  it("有任务且在近期有进展才算在干活", () => {
    const now = 1_000_000;
    assert.equal(isEngineWorking({ activeTasks: 2, progressAgeMs: 5_000 }, now), true);
    assert.equal(isEngineWorking({ activeTasks: 2, progressAgeMs: 400_000 }, now), false);
    assert.equal(isEngineWorking({ activeTasks: 0, progressAgeMs: 1_000 }, now), false);
    assert.equal(isEngineWorking({ activeTasks: 2, progressAgeMs: null }, now), false);
    assert.equal(isEngineWorking(null, now), false);
  });
});

describe("decideEngineRestart", () => {
  const base = {
    probeFailures: WATCHDOG_FAILURE_THRESHOLD,
    suspiciousSinceMs: 0,
    working: false,
    heartbeatAgeMs: 1000,
    nowMs: 200_000,
  };

  it("探针没到阈值就不动手", () => {
    const verdict = decideEngineRestart({ ...base, probeFailures: 1 });
    assert.equal(verdict.restart, false);
    assert.equal(verdict.suspicious, false);
  });

  it("任务在推进时，即使探针一直失败也不许重启", () => {
    const verdict = decideEngineRestart({ ...base, working: true, nowMs: 9_999_999 });
    assert.equal(verdict.restart, false, "在干活的引擎不能被当成死掉");
    assert.match(verdict.reason, /仍在推进/);
  });

  it("首次发现无响应只观察，不立刻杀", () => {
    const verdict = decideEngineRestart({ ...base, suspiciousSinceMs: null });
    assert.equal(verdict.restart, false);
    assert.equal(verdict.suspicious, true, "要记下观察起点");
  });

  it("观察期未满不重启", () => {
    const verdict = decideEngineRestart({ ...base, suspiciousSinceMs: 150_000 });
    assert.equal(verdict.restart, false);
    assert.match(verdict.reason, /未达/);
  });

  it("持续无响应且没有进展才重启，并说明原因", () => {
    const verdict = decideEngineRestart({ ...base, suspiciousSinceMs: 10_000 });
    assert.equal(verdict.restart, true);
    assert.match(verdict.reason, /无响应/);
  });

  it("心跳也停了说明整个进程卡死，原因里点出来", () => {
    const verdict = decideEngineRestart({ ...base, suspiciousSinceMs: 10_000, heartbeatAgeMs: 60_000 });
    assert.equal(verdict.restart, true);
    assert.match(verdict.reason, /心跳停止/);
  });
});
