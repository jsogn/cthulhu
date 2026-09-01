import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  restartDelayMs,
  shouldForceRestart,
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

describe("shouldForceRestart", () => {
  it("只有连续失败达到阈值才强制重启", () => {
    assert.equal(shouldForceRestart(0), false, "无失败不重启");
    assert.equal(shouldForceRestart(WATCHDOG_FAILURE_THRESHOLD - 1), false, "阈值前不重启");
    assert.equal(shouldForceRestart(WATCHDOG_FAILURE_THRESHOLD), true, "达到阈值即重启");
  });
});
