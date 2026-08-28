import { vi } from "vitest";

// jsdom 未实现 ResizeObserver，Radix 的 ScrollArea/Tabs 等组件会用到；
// 用空实现替代，保证组件级测试可以在 jsdom 下渲染。
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal("ResizeObserver", ResizeObserverStub);

// Radix Select 依赖指针捕获 API；jsdom 的 PointerEvent 实现不完整，
// 补齐空实现避免交互测试报 hasPointerCapture 错误。
if (typeof Element.prototype.hasPointerCapture !== "function") {
  Element.prototype.hasPointerCapture = () => false;
}
if (typeof Element.prototype.setPointerCapture !== "function") {
  Element.prototype.setPointerCapture = () => {};
}
if (typeof Element.prototype.releasePointerCapture !== "function") {
  Element.prototype.releasePointerCapture = () => {};
}
if (typeof Element.prototype.scrollIntoView !== "function") {
  Element.prototype.scrollIntoView = () => {};
}
