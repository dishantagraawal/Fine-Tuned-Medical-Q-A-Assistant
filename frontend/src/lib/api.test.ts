import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { getApiBaseUrl } from "./api";

describe("getApiBaseUrl", () => {
  it("strips trailing slash", () => {
    const prev = process.env.NEXT_PUBLIC_API_URL;
    process.env.NEXT_PUBLIC_API_URL = "http://localhost:8000/";
    assert.equal(getApiBaseUrl(), "http://localhost:8000");
    process.env.NEXT_PUBLIC_API_URL = prev;
  });

  it("falls back to default", () => {
    const prev = process.env.NEXT_PUBLIC_API_URL;
    delete process.env.NEXT_PUBLIC_API_URL;
    assert.equal(getApiBaseUrl(), "http://127.0.0.1:8000");
    process.env.NEXT_PUBLIC_API_URL = prev;
  });
});
