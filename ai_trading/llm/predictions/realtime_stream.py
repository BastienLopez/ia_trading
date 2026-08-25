"""Consommateur P1 borné, injecté et observable pour P4 temps réel."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import pandas as pd


class P1PollingSource:
    """Adaptateur concret d'une source P1 historique/live vers le consommateur."""

    def __init__(self, provider: Any, asset: str, timeframe: str):
        self.provider, self.asset, self.timeframe = provider, asset, timeframe
        self._last_timestamp: Optional[str] = None

    def receive(self, timeout: float) -> Optional[Dict[str, Any]]:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p4-p1-poll")
        future = executor.submit(self.provider.fetch, self.asset, self.timeframe)
        try:
            frame = future.result(timeout=timeout)
        except FuturesTimeoutError as error:
            future.cancel()
            raise TimeoutError("P1 polling timeout") from error
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if frame is None or frame.empty:
            return None
        latest = frame.iloc[-1].to_dict()
        timestamp = latest.get("timestamp", getattr(frame.index[-1], "isoformat", lambda: str(frame.index[-1]))())
        timestamp = pd.Timestamp(timestamp)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        timestamp = timestamp.isoformat()
        if timestamp == self._last_timestamp:
            return None
        self._last_timestamp = timestamp
        if "close" not in latest and "price" in latest:
            latest["close"] = float(latest["price"])
        latest.update({"asset": latest.get("asset", self.asset), "timeframe": latest.get("timeframe", self.timeframe),
                       "timestamp": timestamp, "source": latest.get("source", "p1"),
                       "ohlcv_ready": all(key in latest and latest[key] is not None for key in ("open", "high", "low", "close", "volume"))})
        return latest


class P1StreamConsumer:
    """Consomme une source injectée ``receive(timeout)`` avec reprise explicite."""

    def __init__(self, source_factory: Callable[[], Any], handler: Callable[[Dict[str, Any]], Any],
                 max_queue_size: int = 128, receive_timeout: float = 1.0, max_reconnects: int = 3,
                 require_ohlcv: bool = True):
        self.source_factory, self.handler = source_factory, handler
        self.queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=max_queue_size)
        self.receive_timeout, self.max_reconnects = receive_timeout, max_reconnects
        self.require_ohlcv = require_ohlcv
        self.stop_event, self.thread = threading.Event(), None
        self.metrics = {"received": 0, "processed": 0, "dropped": 0, "errors": 0, "reconnects": 0,
                        "latency_seconds": [], "queue_max": 0, "state": "stopped", "cache_hits": 0}

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.metrics["state"] = "running"
        self.thread = threading.Thread(target=self._run, daemon=True, name="p4-p1-stream")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.receive_timeout * 2 + 1)
        self.metrics["state"] = "stopped"

    def _run(self) -> None:
        reconnects = 0
        while not self.stop_event.is_set():
            try:
                source = self.source_factory()
                reconnects = 0
                while not self.stop_event.is_set():
                    item = source.receive(timeout=self.receive_timeout)
                    if item is None:
                        continue
                    self.metrics["received"] += 1
                    item.setdefault("received_at", datetime.now(timezone.utc).isoformat())
                    if self.require_ohlcv and not item.get("ohlcv_ready", False):
                        self.metrics["errors"] += 1
                        self.metrics["state"] = "degraded"
                        continue
                    try:
                        self.queue.put_nowait(item)
                    except queue.Full:
                        self.metrics["dropped"] += 1
                        continue
                    self.metrics["queue_max"] = max(self.metrics["queue_max"], self.queue.qsize())
                    self._drain_one()
            except Exception:
                self.metrics["errors"] += 1
                reconnects += 1
                self.metrics["reconnects"] += 1
                if reconnects > self.max_reconnects:
                    self.metrics["state"] = "degraded"
                    return
                time.sleep(min(0.1 * (2 ** (reconnects - 1)), 1.0))

    def _drain_one(self) -> None:
        item = self.queue.get_nowait()
        started = time.monotonic()
        result = self.handler(item)
        self.metrics["processed"] += 1
        self.metrics["latency_seconds"].append(time.monotonic() - started)
        if isinstance(result, dict) and result.get("cache_hit"):
            self.metrics["cache_hits"] += 1

    def snapshot(self) -> Dict[str, Any]:
        latencies = self.metrics["latency_seconds"]
        return {**self.metrics, "queue_size": self.queue.qsize(), "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
                "memory_queue_capacity": self.queue.maxsize}
