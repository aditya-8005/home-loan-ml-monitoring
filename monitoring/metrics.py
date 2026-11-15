"""Metrics collection and drift monitoring utilities."""

import time
from threading import Lock
from typing import Any, Dict, Optional


class PredictionMetricsTracker:
    """Thread-safe tracker for real-time serving metrics and prediction statistics."""

    def __init__(self, start_time: Optional[float] = None) -> None:
        self._lock = Lock()
        self.start_time: float = start_time if start_time is not None else time.time()
        self.total_predictions: int = 0
        self.successful_predictions: int = 0
        self.failed_predictions: int = 0
        self.approved_count: int = 0
        self.rejected_count: int = 0
        self.total_latency_ms: float = 0.0

    def record_prediction(
        self,
        loan_status: Optional[str] = None,
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """Record an incoming prediction event.

        Parameters
        ----------
        loan_status : str | None
            Predicted loan status ('Y' or 'N').
        latency_ms : float
            Elapsed prediction latency in milliseconds.
        success : bool
            Whether prediction succeeded without internal server errors.
        """
        with self._lock:
            self.total_predictions += 1
            if success:
                self.successful_predictions += 1
                self.total_latency_ms += latency_ms
                if loan_status in ("Y", "1", 1, "Approved", True):
                    self.approved_count += 1
                elif loan_status in ("N", "0", 0, "Rejected", False):
                    self.rejected_count += 1
            else:
                self.failed_predictions += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Compute aggregated metrics dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing prediction counts, approval rates, and latency.
        """
        with self._lock:
            uptime = max(0.0, time.time() - self.start_time)
            avg_latency = (
                (self.total_latency_ms / self.successful_predictions)
                if self.successful_predictions > 0
                else 0.0
            )
            approval_rate = (
                (self.approved_count / self.successful_predictions)
                if self.successful_predictions > 0
                else 0.0
            )
            rejection_rate = (
                (self.rejected_count / self.successful_predictions)
                if self.successful_predictions > 0
                else 0.0
            )

            return {
                "total_predictions": self.total_predictions,
                "successful_predictions": self.successful_predictions,
                "failed_predictions": self.failed_predictions,
                "approved_predictions": self.approved_count,
                "rejected_predictions": self.rejected_count,
                "approval_rate": round(approval_rate, 4),
                "rejection_rate": round(rejection_rate, 4),
                "average_latency_ms": round(avg_latency, 2),
                "uptime_seconds": round(uptime, 2),
            }

    def reset(self) -> None:
        """Reset counters to initial state."""
        with self._lock:
            self.start_time = time.time()
            self.total_predictions = 0
            self.successful_predictions = 0
            self.failed_predictions = 0
            self.approved_count = 0
            self.rejected_count = 0
            self.total_latency_ms = 0.0


# Global singleton instance for metrics collection
metrics_tracker = PredictionMetricsTracker()
