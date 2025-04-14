# utils/progress.py

import threading

class ProgressTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = {}

    def update_status(self, agent, ticker, message):
        with self.lock:
            if agent not in self.status:
                self.status[agent] = {}
            self.status[agent][ticker] = message
            print(f"[{agent}] {ticker}: {message}")

    def get_status(self):
        with self.lock:
            return self.status.copy()

    def reset(self):
        with self.lock:
            self.status.clear()

# Singleton instance
progress = ProgressTracker()
