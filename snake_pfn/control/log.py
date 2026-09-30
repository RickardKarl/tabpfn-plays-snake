"""Bounded in-memory event log for the debug panel. Fed by the `snake_pfn` logger and by
`tabpfn_client`, whose retry warnings would otherwise make a slow request look like a hang."""

import logging
import threading
from collections import deque
from datetime import datetime, timezone

LOGGER = "snake_pfn"
LOGGERS = (LOGGER, "tabpfn_client")


class EventLog(logging.Handler):
    """Keeps the newest records so the UI can poll them; nothing is written to disk."""

    def __init__(self, capacity=1000):
        super().__init__(level=logging.DEBUG)
        self.entries = deque(maxlen=capacity)
        self.counter = 0
        self.lock_ = threading.Lock()
        for name in LOGGERS:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG if name == LOGGER else logging.INFO)
            logger.addHandler(self)

    def emit(self, record):
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + logging.Formatter().formatException(record.exc_info)
        with self.lock_:
            self.counter += 1
            self.entries.append(
                {
                    "id": self.counter,
                    "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                        timespec="milliseconds"
                    ),
                    "level": record.levelname.lower(),
                    "source": record.name.removeprefix(LOGGER + ".").split(".")[0] or LOGGER,
                    "message": message,
                }
            )

    def since(self, after=0, limit=500):
        with self.lock_:
            entries = [e for e in self.entries if e["id"] > after][:limit]
            return {"entries": entries, "latest": self.counter}

    def close(self):
        # Uvicorn's logging setup closes every existing handler; stay attached anyway.
        pass

    def detach(self):
        for name in LOGGERS:
            logging.getLogger(name).removeHandler(self)
        super().close()
