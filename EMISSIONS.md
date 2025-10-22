### Emitting metrics/events as NDJSON

This repository uses the ReSim Open Core emit helper to write metrics and events as newline-delimited JSON (NDJSON).

- **Library**: `resim-open-core`
- **Function**: `emit(topic_name, data, *, timestamp=None, timestamps=None, event=False, file_path=Path("/tmp/resim/outputs/emissions.ndjson"), file=None)`
- **Default output file**: `/tmp/resim/outputs/emissions.ndjson`

#### Installation

- **pip package**: `resim-open-core`
- **Python import module**: `resim_open_core`

Install:
```bash
pip install resim-open-core
```

#### Quick start

```python
from pathlib import Path
# If your environment provides this path (Open Core >= 0.15):
from resim.metrics.python.emit import emit

# Single datapoint at a specific time (nanoseconds since epoch)
emit(
    "battery/state",
    {"voltage_v": 11.9, "current_a": 3.2},
    timestamp=1736371200000000000,
)

# Mark an instantaneous event (must use a single timestamp)
emit(
    "mission/event",
    {"name": "takeoff_started"},
    timestamp=1736371200500000000,
    event=True,
)

# Series of datapoints with aligned timestamps
emit(
    "pose/position_m",
    {"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0], "z": [0.0, 0.1, 0.2]},
    timestamps=[
        1736371200000000000,
        1736371200100000000,
        1736371200200000000,
    ],
)

# Write to a custom file
emit(
    "demo/value",
    {"value": 42},
    timestamp=1736371200300000000,
    file_path=Path("/tmp/resim/outputs/demo.ndjson"),
)

# Or write to an already-open file handle
with open("/tmp/resim/outputs/session.ndjson", "a", encoding="utf8") as f:
    emit("gps/fix", {"lat": 37.4219999, "lon": -122.0840575}, timestamp=1736371200400000000, file=f)
```

#### Example emissions (NDJSON lines)

Each call writes one NDJSON line. The structure is:

```json
{"$metadata": {"topic": "<topic>", "timestamp": <int>, "event": <bool?>}, "$data": {<user fields>}}
```

Sample lines that could appear in `/tmp/resim/outputs/emissions.ndjson`:

```json
{"$metadata": {"topic": "battery/state", "timestamp": 1736371200000000000}, "$data": {"voltage_v": 11.9, "current_a": 3.2}}
{"$metadata": {"topic": "mission/event", "timestamp": 1736371200500000000, "event": true}, "$data": {"name": "takeoff_started"}}
{"$metadata": {"topic": "pose/position_m", "timestamp": 1736371200000000000}, "$data": {"x": 0.0, "y": 0.0, "z": 0.0}}
{"$metadata": {"topic": "pose/position_m", "timestamp": 1736371200100000000}, "$data": {"x": 1.0, "y": 0.0, "z": 0.1}}
{"$metadata": {"topic": "pose/position_m", "timestamp": 1736371200200000000}, "$data": {"x": 2.0, "y": 0.0, "z": 0.2}}
```

#### Argument behavior

- **topic_name**: Logical channel for the emission (string).
- **data**: A dictionary of user payload fields. For series emissions, each value must be a list and all lists must have the same length.
- **timestamp**: Integer nanoseconds since epoch (or a `Timestamp` object). Mutually exclusive with `timestamps`.
- **timestamps**: List/array of per-point timestamps. Must be 1D and the same length as each list in `data`. Mutually exclusive with `timestamp`.
- **event**: If `True`, the emission is annotated as an event. Events require a single `timestamp` (cannot be used with `timestamps`).
- **file_path**: Destination NDJSON file. Ignored if `file` is provided.
- **file**: Open file handle to write to. Mutually exclusive with `file_path`.

Notes:
- If neither `timestamp` nor `timestamps` are provided and all `data` values are lists of equal length, the helper will emit one line per list index by recursively calling itself with scalar values. These lines will not include a `timestamp` unless you provide one.
- If your environment provides a `Timestamp` object, it will be converted to nanoseconds if supported by the library.

#### Error checks

The helper validates common mistakes and will raise an error if:

- Both `timestamp` and `timestamps` are provided.
- `event=True` is used without a single `timestamp`.
- For series emissions, any `data` value is not a list or list lengths do not match `timestamps` length.
- `timestamps` is a NumPy array with dimension other than 1.


