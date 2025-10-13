# Observable Streams & Data Flow Guide

Understanding RxPY patterns and reactive data flow in DIMOS.

## Table of Contents

1. [Introduction to Reactive Programming](#introduction-to-reactive-programming)
2. [Core Observable Types](#core-observable-types)
3. [Creating Observables](#creating-observables)
4. [Transforming Streams](#transforming-streams)
5. [Agent Chaining](#agent-chaining)
6. [Common Patterns](#common-patterns)
7. [Best Practices](#best-practices)

## Introduction to Reactive Programming

DIMOS uses **RxPY (Reactive Extensions for Python)** for asynchronous data flow. This enables:

- **Asynchronous Processing**: Handle multiple data streams concurrently
- **Backpressure Management**: Control data flow rate
- **Stream Composition**: Combine and transform data streams
- **Event-Driven Architecture**: React to robot state changes in real-time

### Why Observables?

Traditional approach (polling):
```python
# ❌ Blocking, inefficient
while True:
    frame = camera.get_frame()
    process(frame)
    time.sleep(0.1)
```

Observable approach:
```python
# ✅ Non-blocking, efficient
camera.video_stream.subscribe(
    on_next=lambda frame: process(frame)
)
```

### Key Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Observable** | Stream of data over time | Video frames, user queries |
| **Observer** | Consumes observable data | Agent processing queries |
| **Subscription** | Connection between observable and observer | `stream.subscribe(...)` |
| **Operator** | Transforms observable data | `map()`, `filter()`, `debounce()` |
| **Subject** | Both observable and observer | User input stream |
| **Scheduler** | Controls execution context | ThreadPoolScheduler |

## Core Observable Types

DIMOS uses observables throughout the framework:

### 1. Video Frame Streams

```python
from reactivex import Observable
import numpy as np

# Type: Observable<np.ndarray>
video_stream: Observable = robot.get_ros_video_stream()

# Subscribe to video frames
video_stream.subscribe(
    on_next=lambda frame: print(f"Frame shape: {frame.shape}"),
    on_error=lambda e: print(f"Error: {e}"),
    on_completed=lambda: print("Stream ended")
)
```

### 2. Query Streams

```python
# Type: Observable<str>
query_stream: Observable = web_interface.query_stream

# Process queries
query_stream.subscribe(
    on_next=lambda query: print(f"Received: {query}")
)
```

### 3. State Streams

```python
# Type: Observable<RobotState>
state_stream: Observable = robot.get_state_observable()

# Monitor robot state
state_stream.subscribe(
    on_next=lambda state: print(f"Battery: {state.battery}%")
)
```

### 4. Response Streams

```python
# Type: Observable<str>
response_stream: Observable = agent.get_response_observable()

# Get agent responses
response_stream.subscribe(
    on_next=lambda response: print(f"Agent: {response}")
)
```

## Creating Observables

### Method 1: From ROS Topics (Built-in)

```python
# DIMOS provides ROS topic → Observable conversion
video_stream = robot.get_ros_video_stream()
state_stream = robot.get_state_observable()
```

### Method 2: From Events

```python
import reactivex as rx

def create_event_stream(event_source):
    """Create observable from event source."""
    def subscribe(observer, scheduler=None):
        def on_event(data):
            observer.on_next(data)
        
        # Register event handler
        event_source.register(on_event)
        
        # Return disposable for cleanup
        return lambda: event_source.unregister(on_event)
    
    return rx.create(subscribe)

# Usage
events = create_event_stream(my_event_source)
```

### Method 3: From Generator

```python
import reactivex as rx
import time

def create_sensor_stream(sensor):
    """Create observable from sensor readings."""
    def subscribe(observer, scheduler=None):
        def read_loop():
            try:
                while True:
                    reading = sensor.read()
                    observer.on_next(reading)
                    time.sleep(0.1)
            except Exception as e:
                observer.on_error(e)
        
        import threading
        thread = threading.Thread(target=read_loop, daemon=True)
        thread.start()
    
    return rx.create(subscribe)

# Usage
sensor_data = create_sensor_stream(my_sensor)
```

### Method 4: Using Subjects

```python
from reactivex.subject import Subject

# Subject is both Observable and Observer
query_subject = Subject()

# Observable interface (for consumers)
query_observable = query_subject.pipe()

# Observer interface (for producers)
query_subject.on_next("Move forward")
query_subject.on_next("Turn left")
query_subject.on_completed()
```

### Method 5: From Iterable

```python
import reactivex as rx

# Create observable from list
commands = rx.from_iterable(["move", "turn", "stop"])

commands.subscribe(
    on_next=lambda cmd: print(f"Command: {cmd}")
)
# Output:
# Command: move
# Command: turn
# Command: stop
```

## Transforming Streams

RxPY operators allow you to transform, filter, and combine observables.

### Map: Transform Each Item

```python
from reactivex import operators as ops

# Convert query to uppercase
upper_stream = query_stream.pipe(
    ops.map(lambda query: query.upper())
)

# Resize video frames
resized_stream = video_stream.pipe(
    ops.map(lambda frame: cv2.resize(frame, (640, 480)))
)
```

### Filter: Select Items

```python
# Only process queries longer than 5 characters
filtered_stream = query_stream.pipe(
    ops.filter(lambda query: len(query) > 5)
)

# Only bright frames (average pixel > threshold)
bright_frames = video_stream.pipe(
    ops.filter(lambda frame: np.mean(frame) > 128)
)
```

### Debounce: Rate Limiting

```python
# Ignore rapid queries (wait 1 second of silence)
debounced_stream = query_stream.pipe(
    ops.debounce(1.0)
)

# Good for handling rapid user input
```

### Sample: Fixed Rate Sampling

```python
# Sample video at 1 FPS (instead of 30 FPS)
sampled_stream = video_stream.pipe(
    ops.sample(1.0)  # 1 second interval
)

# Reduce computational load
```

### Take: Limit Number of Items

```python
# Process only first 10 frames
limited_stream = video_stream.pipe(
    ops.take(10)
)

# Process for 30 seconds
time_limited = video_stream.pipe(
    ops.take_until(rx.timer(30))
)
```

### Scan: Accumulate State

```python
# Count frames
frame_counter = video_stream.pipe(
    ops.scan(lambda count, frame: count + 1, seed=0)
)

# Running average of sensor readings
avg_stream = sensor_stream.pipe(
    ops.scan(
        lambda acc, reading: (acc[0] + reading, acc[1] + 1),
        seed=(0, 0)
    ),
    ops.map(lambda acc: acc[0] / acc[1] if acc[1] > 0 else 0)
)
```

### Combine Latest: Merge Multiple Streams

```python
# Combine video and state
combined = rx.combine_latest(
    video_stream,
    state_stream
).pipe(
    ops.map(lambda pair: {
        'frame': pair[0],
        'state': pair[1]
    })
)

# Now each emission has both video and state
```

### Merge: Interleave Multiple Streams

```python
# Merge multiple query sources
all_queries = rx.merge(
    web_interface.query_stream,
    voice_input.query_stream,
    cli.query_stream
)

# Single stream with all inputs
```

## Agent Chaining

One of DIMOS's most powerful features: chaining agents via observables.

### Basic Chain: Planning → Execution

```python
from dimos.agents.planning_agent import PlanningAgent
from dimos.agents.agent import OpenAIAgent

# 1. Planning Agent (breaks down tasks)
planner = PlanningAgent(
    dev_name="TaskPlanner",
    input_query_stream=web_interface.query_stream,
    skills=robot.get_skills(),
    model_name="gpt-4o",
    system_query="Break down complex tasks into steps"
)

# 2. Execution Agent (executes individual steps)
executor = OpenAIAgent(
    dev_name="StepExecutor",
    input_query_stream=planner.get_response_observable(),  # Chain!
    skills=robot.get_skills(),
    model_name="gpt-4o-mini",
    system_query="Execute the given step"
)

# User: "Patrol the perimeter"
# Planner: "1. Move forward 5m\n2. Turn left 90°\n3. Move forward 5m..."
# Executor: Receives each step and executes
```

### Multi-Agent Pipeline

```python
# 3-agent pipeline: Planner → Validator → Executor

planner = PlanningAgent(
    dev_name="Planner",
    input_query_stream=user_input,
    skills=robot.get_skills(),
    model_name="gpt-4o"
)

validator = OpenAIAgent(
    dev_name="Validator",
    input_query_stream=planner.get_response_observable(),
    skills=robot.get_skills(),
    model_name="gpt-4o",
    system_query="Validate plan safety. Only pass safe plans."
)

executor = OpenAIAgent(
    dev_name="Executor",
    input_query_stream=validator.get_response_observable(),
    skills=robot.get_skills(),
    model_name="gpt-4o-mini"
)
```

### Parallel Processing

```python
# Multiple specialized agents processing same input
general_query = query_subject.pipe()

# Navigation specialist
nav_agent = OpenAIAgent(
    dev_name="NavAgent",
    input_query_stream=general_query,
    skills=navigation_skills,
    system_query="Handle navigation queries"
)

# Manipulation specialist
manip_agent = OpenAIAgent(
    dev_name="ManipAgent",
    input_query_stream=general_query,
    skills=manipulation_skills,
    system_query="Handle manipulation queries"
)

# Vision specialist
vision_agent = OpenAIAgent(
    dev_name="VisionAgent",
    input_query_stream=general_query,
    input_video_stream=robot.get_ros_video_stream(),
    skills=vision_skills,
    system_query="Handle vision queries"
)
```

### Feedback Loops

```python
# Agent output → Processing → Back to agent

# Create feedback subject
feedback_subject = Subject()

# Merge original queries with feedback
combined_input = rx.merge(
    web_interface.query_stream,
    feedback_subject.pipe()
)

# Agent processes both sources
agent = OpenAIAgent(
    dev_name="FeedbackAgent",
    input_query_stream=combined_input,
    skills=robot.get_skills(),
    model_name="gpt-4o"
)

# Monitor responses and provide feedback
agent.get_response_observable().subscribe(
    on_next=lambda response: process_and_feedback(response, feedback_subject)
)

def process_and_feedback(response, feedback_subject):
    """Analyze response and provide feedback if needed."""
    if "error" in response.lower():
        feedback_subject.on_next("Previous command failed. Try alternative approach.")
```

## Common Patterns

### Pattern 1: Video Processing Pipeline

```python
from reactivex import operators as ops
import cv2

# Complete vision processing pipeline
processed_vision = robot.get_ros_video_stream().pipe(
    # Sample at 5 FPS
    ops.sample(0.2),
    
    # Resize frames
    ops.map(lambda frame: cv2.resize(frame, (640, 480))),
    
    # Convert to grayscale (if needed)
    ops.map(lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
    
    # Detect edges
    ops.map(lambda frame: cv2.Canny(frame, 100, 200)),
    
    # Filter out empty frames
    ops.filter(lambda frame: np.sum(frame) > 1000),
    
    # Log for debugging
    ops.do_action(lambda frame: print(f"Processed frame: {frame.shape}"))
)

# Subscribe to processed stream
processed_vision.subscribe(
    on_next=lambda frame: analyze(frame)
)
```

### Pattern 2: State Monitoring

```python
# Monitor robot health
health_monitor = rx.combine_latest(
    robot.get_state_observable(),
    robot.get_battery_observable(),
    robot.get_temperature_observable()
).pipe(
    ops.map(lambda data: {
        'state': data[0],
        'battery': data[1],
        'temp': data[2],
        'healthy': data[1] > 20 and data[2] < 80
    }),
    ops.distinct_until_changed(lambda health: health['healthy'])
)

# Alert on health changes
health_monitor.subscribe(
    on_next=lambda health: 
        print(f"⚠️ Health: {health}") if not health['healthy'] else None
)
```

### Pattern 3: Conditional Processing

```python
# Process queries differently based on content
query_stream.pipe(
    ops.group_by(lambda query: classify_query(query))
).subscribe(
    on_next=lambda grouped: grouped.subscribe(
        on_next=lambda query: process_by_type(grouped.key, query)
    )
)

def classify_query(query: str) -> str:
    """Classify query type."""
    if "move" in query.lower() or "navigate" in query.lower():
        return "navigation"
    elif "look" in query.lower() or "see" in query.lower():
        return "vision"
    else:
        return "general"

def process_by_type(query_type: str, query: str):
    """Route to appropriate agent."""
    if query_type == "navigation":
        nav_agent.process(query)
    elif query_type == "vision":
        vision_agent.process(query)
    else:
        general_agent.process(query)
```

### Pattern 4: Error Recovery

```python
# Robust stream with error recovery
robust_stream = video_stream.pipe(
    ops.catch(lambda error, source: 
        rx.concat(
            rx.of(None),  # Emit null frame on error
            rx.timer(1.0).pipe(ops.flat_map(lambda _: source))  # Retry after 1s
        )
    ),
    ops.filter(lambda frame: frame is not None),  # Skip null frames
    ops.retry(3)  # Retry up to 3 times
)
```

### Pattern 5: Buffering

```python
# Batch process frames
batched_stream = video_stream.pipe(
    ops.buffer_with_time(5.0)  # Collect 5 seconds of frames
)

batched_stream.subscribe(
    on_next=lambda batch: process_batch(batch)
)

def process_batch(frames):
    """Process a batch of frames."""
    print(f"Processing batch of {len(frames)} frames")
    # Batch analysis (e.g., motion detection across frames)
```

### Pattern 6: Throttling

```python
# Prevent overwhelming the system
throttled_stream = query_stream.pipe(
    ops.throttle_first(2.0)  # Allow max 1 query per 2 seconds
)

# Alternative: throttle_latest (keeps most recent)
latest_stream = video_stream.pipe(
    ops.throttle_latest(0.1)  # Max 10 FPS
)
```

## Best Practices

### 1. Always Dispose Subscriptions

```python
from reactivex.disposable import CompositeDisposable

class MyComponent:
    def __init__(self):
        self.disposables = CompositeDisposable()
    
    def start(self):
        # Add subscriptions to composite
        sub1 = stream1.subscribe(on_next=self.handler1)
        sub2 = stream2.subscribe(on_next=self.handler2)
        
        self.disposables.add(sub1)
        self.disposables.add(sub2)
    
    def stop(self):
        # Dispose all at once
        self.disposables.dispose()
```

### 2. Use Schedulers for Threading

```python
from reactivex.scheduler import ThreadPoolScheduler

# Create scheduler
pool_scheduler = ThreadPoolScheduler(max_workers=4)

# Use with observe_on
stream.pipe(
    ops.observe_on(pool_scheduler)  # Process on thread pool
).subscribe(
    on_next=lambda x: heavy_processing(x)
)
```

### 3. Handle Errors Gracefully

```python
stream.subscribe(
    on_next=lambda x: process(x),
    on_error=lambda e: handle_error(e),  # Don't ignore!
    on_completed=lambda: cleanup()
)

# Or use catch operator
safe_stream = stream.pipe(
    ops.catch(lambda error, source: rx.empty())  # Continue on error
)
```

### 4. Log Stream Activity

```python
# Add logging to debug streams
debug_stream = stream.pipe(
    ops.do_action(
        on_next=lambda x: logger.debug(f"Next: {x}"),
        on_error=lambda e: logger.error(f"Error: {e}"),
        on_completed=lambda: logger.info("Completed")
    )
)
```

### 5. Avoid Infinite Streams Without Backpressure

```python
# ❌ Bad: Infinite stream without rate limiting
infinite_stream = rx.interval(0.001)  # 1000 Hz!

# ✅ Good: Rate limited
controlled_stream = rx.interval(0.1).pipe(  # 10 Hz
    ops.take_until(rx.timer(60))  # Also add timeout
)
```

### 6. Test Observables Independently

```python
def test_observable():
    """Test observable behavior."""
    results = []
    
    # Create test observable
    test_stream = rx.from_iterable([1, 2, 3, 4, 5])
    
    # Transform
    processed = test_stream.pipe(
        ops.map(lambda x: x * 2),
        ops.filter(lambda x: x > 5)
    )
    
    # Collect results
    processed.subscribe(on_next=lambda x: results.append(x))
    
    # Verify
    assert results == [6, 8, 10], f"Expected [6, 8, 10], got {results}"
```

## Advanced Topics

### Custom Operators

```python
def custom_operator():
    """Create a custom operator."""
    def _operator(source):
        def subscribe(observer, scheduler=None):
            def on_next(value):
                # Custom processing
                transformed = custom_transform(value)
                observer.on_next(transformed)
            
            return source.subscribe(
                on_next,
                on_error=observer.on_error,
                on_completed=observer.on_completed,
                scheduler=scheduler
            )
        return rx.create(subscribe)
    return _operator

# Usage
stream.pipe(custom_operator()).subscribe(...)
```

### Hot vs Cold Observables

```python
# Cold Observable: Starts producing on subscription
cold = rx.from_iterable([1, 2, 3])

# Hot Observable: Always producing
from reactivex.subject import Subject
hot = Subject()

# Share cold observable (make it hot)
shared = cold.pipe(ops.share())
```

### Multicast and Publish

```python
# Prevent duplicate subscriptions
source = expensive_stream.pipe(
    ops.publish()  # Convert to ConnectableObservable
)

# Subscribe multiple consumers
source.subscribe(consumer1)
source.subscribe(consumer2)

# Start producing (connects to source)
source.connect()
```

## Troubleshooting

### Stream Not Emitting

**Problem**: Subscribed but no data

**Solutions**:
```python
# 1. Check if stream is started
print(f"Stream active: {stream}")

# 2. Add logging
stream.pipe(
    ops.do_action(lambda x: print(f"Emitted: {x}"))
).subscribe(...)

# 3. Check for errors
stream.subscribe(
    on_next=lambda x: print(x),
    on_error=lambda e: print(f"ERROR: {e}")  # Add error handler!
)
```

### Memory Leaks

**Problem**: Memory usage grows over time

**Solutions**:
```python
# 1. Dispose subscriptions
subscription = stream.subscribe(...)
# Later:
subscription.dispose()

# 2. Use take operators
stream.pipe(
    ops.take_until(stop_signal)
).subscribe(...)

# 3. Avoid holding references in closures
# ❌ Bad:
def create_handler():
    large_object = [0] * 1000000
    return lambda x: process(x, large_object)

# ✅ Good:
def create_handler():
    return lambda x: process(x)
```

### Backpressure Issues

**Problem**: System overwhelmed by data rate

**Solutions**:
```python
# 1. Sample at lower rate
stream.pipe(ops.sample(0.5))  # Max 2 Hz

# 2. Buffer and batch
stream.pipe(ops.buffer_with_time(1.0))

# 3. Throttle
stream.pipe(ops.throttle_first(1.0))

# 4. Drop when busy
stream.pipe(ops.observe_on(scheduler, max_pending=1))
```

## Examples in DIMOS

### From dimos/agents/agent.py

```python
# Input stream merger (text + video)
merged_stream = create_stream_merger(
    text_stream=input_query_stream,
    video_stream=input_video_stream,
    scheduler=pool_scheduler
)

# Process merged stream
merged_stream.subscribe(
    on_next=self._process_input,
    on_error=self._handle_error
)
```

### From dimos/skills/observe_stream.py

```python
# Periodic observation skill
interval_observable = rx.interval(
    self.timestep,
    scheduler=self._scheduler
).pipe(
    ops.take_while(lambda _: not self._stop_event.is_set())
)

interval_observable.subscribe(
    on_next=self._monitor_iteration,
    on_error=lambda e: logger.error(f"Error: {e}"),
    on_completed=lambda: logger.info("Observation complete")
)
```

## Next Steps

- **Memory Integration**: Use observables with semantic memory → [Memory Guide](memory.md)
- **Perception**: Observable-based vision pipelines → [Perception Guide](perception.md)
- **Integration**: Complete data flow → [Integration Guide](integration.md)

---

**Previous**: [Skills Extension Guide](skills.md) | **Next**: [Memory & RAG Guide](memory.md)
