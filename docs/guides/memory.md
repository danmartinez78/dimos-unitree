# Semantic Memory & RAG Guide

Learn how to use DIMOS's semantic memory system for spatially-grounded reasoning and retrieval-augmented generation (RAG).

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Memory Implementations](#memory-implementations)
4. [Basic Usage](#basic-usage)
5. [Spatial Memory](#spatial-memory)
6. [RAG Integration](#rag-integration)
7. [Advanced Patterns](#advanced-patterns)
8. [Best Practices](#best-practices)

## Overview

DIMOS's semantic memory enables agents to:
- **Store observations** as embedded vectors in a vector database
- **Retrieve relevant context** based on semantic similarity
- **Ground memories spatially** by linking observations to robot locations
- **Augment LLM context** with retrieved memories (RAG)

### Key Benefits

- **Long-term memory**: Persist observations beyond single conversations
- **Semantic search**: Find relevant context even without exact keyword matches
- **Spatial reasoning**: "What did I see in the kitchen?" works because memories are location-tagged
- **Reduced hallucinations**: Ground agent responses in actual observations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent                                 │
│  ┌──────────┐    Query     ┌─────────────────────┐         │
│  │   LLM    │◄────────────│  Semantic Memory    │         │
│  │          │              │  (ChromaDB)         │         │
│  │          │   Context    │                     │         │
│  │          │◄─────────────┤  - Embeddings       │         │
│  └──────────┘              │  - Vector Store     │         │
│                            │  - Spatial Index    │         │
│                            └─────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ Store
                            ┌───────┴────────┐
                            │  Observations   │
                            │  + Locations    │
                            └────────────────┘
```

### Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **ChromaDB** | Vector database backend | `chromadb` library |
| **Embeddings** | Text → vectors | OpenAI, SentenceTransformers |
| **Memory Interface** | Abstract API | `AbstractAgentSemanticMemory` |
| **Spatial Index** | Location tagging | `SpatialMemory` |
| **RAG System** | Context retrieval | Built into agents |

## Memory Implementations

DIMOS provides three memory implementations:

### 1. OpenAISemanticMemory (Recommended for Production)

Uses OpenAI's embedding API for high-quality embeddings.

```python
from dimos.agents.memory.chroma_impl import OpenAISemanticMemory

memory = OpenAISemanticMemory(
    collection_name="robot_memory",
    model="text-embedding-3-large",  # Latest model
    dimensions=1024                  # Embedding dimensions
)
memory.create()  # Initialize
```

**Pros**:
- High-quality embeddings
- Fast inference
- No GPU required

**Cons**:
- Requires OpenAI API key
- API costs
- Requires internet connection

### 2. LocalSemanticMemory (Recommended for Offline/Development)

Uses local SentenceTransformer models.

```python
from dimos.agents.memory.chroma_impl import LocalSemanticMemory

memory = LocalSemanticMemory(
    collection_name="robot_memory",
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Fast, lightweight
)
memory.create()  # Downloads model if needed
```

**Pros**:
- No API costs
- Works offline
- Privacy (data stays local)

**Cons**:
- Slower on CPU
- Requires GPU for best performance
- Larger memory footprint

**Popular Models**:
- `all-MiniLM-L6-v2`: Fast, lightweight (384 dims)
- `all-mpnet-base-v2`: Better quality (768 dims)
- `multi-qa-mpnet-base-dot-v1`: Optimized for Q&A

### 3. Custom Memory Implementation

Implement your own backend:

```python
from dimos.agents.memory.base import AbstractAgentSemanticMemory

class MyCustomMemory(AbstractAgentSemanticMemory):
    """Custom memory implementation."""
    
    def __init__(self, config):
        super().__init__(connection_type='custom')
        self.config = config
    
    def connect(self):
        """Connect to your database."""
        self.db = MyDatabase(self.config)
    
    def create(self):
        """Initialize embedding model and database."""
        self.embedder = MyEmbedder()
        self.db.init()
    
    def add_vector(self, vector_id, vector_data):
        """Store a memory."""
        embedding = self.embedder.embed(vector_data)
        self.db.insert(vector_id, embedding, vector_data)
    
    def query(self, query_text, n_results=4, similarity_threshold=None):
        """Retrieve similar memories."""
        query_embedding = self.embedder.embed(query_text)
        results = self.db.search(query_embedding, k=n_results)
        return results
```

## Basic Usage

### Setting Up Memory with an Agent

```python
from dimos.agents.agent import OpenAIAgent
from dimos.agents.memory.chroma_impl import OpenAISemanticMemory

# 1. Create memory
memory = OpenAISemanticMemory(collection_name="my_robot_memory")
memory.create()

# 2. Create agent with memory
agent = OpenAIAgent(
    dev_name="MemoryAgent",
    input_query_stream=query_stream,
    agent_memory=memory,  # Pass memory instance
    skills=robot.get_skills(),
    model_name="gpt-4o",
    rag_query_n=5,  # Retrieve top 5 memories
    rag_similarity_threshold=0.7  # Min similarity score
)
```

### Storing Observations

#### Manual Storage

```python
# Store individual observations
memory.add_vector(
    vector_id="obs_001",
    vector_data="Saw a red ball in the living room"
)

memory.add_vector(
    vector_id="obs_002",
    vector_data="Detected a person near the doorway"
)

memory.add_vector(
    vector_id="obs_003",
    vector_data="Found an obstacle blocking the hallway"
)
```

#### Automatic Storage (via Agent)

Agents can automatically store observations:

```python
agent = OpenAIAgent(
    dev_name="ObservingAgent",
    input_query_stream=query_stream,
    input_video_stream=robot.get_ros_video_stream(),
    agent_memory=memory,
    skills=robot.get_skills(),
    system_query="""
    You are an observant robot. When you see something interesting,
    describe it concisely and I will store it in your memory.
    """,
    model_name="gpt-4o"
)

# Agent's observations are automatically stored
```

### Querying Memory

#### Direct Query

```python
# Search for similar memories
results = memory.query(
    query_text="What did I see in the living room?",
    n_results=5,
    similarity_threshold=0.7
)

# Process results
for doc, score in results:
    if score is not None:
        print(f"[{score:.2f}] {doc.page_content}")
    else:
        print(doc.page_content)
```

#### Automatic RAG (via Agent)

When RAG is enabled, agents automatically retrieve relevant memories:

```python
# Agent with RAG
agent = OpenAIAgent(
    dev_name="RAGAgent",
    input_query_stream=query_stream,
    agent_memory=memory,
    skills=robot.get_skills(),
    rag_query_n=5,  # Top 5 results
    rag_similarity_threshold=0.7,
    model_name="gpt-4o"
)

# User: "What did I tell you about the living room?"
# Agent automatically:
# 1. Queries memory with the question
# 2. Retrieves: "Saw a red ball in the living room"
# 3. Includes in context for LLM
# 4. Responds: "You told me you saw a red ball in the living room."
```

## Spatial Memory

**Spatial memory** links observations to robot locations, enabling location-based queries.

### Architecture

```python
from dimos.perception.spatial_perception import SpatialMemory

# Initialize spatial memory (happens in Robot.__init__)
spatial_memory = SpatialMemory(
    video_stream=robot.get_ros_video_stream(),
    transform_provider=lambda: robot.get_transform('base_link', 'map'),
    collection_name="spatial_observations",
    new_memory=False  # False to keep existing memories
)
```

### How It Works

```
1. Robot at (x=2.0, y=3.5) sees "red ball"
2. Observation stored with metadata: {"x": 2.0, "y": 3.5}
3. Later query: "What's at position (2, 3)?"
4. Retrieval filters by location proximity
5. Returns: "red ball"
```

### Using Spatial Queries

```python
# Query by location
results = memory.query(
    query_text="objects in living room",
    n_results=10,
    where={"x": {"$gte": 0, "$lte": 5}, "y": {"$gte": 0, "$lte": 5}}
)

# Or with agent
agent.run_query("What did I see in the kitchen?")
# Agent uses spatial context to filter memories
```

### Storing Spatial Observations

```python
from dimos.perception.spatial_perception import SpatialMemory

# Get current pose
pose = robot.get_pose()

# Store observation with location
memory.add_vector(
    vector_id=f"obs_{timestamp}",
    vector_data="Red ball on the floor",
    metadata={
        "x": pose.x,
        "y": pose.y,
        "theta": pose.theta,
        "timestamp": time.time()
    }
)
```

### Example: Navigation with Spatial Memory

```python
from dimos.agents.agent import OpenAIAgent
from dimos.agents.memory.chroma_impl import OpenAISemanticMemory

# Setup
memory = OpenAISemanticMemory(collection_name="nav_memory")
memory.create()

agent = OpenAIAgent(
    dev_name="NavAgent",
    input_query_stream=query_stream,
    input_video_stream=robot.get_ros_video_stream(),
    agent_memory=memory,
    skills=robot.get_skills(),
    system_query="""
    You are a navigation agent with spatial memory.
    Remember what you see at each location.
    When asked to navigate, recall relevant observations.
    """,
    model_name="gpt-4o",
    rag_query_n=5
)

# Usage
# User: "Go to where you saw the red ball"
# Agent:
# 1. Queries memory: "red ball"
# 2. Retrieves: "Red ball at (x=2.0, y=3.5)"
# 3. Navigates to (2.0, 3.5)
```

## RAG Integration

### How RAG Works in DIMOS

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Query Processing                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. User Query: "What did I see in the kitchen?"            │
│                                                              │
│  2. Retrieve from Memory:                                   │
│     ┌─────────────────────────────────────────────┐        │
│     │ memory.query("what in kitchen", n=5)        │        │
│     │ → ["red pot on stove",                       │        │
│     │    "white fridge with magnets",              │        │
│     │    "wooden table with chairs"]               │        │
│     └─────────────────────────────────────────────┘        │
│                                                              │
│  3. Build Context:                                          │
│     ┌─────────────────────────────────────────────┐        │
│     │ System: "You are a helpful robot..."        │        │
│     │ Memory Context:                              │        │
│     │   - red pot on stove                         │        │
│     │   - white fridge with magnets                │        │
│     │   - wooden table with chairs                 │        │
│     │ User: "What did I see in the kitchen?"      │        │
│     └─────────────────────────────────────────────┘        │
│                                                              │
│  4. LLM Response:                                           │
│     "In the kitchen, you saw a red pot on the stove,       │
│      a white fridge with magnets, and a wooden table."     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Configuration

```python
agent = OpenAIAgent(
    dev_name="RAGAgent",
    agent_memory=memory,
    
    # RAG Configuration
    rag_query_n=5,  # Number of memories to retrieve
    rag_similarity_threshold=0.7,  # Minimum similarity (0-1)
    
    # Token management
    max_input_tokens_per_request=4000,  # Total context budget
    
    model_name="gpt-4o"
)
```

### RAG Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rag_query_n` | 4 | Number of memories to retrieve |
| `rag_similarity_threshold` | None | Min cosine similarity (0-1) |
| `max_input_tokens_per_request` | 2000 | Total context budget |

### Example: Multi-Turn Conversation with Memory

```python
# Setup agent with memory
memory = OpenAISemanticMemory(collection_name="conversation_memory")
memory.create()

agent = OpenAIAgent(
    dev_name="ConversationAgent",
    input_query_stream=query_stream,
    agent_memory=memory,
    rag_query_n=5,
    model_name="gpt-4o",
    system_query="Remember our conversation and recall relevant details."
)

# Conversation
# Turn 1
agent.run_query("My favorite color is blue")
# Agent stores: "User's favorite color is blue"

# Turn 2 (later)
agent.run_query("What's my favorite color?")
# Agent retrieves memory and responds: "Your favorite color is blue."

# Turn 3
agent.run_query("I also like green")
# Agent stores: "User also likes green"

# Turn 4
agent.run_query("What colors do I like?")
# Agent retrieves both memories and responds:
# "You like blue and green."
```

## Advanced Patterns

### Pattern 1: Multi-Modal Memory

Store observations with images:

```python
import base64

def store_visual_observation(memory, frame, description, pose):
    """Store observation with visual data."""
    # Encode frame
    _, buffer = cv2.imencode('.jpg', frame)
    image_b64 = base64.b64encode(buffer).decode()
    
    # Store with metadata
    memory.add_vector(
        vector_id=f"obs_{time.time()}",
        vector_data=description,
        metadata={
            "x": pose.x,
            "y": pose.y,
            "image": image_b64,  # Store image
            "timestamp": time.time()
        }
    )

# Usage
frame = robot.get_latest_frame()
pose = robot.get_pose()
store_visual_observation(
    memory, frame,
    "Red ball on the floor",
    pose
)
```

### Pattern 2: Temporal Memory

Track changes over time:

```python
def store_temporal_observation(memory, observation, pose):
    """Store observation with temporal metadata."""
    timestamp = time.time()
    
    memory.add_vector(
        vector_id=f"obs_{timestamp}",
        vector_data=observation,
        metadata={
            "x": pose.x,
            "y": pose.y,
            "timestamp": timestamp,
            "date": datetime.now().isoformat()
        }
    )

# Query recent observations
recent_results = memory.query(
    query_text="objects seen",
    n_results=10,
    where={"timestamp": {"$gte": time.time() - 3600}}  # Last hour
)
```

### Pattern 3: Hierarchical Memory

Different memory levels:

```python
# Short-term memory (recent observations)
short_term = OpenAISemanticMemory(collection_name="short_term")
short_term.create()

# Long-term memory (important observations)
long_term = OpenAISemanticMemory(collection_name="long_term")
long_term.create()

def store_observation(observation, importance):
    """Store in appropriate memory."""
    # Always store in short-term
    short_term.add_vector(
        vector_id=f"st_{time.time()}",
        vector_data=observation
    )
    
    # Store important ones in long-term
    if importance > 0.8:
        long_term.add_vector(
            vector_id=f"lt_{time.time()}",
            vector_data=observation
        )

# Agent queries both
def query_all_memory(query, n_results=5):
    """Query both memory types."""
    st_results = short_term.query(query, n_results)
    lt_results = long_term.query(query, n_results)
    
    # Combine and deduplicate
    all_results = st_results + lt_results
    return all_results[:n_results]
```

### Pattern 4: Semantic Clustering

Group related observations:

```python
def cluster_observations(memory, n_clusters=5):
    """Cluster observations by semantic similarity."""
    from sklearn.cluster import KMeans
    
    # Get all embeddings
    all_docs = memory.db_connection.get()
    embeddings = [doc['embedding'] for doc in all_docs]
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings)
    
    # Group by cluster
    grouped = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in grouped:
            grouped[cluster_id] = []
        grouped[cluster_id].append(all_docs[idx]['text'])
    
    return grouped

# Usage
clusters = cluster_observations(memory, n_clusters=3)
for cluster_id, observations in clusters.items():
    print(f"\nCluster {cluster_id}:")
    for obs in observations:
        print(f"  - {obs}")
```

## Best Practices

### 1. Memory Organization

✅ **Do**:
- Use separate collections for different types of data
- Add rich metadata (location, time, importance)
- Clean up old observations periodically

❌ **Don't**:
- Store everything in one collection
- Forget to add metadata
- Let memory grow indefinitely

### 2. Embedding Quality

✅ **Do**:
```python
# Good observations (specific, descriptive)
memory.add_vector("obs1", "Red ball under the coffee table in the living room")
memory.add_vector("obs2", "White refrigerator with family photos on the left wall")
```

❌ **Don't**:
```python
# Bad observations (vague, unclear)
memory.add_vector("obs1", "thing")
memory.add_vector("obs2", "stuff here")
```

### 3. Query Design

✅ **Do**:
```python
# Specific queries
results = memory.query("red objects in the living room", n_results=5)
```

❌ **Don't**:
```python
# Overly broad queries
results = memory.query("stuff", n_results=100)
```

### 4. Similarity Thresholds

```python
# Tune based on use case
memory.query(
    query_text="kitchen items",
    n_results=10,
    similarity_threshold=0.7  # Experiment with this value
)

# Low threshold (0.5-0.6): More results, less relevant
# Medium threshold (0.7-0.8): Balanced
# High threshold (0.9+): Very few, highly relevant results
```

### 5. Memory Persistence

```python
# Persist memory to disk (ChromaDB does this automatically)
memory = OpenAISemanticMemory(collection_name="persistent_memory")
memory.create()

# Data is automatically persisted to:
# ./chroma/ (default location)

# To specify location:
import chromadb
client = chromadb.PersistentClient(path="/path/to/memory")
```

### 6. Memory Cleanup

```python
def cleanup_old_memories(memory, days=30):
    """Remove memories older than specified days."""
    cutoff = time.time() - (days * 24 * 60 * 60)
    
    # Get all documents
    all_docs = memory.db_connection.get()
    
    # Delete old ones
    for doc in all_docs:
        if doc.metadata.get('timestamp', float('inf')) < cutoff:
            memory.delete_vector(doc.id)

# Run periodically
cleanup_old_memories(memory, days=30)
```

## Troubleshooting

### Memory Not Retrieving Results

**Problem**: Queries return no results

**Solutions**:
```python
# 1. Check if data is stored
all_docs = memory.db_connection.get()
print(f"Total memories: {len(all_docs['ids'])}")

# 2. Lower similarity threshold
results = memory.query(query, similarity_threshold=0.5)  # Lower threshold

# 3. Check embedding model
print(f"Embedding model: {memory.embeddings}")
```

### Poor Quality Retrievals

**Problem**: Irrelevant memories retrieved

**Solutions**:
1. Improve observation descriptions (more specific)
2. Add metadata for filtering
3. Increase similarity threshold
4. Use better embedding model

### Memory Growing Too Large

**Problem**: ChromaDB size increasing

**Solutions**:
```python
# 1. Implement cleanup
cleanup_old_memories(memory, days=7)

# 2. Use multiple collections
recent = OpenAISemanticMemory(collection_name="recent")
archive = OpenAISemanticMemory(collection_name="archive")

# 3. Limit collection size
if len(memory.db_connection.get()['ids']) > 10000:
    # Archive oldest memories
    pass
```

## Next Steps

- **Perception**: Integrate vision with memory → [Perception Guide](perception.md)
- **Skills**: Create memory-aware skills → [Skills Guide](skills.md)
- **API Reference**: Detailed memory API → [Agents API](../api/agents.md)

---

**Previous**: [Observable Streams Guide](observables.md) | **Next**: [Perception & Vision Guide](perception.md)
