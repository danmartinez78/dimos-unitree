# Agents API Reference

Complete API documentation for DIMOS agent classes.

## Table of Contents

1. [Agent Base Classes](#agent-base-classes)
2. [OpenAIAgent](#openaiagent)
3. [PlanningAgent](#planningagent)
4. [ClaudeAgent](#claudeagent)
5. [Memory Interfaces](#memory-interfaces)

## Agent Base Classes

### Agent

Base class for all DIMOS agents, managing memory and subscriptions.

```python
class Agent:
    def __init__(self,
                 dev_name: str = "NA",
                 agent_type: str = "Base",
                 agent_memory: Optional[AbstractAgentSemanticMemory] = None,
                 pool_scheduler: Optional[ThreadPoolScheduler] = None)
```

**Parameters:**
- `dev_name` (str): Agent identifier for logging and debugging
- `agent_type` (str): Type classification (e.g., "Base", "Vision", "Planning")
- `agent_memory` (AbstractAgentSemanticMemory): Optional semantic memory system
- `pool_scheduler` (ThreadPoolScheduler): Thread pool for async operations. If None, uses global scheduler

**Attributes:**
- `dev_name` (str): Agent name
- `agent_type` (str): Agent type
- `agent_memory` (AbstractAgentSemanticMemory): Memory instance
- `disposables` (CompositeDisposable): Resource cleanup manager
- `pool_scheduler` (ThreadPoolScheduler): Execution scheduler

**Methods:**

#### dispose_all()
```python
def dispose_all(self) -> None
```
Disposes of all active subscriptions managed by the agent.

---

### LLMAgent

Abstract base class for LLM-based agents.

```python
class LLMAgent(Agent):
    def __init__(self,
                 dev_name: str = "NA",
                 input_query_stream: Optional[Observable] = None,
                 input_video_stream: Optional[Observable] = None,
                 system_query: str = "",
                 skills: Union[AbstractSkill, List[AbstractSkill], SkillLibrary] = None,
                 agent_memory: Optional[AbstractAgentSemanticMemory] = None,
                 pool_scheduler: Optional[ThreadPoolScheduler] = None)
```

**Additional Parameters:**
- `input_query_stream` (Observable[str]): Stream of text queries
- `input_video_stream` (Observable[np.ndarray]): Stream of video frames
- `system_query` (str): System prompt defining agent behavior
- `skills` (Union[AbstractSkill, List, SkillLibrary]): Available skills/tools

**Additional Attributes:**
- `skill_library` (SkillLibrary): Registered skills
- `response_subject` (Subject): Observable for agent responses

**Additional Methods:**

#### get_response_observable()
```python
def get_response_observable(self) -> Observable
```
Get observable stream of agent responses for chaining.

**Returns:** Observable[str] emitting agent responses

**Example:**
```python
planner = PlanningAgent(...)
executor = OpenAIAgent(
    input_query_stream=planner.get_response_observable()
)
```

## OpenAIAgent

Agent powered by OpenAI's GPT models (GPT-4, GPT-4o, GPT-4o-mini, etc.).

### Constructor

```python
class OpenAIAgent(LLMAgent):
    def __init__(self,
                 dev_name: str = "NA",
                 input_query_stream: Optional[Observable] = None,
                 input_video_stream: Optional[Observable] = None,
                 input_video_url_stream: Optional[Observable] = None,
                 system_query: str = "",
                 agent_memory: Optional[AbstractAgentSemanticMemory] = None,
                 max_input_tokens_per_request: int = 2000,
                 max_output_tokens_per_request: int = 2000,
                 model_name: str = "gpt-4o-mini",
                 prompt_builder: Optional[PromptBuilder] = None,
                 tokenizer: Optional[AbstractTokenizer] = None,
                 rag_query_n: int = 4,
                 rag_similarity_threshold: Optional[float] = None,
                 skills: Union[AbstractSkill, List[AbstractSkill], SkillLibrary] = None,
                 response_model: Optional[BaseModel] = None,
                 frame_processor: Optional[FrameProcessor] = None,
                 image_detail: str = "auto",
                 pool_scheduler: Optional[ThreadPoolScheduler] = None,
                 process_all_inputs: Optional[bool] = None)
```

### Parameters

#### Input/Output
- **`input_query_stream`** (Observable[str]): Text query input stream
- **`input_video_stream`** (Observable[np.ndarray]): Video frame stream for vision models
- **`input_video_url_stream`** (Observable[str]): Video URL stream (alternative to frame stream)
- **`system_query`** (str): System prompt defining agent persona and behavior

#### Model Configuration
- **`model_name`** (str): OpenAI model identifier
  - Text-only: `"gpt-4o-mini"`, `"gpt-4"`
  - Vision: `"gpt-4o"`, `"gpt-4-turbo"`, `"gpt-4o-mini"`
  - Default: `"gpt-4o-mini"`

- **`max_input_tokens_per_request`** (int): Maximum tokens for input context (default: 2000)
- **`max_output_tokens_per_request`** (int): Maximum tokens for response (default: 2000)

#### Vision Configuration
- **`image_detail`** (str): Image processing quality
  - `"low"`: Faster, cheaper, less detail (512x512)
  - `"high"`: Slower, more expensive, better detail
  - `"auto"`: Automatic based on image size (default)
  
- **`frame_processor`** (FrameProcessor): Custom video frame processing

#### Memory & RAG
- **`agent_memory`** (AbstractAgentSemanticMemory): Semantic memory instance
- **`rag_query_n`** (int): Number of memory items to retrieve (default: 4)
- **`rag_similarity_threshold`** (float): Minimum similarity for retrieval (0-1, optional)

#### Skills
- **`skills`** (Union[AbstractSkill, List, SkillLibrary]): Available skills for tool calling
  - Single skill class: `skills=MySkill`
  - List of skills: `skills=[Skill1, Skill2]`
  - Skill library: `skills=MySkillLibrary()`

#### Advanced
- **`prompt_builder`** (PromptBuilder): Custom prompt construction logic
- **`tokenizer`** (AbstractTokenizer): Custom token counter
- **`response_model`** (BaseModel): Pydantic model for structured outputs
- **`pool_scheduler`** (ThreadPoolScheduler): Custom thread pool
- **`process_all_inputs`** (bool): Whether to process all inputs or skip when busy

### Methods

#### run_query()
```python
def run_query(self, query: str, 
              base64_image: Optional[str] = None,
              thinking_budget_tokens: int = 0) -> str
```

Execute a single query synchronously.

**Parameters:**
- `query` (str): Text query
- `base64_image` (str): Optional base64-encoded image
- `thinking_budget_tokens` (int): Extended thinking tokens (o1 models)

**Returns:** str - Agent response

**Example:**
```python
agent = OpenAIAgent(
    dev_name="QueryAgent",
    skills=robot.get_skills(),
    model_name="gpt-4o-mini"
)

response = agent.run_query("Move forward 2 meters")
print(response)
```

#### run_observable_query()
```python
def run_observable_query(self, query: str,
                        base64_image: Optional[str] = None,
                        thinking_budget_tokens: int = 0) -> Observable
```

Execute query and return response as observable.

**Parameters:** Same as `run_query()`

**Returns:** Observable[str] - Response stream

**Example:**
```python
response_stream = agent.run_observable_query("What do you see?")
response_stream.subscribe(
    on_next=lambda r: print(f"Response: {r}")
)
```

### Usage Examples

#### Basic Text Agent
```python
from dimos.agents.agent import OpenAIAgent
from reactivex.subject import Subject

query_subject = Subject()

agent = OpenAIAgent(
    dev_name="TextAgent",
    input_query_stream=query_subject.pipe(),
    system_query="You are a helpful assistant.",
    model_name="gpt-4o-mini"
)

query_subject.on_next("Hello!")
```

#### Vision Agent
```python
agent = OpenAIAgent(
    dev_name="VisionAgent",
    input_query_stream=query_stream,
    input_video_stream=robot.get_ros_video_stream(),
    system_query="Describe what you see and navigate safely.",
    model_name="gpt-4o",
    image_detail="high"
)
```

#### Agent with Skills
```python
agent = OpenAIAgent(
    dev_name="SkillAgent",
    input_query_stream=query_stream,
    skills=robot.get_skills(),
    system_query="Execute user commands using your skills.",
    model_name="gpt-4o-mini"
)
```

#### Agent with Memory
```python
from dimos.agents.memory.chroma_impl import OpenAISemanticMemory

memory = OpenAISemanticMemory(collection_name="robot_memory")
memory.create()

agent = OpenAIAgent(
    dev_name="MemoryAgent",
    input_query_stream=query_stream,
    agent_memory=memory,
    rag_query_n=5,
    rag_similarity_threshold=0.7,
    model_name="gpt-4o"
)
```

## PlanningAgent

Specialized agent for breaking down complex tasks into executable steps.

### Constructor

```python
class PlanningAgent(OpenAIAgent):
    def __init__(self,
                 dev_name: str = "PlanningAgent",
                 input_query_stream: Optional[Observable] = None,
                 skills: Union[AbstractSkill, List[AbstractSkill], SkillLibrary] = None,
                 model_name: str = "gpt-4o",
                 system_query: str = None,
                 **kwargs)
```

### Parameters

- **`dev_name`** (str): Agent identifier (default: "PlanningAgent")
- **`input_query_stream`** (Observable[str]): User queries
- **`skills`** (SkillLibrary): Available skills for planning
- **`model_name`** (str): LLM model (default: "gpt-4o")
- **`system_query`** (str): Custom system prompt (optional)
- **`**kwargs`**: Additional OpenAIAgent parameters

### Default System Prompt

If not provided, uses a specialized planning prompt that instructs the agent to:
1. Break down complex tasks into steps
2. Consider available skills
3. Create logical, executable sequences
4. Output one step at a time

### Usage Example

```python
from dimos.agents.planning_agent import PlanningAgent
from dimos.agents.agent import OpenAIAgent

# Planning agent
planner = PlanningAgent(
    dev_name="MissionPlanner",
    input_query_stream=web_interface.query_stream,
    skills=robot.get_skills(),
    model_name="gpt-4o"
)

# Execution agent
executor = OpenAIAgent(
    dev_name="StepExecutor",
    input_query_stream=planner.get_response_observable(),
    skills=robot.get_skills(),
    model_name="gpt-4o-mini",
    system_query="Execute the given step."
)

# User: "Patrol the perimeter"
# Planner outputs: "Step 1: Move forward 5 meters"
# Executor executes: Move(distance=5)
# Planner outputs: "Step 2: Turn left 90 degrees"
# Executor executes: SpinLeft(degrees=90)
# ...
```

## ClaudeAgent

Agent powered by Anthropic's Claude models.

### Constructor

```python
class ClaudeAgent(LLMAgent):
    def __init__(self,
                 dev_name: str = "NA",
                 input_query_stream: Optional[Observable] = None,
                 input_video_stream: Optional[Observable] = None,
                 system_query: str = "",
                 agent_memory: Optional[AbstractAgentSemanticMemory] = None,
                 max_input_tokens_per_request: int = 2000,
                 max_output_tokens_per_request: int = 2000,
                 model_name: str = "claude-3-5-sonnet-20241022",
                 rag_query_n: int = 4,
                 rag_similarity_threshold: Optional[float] = None,
                 skills: Union[AbstractSkill, List[AbstractSkill], SkillLibrary] = None,
                 thinking_budget_tokens: int = 0,
                 pool_scheduler: Optional[ThreadPoolScheduler] = None)
```

### Parameters

Similar to OpenAIAgent with Claude-specific differences:

- **`model_name`** (str): Claude model identifier
  - `"claude-3-5-sonnet-20241022"` (default)
  - `"claude-3-opus-20240229"`
  - `"claude-3-sonnet-20240229"`
  - `"claude-3-haiku-20240307"`

- **`thinking_budget_tokens`** (int): Extended thinking tokens for reasoning

### Key Differences from OpenAI

1. **Tool calling format**: Uses Claude's native tool format
2. **Vision handling**: Different image format requirements
3. **API authentication**: Requires `ANTHROPIC_API_KEY` environment variable

### Usage Example

```python
from dimos.agents.claude_agent import ClaudeAgent

agent = ClaudeAgent(
    dev_name="ClaudeNavigator",
    input_query_stream=query_stream,
    input_video_stream=robot.get_ros_video_stream(),
    skills=robot.get_skills(),
    system_query="You are a navigation assistant.",
    model_name="claude-3-5-sonnet-20241022"
)
```

## Memory Interfaces

### AbstractAgentSemanticMemory

Base interface for memory implementations.

```python
class AbstractAgentSemanticMemory(ABC):
    def __init__(self, connection_type: str = 'local')
    
    @abstractmethod
    def connect(self)
    
    @abstractmethod
    def create(self)
    
    @abstractmethod
    def add_vector(self, vector_id: str, vector_data: str)
    
    @abstractmethod
    def get_vector(self, vector_id: str)
    
    @abstractmethod
    def query(self, query_texts: str, 
              n_results: int = 4,
              similarity_threshold: Optional[float] = None)
    
    @abstractmethod
    def delete_vector(self, vector_id: str)
```

### OpenAISemanticMemory

ChromaDB-backed memory using OpenAI embeddings.

```python
class OpenAISemanticMemory(ChromaAgentSemanticMemory):
    def __init__(self,
                 collection_name: str = "my_collection",
                 model: str = "text-embedding-3-large",
                 dimensions: int = 1024)
```

**Parameters:**
- `collection_name` (str): ChromaDB collection name
- `model` (str): OpenAI embedding model
- `dimensions` (int): Embedding vector dimensions

**Example:**
```python
from dimos.agents.memory.chroma_impl import OpenAISemanticMemory

memory = OpenAISemanticMemory(
    collection_name="robot_observations",
    model="text-embedding-3-large",
    dimensions=1024
)
memory.create()

# Store
memory.add_vector("obs1", "Saw a red ball in the living room")

# Query
results = memory.query("objects in living room", n_results=5)
```

### LocalSemanticMemory

ChromaDB-backed memory using local SentenceTransformer models.

```python
class LocalSemanticMemory(ChromaAgentSemanticMemory):
    def __init__(self,
                 collection_name: str = "my_collection",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2")
```

**Parameters:**
- `collection_name` (str): ChromaDB collection name
- `model_name` (str): HuggingFace model identifier

**Example:**
```python
from dimos.agents.memory.chroma_impl import LocalSemanticMemory

memory = LocalSemanticMemory(
    collection_name="local_memory",
    model_name="sentence-transformers/all-mpnet-base-v2"
)
memory.create()
```

## Common Patterns

### Agent Chaining

```python
# Plan → Execute pattern
planner = PlanningAgent(
    input_query_stream=user_input,
    skills=robot.get_skills()
)

executor = OpenAIAgent(
    input_query_stream=planner.get_response_observable(),
    skills=robot.get_skills()
)
```

### Multi-Input Agent

```python
from dimos.stream.stream_merger import create_stream_merger

# Merge text and video
merged = create_stream_merger(
    text_stream=query_stream,
    video_stream=video_stream,
    scheduler=pool_scheduler
)

agent = OpenAIAgent(
    input_query_stream=merged,
    skills=robot.get_skills(),
    model_name="gpt-4o"
)
```

### Response Monitoring

```python
agent = OpenAIAgent(...)

# Subscribe to responses
agent.get_response_observable().subscribe(
    on_next=lambda r: print(f"Agent said: {r}"),
    on_error=lambda e: print(f"Error: {e}")
)
```

## Error Handling

All agents handle errors gracefully:

```python
agent.get_response_observable().subscribe(
    on_next=lambda response: handle_response(response),
    on_error=lambda error: handle_error(error)
)

def handle_error(error):
    if isinstance(error, TimeoutError):
        print("Agent timed out")
    elif isinstance(error, APIError):
        print("API error occurred")
    else:
        print(f"Unexpected error: {error}")
```

## Environment Variables

Required environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic (for ClaudeAgent)
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: API base URL override
export OPENAI_API_BASE="https://custom.api.endpoint"
```

---

**Related Documentation:**
- [Integration Guide](../guides/integration.md)
- [Skills API](skills.md)
- [Memory Guide](../guides/memory.md)
