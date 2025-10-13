# DIMOS Framework Documentation

Welcome to the DIMOS (Dimensional OS) framework documentation. This guide will help you understand, extend, and integrate DIMOS into your robotics projects.

## 📚 Documentation Structure

### 🚀 Getting Started
- [Main README](../README.md) - Quick start and basic setup
- [Installation Guide](../README.md#python-quick-start-) - Detailed installation instructions

### 📖 Guides

#### Core Integration (Start Here)
- **[Agent-to-Robot Integration](guides/integration.md)** - Complete pipeline from natural language to robot actions
- **[Skills Extension](guides/skills.md)** - Creating custom skills and extending robot capabilities
- **[Observable Streams](guides/observables.md)** - RxPY patterns and data flow in DIMOS

#### Advanced Features
- **[Semantic Memory & RAG](guides/memory.md)** - Spatial reasoning and retrieval-augmented generation
- **[Perception & Vision](guides/perception.md)** - Vision pipelines, VLM integration, object detection
- **[Robot Platform Abstraction](guides/robot-platforms.md)** - Adapting DIMOS to new robot platforms

### 📑 API Reference
- **[Agents API](api/agents.md)** - OpenAIAgent, PlanningAgent, ClaudeAgent, and base classes
- **[Skills API](api/skills.md)** - AbstractSkill, AbstractRobotSkill, SkillLibrary
- **[Robot API](api/robot.md)** - Robot base class and platform implementations

### 🎓 Tutorials
- **[Basic Agent Setup](tutorials/basic-agent.md)** - Your first DIMOS agent
- **[Custom Skill Development](tutorials/custom-skill.md)** - Step-by-step skill creation
- **[Multi-Agent Orchestration](tutorials/multi-agent.md)** - Chaining agents for complex tasks

## 🎯 Quick Navigation by Use Case

### I want to...

**Build a basic robot agent**
1. Read [Agent-to-Robot Integration](guides/integration.md)
2. Follow [Basic Agent Setup](tutorials/basic-agent.md)
3. Reference [Agents API](api/agents.md)

**Create custom robot behaviors**
1. Read [Skills Extension](guides/skills.md)
2. Follow [Custom Skill Development](tutorials/custom-skill.md)
3. Reference [Skills API](api/skills.md)

**Add vision capabilities**
1. Read [Perception & Vision](guides/perception.md)
2. Reference [Agents API](api/agents.md) for vision-enabled agents

**Implement spatial memory**
1. Read [Semantic Memory & RAG](guides/memory.md)
2. Reference [Agents API](api/agents.md) for memory configuration

**Port DIMOS to a new robot**
1. Read [Robot Platform Abstraction](guides/robot-platforms.md)
2. Reference [Robot API](api/robot.md)

**Understand data flow**
1. Read [Observable Streams](guides/observables.md)
2. Read [Agent-to-Robot Integration](guides/integration.md)

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                   (Web UI / CLI / Voice)                         │
└────────────────┬────────────────────────────────────────────────┘
                 │ Observable<str> (queries)
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Layer                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐        │
│  │ Planning    │─▶│ Execution    │  │ Semantic Memory │        │
│  │ Agent       │  │ Agent        │  │ (ChromaDB)      │        │
│  └─────────────┘  └──────┬───────┘  └─────────────────┘        │
└────────────────────────────┼─────────────────────────────────────┘
                             │ Tool Calls (Skills)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Skills Layer                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Move     │  │ Navigate │  │ Detect   │  │ Custom   │       │
│  │ Skills   │  │ Skills   │  │ Objects  │  │ Skills   │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
└───────┼─────────────┼─────────────┼─────────────┼──────────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Robot Interface Layer                          │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │ ROS2 Control     │         │ WebRTC API       │             │
│  │ (velocity, pose) │         │ (behaviors)      │             │
│  └────────┬─────────┘         └─────────┬────────┘             │
└───────────┼───────────────────────────────┼─────────────────────┘
            │                               │
            ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Physical Robot                              │
│               (Unitree Go2, Custom Robot, etc.)                  │
└─────────────────────────────────────────────────────────────────┘
            │
            │ Feedback (state, video, sensors)
            ▼
    Observable Streams (RxPY)
```

## 🔑 Key Concepts

### Agents
Agents are LLM-powered decision-makers that process inputs, reason about goals, and invoke skills to achieve objectives. DIMOS supports multiple agent types (OpenAI, Claude, local models) and agent chaining.

### Skills
Skills are discrete robot capabilities (move, navigate, detect objects, etc.) that agents can invoke via tool calling. Skills are Pydantic models with validation and execution logic.

### Observable Streams
DIMOS uses RxPY for reactive programming, allowing asynchronous data flow between components (video frames, commands, state updates).

### Semantic Memory
ChromaDB-backed vector storage enables agents to store and retrieve spatially-grounded observations for context-aware reasoning.

## 📦 Project Structure Reference

```
dimos/
├── agents/           # Agent implementations and memory systems
│   ├── agent.py      # OpenAIAgent, LLMAgent base classes
│   ├── planning_agent.py
│   ├── claude_agent.py
│   └── memory/       # Semantic memory implementations
│       ├── base.py
│       └── chroma_impl.py
├── robot/            # Robot abstraction and platform implementations
│   ├── robot.py      # Robot base class
│   ├── ros_control.py
│   └── unitree/      # Unitree Go2 specific implementations
├── skills/           # Skill definitions and library
│   ├── skills.py     # AbstractSkill, SkillLibrary
│   ├── navigation.py
│   └── observe_stream.py
├── perception/       # Computer vision and sensing
│   ├── detection2d/
│   ├── spatial_perception.py
│   └── object_detection_stream.py
├── stream/           # Video and data streaming utilities
│   ├── frame_processor.py
│   └── video_providers/
└── web/              # Web interface and API
    └── dimos_interface/
```

## 🤝 Contributing

We welcome contributions to the documentation! If you find errors, have suggestions, or want to add examples:

1. Fork the repository
2. Create a documentation branch
3. Submit a pull request with your improvements

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/dimensionalOS/dimos-unitree/issues)
- **Email**: build@dimensionalOS.com
- **Community**: Join the [Roboverse Discord](https://discord.gg/HEXNMCNhEh)

## 📄 License

DIMOS is licensed under the Apache 2.0 License. See [LICENSE](../LICENSE) for details.

---

**Next Steps**: Start with the [Agent-to-Robot Integration Guide](guides/integration.md) to understand the complete pipeline from natural language to robot actions.
