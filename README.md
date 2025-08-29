# AgentChord
  
AgentChord is a flexible framework for building, simulating, and optimizing multi-agent systems. It provides core abstractions and utilities to help you design, implement, and experiment with agent-based models efficiently.

## Features

- Gradient-Based Connections (GBC) for automataed Multi-Agent System (MAS) optimization.
- Handy visualization of Gradient-Based Connections (GBC).
- Connected to [WandB](https://wandb.ai/site/sdk) to easily monitor the optimization process.
- Based on [LiteLLM](https://github.com/BerriAI/litellm) to support APIs from various providers.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/AgentChord.git
    cd AgentChord
    ```

2. Make sure you have Astral UV installed in your environment. Please refer to [the official installation instruaction](https://docs.astral.sh/uv/getting-started/installation/) if you have not yet.

3. Synchronize the virtual environment using UV:
    ```bash
    uv sync
    ```

4. Configure the LiteLLM environment variables in `.env`.

5. Explore the `examples/` directory for ready-to-run demonstrations.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and new features.

## License

This project is licensed under the MIT License.