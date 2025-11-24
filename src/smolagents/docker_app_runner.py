"""Runner script for executing agents inside Docker containers for SEC-bench evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Union

# Import smolagents - should be installed locally via install_local_smolagents before this script runs
from smolagents import CodeAgent, ToolCallingAgent
from smolagents.default_tools import TOOL_MAPPING
from smolagents.models import InferenceClientModel, LiteLLMModel, OpenAIModel, TransformersModel
from smolagents.monitoring import LogLevel


def _build_model(model_config: dict[str, Any]) -> Any:
    """Build a model from configuration."""
    model_type = model_config.get("type", "InferenceClientModel")
    model_id = model_config.get("model_id", "")

    if model_type == "InferenceClientModel":
        return InferenceClientModel(
            model_id=model_id,
            token=model_config.get("api_key") or os.getenv("HF_API_KEY"),
            provider=model_config.get("provider"),
        )
    elif model_type == "OpenAIModel":
        return OpenAIModel(
            model_id=model_id,
            api_key=model_config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            api_base=model_config.get("api_base"),
        )
    elif model_type == "LiteLLMModel":
        return LiteLLMModel(
            model_id=model_id,
            api_key=model_config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            api_base=model_config.get("api_base"),
        )
    elif model_type == "TransformersModel":
        return TransformersModel(model_id=model_id, device_map="auto")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _build_tools(tool_names: list[str]) -> list[Any]:
    """Build tools from tool names."""
    tools = []
    for tool_name in tool_names:
        if tool_name in TOOL_MAPPING:
            tools.append(TOOL_MAPPING[tool_name]())
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    return tools


def _write_meta_json(artifacts_dir: str, agent: Union[ToolCallingAgent, CodeAgent], result: Any) -> None:
    """Write metadata JSON file for the agent run."""
    try:
        # Ensure artifacts directory exists
        out_dir = Path(artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract model name
        model_name = getattr(agent.model, "model_id", None) or agent.model.__class__.__name__

        # Extract agent name
        agent_name = agent.__class__.__name__

        # Extract tool names
        tools_attr = getattr(agent, "tools", {})
        if isinstance(tools_attr, dict):
            tool_names = sorted(list(tools_attr.keys()))
        elif isinstance(tools_attr, list):
            tool_names = sorted([getattr(t, "name", t.__class__.__name__) for t in tools_attr])
        else:
            tool_names = []

        # Count steps and aggregate token usage from agent.memory.steps
        steps_count = 0
        input_tokens = 0
        output_tokens = 0

        for step in agent.memory.steps:
            # Count steps that have step_number (ActionStep)
            if hasattr(step, "step_number"):
                steps_count += 1

            # Extract token usage from step
            tu = getattr(step, "token_usage", None)
            if tu is not None:
                input_tokens += getattr(tu, "input_tokens", 0)
                output_tokens += getattr(tu, "output_tokens", 0)

        # If result has token_usage, prefer that (more accurate aggregation)
        if hasattr(result, "token_usage") and result.token_usage is not None:
            input_tokens = result.token_usage.input_tokens
            output_tokens = result.token_usage.output_tokens

        # Calculate cost using litellm
        cost = 0.0
        try:
            from litellm import completion_cost

            fake_response = {
                "model": model_name.split("/")[-1],
                "usage": {"prompt_tokens": int(input_tokens), "completion_tokens": int(output_tokens)},
            }
            cost = float(completion_cost(fake_response))
        except Exception:
            # If litellm is not available or cost calculation fails, cost remains 0.0
            pass

        # Build metadata dictionary
        meta = {
            "model": model_name,
            "agent": agent_name,
            "tools": tool_names,
            "steps": steps_count,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        # Add docker_image if available
        docker_image = os.getenv("DOCKER_IMAGE")
        if docker_image:
            meta["docker_image"] = docker_image

        # Write meta.json using Path.write_text (more robust)
        meta_path = out_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8")

    except Exception as e:
        # Log warning but don't fail the entire run
        import traceback

        print(f"Warning: failed to write meta.json: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


def main() -> None:
    """Main entry point for the Docker container runner."""
    # Parse command-line arguments (fallback to environment variables)
    parser = argparse.ArgumentParser(description="Run smolagents application inside container")
    parser.add_argument("--config", help="Path to agent config JSON inside container")
    parser.add_argument("--task", help="Task string to run")
    parser.add_argument("--artifacts-dir", help="Directory to write trajectory and outputs")
    parser.add_argument("--max-steps", type=int, help="Optional max steps override")
    args = parser.parse_args()

    # Load agent config (prefer CLI arg, fallback to env var, then default)
    config_path = args.config or os.environ.get("SMOLAGENTS_CONFIG_PATH", "/app/agent_config.json")
    with open(config_path, "r") as f:
        agent_config = json.load(f)

    # Load task (prefer CLI arg, fallback to env var, then default)
    if args.task:
        task = args.task
    else:
        task_path = os.environ.get("SMOLAGENTS_TASK_PATH", "/app/task.txt")
        with open(task_path, "r") as f:
            task = f.read()

    # Create model
    model_config = agent_config.get("model", {})
    model = _build_model(model_config)

    # Create tools
    tools = _build_tools(agent_config.get("tools", []))

    # Create agent
    agent_type = agent_config.get("agent_type", "ToolCallingAgent")
    # Prefer CLI arg for max_steps, then config, then default
    max_steps = args.max_steps if args.max_steps is not None else agent_config.get("max_steps", 20)
    verbosity_level = LogLevel(agent_config.get("verbosity_level", 1))

    # Declare agent variable with union type to allow both agent types
    agent: Union[ToolCallingAgent, CodeAgent]
    if agent_type == "ToolCallingAgent":
        agent = ToolCallingAgent(
            tools=tools,
            model=model,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            stream_outputs=False,
        )
    elif agent_type == "CodeAgent":
        agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            additional_authorized_imports=agent_config.get("additional_imports", []),
            stream_outputs=False,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    # Run agent
    try:
        result = agent.run(task, return_full_result=True)

        # Extract output
        if hasattr(result, "output"):
            output = result.output
        else:
            output = result

        # Save result
        # Prefer CLI arg for artifacts_dir, fallback to env var, then default
        artifacts_dir = args.artifacts_dir or os.environ.get("SMOLAGENTS_ARTIFACTS_DIR", "/app/artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        with open(os.path.join(artifacts_dir, "output.json"), "w") as f:
            json.dump(
                {
                    "output": str(output) if output is not None else "",
                    "steps": result.steps if hasattr(result, "steps") else [],
                },
                f,
            )

        # Save trajectory
        if hasattr(result, "steps"):
            with open(os.path.join(artifacts_dir, "trajectory.jsonl"), "w") as f:
                for step in result.steps:
                    f.write(json.dumps(step) + "\n")

        # Save metadata
        _write_meta_json(artifacts_dir, agent, result)

        sys.exit(0)
    except Exception as e:
        import traceback

        # Prefer CLI arg for artifacts_dir, fallback to env var, then default
        artifacts_dir = args.artifacts_dir or os.environ.get("SMOLAGENTS_ARTIFACTS_DIR", "/app/artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(os.path.join(artifacts_dir, "error.txt"), "w") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
