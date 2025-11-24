#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


try:
    import tomli
except ImportError:
    try:
        import tomllib as tomli  # type: ignore  # Python 3.11+
    except ImportError:
        raise ImportError("Please install 'tomli' package: pip install tomli")

# datasets is imported lazily in run_secb_evaluation to avoid import errors when not using secb-run

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table

from smolagents import (
    CodeAgent,
    InferenceClientModel,
    LiteLLMModel,
    Model,
    OpenAIModel,
    Tool,
    ToolCallingAgent,
    TransformersModel,
)
from smolagents.default_tools import TOOL_MAPPING
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.remote_executors import DockerAgentRuntime


console = Console()

leopard_prompt = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a CodeAgent with all specified parameters")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=None,
        help="The prompt to run with the agent. If no prompt is provided, interactive mode will be launched to guide user through agent setup",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="InferenceClientModel",
        help="The model type to use (e.g., InferenceClientModel, OpenAIModel, LiteLLMModel, TransformersModel)",
    )
    parser.add_argument(
        "--action-type",
        type=str,
        default="code",
        help="The action type to use (e.g., code, tool_calling)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-Next-80B-A3B-Thinking",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--imports",
        nargs="*",  # accepts zero or more arguments
        default=[],
        help="Space-separated list of imports to authorize (e.g., 'numpy pandas')",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=["web_search"],
        help="Space-separated list of tools that the agent can use (e.g., 'tool1 tool2 tool3')",
    )
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=1,
        help="The verbosity level, as an int in [0, 1, 2].",
    )
    group = parser.add_argument_group("api options", "Options for API-based model types")
    group.add_argument(
        "--provider",
        type=str,
        default=None,
        help="The inference provider to use for the model",
    )
    group.add_argument(
        "--api-base",
        type=str,
        help="The base URL for the model",
    )
    group.add_argument(
        "--api-key",
        type=str,
        help="The API key for the model",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # SEC-bench batch runner
    secbench = subparsers.add_parser("secb-run", help="Run multiple SEC-bench instances across Docker images")
    secbench.add_argument("--config", required=True, help="Local path to agent config TOML file")
    secbench.add_argument("--output-dir", help="Output directory for evaluation results")
    secbench.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    secbench.add_argument("--instance-id", help="Run evaluation for a specific instance ID only")

    return parser.parse_args()


def interactive_mode():
    """Run the CLI in interactive mode"""
    console.print(
        Panel.fit(
            "[bold magenta]🤖 SmolaGents CLI[/]\n[dim]Intelligent agents at your service[/]", border_style="magenta"
        )
    )

    console.print("\n[bold yellow]Welcome to smolagents![/] Let's set up your agent step by step.\n")

    # Get user input step by step
    console.print(Rule("[bold yellow]⚙️  Configuration", style="bold yellow"))

    # Get agent action type
    action_type = Prompt.ask(
        "[bold white]What action type would you like to use? 'code' or 'tool_calling'?[/]",
        default="code",
        choices=["code", "tool_calling"],
    )

    # Show available tools
    tools_table = Table(title="[bold yellow]🛠️  Available Tools", show_header=True, header_style="bold yellow")
    tools_table.add_column("Tool Name", style="bold yellow")
    tools_table.add_column("Description", style="white")

    for tool_name, tool_class in TOOL_MAPPING.items():
        # Get description from the tool class if available
        try:
            tool_instance = tool_class()
            description = getattr(tool_instance, "description", "No description available")
        except Exception:
            description = "Built-in tool"
        tools_table.add_row(tool_name, description)

    console.print(tools_table)
    console.print(
        "\n[dim]You can also use HuggingFace Spaces by providing the full path (e.g., 'username/spacename')[/]"
    )

    console.print("[dim]Enter tool names separated by spaces (e.g., 'web_search python_interpreter')[/]")
    tools_input = Prompt.ask("[bold white]Select tools for your agent[/]", default="web_search")
    tools = tools_input.split()

    # Get model configuration
    console.print("\n[bold yellow]Model Configuration:[/]")
    model_type = Prompt.ask(
        "[bold]Model type[/]",
        default="InferenceClientModel",
        choices=["InferenceClientModel", "OpenAIServerModel", "LiteLLMModel", "TransformersModel"],
    )

    model_id = Prompt.ask("[bold white]Model ID[/]", default="Qwen/Qwen2.5-Coder-32B-Instruct")

    # Optional configurations
    provider = None
    api_base = None
    api_key = None
    imports = []
    action_type = "code"

    if Confirm.ask("\n[bold white]Configure advanced options?[/]", default=False):
        if model_type in ["InferenceClientModel", "OpenAIServerModel", "LiteLLMModel"]:
            provider = Prompt.ask("[bold white]Provider[/]", default="")
            api_base = Prompt.ask("[bold white]API Base URL[/]", default="")
            api_key = Prompt.ask("[bold white]API Key[/]", default="", password=True)

        imports_input = Prompt.ask("[bold white]Additional imports (space-separated)[/]", default="")
        if imports_input:
            imports = imports_input.split()

    # Get prompt
    prompt = Prompt.ask(
        "[bold white]Now the final step; what task would you like the agent to perform?[/]", default=leopard_prompt
    )

    return prompt, tools, model_type, model_id, provider, api_base, api_key, imports, action_type


def load_model(
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> Model:
    if model_type == "OpenAIModel":
        return OpenAIModel(
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
            api_base=api_base or "https://api.fireworks.ai/inference/v1",
            model_id=model_id,
        )
    elif model_type == "LiteLLMModel":
        return LiteLLMModel(
            model_id=model_id,
            api_key=api_key,
            api_base=api_base,
        )
    elif model_type == "TransformersModel":
        return TransformersModel(model_id=model_id, device_map="auto")
    elif model_type == "InferenceClientModel":
        return InferenceClientModel(
            model_id=model_id,
            token=api_key or os.getenv("HF_API_KEY"),
            provider=provider,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_smolagent(
    prompt: str,
    tools: list[str],
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    imports: list[str] | None = None,
    provider: str | None = None,
    action_type: str = "code",
) -> None:
    load_dotenv()

    model = load_model(model_type, model_id, api_base=api_base, api_key=api_key, provider=provider)

    available_tools = []

    for tool_name in tools:
        if "/" in tool_name:
            space_name = tool_name.split("/")[-1].lower().replace("-", "_").replace(".", "_")
            description = f"Tool loaded from Hugging Face Space: {tool_name}"
            available_tools.append(Tool.from_space(space_id=tool_name, name=space_name, description=description))
        else:
            if tool_name in TOOL_MAPPING:
                available_tools.append(TOOL_MAPPING[tool_name]())
            else:
                raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

    if action_type == "code":
        agent: CodeAgent | ToolCallingAgent = CodeAgent(
            tools=available_tools,
            model=model,
            additional_authorized_imports=imports,
            stream_outputs=True,
        )
    elif action_type == "tool_calling":
        agent = ToolCallingAgent(tools=available_tools, model=model, stream_outputs=True)
    else:
        raise ValueError(f"Unsupported action type: {action_type}")

    agent.run(prompt)


def run_secb_evaluation(args) -> None:
    """Run SEC-bench evaluation using Docker containers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    load_dotenv()

    # Load TOML config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as f:
        config = tomli.load(f)

    # Extract configuration
    dataset_config = config.get("dataset", {})
    docker_config = config.get("docker", {})
    task_config = config.get("task", {})
    output_config = config.get("output", {})

    # Load dataset - import datasets lazily to avoid import errors when not using secb-run
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Please install 'datasets' package for SEC-bench evaluation: pip install 'smolagents[secb]'"
        ) from e

    dataset_name = dataset_config.get("name", "SEC-bench/SEC-bench")
    dataset_split = dataset_config.get("split", "eval")  # eval, cve, or oss
    selected_ids = dataset_config.get("instance_ids", [])

    console.print(f"[bold]Loading dataset: {dataset_name} (split: {dataset_split})[/]")
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Filter by instance IDs if specified
    if selected_ids:
        dataset = dataset.filter(lambda x: x["instance_id"] in selected_ids)
        console.print(f"[bold]Filtered to {len(selected_ids)} instance(s)[/]")

    # Filter by specific instance_id if provided via CLI
    if args.instance_id:
        dataset = dataset.filter(lambda x: x["instance_id"] == args.instance_id)
        console.print(f"[bold]Running for instance: {args.instance_id}[/]")

    # Convert to list for easier processing
    instances = list(dataset)

    if not instances:
        console.print("[bold red]No instances found matching the criteria[/]")
        return

    console.print(f"[bold green]Found {len(instances)} instance(s) to evaluate[/]")

    # Setup output directory with timestamp-based subdirectory
    # Priority: CLI arg > config file > default
    base_output_dir = (
        Path(args.output_dir) if args.output_dir else Path(output_config.get("output_dir", "./secb_results"))
    )
    # Create timestamp-based subdirectory for this run session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Output directory: {output_dir}[/]")

    # Process instances
    if args.num_workers <= 1:
        # Sequential processing
        for instance in instances:
            _process_secb_instance(
                instance,
                config,
                output_dir,
                docker_config,
                task_config,
            )
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    _process_secb_instance,
                    instance,
                    config,
                    output_dir,
                    docker_config,
                    task_config,
                ): instance
                for instance in instances
            }
            for future in as_completed(futures):
                instance = futures[future]
                try:
                    future.result()
                    console.print(f"[green]Completed: {instance['instance_id']}[/]")
                except Exception as e:
                    console.print(f"[red]Failed {instance['instance_id']}: {e}[/]")


def _process_secb_instance(
    instance: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    docker_config: dict[str, Any],
    task_config: dict[str, Any],
) -> None:
    """Process a single SEC-bench instance."""
    instance_id = instance["instance_id"]
    task_type = task_config.get("type", "patch")  # patch, poc-repo, poc-desc, poc-san

    console.print(f"[bold]Processing instance: {instance_id} (task: {task_type})[/]")

    # Create task prompt
    task_prompt = _create_task_prompt(instance, task_type)

    # Create agent config JSON
    agent_config_json = _create_agent_config_json(config)

    # Create temporary directory for this instance
    instance_output_dir = output_dir / instance_id
    instance_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine Docker image name
    image_prefix = docker_config.get("image_prefix", "hwiwonlee/secb.eval.x86_64")
    image_tag = "poc" if task_type.startswith("poc") else "patch"
    docker_image = f"{image_prefix}.{instance_id}:{image_tag}"

    # Create logger
    logger = AgentLogger(level=LogLevel.INFO)

    # Setup Docker runtime
    docker_kwargs = docker_config.get("run_kwargs", {})
    docker_kwargs.setdefault("mem_limit", "8g")
    docker_kwargs.setdefault("network_mode", "host")
    docker_kwargs.setdefault("auto_remove", True)

    # Use work_dir from instance as default workdir, fallback to /app if not present
    workdir = instance.get("work_dir", "/app")

    runtime = DockerAgentRuntime(
        image_name=docker_image,
        workdir=workdir,
        artifacts_dir=str(instance_output_dir / "output"),
        runtime_logger=logger,
        docker_run_kwargs=docker_kwargs,
    )

    try:
        # Start container
        runtime.start()

        # Install local smolagents source to ensure CmdTool and other local changes are available
        # Find the repository root (assuming cli.py is in src/smolagents/)
        repo_root = Path(__file__).parent.parent.parent
        runtime.install_local_smolagents(str(repo_root))

        # Copy runner script from source directory
        runner_script_path = Path(__file__).parent / "docker_app_runner.py"
        if not runner_script_path.exists():
            raise FileNotFoundError(f"Runner script not found: {runner_script_path}")
        runtime.copy_into_container(str(runner_script_path), "/app/runner.py")

        # Copy agent config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(agent_config_json, f)
            agent_config_path = f.name

        runtime.copy_into_container(agent_config_path, "/app/agent_config.json")
        os.unlink(agent_config_path)

        # Copy task
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(task_prompt)
            task_path = f.name

        runtime.copy_into_container(task_path, "/app/task.txt")
        os.unlink(task_path)

        # Set environment variables
        # Pass API keys from config to container environment
        model_config = config.get("model", {})
        env_vars = {
            "SMOLAGENTS_CONFIG_PATH": "/app/agent_config.json",
            "SMOLAGENTS_TASK_PATH": "/app/task.txt",
            "SMOLAGENTS_ARTIFACTS_DIR": "/app/artifacts",
            "SMOLAGENTS_SECB_RUN": "1",  # Disable sandbox checks for SEC-bench (runs in Docker)
        }

        # Add API keys to environment if provided in config
        if api_key := model_config.get("api_key"):
            # Set appropriate environment variable based on model type
            model_type = model_config.get("type", "")
            if model_type == "InferenceClientModel":
                env_vars["HF_API_KEY"] = api_key
            elif model_type in ["OpenAIModel", "LiteLLMModel"]:
                env_vars["OPENAI_API_KEY"] = api_key

        runtime.environment.update(env_vars)

        # Run agent
        timeout_seconds = task_config.get("timeout_seconds", 7200)  # 2 hours default
        # The runner script reads config and task from environment variables
        exit_code = runtime.run_agent(
            agent_runner_path_in_container="/app/runner.py",
            agent_config_path_in_container="/app/agent_config.json",
            task=task_prompt,
            max_steps=config.get("agent", {}).get("max_steps", 20),
            timeout_seconds=timeout_seconds,
            stream=True,
        )

        # Collect artifacts
        _collect_artifacts(instance, instance_output_dir, task_type, runtime)

        # Copy meta.json from container's artifacts directory if it exists
        # Note: artifacts_dir is set to instance_output_dir / "output"
        # In remote_executors.py, artifacts_subdir = artifacts_dir / "artifacts" = instance_output_dir / "output" / "artifacts"
        # /app/artifacts is mounted to artifacts_subdir
        meta_json_source = instance_output_dir / "output" / "artifacts" / "meta.json"
        if meta_json_source.exists():
            meta_json_dest = instance_output_dir / "meta.json"
            shutil.copy2(meta_json_source, meta_json_dest)

        # Save result
        _save_result(instance_id, instance_output_dir, exit_code, task_type)

    except Exception as e:
        console.print(f"[red]Error processing {instance_id}: {e}[/]")
        raise
    finally:
        runtime.cleanup()


def _create_task_prompt(instance: dict[str, Any], task_type: str) -> str:
    """Create task prompt based on task type using Jinja2 templates."""
    # Find prompts directory relative to this file
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    # Load Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(prompts_dir)))

    # Map task types to template files
    template_map = {
        "patch": "patch.j2",
        "poc-repo": "poc-repo.j2",
        "poc-desc": "poc-desc.j2",
        "poc-san": "poc-san.j2",
    }

    template_name = template_map.get(task_type)
    if not template_name:
        raise ValueError(f"Unknown task type: {task_type}")

    template = env.get_template(template_name)

    # Prepare context variables
    context = {
        "work_dir": instance.get("work_dir", ""),
        "bug_report": instance.get("bug_report", ""),
        "bug_description": instance.get("bug_description", ""),
        "sanitizer_report": instance.get("sanitizer_report", ""),
    }

    return template.render(**context)


def _create_agent_config_json(config: dict[str, Any]) -> dict[str, Any]:
    """Create agent configuration JSON from TOML config."""
    return {
        "model": {
            "type": config.get("model", {}).get("type", "InferenceClientModel"),
            "model_id": config.get("model", {}).get("model_id", ""),
            "api_key": config.get("model", {}).get("api_key"),
            "api_base": config.get("model", {}).get("api_base"),
            "provider": config.get("model", {}).get("provider"),
        },
        "agent_type": config.get("agent", {}).get("type", "ToolCallingAgent"),
        "tools": config.get("agent", {}).get("tools", []),
        "max_steps": config.get("agent", {}).get("max_steps", 20),
        "verbosity_level": config.get("agent", {}).get("verbosity_level", 1),
        "additional_imports": config.get("agent", {}).get("additional_imports", []),
    }


def _collect_artifacts(
    instance: dict[str, Any],
    instance_output_dir: Path,
    task_type: str,
    runtime: DockerAgentRuntime,
) -> None:
    """Collect artifacts from container based on task type."""
    work_dir = instance.get("work_dir", "")

    if runtime.container is None:
        return

    container = runtime.container

    if task_type == "patch":
        # Collect git patch - execute git diff and capture output
        exec_result = container.exec_run(
            [
                "bash",
                "-c",
                f"cd {work_dir} && git config --global core.pager '' && git add -A && git diff --no-color --cached {instance.get('base_commit', 'HEAD')} '*.c' '*.cpp' '*.h' '*.hpp' '*.cc' '*.hh'",
            ],
            workdir=runtime.workdir,
        )
        if exec_result.exit_code == 0:
            patch_content = exec_result.output.decode("utf-8", errors="ignore").strip()
        else:
            patch_content = ""

        # Save patch
        with (instance_output_dir / "git_patch.diff").open("w") as f:
            f.write(patch_content)

    elif task_type.startswith("poc"):
        # Collect PoC artifact (base64 encoded tar.gz)
        # Compress and encode PoC
        runtime.exec(
            [
                "bash",
                "-c",
                'tar --exclude="base_commit_hash" -czf /tmp/poc.tar.gz -C /testcase . 2>/dev/null || echo ""',
            ]
        )

        # Encode to base64 and save to artifacts directory (which is mounted)
        exec_result = container.exec_run(
            [
                "bash",
                "-c",
                "cat /tmp/poc.tar.gz | base64 -w 0 > /app/artifacts/poc.tar.gz.base64 2>/dev/null || echo ''",
            ],
            workdir=runtime.workdir,
        )

        # Read base64 content from mounted artifacts directory
        # Note: artifacts_dir is set to instance_output_dir / "output"
        # In remote_executors.py, artifacts_subdir = artifacts_dir / "artifacts" = instance_output_dir / "output" / "artifacts"
        # /app/artifacts is mounted to artifacts_subdir
        poc_file = instance_output_dir / "output" / "artifacts" / "poc.tar.gz.base64"
        if poc_file.exists():
            with poc_file.open() as f:
                poc_content = f.read().strip()
        else:
            poc_content = ""

        # Save PoC artifact
        with (instance_output_dir / "poc_artifact.txt").open("w") as f:
            f.write(poc_content)


def _save_result(
    instance_id: str,
    instance_output_dir: Path,
    exit_code: int,
    task_type: str,
) -> None:
    """Save evaluation result in compatible format."""
    result: dict[str, Any] = {
        "instance_id": instance_id,
    }

    if task_type == "patch":
        # Read git patch
        patch_file = instance_output_dir / "git_patch.diff"
        git_patch: str = ""
        if patch_file.exists():
            with patch_file.open() as f:
                git_patch = f.read()

        result["test_result"] = {
            "git_patch": git_patch,
        }
    else:  # poc tasks
        # Read PoC artifact
        poc_file = instance_output_dir / "poc_artifact.txt"
        poc_artifact: str = ""
        if poc_file.exists():
            with poc_file.open() as f:
                poc_artifact = f.read().strip()

        result["test_result"] = {
            "poc_artifact": poc_artifact,
        }

    # Save as JSONL (compatible with eval_instances.py)
    output_file = instance_output_dir / "output.jsonl"
    with output_file.open("w") as f:
        f.write(json.dumps(result) + "\n")


def main() -> None:
    args = parse_arguments()

    # Handle secb-run subcommand
    if args.subcommand == "secb-run":
        run_secb_evaluation(args)
        return

    # Check if we should run in interactive mode
    # Interactive mode is triggered when no prompt is provided
    if args.prompt is None:
        prompt, tools, model_type, model_id, provider, api_base, api_key, imports, action_type = interactive_mode()
    else:
        prompt = args.prompt
        tools = args.tools
        model_type = args.model_type
        model_id = args.model_id
        provider = args.provider
        api_base = args.api_base
        api_key = args.api_key
        imports = args.imports
        action_type = args.action_type

    run_smolagent(
        prompt,
        tools,
        model_type,
        model_id,
        provider=provider,
        api_base=api_base,
        api_key=api_key,
        imports=imports,
        action_type=action_type,
    )


if __name__ == "__main__":
    main()
