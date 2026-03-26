# Copyright 2025 the LlamaFactory team.
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

import os
import signal
import subprocess
import sys
from copy import deepcopy


def _register_usr1_faulthandler() -> None:
    if str(os.getenv("LLAMAFACTORY_ENABLE_USR1_FAULTHANDLER", "0")).lower() not in {"1", "true", "yes", "on"}:
        return
    if not hasattr(signal, "SIGUSR1"):
        return

    try:
        import faulthandler
    except Exception:
        return

    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        pass

    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
    except Exception:
        pass


_register_usr1_faulthandler()


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"
    + "|   llamafactory-cli env: show environment info                      |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "| Hint: You can use `lmf` as a shortcut for `llamafactory-cli`.      |\n"
    + "-" * 70
)


def launch():
    from .extras import logging
    from .extras.env import VERSION, print_env
    from .extras.misc import find_available_port, get_device_count, is_env_enabled, use_kt, use_ray

    logger = logging.get_logger(__name__)
    WELCOME = (
        "-" * 58
        + "\n"
        + f"| Welcome to LLaMA Factory, version {VERSION}"
        + " " * (21 - len(VERSION))
        + "|\n|"
        + " " * 56
        + "|\n"
        + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
        + "-" * 58
    )

    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    if is_env_enabled("USE_MCA"):  # force use torchrun
        os.environ["FORCE_TORCHRUN"] = "1"

    def _train_uses_grpo_vllm(argv: list[str]) -> bool:
        for arg in argv:
            if arg.startswith("-"):
                continue
            if not os.path.isfile(arg):
                continue
            if not arg.lower().endswith((".yaml", ".yml", ".json")):
                continue
            try:
                with open(arg, encoding="utf-8") as fp:
                    text = fp.read().lower()
            except Exception:
                continue
            if ("stage: grpo" in text or 'stage: "grpo"' in text or "stage: 'grpo'" in text) and (
                "grpo_use_vllm: true" in text or "use_vllm: true" in text
            ):
                return True
        return False

    def _kill_process_tree(process: subprocess.Popen, *, timeout: float = 30.0) -> None:
        """Best-effort terminate a process *and* its children (torchrun/DDP workers)."""
        if process.poll() is not None:
            return

        def _wait_or_none(seconds: float) -> bool:
            try:
                process.wait(timeout=seconds)
                return True
            except subprocess.TimeoutExpired:
                return False

        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGINT)
            except Exception:
                try:
                    process.send_signal(signal.SIGINT)
                except Exception:
                    pass
            if _wait_or_none(min(10.0, timeout)):
                return
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception:
                try:
                    process.terminate()
                except Exception:
                    pass
            if _wait_or_none(min(10.0, timeout)):
                return
            try:
                os.killpg(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            _wait_or_none(min(10.0, timeout))
        else:
            # Windows: best-effort terminate/kill.
            try:
                process.send_signal(signal.SIGINT)
            except Exception:
                pass
            if _wait_or_none(min(10.0, timeout)):
                return
            try:
                process.terminate()
            except Exception:
                pass
            if _wait_or_none(min(10.0, timeout)):
                return
            try:
                process.kill()
            except Exception:
                pass
            _wait_or_none(min(10.0, timeout))

    def _run_subprocess(cmd: list[str], *, env: dict[str, str]) -> int:
        popen_kwargs: dict[str, object] = {"env": env}
        # Make the subprocess a process group leader so Ctrl+C can be forwarded reliably.
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        else:  # pragma: no cover
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        process = subprocess.Popen(cmd, **popen_kwargs)  # noqa: S603
        try:
            return process.wait()
        except KeyboardInterrupt:
            logger.warning_rank0("KeyboardInterrupt received, terminating distributed workers...")
            _kill_process_tree(process)
            return 130

    if command == "train" and (
        is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and not use_ray() and not use_kt())
    ):
        # launch distributed training
        nnodes = os.getenv("NNODES", "1")
        node_rank = os.getenv("NODE_RANK", "0")
        nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(find_available_port()))
        logger.info_rank0(f"Initializing {nproc_per_node} distributed tasks at: {master_addr}:{master_port}")
        if int(nnodes) > 1:
            logger.info_rank0(f"Multi-node training enabled: num nodes: {nnodes}, node rank: {node_rank}")

        # elastic launch support
        max_restarts = os.getenv("MAX_RESTARTS", "0")
        rdzv_id = os.getenv("RDZV_ID")
        min_nnodes = os.getenv("MIN_NNODES")
        max_nnodes = os.getenv("MAX_NNODES")

        env = deepcopy(os.environ)
        disable_expandable_segments = is_env_enabled("DISABLE_EXPANDABLE_SEGMENTS") or (
            command == "train" and _train_uses_grpo_vllm(sys.argv)
        )
        if disable_expandable_segments:
            env.pop("PYTORCH_ALLOC_CONF", None)
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        if is_env_enabled("OPTIM_TORCH", "1") and not disable_expandable_segments:
            # optimize DDP, see https://zhuanlan.zhihu.com/p/671834539
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        if rdzv_id is not None:
            # launch elastic job with fault tolerant support when possible
            # see also https://docs.pytorch.org/docs/stable/elastic/train_script.html
            rdzv_nnodes = nnodes
            # elastic number of nodes if MIN_NNODES and MAX_NNODES are set
            if min_nnodes is not None and max_nnodes is not None:
                rdzv_nnodes = f"{min_nnodes}:{max_nnodes}"

            cmd = (
                "torchrun --nnodes {rdzv_nnodes} --nproc-per-node {nproc_per_node} "
                "--rdzv-id {rdzv_id} --rdzv-backend c10d --rdzv-endpoint {master_addr}:{master_port} "
                "--max-restarts {max_restarts} {file_name} {args}"
            ).format(
                rdzv_nnodes=rdzv_nnodes,
                nproc_per_node=nproc_per_node,
                rdzv_id=rdzv_id,
                master_addr=master_addr,
                master_port=master_port,
                max_restarts=max_restarts,
                file_name=__file__,
                args=" ".join(sys.argv[1:]),
            )
        else:
            # NOTE: DO NOT USE shell=True to avoid security risk
            cmd = (
                "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
            ).format(
                nnodes=nnodes,
                node_rank=node_rank,
                nproc_per_node=nproc_per_node,
                master_addr=master_addr,
                master_port=master_port,
                file_name=__file__,
                args=" ".join(sys.argv[1:]),
            )

        # torchrun prints its own errors; propagate return code cleanly.
        sys.exit(_run_subprocess(cmd.split(), env=env))

    elif command == "api":
        from .api.app import run_api

        run_api()

    elif command == "chat":
        from .chat.chat_model import run_chat

        run_chat()

    elif command == "eval":
        raise NotImplementedError("Evaluation will be deprecated in the future.")

    elif command == "export":
        from .train.tuner import export_model

        export_model()

    elif command == "train":
        from .train.tuner import run_exp

        run_exp()

    elif command == "webchat":
        from .webui.interface import run_web_demo

        run_web_demo()

    elif command == "webui":
        from .webui.interface import run_web_ui

        run_web_ui()

    elif command == "env":
        print_env()

    elif command == "version":
        print(WELCOME)

    elif command == "help":
        print(USAGE)

    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    from llamafactory.train.tuner import run_exp  # use absolute import

    run_exp()
