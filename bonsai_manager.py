from __future__ import annotations

import atexit
import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


CONFIG_FILE_NAME = "config.json"


@dataclass(frozen=True)
class BonsaiConfig:
    llama_server_exe: Path
    model_path: Path
    host: str
    port: int
    ctx_size: int
    gpu_layers: int
    parallel: int
    startup_timeout_sec: int
    request_timeout_sec: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/v1/models"

    @classmethod
    def from_file(cls, base_dir: Path) -> "BonsaiConfig":
        config_path = base_dir / CONFIG_FILE_NAME
        if not config_path.is_file():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

        if not isinstance(raw, dict):
            raise RuntimeError("config.json の内容が不正です。")

        llama_server_value = _get_required_string(raw, "llama_server_exe")
        model_path_value = _get_required_string(raw, "model_path")
        host = _get_optional_string(raw, "host", "127.0.0.1")
        port = _get_optional_int(raw, "port", 8080)
        ctx_size = _get_optional_int(raw, "ctx_size", 8192)
        gpu_layers = _get_optional_int(raw, "gpu_layers", 99)
        parallel = _get_optional_int(raw, "parallel", 2)
        startup_timeout_sec = _get_optional_int(raw, "startup_timeout_sec", 120)
        request_timeout_sec = _get_optional_int(raw, "request_timeout_sec", 180)

        return cls(
            llama_server_exe=_resolve_config_path(base_dir, llama_server_value),
            model_path=_resolve_config_path(base_dir, model_path_value),
            host=host,
            port=port,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers,
            parallel=parallel,
            startup_timeout_sec=startup_timeout_sec,
            request_timeout_sec=request_timeout_sec,
        )


@dataclass(frozen=True)
class BonsaiContextSizeError(RuntimeError):
    message: str
    prompt_tokens: int
    context_size: int

    def __str__(self) -> str:
        return self.message


class BonsaiServerManager:
    _instance: "BonsaiServerManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._started_once = False
        self._config_error: str | None = None
        self._base_dir = Path(__file__).resolve().parent

    @classmethod
    def instance(cls) -> "BonsaiServerManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def warmup_async(self) -> None:
        thread = threading.Thread(target=self._warmup_worker, daemon=True)
        thread.start()

    def _warmup_worker(self) -> None:
        try:
            self.ensure_started(wait=True)
        except Exception as exc:
            self._config_error = str(exc)

    def ensure_started(self, wait: bool = True) -> None:
        config = self._load_config()

        with self._lock:
            if self._is_ready(config):
                return

            if self._process is not None and self._process.poll() is None:
                if wait:
                    self._wait_until_ready(config)
                return

            self._validate_paths(config)
            self._process = subprocess.Popen(
                self._build_command(config),
                cwd=str(config.llama_server_exe.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=self._creationflags(),
            )
            self._started_once = True

        if wait:
            self._wait_until_ready(config)

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None

        if process is None or process.poll() is not None:
            return

        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=5)
            except Exception:
                return

    def status(self) -> dict[str, object]:
        config = self._safe_config()
        process = self._process
        running = config is not None and self._is_ready(config)
        pid = process.pid if process is not None and process.poll() is None else None
        return {
            "configured": config is not None,
            "started_once": self._started_once,
            "running": running,
            "pid": pid,
            "base_url": config.base_url if config is not None else None,
            "model_path": str(config.model_path) if config is not None else None,
            "error": self._config_error,
        }

    def get_context_size(self) -> int:
        config = self._load_config()
        return config.ctx_size

    @staticmethod
    def estimate_token_count(text: str) -> int:
        if not text:
            return 0
        utf8_length = len(text.encode("utf-8"))
        return max(1, (utf8_length + 2) // 3)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
    ) -> str:
        config = self._load_config()
        self.ensure_started(wait=True)

        payload: dict[str, object] = {
            "model": config.model_path.name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }

        request = urllib.request.Request(
            config.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=config.request_timeout_sec) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            context_size_error = self._parse_context_size_error(detail)
            if context_size_error is not None:
                raise context_size_error from exc
            raise RuntimeError(f"Bonsai サーバー HTTP エラー: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Bonsai サーバーに接続できません: {exc}") from exc

        parsed = json.loads(body)
        return self._extract_content(parsed)

    @staticmethod
    def _parse_context_size_error(detail: str) -> BonsaiContextSizeError | None:
        try:
            parsed = json.loads(detail)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None
        error = parsed.get("error")
        if not isinstance(error, dict):
            return None

        error_type = error.get("type")
        message = error.get("message")
        prompt_tokens = error.get("n_prompt_tokens")
        context_size = error.get("n_ctx")
        if (
            error_type != "exceed_context_size_error"
            or not isinstance(message, str)
            or not isinstance(prompt_tokens, int)
            or not isinstance(context_size, int)
        ):
            return None

        return BonsaiContextSizeError(
            message=f"Bonsai サーバー HTTP エラー: 400 {detail}",
            prompt_tokens=prompt_tokens,
            context_size=context_size,
        )

    def _load_config(self) -> BonsaiConfig:
        try:
            config = BonsaiConfig.from_file(self._base_dir)
        except Exception as exc:
            self._config_error = str(exc)
            raise

        self._config_error = None
        return config

    def _safe_config(self) -> BonsaiConfig | None:
        try:
            return self._load_config()
        except Exception:
            return None

    def _validate_paths(self, config: BonsaiConfig) -> None:
        if not config.llama_server_exe.is_file():
            raise FileNotFoundError(f"LLAMA_SERVER_EXE が見つかりません: {config.llama_server_exe}")
        if not config.model_path.is_file():
            raise FileNotFoundError(f"BONSAI_MODEL_PATH が見つかりません: {config.model_path}")

    def _build_command(self, config: BonsaiConfig) -> list[str]:
        return [
            str(config.llama_server_exe),
            "-m",
            str(config.model_path),
            "--host",
            config.host,
            "--port",
            str(config.port),
            "-ngl",
            str(config.gpu_layers),
            "--ctx-size",
            str(config.ctx_size),
            "-np",
            str(config.parallel),
        ]

    def _wait_until_ready(self, config: BonsaiConfig) -> None:
        deadline = time.time() + float(config.startup_timeout_sec)
        while time.time() < deadline:
            process = self._process
            if process is not None and process.poll() is not None:
                raise RuntimeError("Bonsai サーバーが ready になる前に終了しました。")
            if self._is_ready(config):
                return
            time.sleep(0.5)
        raise TimeoutError("Bonsai サーバーの起動待機がタイムアウトしました。")

    def _is_ready(self, config: BonsaiConfig) -> bool:
        if not self._is_port_open(config.host, config.port):
            return False

        request = urllib.request.Request(config.models_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    @staticmethod
    def _is_port_open(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0

    @staticmethod
    def _extract_content(parsed: object) -> str:
        if not isinstance(parsed, dict):
            raise RuntimeError("Bonsai サーバーの応答形式が不正です。")

        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Bonsai サーバーの応答に choices がありません。")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("Bonsai サーバーの応答形式が不正です。")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Bonsai サーバーの応答に message がありません。")

        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Bonsai サーバーの応答に文字列 content がありません。")

        return content

    @staticmethod
    def _creationflags() -> int:
        return (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )


def setup_bonsai_lifecycle() -> None:
    manager = BonsaiServerManager.instance()
    manager.warmup_async()

    prompt_server = _get_prompt_server_instance()
    if prompt_server is not None:
        _register_routes(prompt_server, manager)
        _register_shutdown(prompt_server, manager)

    atexit.register(manager.stop)


def _get_prompt_server_instance() -> object | None:
    try:
        server_module = import_module("server")
    except Exception:
        return None

    prompt_server_cls = getattr(server_module, "PromptServer", None)
    if prompt_server_cls is None:
        return None

    return getattr(prompt_server_cls, "instance", None)


def _register_shutdown(prompt_server: object, manager: BonsaiServerManager) -> None:
    app = getattr(prompt_server, "app", None)
    if app is None:
        return

    on_shutdown = getattr(app, "on_shutdown", None)
    if on_shutdown is None:
        return

    async def _shutdown(_: object) -> None:
        manager.stop()

    try:
        on_shutdown.append(_shutdown)
    except Exception:
        return


def _register_routes(prompt_server: object, manager: BonsaiServerManager) -> None:
    routes = getattr(prompt_server, "routes", None)
    if routes is None:
        return

    try:
        from aiohttp import web
    except Exception:
        return

    @routes.get("/bonsai/status")
    async def bonsai_status(_: object) -> "web.Response":
        return web.json_response(manager.status())

    @routes.post("/bonsai/restart")
    async def bonsai_restart(_: object) -> "web.Response":
        manager.stop()
        manager.ensure_started(wait=True)
        return web.json_response(manager.status())


def _get_required_string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"config.json の {key} は空でない文字列で指定してください。")
    return value


def _get_optional_string(data: dict[str, object], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"config.json の {key} は文字列で指定してください。")
    return value


def _get_optional_int(data: dict[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise RuntimeError(f"config.json の {key} は整数で指定してください。")
    return value


def _resolve_config_path(base_dir: Path, value: str) -> Path:
    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (base_dir / raw_path).resolve()
