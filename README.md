# bonsai_node

ComfyUI の custom node として配置し、ComfyUI 起動時に Bonsai 用 `llama-server` を自動起動し、ComfyUI 終了時に停止します。

## 使い方

このフォルダを `ComfyUI/custom_nodes/bonsai_node` として配置します。

`config.json` でサーバー設定とモデルパスを指定します。

既定の `config.json` は次のコマンドに対応しています。

```powershell
.\bin\cuda\llama-server.exe -m .\models\gguf\8B\Bonsai-8B.gguf --host 127.0.0.1 --port 8080 -ngl 99 --ctx-size 8192
```

設定ファイル:

```json
{
  "llama_server_exe": "./bin/cuda/llama-server.exe",
  "model_path": "./models/gguf/8B/Bonsai-8B.gguf",
  "host": "127.0.0.1",
  "port": 8080,
  "ctx_size": 8192,
  "gpu_layers": 99,
  "parallel": 2,
  "startup_timeout_sec": 120,
  "request_timeout_sec": 180
}
```

`llama_server_exe` と `model_path` はこのリポジトリ基準の相対パスでも、絶対パスでも指定できます。

起動確認:

```powershell
python .\main.py
```

## ノード

ComfyUI 上では `Bonsai Tag Generator` として表示されます。

入力:

- `instruction_ja`: 日本語の指示文
- `system_prompt`: タグ生成ルール。既定値はタグ専用
- `temperature`
- `max_tokens`
- `top_p`
- `top_k`

出力:

- `tags`: 1 行のカンマ区切りタグ

想定用途:

- 日本語のプロンプト案から画像生成用タグを作る
- 構図、人物、背景、雰囲気、品質タグを Bonsai で補完する

追加ルート:

- `GET /bonsai/status`
- `POST /bonsai/restart`
