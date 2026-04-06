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

### Bonsai Tag Generator

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

### Bonsai CSV Tag Selector

ComfyUI 上では `Bonsai CSV Tag Selector` として表示されます。

入力:

- `instruction_ja`: 日本語の指示文
- `max_candidates`: `tags.csv` から Bonsai に渡す候補数上限
- `max_selected_tags`: 最終的に返すタグ数上限
- `creativity`: 検索語の意味展開と最終選択の独創性。`0` で忠実、`1` で広めに補完

出力:

- `tags`: 1 行のカンマ区切りタグ

挙動:

- `tags.csv` の 1 列目をタグ、2 列目を頻度として読み込みます
- まず Bonsai が日本語指示から `tags.csv` 検索用の英語タグ候補を生成します
- 元の入力文と検索用タグ候補の両方を使って `tags.csv` を探索します
- 候補抽出では部分一致とトークン一致をスコア化し、上位候補を Bonsai に渡します
- Bonsai は候補一覧からのみタグを選択します
- 最終出力では `tags.csv` に存在しないタグ、候補外タグ、重複タグを除外します

既存ノードとの違い:

- `Bonsai Tag Generator` はタグを自由生成します
- `Bonsai CSV Tag Selector` は `tags.csv` にあるタグだけを返します

注意:

- 抽象的な日本語指示でも、検索用タグ候補の意味展開で候補を拾いやすくしています
- `creativity` を上げるほど、雰囲気や構図の補完を含む広めの候補選定になります
- 最終的に候補が抽出できない場合はエラーになります
- `tags.csv` は初回読み込み後にメモリへ保持されます

追加ルート:

- `GET /bonsai/status`
- `POST /bonsai/restart`
