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

### Bonsai Semantic Tag Selector

ComfyUI 上では `Bonsai Semantic Tag Selector` として表示されます。

入力:

- `instruction_ja`: 日本語の指示文
- `max_candidates`: `tags.json` から Bonsai に渡す候補数上限
- `max_selected_tags`: 最終的に返すタグ数上限
- `category_profile`: 再ランク時のカテゴリ重み。`balanced` / `character_focus` / `style_pose_focus`
- `rebuild_index`: `true` のとき埋め込み索引を再構築

出力:

- `tags`: 1 行のカンマ区切りタグ

挙動:

- `tags.json` の `name`, `words`, `category`, `post_count`, `is_deprecated` を読み込みます
- 初回実行時に `intfloat/multilingual-e5-small` でタグ埋め込み索引を生成し、`tag_index_meta.json` と `tag_index_vectors.npz` を保存します
- 2 回目以降は保存済み索引を再利用し、`tags.json` 更新時または `rebuild_index=true` のときだけ再生成します
- 日本語指示を埋め込みベクトル化し、類似度で上位候補を取得します
- その後 `category_profile`, `post_count`, `is_deprecated` を使って再ランクします
- Bonsai は候補一覧からのみタグを選択します
- 最終出力では `tags.json` に存在しないタグ、候補外タグ、重複タグを除外します

既存ノードとの違い:

- `Bonsai Tag Generator` はタグを自由生成します
- `Bonsai Semantic Tag Selector` は `tags.json` にあるタグだけを返します

注意:

- 初回索引生成はタグ数に応じて時間がかかります
- `style_pose_focus` は厳密な style / pose 抽出ではなく、general 系を優先する重み付けです
- `sentence-transformers`, `torch`, `numpy` が必要です
- 最終的に候補が抽出できない場合はエラーになります
- 埋め込み索引は初回読み込み後にメモリへ保持されます

追加ルート:

- `GET /bonsai/status`
- `POST /bonsai/restart`
