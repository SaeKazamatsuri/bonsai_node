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

### Bonsai Direct Tag Generator

ComfyUI 上では `Bonsai Direct Tag Generator` として表示されます。

入力:

- `instruction_ja`: 日本語の指示文
- `system_prompt`: 既存の SDXL / Danbooru 向けシステムプロンプトを既定値として流用
- `temperature`
- `max_tokens`
- `top_p`
- `top_k`

出力:

- `tags`: 1 行のカンマ区切りタグ

想定用途:

- `tags.json` や埋め込み索引を使わず、Bonsai のみでタグ列を生成する
- 既存のシステムプロンプトをそのまま使って、軽量にタグ生成したい

### Bonsai Tag Generator

ComfyUI 上では `Bonsai Tag Generator` として表示されます。

入力:

- `instruction_ja`: 日本語の指示文
- `system_prompt`: 補助指示。互換入力として残していますが、候補制約と strict policy を上書きしません
- `temperature`
- `max_tokens`
- `top_p`
- `top_k`

出力:

- `tags`: 1 行のカンマ区切りタグ

想定用途:

- 日本語の指示文から、DeepDanbooru / Danbooru 風の題材直結タグ列を作る
- `tags.json` の候補から、品質タグや画風タグを混ぜずに整ったタグ列を得る

### Bonsai Semantic Tag Selector

ComfyUI 上では `Bonsai Semantic Tag Selector` として表示されます。

入力:

- `instruction_ja`: 日本語の指示文
- `max_candidates`: `tags.json` から Bonsai に渡す候補数上限
- `max_selected_tags`: 最終的に返すタグ数上限
- `category_profile`: 再ランク時のカテゴリ重み。`balanced` / `character_focus` / `style_pose_focus`
- `temperature`: 最終タグ選択時の多様性
- `top_p`
- `top_k`
- `rebuild_index`: `true` のとき埋め込み索引を再構築

出力:

- `tags`: 1 行のカンマ区切りタグ

挙動:

- `tags.json` の `name`, `words`, `category`, `post_count`, `is_deprecated` を読み込みます
- 初回実行時に `intfloat/multilingual-e5-small` でタグ埋め込み索引を生成し、`tag_index_meta.json` と `tag_index_vectors.npz` を保存します
- 2 回目以降は保存済み索引を再利用し、`tags.json` 更新時または `rebuild_index=true` のときだけ再生成します
- 日本語指示を埋め込みベクトル化し、類似度で上位候補を取得します
- その後 `category_profile`, `post_count`, `is_deprecated`, strict policy, 指示文中の色指定を使って再ランクします
- Bonsai は候補一覧からのみタグを選択します
- 候補一覧は主題、顔髪、服、ポーズ、背景の bucket ごとに整理して渡します
- 最終出力では `tags.json` に存在しないタグ、候補外タグ、重複タグ、品質タグ、画風タグ、artist/meta タグ、明白な競合タグを整理します
- 出力順は `subject count/focus -> character/copyright -> body/global attributes -> face/expression/eyes -> hair/head -> clothes/accessories -> pose/composition -> background/environment` に固定します

既存ノードとの違い:

- `Bonsai Direct Tag Generator` は `tags.json` を使わず、Bonsai に直接タグ生成させます
- `Bonsai Tag Generator` は簡易インターフェースで strict pipeline を使います
- `Bonsai Semantic Tag Selector` は候補数やカテゴリ重みを調整できる詳細版です

注意:

- 初回索引生成はタグ数に応じて時間がかかります
- `style_pose_focus` は厳密な style / pose 抽出ではなく、general 系を優先する重み付けです
- `sentence-transformers`, `torch`, `numpy` が必要です
- 最終的に候補が抽出できない場合はエラーになります
- 埋め込み索引は初回読み込み後にメモリへ保持されます
- 既定では品質タグ、画風タグ、artist/meta タグは自動挿入しません

追加ルート:

- `GET /bonsai/status`
- `POST /bonsai/restart`
