# 看动图（astrbot_plugin_gif_vision）

让 AstrBot 真正看懂 QQ 里的 **GIF / 动态表情**：

1. 收到动图后按时间顺序抽帧（默认 6 帧）；
2. 交给当前多模态 provider 生成中文描述（主体、动作变化、文字、表达意图）；
3. 优先通过「我会永远陪着你」（`astrbot_plugin_private_companion`）的公开扩展协议
   `add_context_contribution` 回传为 `scene` 上下文；它没安装/不可用时直接注入主 LLM；
4. 把原始 GIF、描述、embedding、pHash 存进本地库；
5. 同一张图下次直接命中描述，不再调用视觉模型；近似图走 pHash；
6. 提供 WebUI「动图库」：浏览、语义搜索、播放/下载、导入、重新描述、删除、重建缺失向量。

## 安装

- AstrBot `>=4.28.0,<5`，QQ 走 `aiocqhttp` / NapCat。
- 本插件依赖 Pillow（AstrBot 已自带）。`requirements.txt` 只是声明。
- 通过 AstrBot WebUI 的“插件 → 安装插件 → 上传 zip”安装，或在插件目录放入本文件夹后重载。
- 安装后在 WebUI“插件 → 看动图”里确认已启用；页面入口为“动图库”。

## 与「我会永远陪着你」的联动

插件不会修改陪伴插件的源码。运行时它会：

- 通过 `sys.modules` 查找 `astrbot_plugin_private_companion.main.get_private_companion_api()`，
  找不到时再通过 `context.get_registered_star(...).star_cls.extension_api` 兜底；
- 调用 `register_extension({...})` 注册 `gif_vision`（控制面可在
  `/page/extensions/status` 看到）；
- 在 `on_llm_request(priority=-259500)` 中调用：

```python
api.add_context_contribution(
    req,
    {
        "lane": "scene",
        "key": "gif_<sha12>",
        "content": "[gif_vision:<sha12>] ...画面描述...",
        "evidence": "gif_vision",
        "priority": 55,
        "max_age_seconds": 900,
        "visibility": "public" if group else "private",
        "source": "gif_vision",
        "source_refs": ["gif:<sha12>"],
        "revision": "<sha12>",
    },
    event=event,
    source_id="gif_vision",
)
```

陪伴插件仍掌握最终 prompt 放置、预算、合并和审计。若该 API 不存在、返回 `False`
或最终请求里没有 `[gif_vision:<sha12>]` 标记，插件会在更晚的
`on_llm_request(priority=-300000)` 里回退为 `<gif_analysis>` 文字注入。

> 陪伴模式下不修改 `req.image_urls`（图片归陪伴插件管理）；独立模式才移除已处理的
> 原 GIF，避免主模型同时收到拼图和描述。

## WebUI 动图库

WebUI → 插件 → 看动图 → **动图库**。

- 统计：总数、磁盘、向量覆盖、视觉/向量 provider、陪伴连接状态；
- 搜索：关键词或语义检索（无向量 provider 时自动降级 LIKE）；
- 卡片：缩略图、描述摘要、#id、大小/帧数/尺寸；
- 详情：完整描述、原 GIF 播放（≤ `webui_preview_max_mb`）、下载、复制
  `/gifsend <id>`；
- 管理：导入 GIF、重新描述、删除、重建缺失向量、导出 JSON、健康检查。

页面是 AstrBot 插件页面，不另起端口；API 注册在
`/<plugin_name>/library/*`，由 AstrBot 页面桥接鉴权后转发。

## QQ 命令

| 命令 | 别名 | 说明 |
|---|---|---|
| `/gif` | `/动图` | 处理引用/当前消息里的动图并保存 |
| `/gifsearch <文本>` | `/找动图` | 语义/关键词检索 |
| `/gifsend <id或文本>` | `/发动图` | 发送已保存的原始 GIF |
| `/gifstats` | `/动图统计` | 管理员：统计与健康状态 |
| `/gifforget <id>` | `/忘掉动图` | 管理员：删除记录和文件 |
| `/gifimport <路径>` | `/导入动图` | 管理员：导入本地 GIF |

## 配置

WebUI 配置页由 `_conf_schema.json` 自动生成。常用项：

- `provider_id`：视觉 provider，留空=当前对话模型；必须支持图片。
- `embedding_provider_id`：向量 provider，留空=第一个可用。
- `return_to_companion`：优先回传“我会永远陪着你”。
- `session_scope`：`all` / `group` / `private`。
- `frame_count` / `frame_max_edge`：抽帧数量与清晰度。
- `library_max_mb` / `retention_days`：动图库磁盘与过期。
- `webui_preview_max_mb`：WebUI 内联预览上限。

## 数据目录

默认：`~/.astrbot/data/plugin_data/astrbot_plugin_gif_vision/`

```
gifs/ab/cd/<sha256>.gif   原始动图
thumbs/ab/<sha256>.jpg    2x2 缩略图
library.sqlite3           描述、向量、pHash、使用记录
```

删除单条用 `/gifforget <id>` 或 WebUI；清空可直接停 AstrBot 后删除整个目录。

## 故障排查

- **日志**：`backend.log` 搜索 `[GifVision]`。
- **没有回传到陪伴插件**：看 `companion` 状态；确认陪伴插件版本暴露
  `get_private_companion_api()`，且其 `finalize_conversation_injection_plan`
  仍在 `-260000`。回退路径会直接注入主 LLM，不影响聊天。
- **视觉失败**：检查 `provider_id` 是否支持图片；DeepSeek/Gemini 多模态 provider
  可用。失败只记警告，GIF 仍交给核心/陪伴插件处理。
- **向量失败**：描述仍会保存；搜索自动降级为 LIKE。可在 WebUI 点“重建缺失向量”。
- **QQ 空框**：若 NapCat/QQ 根本没把图片字节交给 AstrBot，插件无法恢复；只能记录跳过。

## 测试

```powershell
# 在插件根目录
python -m unittest discover -s tests -v
```

`tests/manual_probe.py` 可用真实 GIF 直连当前 DeepSeek provider 做一次人工验证。
