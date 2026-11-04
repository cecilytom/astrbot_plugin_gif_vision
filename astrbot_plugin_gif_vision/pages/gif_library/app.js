const bridge = window.AstrBotPluginPage || window.AstrBotPluginView;
const PLACEHOLDER =
  "data:image/svg+xml;charset=utf-8," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240"><rect width="100%" height="100%" fill="#8aa0bb" opacity=".18"/><text x="50%" y="52%" text-anchor="middle" font-family="system-ui" font-size="18" fill="#6b7f99">GIF</text></svg>',
  );

const state = {
  page: 1,
  pageSize: 24,
  query: "",
  sort: "recent",
  semantic: true,
  missing: false,
  total: 0,
  items: [],
  mode: "list",
  stats: null,
  detail: null,
  thumbCache: new Map(),
  busy: false,
};

const el = (id) => document.getElementById(id);
const grid = el("grid");
const empty = el("empty");
const loading = el("loading");
const pagination = el("pagination");
const toastEl = el("toast");
let toastTimer = null;
let searchTimer = null;
let observer = null;

function unwrap(payload) {
  if (payload && payload.status === "error") {
    throw new Error(payload.message || "请求失败");
  }
  if (payload && payload.status === "ok" && "data" in payload) {
    return payload.data;
  }
  return payload;
}

async function apiGet(endpoint, params = {}) {
  if (!bridge) throw new Error("AstrBot 页面桥接不可用，请刷新 WebUI");
  return unwrap(await bridge.apiGet(endpoint, params));
}

async function apiPost(endpoint, body = {}) {
  if (!bridge) throw new Error("AstrBot 页面桥接不可用，请刷新 WebUI");
  return unwrap(await bridge.apiPost(endpoint, body));
}

function toast(message, { error = false, duration = 3200 } = {}) {
  if (!toastEl) return;
  toastEl.textContent = String(message || "");
  toastEl.classList.toggle("error", !!error);
  toastEl.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.hidden = true;
  }, duration);
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatTime(value) {
  const seconds = Number(value || 0);
  if (!seconds) return "-";
  try {
    return new Intl.DateTimeFormat("zh-CN", {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(seconds * 1000));
  } catch {
    return new Date(seconds * 1000).toLocaleString();
  }
}

function shortText(value, limit = 120) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(1, limit - 1)).trimEnd()}…`;
}

function setBusy(value) {
  state.busy = !!value;
  loading.hidden = !state.busy || state.items.length > 0;
}

async function loadStats() {
  try {
    const data = await apiGet("library/stats");
    state.stats = data;
    const library = data.library || {};
    const vision = data.vision || {};
    const embedding = data.embedding || {};
    const companion = data.companion || {};
    el("stat-total").textContent = String(library.total ?? 0);
    el("stat-root").textContent = library.root_name || "plugin_data";
    el("stat-disk").textContent = formatBytes(library.disk_bytes || 0);
    el("stat-db").textContent = library.db_name
      ? `DB ${formatBytes(library.db_bytes || 0)}`
      : "";
    el("stat-vectors").textContent = `${library.with_embedding || 0} / ${library.total || 0}`;
    el("stat-missing").textContent = `缺 ${library.without_embedding || 0}`;
    if (companion.available) {
      el("stat-companion").textContent = "已连接";
    } else if (companion.enabled) {
      el("stat-companion").textContent = "未连接";
    } else {
      el("stat-companion").textContent = "已关闭";
    }
    const visionName = vision.id || vision.model || vision.configured_id || "当前模型";
    const embeddingName = embedding.id
      ? `${embedding.id}${embedding.dimension ? ` (${embedding.dimension}d)` : ""}`
      : embedding.enabled
        ? "未配置"
        : "已关闭";
    el("stat-model").textContent = `视觉 ${visionName} · 向量 ${embeddingName}`;
  } catch (error) {
    toast(`统计加载失败：${error.message || error}`, { error: true });
  }
}

function makeChip(text, className = "") {
  const chip = document.createElement("span");
  chip.className = `chip ${className}`.trim();
  chip.textContent = text;
  return chip;
}

function makeCard(item) {
  const card = document.createElement("article");
  card.className = "card";
  card.tabIndex = 0;
  card.setAttribute("role", "button");
  card.setAttribute("aria-label", `动图 #${item.id}`);

  const thumb = document.createElement("div");
  thumb.className = "card-thumb";
  const image = document.createElement("img");
  image.loading = "lazy";
  image.alt = "";
  image.dataset.id = String(item.id);
  image.src = PLACEHOLDER;
  thumb.appendChild(image);

  const body = document.createElement("div");
  body.className = "card-body";
  const head = document.createElement("div");
  head.className = "card-head";
  const idText = document.createElement("span");
  idText.textContent = `#${item.id}`;
  const timeText = document.createElement("span");
  timeText.textContent = formatTime(item.created_at);
  head.append(idText, timeText);

  const desc = document.createElement("p");
  desc.className = "card-desc";
  desc.textContent = shortText(item.description || "（无描述）", 150);

  const chips = document.createElement("div");
  chips.className = "chips";
  chips.append(
    makeChip(formatBytes(item.size)),
    makeChip(`${item.frames || 0} 帧`),
    makeChip(
      item.has_embedding ? "已向量" : "缺向量",
      item.has_embedding ? "ok" : "warn",
    ),
  );

  body.append(head, desc, chips);
  card.append(thumb, body);
  card.addEventListener("click", () => openDetail(item.id));
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openDetail(item.id);
    }
  });

  if (observer && image) observer.observe(image);
  return card;
}

async function loadThumb(image) {
  const id = image.dataset.id;
  if (!id) return;
  if (state.thumbCache.has(id)) {
    image.src = state.thumbCache.get(id);
    return;
  }
  try {
    const data = await apiGet("library/thumb", { id });
    if (data && data.data_url) {
      state.thumbCache.set(id, data.data_url);
      image.src = data.data_url;
    }
  } catch {
    // Keep the placeholder; a missing thumbnail should not break the grid.
  }
}

function setupObserver() {
  if (typeof IntersectionObserver !== "function") return;
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        observer.unobserve(entry.target);
        loadThumb(entry.target);
      }
    },
    { rootMargin: "260px 0px" },
  );
}

function renderGrid() {
  grid.replaceChildren();
  if (!state.items.length) {
    empty.hidden = false;
    pagination.hidden = true;
    return;
  }
  empty.hidden = true;
  for (const item of state.items) grid.appendChild(makeCard(item));
}

function renderPagination() {
  const pages = Math.max(1, Math.ceil((state.total || 0) / state.pageSize));
  if (pages <= 1) {
    pagination.hidden = true;
    return;
  }
  pagination.hidden = false;
  el("page-info").textContent = `${state.page} / ${pages}`;
  el("page-prev").disabled = state.page <= 1;
  el("page-next").disabled = state.page >= pages;
}

async function loadList() {
  setBusy(true);
  try {
    const params = {
      page: state.page,
      page_size: state.pageSize,
      sort: state.sort,
      missing_embedding: state.missing ? 1 : 0,
      mode: state.semantic ? "auto" : "like",
    };
    if (state.query) params.query = state.query;
    const data = await apiGet("library/list", params);
    state.items = Array.isArray(data.items) ? data.items : [];
    state.total = Number(data.total || 0);
    state.mode = String(data.mode || "list");
    renderGrid();
    renderPagination();
    const modeText =
      state.mode === "vector"
        ? "语义检索"
        : state.mode === "like"
          ? "关键词检索（LIKE）"
          : "列表";
    el("mode-hint").textContent = state.query
      ? `${modeText} · ${state.total} 条结果`
      : `共 ${state.total} 条 · 第 ${state.page} 页`;
  } catch (error) {
    state.items = [];
    renderGrid();
    toast(`列表加载失败：${error.message || error}`, { error: true });
  } finally {
    setBusy(false);
  }
}

async function refreshAll() {
  await Promise.all([loadStats(), loadList()]);
}

function renderMeta(entries) {
  const container = el("detail-meta");
  container.replaceChildren();
  for (const [label, value] of entries) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value == null || value === "" ? "-" : String(value);
    container.append(dt, dd);
  }
}

async function openDetail(id) {
  try {
    const data = await apiGet("library/item", { id, include_gif: 1 });
    state.detail = data;
    el("detail-title").textContent = `动图 #${data.id}`;
    el("detail-sub").textContent = `SHA ${data.sha256 || ""}`;
    const image = el("detail-image");
    image.src = data.gif_data_url || data.preview_data_url || PLACEHOLDER;
    if (data.gif_data_url) {
      el("detail-preview-note").textContent = "动图预览";
    } else if (data.too_large_for_preview) {
      el("detail-preview-note").textContent =
        "原图超过预览上限，请点击「下载原图」查看完整动图。";
    } else {
      el("detail-preview-note").textContent =
        "当前显示缩略图；完整动图请下载查看。";
    }
    el("detail-description").textContent = data.description || "（无描述）";
    renderMeta([
      ["ID", data.id],
      ["大小", formatBytes(data.size)],
      ["尺寸", data.width && data.height ? `${data.width} × ${data.height}` : "-"],
      ["帧数", data.frames || "-"],
      ["保存时间", formatTime(data.created_at)],
      ["最近使用", formatTime(data.last_used_at)],
      ["使用次数", data.use_count || 0],
      ["来源", data.source || "-"],
      ["向量", data.has_embedding ? "已生成" : "缺失"],
      ["pHash", data.phash || "-"],
      ["规范记录", data.canonical_id || "-"],
    ]);
    el("btn-copy-command").disabled = false;
    el("btn-download").disabled = !data.id;
    el("btn-redescribe").disabled = !data.id;
    el("btn-delete").disabled = !data.id;
    el("btn-copy-command").textContent = `复制 ${data.command || `/gifsend ${data.id}`}`;
    const dialog = el("detail");
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
  } catch (error) {
    toast(`详情加载失败：${error.message || error}`, { error: true });
  }
}

function closeDetail() {
  const dialog = el("detail");
  if (dialog.open && typeof dialog.close === "function") dialog.close();
  else dialog.removeAttribute("open");
}

async function copyCommand() {
  const command = state.detail?.command || `/gifsend ${state.detail?.id || ""}`;
  try {
    await navigator.clipboard.writeText(command);
    toast(`已复制：${command}`);
  } catch {
    window.prompt("复制发送指令：", command);
  }
}

async function downloadCurrent() {
  if (!state.detail?.id) return;
  try {
    if (!bridge?.download) throw new Error("页面桥接不支持下载");
    await bridge.download(
      "library/download",
      { id: state.detail.id },
      `gif_${state.detail.id}.gif`,
    );
    toast("已开始下载");
  } catch (error) {
    toast(`下载失败：${error.message || error}`, { error: true });
  }
}

async function redescribeCurrent() {
  if (!state.detail?.id) return;
  if (!window.confirm(`重新调用视觉模型描述 #${state.detail.id}？`)) return;
  try {
    toast("正在重新描述，请稍等…", { duration: 6000 });
    const updated = await apiPost("library/redescribe", { id: state.detail.id });
    state.detail = { ...state.detail, ...(updated || {}) };
    toast("已更新描述");
    await openDetail(state.detail.id);
    await refreshAll();
  } catch (error) {
    toast(`重新描述失败：${error.message || error}`, { error: true });
  }
}

async function deleteCurrent() {
  if (!state.detail?.id) return;
  if (!window.confirm(`删除动图 #${state.detail.id}？原图和记录都会删除。`)) return;
  try {
    await apiPost("library/delete", { id: state.detail.id });
    closeDetail();
    state.detail = null;
    toast("已删除");
    await refreshAll();
  } catch (error) {
    toast(`删除失败：${error.message || error}`, { error: true });
  }
}

async function rebuildEmbeddings() {
  if (!window.confirm("为缺失向量的记录重新生成向量？单次最多 20 条。")) return;
  try {
    const result = await apiPost("library/rebuild_embeddings", { limit: 20 });
    toast(
      `扫描 ${result.scanned || 0} 条，更新 ${result.updated || 0} 条，失败 ${result.failed || 0} 条`,
      { duration: 5000 },
    );
    await refreshAll();
  } catch (error) {
    toast(`重建失败：${error.message || error}`, { error: true });
  }
}

async function exportJson() {
  try {
    const records = await apiGet("library/export");
    const blob = new Blob([JSON.stringify(records, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `gif_library_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast(`已导出 ${Array.isArray(records) ? records.length : 0} 条记录`);
  } catch (error) {
    toast(`导出失败：${error.message || error}`, { error: true });
  }
}

async function runHealth() {
  try {
    const result = await apiGet("library/health");
    const companion = result.companion || {};
    const embedding = result.embedding_ok
      ? `向量正常 (${result.embedding_dimension || 0}d)`
      : `向量异常：${result.embedding_error || "不可用"}`;
    toast(
      [
        embedding,
        `陪伴：${companion.available ? "已连接" : "未连接"}`,
        `视觉：${result.vision?.id || result.vision?.model || "当前模型"}`,
        result.last_error ? `最近错误：${result.last_error}` : "无最近错误",
      ].join("\n"),
      { duration: 6000 },
    );
    await loadStats();
  } catch (error) {
    toast(`健康检查失败：${error.message || error}`, { error: true });
  }
}

async function importFile(file) {
  if (!file) return;
  try {
    toast(`正在导入 ${file.name}…`, { duration: 8000 });
    if (!bridge?.upload) throw new Error("页面桥接不支持上传");
    const result = await bridge.upload("library/import", file);
    toast(
      result?.id
        ? `已导入 #${result.id}${result.new ? "" : "（已存在）"}`
        : "导入完成",
    );
    await refreshAll();
  } catch (error) {
    toast(`导入失败：${error.message || error}`, { error: true });
  }
}

function bindEvents() {
  el("btn-refresh").addEventListener("click", () => refreshAll());
  el("btn-health").addEventListener("click", runHealth);
  el("btn-rebuild").addEventListener("click", rebuildEmbeddings);
  el("btn-export").addEventListener("click", exportJson);
  el("btn-import").addEventListener("click", () => el("file-input").click());
  el("file-input").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) importFile(file);
  });

  const search = el("search-input");
  search.addEventListener("input", () => {
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.query = search.value.trim();
      state.page = 1;
      loadList();
    }, 300);
  });
  search.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      if (searchTimer) clearTimeout(searchTimer);
      state.query = search.value.trim();
      state.page = 1;
      loadList();
    }
  });
  el("semantic-toggle").addEventListener("change", (event) => {
    state.semantic = !!event.target.checked;
    if (state.query) loadList();
  });
  el("missing-toggle").addEventListener("change", (event) => {
    state.missing = !!event.target.checked;
    state.page = 1;
    loadList();
  });
  el("sort-select").addEventListener("change", (event) => {
    state.sort = String(event.target.value || "recent");
    state.page = 1;
    loadList();
  });

  el("page-prev").addEventListener("click", () => {
    if (state.page <= 1) return;
    state.page -= 1;
    loadList();
  });
  el("page-next").addEventListener("click", () => {
    state.page += 1;
    loadList();
  });

  el("detail-close").addEventListener("click", closeDetail);
  el("detail").addEventListener("click", (event) => {
    if (event.target === el("detail")) closeDetail();
  });
  el("btn-copy-command").addEventListener("click", copyCommand);
  el("btn-download").addEventListener("click", downloadCurrent);
  el("btn-redescribe").addEventListener("click", redescribeCurrent);
  el("btn-delete").addEventListener("click", deleteCurrent);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && el("detail").open) closeDetail();
  });
}

async function init() {
  if (!bridge) {
    toast("AstrBot 页面桥接不可用，请刷新 WebUI", { error: true, duration: 8000 });
    return;
  }
  try {
    await bridge.ready();
  } catch {
    // Older AstrBot versions may not expose ready(); continue anyway.
  }
  setupObserver();
  bindEvents();
  await refreshAll();
}

init();
