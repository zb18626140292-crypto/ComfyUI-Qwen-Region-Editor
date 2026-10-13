import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const CLASS = "QwenRegionSelect";
const widget = (node, name) => node.widgets?.find((w) => w.name === name);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function imageFile(value) {
    if (typeof value === "string" && value) return value;
    // Load Image+Thumbnails may group files into {name, files} objects.
    if (value && Array.isArray(value.files) && value.files.length === 1) {
        const file = value.files[0];
        if (typeof file === "string") return value.name ? `${value.name}/${file}` : file;
    }
    return null;
}

function upstreamImage(node) {
    const seen = new Set();
    let current = node;
    let provisional = false;
    for (let depth = 0; depth < 16 && current; depth++) {
        if (seen.has(current.id)) break;
        seen.add(current.id);
        const input = current.inputs?.find((i) => i.name === "image" || i.name === "original")
            ?? current.inputs?.find((i) => i.type === "IMAGE");
        const link = current.graph?.links?.[input?.link];
        const source = link && current.graph?.getNodeById(link.origin_id);
        if (!source) break;
        // Do not present an unprocessed loader preview for arbitrary image-processing nodes.
        const type = source.comfyClass ?? source.type;
        if (["LoadImage", "LoadImageOutput"].includes(type)) {
            const name = imageFile(widget(source, "image")?.value);
            return name ? { filename: name, type: type === "LoadImageOutput" ? "output" : "input", provisional } : null;
        }
        // This compositor preserves the original image's coordinate system.
        // Its original is a useful initial preview for a later selection, until
        // that selection executes with the preceding edit actually applied.
        if (type === "QwenRegionComposite") {
            provisional = true;
            current = source;
            continue;
        }
        if (!String(type).toLowerCase().includes("reroute")) break;
        current = source;
    }
    return null;
}

function viewUrl(file) {
    let filename = file.filename, subfolder = file.subfolder ?? "", type = file.type ?? "input";
    const annotation = /\s*\[(input|output|temp)\]$/.exec(filename);
    if (annotation) { type = annotation[1]; filename = filename.slice(0, annotation.index); }
    const normalized = filename.replaceAll("\\", "/");
    if (!subfolder && normalized.includes("/")) {
        const cut = normalized.lastIndexOf("/");
        subfolder = normalized.slice(0, cut); filename = normalized.slice(cut + 1);
    } else filename = normalized;
    return api.apiURL(`/view?${new URLSearchParams({ filename, subfolder, type })}`);
}

function install(node) {
    if (node._qwenRegionEditor || typeof node.addDOMWidget !== "function") return;
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:flex;flex-direction:column;gap:6px;height:100%;min-height:280px;box-sizing:border-box;padding:6px;color:#e8e8e8;background:#20252d;border-radius:6px;";
    wrap.dataset.qwenRegionNode = String(node.id);
    const toolbar = document.createElement("div");
    toolbar.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:wrap";
    const button = (text) => {
        const b = document.createElement("button"); b.textContent = text;
        b.style.cssText = "background:#384452;color:#fff;border:1px solid #68798a;border-radius:4px;padding:4px 8px;cursor:pointer;font-size:12px";
        toolbar.append(b); return b;
    };
    const refresh = button("同步原图");
    const redraw = button("重新画框");
    const enlarge = button("放大框选");
    const full = button("整张图");
    const tip = document.createElement("div");
    tip.style.cssText = "font-size:12px;line-height:1.4;color:#a9c2d7";
    tip.textContent = "拖动空白处画框；框内拖动移动；拖四角缩放。";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;width:100%;flex:1;min-height:220px;background:#13181f;touch-action:none;cursor:crosshair";
    canvas.setAttribute("aria-label", "Qwen 编辑区域框选画布");
    const status = document.createElement("div");
    status.style.cssText = "font-size:12px;min-height:18px;color:#c7d5e2";
    wrap.append(toolbar, tip, canvas, status);
    node.addDOMWidget("qwen_region_canvas", "QwenRegionCanvas", wrap, {
        serialize: false, hideOnZoom: false, getMinHeight: () => 320,
    });
    if (node.size[0] < 430 || node.size[1] < 720) node.setSize([Math.max(460, node.size[0]), Math.max(760, node.size[1])]);
    const state = { image: null, url: null, requested: null, fit: null, drag: null, newBox: false, disposed: false, modal: null, sourceKey: null, executedPreview: null };
    function getBox() {
        return Object.fromEntries(["x", "y", "width", "height"].map((n) => [n, Number(widget(node, n)?.value) || 0]));
    }
    function setBox(box) {
        for (const n of ["x", "y", "width", "height"]) {
            const w = widget(node, n); if (!w) continue;
            w.value = Math.round(box[n] * 100) / 100;
            w.callback?.(w.value);
        }
        node.graph?.setDirtyCanvas?.(true, true);
        paint();
    }
    function paint() {
        if (state.disposed) return;
        const bounds = canvas.getBoundingClientRect();
        const cw = Math.max(100, Math.round(bounds.width)), ch = Math.max(220, Math.round(bounds.height));
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        if (canvas.width !== Math.round(cw*dpr) || canvas.height !== Math.round(ch*dpr)) {
            canvas.width = Math.round(cw*dpr); canvas.height = Math.round(ch*dpr);
        }
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, cw, ch);
        if (!state.image) {
            ctx.fillStyle = "#a9c2d7"; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
            ctx.fillText("连接加载图像节点后，点击“同步原图”", cw/2, ch/2-8);
            ctx.fillText("其他上游节点需先执行一次区域预览", cw/2, ch/2+14);
            return;
        }
        const im = state.image;
        const scale = Math.min((cw-16)/im.naturalWidth, (ch-16)/im.naturalHeight);
        const iw = im.naturalWidth*scale, ih = im.naturalHeight*scale;
        const ox = (cw-iw)/2, oy = (ch-ih)/2;
        state.fit = { ox, oy, iw, ih };
        ctx.drawImage(im, ox, oy, iw, ih);
        const b = getBox(); const right = Math.min(100, b.x+b.width), bottom = Math.min(100, b.y+b.height);
        const px = ox+iw*b.x/100, py = oy+ih*b.y/100, pw = iw*(right-b.x)/100, ph = ih*(bottom-b.y)/100;
        ctx.fillStyle = "rgba(8,13,20,0.38)";
        ctx.fillRect(ox, oy, iw, py-oy); ctx.fillRect(ox, py+ph, iw, oy+ih-py-ph);
        ctx.fillRect(ox, py, px-ox, ph); ctx.fillRect(px+pw, py, ox+iw-px-pw, ph);
        ctx.strokeStyle = "#4be1ff"; ctx.lineWidth = 2; ctx.strokeRect(px, py, pw, ph);
        ctx.fillStyle = "#4be1ff";
        for (const [cx, cy] of [[px,py],[px+pw,py],[px,py+ph],[px+pw,py+ph]]) ctx.fillRect(cx-4, cy-4, 8, 8);
        status.textContent = `选框：x ${b.x}% · y ${b.y}% · 宽 ${b.width}% · 高 ${b.height}%　框外保留原图`;
    }
    async function load(file, force=false) {
        if (!file || state.disposed) return;
        const url = viewUrl(file);
        if (!force && (state.url === url || state.requested === url)) return;
        state.requested = url; status.textContent = "正在加载原图…";
        const im = new Image();
        im.onload = () => {
            if (state.disposed || state.requested !== url) return;
            state.image = im; state.url = url; state.requested = null; paint();
        };
        im.onerror = () => { if (!state.disposed && state.requested === url) { state.requested=null; status.textContent="预览读取失败，请重新选择输入图像。"; } };
        im.src = url + (force ? `&t=${Date.now()}` : "");
    }
    function sync(force=false) {
        const file = upstreamImage(node);
        if (file) {
            const key = viewUrl(file);
            if (key !== state.sourceKey) {
                state.sourceKey = key;
                state.executedPreview = null;
            }
            if (!force && file.provisional && state.executedPreview === key) return;
            if (force) state.executedPreview = null;
            load(file, force);
        }
        else if (force) status.textContent = "上游不是直接加载图像；先运行一次仅区域预览，或查看上次执行预览。";
    }
    refresh.onclick = () => sync(true);
    redraw.onclick = () => { state.newBox = true; status.textContent = "在原图上按下并拖动，重新画一个框。"; };
    full.onclick = () => { state.newBox = false; setBox({x:0,y:0,width:100,height:100}); };
    function closeModal() {
        if (!state.modal) return;
        const modal = state.modal; state.modal = null;
        wrap.append(canvas, status); modal.remove();
        document.removeEventListener("keydown", modalEscape);
        requestAnimationFrame(paint);
    }
    function modalEscape(e) { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeModal(); } }
    enlarge.onclick = () => {
        if (state.modal) return;
        const overlay = document.createElement("div");
        overlay.style.cssText = "position:fixed;inset:0;z-index:100000;background:rgba(0,0,0,.75);display:flex;align-items:center;justify-content:center";
        const panel = document.createElement("div");
        panel.style.cssText = "width:min(900px,95vw);height:min(940px,92vh);padding:12px;box-sizing:border-box;background:#20252d;color:#e8e8e8;border:1px solid #718295;border-radius:8px;display:flex;flex-direction:column;gap:8px";
        const actions = document.createElement("div");
        actions.style.cssText = "display:flex;align-items:center;gap:10px;font:14px sans-serif";
        const done = document.createElement("button"); done.textContent = "完成框选";
        done.style.cssText = "padding:6px 14px;background:#315b72;color:white;border:1px solid #8cb4cb;border-radius:4px;cursor:pointer";
        done.onclick = closeModal;
        const newButton = done.cloneNode(true); newButton.textContent = "重新画框"; newButton.onclick = redraw.onclick;
        const note = document.createElement("span"); note.textContent = "框内移动，四角缩放；Esc 返回。";
        actions.append(done, newButton, note); panel.append(actions, canvas, status); overlay.append(panel);
        overlay.addEventListener("pointerdown", (e) => { if (e.target === overlay) closeModal(); });
        state.modal = overlay; document.body.append(overlay);
        document.addEventListener("keydown", modalEscape); requestAnimationFrame(paint);
    };
    function point(event) {
        if (!state.fit) return null;
        const r = canvas.getBoundingClientRect(), f = state.fit;
        return {x:clamp((event.clientX-r.left-f.ox)/f.iw*100,0,100),
                y:clamp((event.clientY-r.top-f.oy)/f.ih*100,0,100)};
    }
    canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 0 || !state.image) return;
        e.preventDefault(); e.stopPropagation();
        const p = point(e), b = getBox(); if (!p) return;
        const r = Math.min(100,b.x+b.width), bot = Math.min(100,b.y+b.height), f = state.fit;
        const corners = [[b.x,b.y,"nw"],[r,b.y,"ne"],[b.x,bot,"sw"],[r,bot,"se"]];
        const cornerRadius = Math.min(10, (r-b.x)*f.iw/400, (bot-b.y)*f.ih/400);
        const hit = corners.find(([x,y]) => Math.hypot((p.x-x)*f.iw/100,(p.y-y)*f.ih/100) < Math.max(2,cornerRadius));
        const mode = state.newBox ? "new" : hit?.[2] ?? ((p.x>b.x && p.x<r && p.y>b.y && p.y<bot) ? "move" : "new");
        state.drag = { start:p, box:{...b,width:r-b.x,height:bot-b.y}, mode };
        state.newBox = false; canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
        const d = state.drag; if (!d) return;
        e.preventDefault(); e.stopPropagation(); const p = point(e); if (!p) return;
        const b = d.box;
        if (d.mode === "move") {
            setBox({...b,x:clamp(b.x+p.x-d.start.x,0,100-b.width),y:clamp(b.y+p.y-d.start.y,0,100-b.height)}); return;
        }
        let ax, ay, bx=p.x, by=p.y;
        if (d.mode === "new") { ax=d.start.x; ay=d.start.y; }
        else { ax=d.mode.includes("w") ? b.x+b.width : b.x; ay=d.mode.includes("n") ? b.y+b.height : b.y; }
        let x=Math.min(ax,bx), y=Math.min(ay,by), width=Math.abs(ax-bx), height=Math.abs(ay-by);
        width=Math.max(0.01,width); height=Math.max(0.01,height);
        x=Math.min(x,100-width); y=Math.min(y,100-height);
        setBox({x,y,width,height});
    });
    const endDrag = (e) => { if (state.drag) { state.drag=null; e.stopPropagation(); if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId); } };
    canvas.addEventListener("pointerup", endDrag); canvas.addEventListener("pointercancel", endDrag);
    const observer = new ResizeObserver(paint); observer.observe(canvas);
    const oldExecuted = node.onExecuted;
    node.onExecuted = function (output) {
        const result = oldExecuted?.apply(this, arguments);
        const file = upstreamImage(node);
        if (output?.qwen_region_preview?.[0] && (!file || file.provisional)) {
            if (file) state.sourceKey = state.executedPreview = viewUrl(file);
            load(output.qwen_region_preview[0], true);
        }
        return result;
    };
    const oldRemoved = node.onRemoved;
    node.onRemoved = function () {
        closeModal(); state.disposed = true; observer.disconnect(); clearInterval(timer);
        return oldRemoved?.apply(this, arguments);
    };
    const timer = setInterval(() => { sync(); paint(); }, 1200);
    node._qwenRegionEditor = { sync, paint, getBox, setBox };
    requestAnimationFrame(() => { sync(); paint(); });
}

function installAlignmentStatus(node) {
    if (node._qwenAlignmentStatus || typeof node.addDOMWidget !== "function") return;
    const status = document.createElement("div");
    status.style.cssText = "font:12px/1.5 sans-serif;white-space:normal;padding:5px 7px;color:#c7d5e2;background:#20252d;border-radius:4px;box-sizing:border-box;overflow:auto;max-height:85px";
    status.textContent = "运行后显示回贴、颜色校正和微羽化结果。微羽化只向选区内过渡，建议 1 像素。";
    node.addDOMWidget("qwen_alignment_status", "QwenAlignmentStatus", status, {
        serialize: false, hideOnZoom: false, getMinHeight: () => 58,
    });
    const oldExecuted = node.onExecuted;
    node.onExecuted = function (output) {
        const result = oldExecuted?.apply(this, arguments);
        if (output?.qwen_alignment_info?.[0]) status.textContent = output.qwen_alignment_info[0];
        return result;
    };
    node._qwenAlignmentStatus = status;
    watchAlignmentParams(node);
}

function watchAlignmentParams(node) {
    const status = node._qwenAlignmentStatus;
    if (!status) return;
    const labels = {color_mode: "颜色校正", color_strength: "校正强度", edge_feather_pixels: "微羽化像素"};
    for (const name of ["alignment", "offset_x", "offset_y", "scale", "color_mode", "color_strength", "edge_feather_pixels"]) {
        const field = widget(node, name);
        if (field && labels[name]) field.label = labels[name];
        if (!field || field._qwenAlignmentCallback) continue;
        const oldCallback = field.callback;
        field.callback = function () {
            const result = oldCallback?.apply(this, arguments);
            status.textContent = "参数已改变，请重新运行查看回贴结果。";
            return result;
        };
        field._qwenAlignmentCallback = true;
    }
}

function installInpaintStatus(node) {
    if (node._qwenInpaintStatus || typeof node.addDOMWidget !== "function") return;
    const status = document.createElement("div");
    status.style.cssText = "font:12px/1.5 sans-serif;white-space:normal;padding:5px 7px;color:#c7d5e2;background:#20252d;border-radius:4px;box-sizing:border-box;overflow:auto;max-height:110px";
    status.textContent = "从原参考图潜变量开始，只在实际蒙版内重绘。定位线请关闭；回贴接本节点第三个输出。";
    node.addDOMWidget("qwen_inpaint_status", "QwenInpaintStatus", status, {
        serialize: false, hideOnZoom: false, getMinHeight: () => 82,
    });
    const oldExecuted = node.onExecuted;
    node.onExecuted = function (output) {
        const result = oldExecuted?.apply(this, arguments);
        if (output?.qwen_inpaint_info?.[0]) status.textContent = output.qwen_inpaint_info[0];
        return result;
    };
    node._qwenInpaintStatus = status;
}

app.registerExtension({
    name: "Qwen.RegionEditor",
    beforeConfigureGraph(data) {
        for (const node of data?.nodes ?? []) {
            if (node.type !== "QwenRegionComposite" || !Array.isArray(node.widgets_values)) continue;
            const values = node.widgets_values;
            if (values.length === 4 || (values.length === 5 && values[4] === "")) {
                node.widgets_values = [...values.slice(0, 4), "关闭", 0.3, 0];
            }
        }
    },
    nodeCreated(node) {
        const type = node.comfyClass ?? node.type;
        if (type === CLASS) install(node);
        if (type === "QwenRegionComposite") installAlignmentStatus(node);
        if (type === "QwenRegionMaskedLatent") installInpaintStatus(node);
    },
    afterConfigureGraph() {
        for (const node of app.graph?._nodes ?? []) {
            if (node._qwenRegionEditor) {
                node._qwenRegionEditor.sync(); node._qwenRegionEditor.paint();
            }
            if (node._qwenAlignmentStatus) watchAlignmentParams(node);
        }
    },
});
