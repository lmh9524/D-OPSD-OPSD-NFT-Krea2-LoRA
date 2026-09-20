const scenes = {
  fashion: {
    prompt: "材质透明度、人体完整性、服装结构与摄影质感",
    path: "01-fashion",
    note: ["原始能力", "负样本感知微调", "策略内自蒸馏"]
  },
  typography: {
    prompt: "复杂文字可读性、字形完整性与版面层级",
    path: "02-typography",
    note: ["原始能力", "奖励对齐", "端点奖励转显式监督"]
  },
  mech: {
    prompt: "机械结构、局部细节、主体一致性与复杂构图",
    path: "08-mech",
    note: ["原始能力", "奖励对齐", "策略内目标刷新"]
  },
  lighthouse: {
    prompt: "远近空间、体积光、主体位置与整体氛围",
    path: "09-lighthouse",
    note: ["原始能力", "奖励对齐", "策略内目标刷新"]
  }
};

const methods = [
  { key: "base", label: "Base", cls: "base" },
  { key: "nft", label: "NFT", cls: "nft" },
  { key: "opsd", label: "OPSD", cls: "opsd" }
];

const comparisonGrid = document.querySelector("#comparison-grid");
const scenePrompt = document.querySelector("#scene-prompt");

function renderScene(sceneKey) {
  const scene = scenes[sceneKey];
  scenePrompt.textContent = scene.prompt;
  comparisonGrid.innerHTML = methods.map((method, index) => `
    <article class="compare-card ${method.cls}">
      <div class="image-wrap">
        <img src="https://diffusionopsd.github.io/assets/comparisons/${scene.path}/${method.key}.jpg" alt="${method.label} 在${scene.prompt}场景的公开对照样例">
      </div>
      <div class="compare-label"><b>${method.label}</b><span>${scene.note[index]}</span></div>
    </article>
  `).join("");
  comparisonGrid.querySelectorAll("img").forEach(addImageFallback);
}

document.querySelectorAll(".scene-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".scene-tab").forEach((item) => {
      item.classList.toggle("active", item === button);
      item.setAttribute("aria-selected", item === button ? "true" : "false");
    });
    renderScene(button.dataset.scene);
  });
});

function addImageFallback(image) {
  image.addEventListener("error", () => {
    const parent = image.parentElement;
    image.remove();
    const fallback = document.createElement("div");
    fallback.className = "image-error";
    fallback.textContent = "外部图片暂时无法加载\n请点击来源链接查看原图";
    parent.append(fallback);
  }, { once: true });
}

renderScene("fashion");
document.querySelectorAll(".gallery img, .tdm-shot img, .video-poster img").forEach(addImageFallback);

const lightbox = document.querySelector("#lightbox");
const lightboxImage = lightbox.querySelector("img");
document.querySelectorAll(".gallery-item, .tdm-shot").forEach((item) => {
  item.addEventListener("click", () => {
    lightboxImage.src = item.dataset.full;
    lightboxImage.alt = item.querySelector("img")?.alt || "放大的效果图";
    lightbox.showModal();
  });
});
lightbox.querySelector("button").addEventListener("click", () => lightbox.close());
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) lightbox.close();
});

const fallbackDopsd = [
  [1, .123], [100, .086], [200, .071], [400, .058], [600, .044], [800, .063],
  [1000, .039], [1200, .048], [1400, .031], [1600, .042], [1800, .029], [2000, .0506121]
];
const fallbackNft = [
  [8, .68], [48, .73], [96, .78], [144, .81], [200, .76], [248, .84],
  [304, .826], [352, .665], [400, .796], [448, .852], [480, .901], [500, .856321]
];

function parseDopsd(text) {
  const points = [];
  for (const line of text.split(/\r?\n/)) {
    const match = line.match(/step\s+(\d+):\s+loss=([\d.eE+-]+)/);
    if (match) points.push([Number(match[1]), Number(match[2])]);
  }
  return points;
}

function parseNft(text) {
  const points = [];
  for (const line of text.split(/\r?\n/)) {
    const match = line.match(/step\s+(\d+):.*?reward\/mean=([\d.eE+-]+)/);
    if (match) points.push([Number(match[1]), Number(match[2])]);
  }
  return points;
}

function movingAverage(points, windowSize) {
  return points.map((point, index) => {
    const start = Math.max(0, index - windowSize + 1);
    const slice = points.slice(start, index + 1);
    return [point[0], slice.reduce((sum, item) => sum + item[1], 0) / slice.length];
  });
}

function drawChart(canvas, points, color, label) {
  const ratio = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth;
  const cssHeight = canvas.clientHeight;
  canvas.width = cssWidth * ratio;
  canvas.height = cssHeight * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);

  const pad = { top: 18, right: 18, bottom: 32, left: 48 };
  const width = cssWidth - pad.left - pad.right;
  const height = cssHeight - pad.top - pad.bottom;
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  const range = Math.max(yMax - yMin, .001);
  yMin = Math.max(0, yMin - range * .12);
  yMax += range * .12;
  const x = (value) => pad.left + ((value - xMin) / (xMax - xMin || 1)) * width;
  const y = (value) => pad.top + (1 - (value - yMin) / (yMax - yMin || 1)) * height;

  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = "#64707b";
  ctx.strokeStyle = "rgba(255,255,255,.07)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const yy = pad.top + (height / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(cssWidth - pad.right, yy); ctx.stroke();
    const value = yMax - ((yMax - yMin) / 4) * i;
    ctx.fillText(value.toFixed(label === "reward" ? 2 : 3), 3, yy + 3);
  }
  ctx.fillText(String(xMin), pad.left, cssHeight - 8);
  ctx.fillText(String(xMax), cssWidth - pad.right - String(xMax).length * 6, cssHeight - 8);

  const gradient = ctx.createLinearGradient(0, pad.top, 0, cssHeight - pad.bottom);
  gradient.addColorStop(0, `${color}42`);
  gradient.addColorStop(1, `${color}00`);
  ctx.beginPath();
  points.forEach((point, index) => index ? ctx.lineTo(x(point[0]), y(point[1])) : ctx.moveTo(x(point[0]), y(point[1])));
  ctx.lineTo(x(points.at(-1)[0]), cssHeight - pad.bottom);
  ctx.lineTo(x(points[0][0]), cssHeight - pad.bottom);
  ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();

  ctx.beginPath();
  points.forEach((point, index) => index ? ctx.lineTo(x(point[0]), y(point[1])) : ctx.moveTo(x(point[0]), y(point[1])));
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();

  const last = points.at(-1);
  ctx.beginPath(); ctx.arc(x(last[0]), y(last[1]), 4, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
}

function updateStats(element, points, type) {
  if (!points.length) return;
  const last = points.at(-1);
  const tail = points.slice(-Math.min(20, points.length));
  const average = tail.reduce((sum, point) => sum + point[1], 0) / tail.length;
  if (type === "loss") {
    element.innerHTML = `<span><b>${last[1].toFixed(4)}</b> final loss</span><span><b>${average.toFixed(4)}</b> recent mean</span><span><b>完成</b> step ${last[0]}</span>`;
  } else {
    element.innerHTML = `<span><b>${last[1].toFixed(4)}</b> final reward</span><span><b>${average.toFixed(4)}</b> recent mean</span><span><b>完成</b> step ${last[0]}</span>`;
  }
}

let chartData = { dopsd: fallbackDopsd, nft: fallbackNft };
function renderCharts() {
  drawChart(document.querySelector("#dopsd-chart"), movingAverage(chartData.dopsd, Math.max(1, Math.floor(chartData.dopsd.length / 70))), "#65d6ff", "loss");
  drawChart(document.querySelector("#nft-chart"), movingAverage(chartData.nft, Math.max(1, Math.floor(chartData.nft.length / 30))), "#c5ff54", "reward");
  updateStats(document.querySelector("#dopsd-stats"), chartData.dopsd, "loss");
  updateStats(document.querySelector("#nft-stats"), chartData.nft, "reward");
}

renderCharts();
Promise.all([
  fetch("../../records/dopsd-2000/train.log").then((response) => response.ok ? response.text() : Promise.reject()),
  fetch("../../records/opsd-nft-500/train.log").then((response) => response.ok ? response.text() : Promise.reject())
]).then(([dopsdText, nftText]) => {
  const dopsd = parseDopsd(dopsdText);
  const nft = parseNft(nftText);
  if (dopsd.length) chartData.dopsd = dopsd;
  if (nft.length) chartData.nft = nft;
  renderCharts();
}).catch(() => {
  // The fallback makes the page usable when opened directly from disk.
});

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderCharts, 120);
});
