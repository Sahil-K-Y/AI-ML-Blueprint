/* ═══════════════════════════════════════════════════
   Boosting Masterclass — Interactive Visualizations
   ═══════════════════════════════════════════════════ */

// ─── Utility Functions ───
function scrollToSection(id) {
  document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
}

function lerp(a, b, t) { return a + (b - a) * t; }
function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function drawRoundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function getCanvasContext(id) {
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  return { ctx, w: rect.width, h: rect.height };
}

// Colors
const C = {
  blue: '#3b82f6', purple: '#8b5cf6', cyan: '#06b6d4',
  green: '#10b981', orange: '#f59e0b', red: '#ef4444',
  pink: '#ec4899', text: '#e8ecf4', muted: '#5a6580',
  bg: '#0a0e1a', card: '#1a1f35', grid: 'rgba(255,255,255,0.05)'
};

// ═══════════════════════════════════════════════════
// SECTION 1: GRADIENT BOOSTING — REGRESSION
// ═══════════════════════════════════════════════════

const gbRegData = {
  x: [1, 2, 3, 4, 5, 6, 7, 8],
  y: [2.1, 4.8, 3.5, 7.2, 5.9, 9.1, 7.5, 10.3],
  predictions: [],
  iteration: 0,
  lr: 0.5,
  maxIter: 12,
  autoPlaying: false,
  autoTimer: null,
  treeOutputs: []
};

function gbRegInit() {
  const mean = gbRegData.y.reduce((a, b) => a + b, 0) / gbRegData.y.length;
  gbRegData.predictions = gbRegData.y.map(() => mean);
  gbRegData.iteration = 0;
  gbRegData.treeOutputs = [];
  gbRegUpdateUI();
  gbRegDraw();
}

function gbRegReset() {
  if (gbRegData.autoTimer) clearInterval(gbRegData.autoTimer);
  gbRegData.autoPlaying = false;
  gbRegInit();
}

function gbRegUpdateLR(val) {
  gbRegData.lr = parseFloat(val);
  document.getElementById('gb-reg-lr-val').textContent = val;
}

function gbRegStep() {
  if (gbRegData.iteration >= gbRegData.maxIter) return;

  const { y, predictions, lr } = gbRegData;
  const n = y.length;

  // Calculate residuals
  const residuals = y.map((yi, i) => yi - predictions[i]);

  // Simulate tree fitting (simple: split into 2 groups and predict mean residual per group)
  const mid = Math.floor(n / 2);
  const leftResiduals = residuals.slice(0, mid);
  const rightResiduals = residuals.slice(mid);
  const leftMean = leftResiduals.reduce((a, b) => a + b, 0) / leftResiduals.length;
  const rightMean = rightResiduals.reduce((a, b) => a + b, 0) / rightResiduals.length;

  const treeOutput = y.map((_, i) => i < mid ? leftMean : rightMean);
  gbRegData.treeOutputs.push(treeOutput);

  // Update predictions
  for (let i = 0; i < n; i++) {
    gbRegData.predictions[i] += lr * treeOutput[i];
  }

  gbRegData.iteration++;
  gbRegUpdateUI();
  gbRegDraw();
}

function gbRegAuto() {
  if (gbRegData.autoPlaying) {
    clearInterval(gbRegData.autoTimer);
    gbRegData.autoPlaying = false;
    return;
  }
  gbRegData.autoPlaying = true;
  gbRegData.autoTimer = setInterval(() => {
    if (gbRegData.iteration >= gbRegData.maxIter) {
      clearInterval(gbRegData.autoTimer);
      gbRegData.autoPlaying = false;
      return;
    }
    gbRegStep();
  }, 800);
}

function gbRegUpdateUI() {
  const { y, predictions, iteration } = gbRegData;
  const mse = y.reduce((sum, yi, i) => sum + (yi - predictions[i]) ** 2, 0) / y.length;

  document.querySelector('#gb-reg-iteration .info-value').textContent = iteration;
  document.querySelector('#gb-reg-mse .info-value').textContent = mse.toFixed(4);
  document.querySelector('#gb-reg-trees .info-value').textContent = iteration;

  // Highlight current math step
  const steps = document.querySelectorAll('#gb-reg-math .math-step');
  steps.forEach((s, i) => {
    s.classList.remove('highlight');
    s.classList.add('active');
  });
  if (iteration > 0 && iteration <= gbRegData.maxIter) {
    const stepIdx = ((iteration - 1) % 3) + 1;
    if (steps[stepIdx]) steps[stepIdx].classList.add('highlight');
  }

  // Update table
  const tbody = document.getElementById('gb-reg-table-body');
  tbody.innerHTML = '';
  gbRegData.x.forEach((xi, i) => {
    const residual = y[i] - predictions[i];
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${xi}</td>
      <td>${y[i].toFixed(2)}</td>
      <td>${predictions[i].toFixed(3)}</td>
      <td class="${residual >= 0 ? 'positive' : 'negative'}">${residual >= 0 ? '+' : ''}${residual.toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function gbRegDraw() {
  const result = getCanvasContext('gb-reg-canvas');
  if (!result) return;
  const { ctx, w, h } = result;

  // Background
  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const pad = { top: 40, right: 40, bottom: 50, left: 60 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const { x, y, predictions } = gbRegData;
  const allVals = [...y, ...predictions];
  const yMin = Math.min(...allVals) - 1;
  const yMax = Math.max(...allVals) + 1;
  const xMin = 0;
  const xMax = Math.max(...x) + 1;

  function toCanvasX(v) { return pad.left + ((v - xMin) / (xMax - xMin)) * plotW; }
  function toCanvasY(v) { return pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH; }

  // Grid
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const yv = yMin + (yMax - yMin) * i / 5;
    const cy = toCanvasY(yv);
    ctx.beginPath();
    ctx.moveTo(pad.left, cy);
    ctx.lineTo(w - pad.right, cy);
    ctx.stroke();
    ctx.fillStyle = C.muted;
    ctx.font = '11px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(yv.toFixed(1), pad.left - 8, cy + 4);
  }

  // Axis labels
  ctx.fillStyle = C.muted;
  ctx.font = '12px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('x (feature)', w / 2, h - 8);
  ctx.save();
  ctx.translate(15, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('y (target)', 0, 0);
  ctx.restore();

  // Title
  ctx.fillStyle = C.text;
  ctx.font = 'bold 14px Inter';
  ctx.textAlign = 'left';
  ctx.fillText(`Gradient Boosting Regression — Iteration ${gbRegData.iteration}`, pad.left, 25);

  // Draw residual arrows (actual → predicted)
  x.forEach((xi, i) => {
    const cx = toCanvasX(xi);
    const cyActual = toCanvasY(y[i]);
    const cyPred = toCanvasY(predictions[i]);

    // Arrow line
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, cyActual);
    ctx.lineTo(cx, cyPred);
    ctx.stroke();
    ctx.setLineDash([]);

    // Residual label
    const residual = y[i] - predictions[i];
    if (Math.abs(residual) > 0.05) {
      ctx.fillStyle = Math.abs(residual) > 1 ? C.red : C.orange;
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText((residual >= 0 ? '+' : '') + residual.toFixed(2), cx + 20, (cyActual + cyPred) / 2);
    }
  });

  // Draw prediction line
  ctx.strokeStyle = C.blue;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  const sortedIdx = x.map((_, i) => i).sort((a, b) => x[a] - x[b]);
  sortedIdx.forEach((i, idx) => {
    const cx = toCanvasX(x[i]);
    const cy = toCanvasY(predictions[i]);
    if (idx === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();

  // Prediction dots
  x.forEach((xi, i) => {
    const cx = toCanvasX(xi);
    const cy = toCanvasY(predictions[i]);
    ctx.fillStyle = C.blue;
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fill();
  });

  // Actual data points (filled circles)
  x.forEach((xi, i) => {
    const cx = toCanvasX(xi);
    const cy = toCanvasY(y[i]);
    ctx.fillStyle = C.green;
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });

  // Legend
  const legendY = h - 15;
  ctx.fillStyle = C.green;
  ctx.beginPath();
  ctx.arc(w - 220, legendY, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = C.muted;
  ctx.font = '11px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('Actual (y)', w - 210, legendY + 4);

  ctx.fillStyle = C.blue;
  ctx.beginPath();
  ctx.arc(w - 120, legendY, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = C.muted;
  ctx.fillText('Predicted', w - 110, legendY + 4);
}


// ═══════════════════════════════════════════════════
// SECTION 2: GRADIENT BOOSTING — CLASSIFICATION
// ═══════════════════════════════════════════════════

const gbClsData = {
  x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  y: [0, 0, 0, 0, 1, 0, 1, 1, 1, 1], // binary labels
  logOdds: [],
  probs: [],
  iteration: 0,
  lr: 0.5,
  maxIter: 15,
  autoPlaying: false,
  autoTimer: null
};

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function gbClsInit() {
  const n1 = gbClsData.y.filter(v => v === 1).length;
  const n0 = gbClsData.y.length - n1;
  const initLogOdds = Math.log(n1 / n0);
  gbClsData.logOdds = gbClsData.y.map(() => initLogOdds);
  gbClsData.probs = gbClsData.logOdds.map(sigmoid);
  gbClsData.iteration = 0;
  gbClsUpdateUI();
  gbClsDraw();
  drawSigmoid();
}

function gbClsReset() {
  if (gbClsData.autoTimer) clearInterval(gbClsData.autoTimer);
  gbClsData.autoPlaying = false;
  gbClsInit();
}

function gbClsUpdateLR(val) {
  gbClsData.lr = parseFloat(val);
  document.getElementById('gb-cls-lr-val').textContent = val;
}

function gbClsStep() {
  if (gbClsData.iteration >= gbClsData.maxIter) return;

  const { y, logOdds, probs, lr } = gbClsData;
  const n = y.length;

  // Calculate pseudo-residuals: r = y - p
  const residuals = y.map((yi, i) => yi - probs[i]);

  // Simulate tree: split into 2 groups, calculate leaf output
  const mid = 5;
  const leftResiduals = residuals.slice(0, mid);
  const rightResiduals = residuals.slice(mid);
  const leftProbs = probs.slice(0, mid);
  const rightProbs = probs.slice(mid);

  // Leaf output = sum(residuals) / sum(p*(1-p))  — this is the Newton-Raphson step
  const leftHessian = leftProbs.reduce((s, p) => s + p * (1 - p), 0);
  const rightHessian = rightProbs.reduce((s, p) => s + p * (1 - p), 0);
  const leftOutput = leftResiduals.reduce((a, b) => a + b, 0) / (leftHessian || 1);
  const rightOutput = rightResiduals.reduce((a, b) => a + b, 0) / (rightHessian || 1);

  // Update log-odds
  for (let i = 0; i < n; i++) {
    const output = i < mid ? leftOutput : rightOutput;
    gbClsData.logOdds[i] += lr * output;
    gbClsData.probs[i] = sigmoid(gbClsData.logOdds[i]);
  }

  gbClsData.iteration++;
  gbClsUpdateUI();
  gbClsDraw();
  drawSigmoid();
}

function gbClsAuto() {
  if (gbClsData.autoPlaying) {
    clearInterval(gbClsData.autoTimer);
    gbClsData.autoPlaying = false;
    return;
  }
  gbClsData.autoPlaying = true;
  gbClsData.autoTimer = setInterval(() => {
    if (gbClsData.iteration >= gbClsData.maxIter) {
      clearInterval(gbClsData.autoTimer);
      gbClsData.autoPlaying = false;
      return;
    }
    gbClsStep();
  }, 800);
}

function gbClsUpdateUI() {
  const { y, probs, logOdds, iteration } = gbClsData;
  const n = y.length;

  // Log loss
  const logLoss = -y.reduce((sum, yi, i) => {
    const p = clamp(probs[i], 0.001, 0.999);
    return sum + (yi * Math.log(p) + (1 - yi) * Math.log(1 - p));
  }, 0) / n;

  // Accuracy
  const correct = y.filter((yi, i) => (probs[i] >= 0.5 ? 1 : 0) === yi).length;
  const accuracy = (correct / n * 100).toFixed(0);

  document.querySelector('#gb-cls-iteration .info-value').textContent = iteration;
  document.querySelector('#gb-cls-logloss .info-value').textContent = logLoss.toFixed(4);
  document.querySelector('#gb-cls-accuracy .info-value').textContent = accuracy + '%';

  // Update table
  const tbody = document.getElementById('gb-cls-table-body');
  tbody.innerHTML = '';
  gbClsData.x.forEach((xi, i) => {
    const residual = y[i] - probs[i];
    const predicted = probs[i] >= 0.5 ? 1 : 0;
    const isCorrect = predicted === y[i];
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${xi}</td>
      <td><span style="color:${y[i] === 1 ? C.green : C.red}">${y[i]}</span></td>
      <td>${logOdds[i].toFixed(3)}</td>
      <td>${probs[i].toFixed(4)}</td>
      <td class="${residual >= 0 ? 'positive' : 'negative'}">${residual >= 0 ? '+' : ''}${residual.toFixed(4)}</td>
      <td class="${isCorrect ? 'correct' : 'wrong'}">${predicted} ${isCorrect ? '✅' : '❌'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function gbClsDraw() {
  const result = getCanvasContext('gb-cls-canvas');
  if (!result) return;
  const { ctx, w, h } = result;

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const pad = { top: 40, right: 40, bottom: 50, left: 60 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const { x, y, probs } = gbClsData;
  const xMin = 0, xMax = 11;
  const yMin = -0.1, yMax = 1.1;

  function toX(v) { return pad.left + ((v - xMin) / (xMax - xMin)) * plotW; }
  function toY(v) { return pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH; }

  // Grid lines
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  [0, 0.25, 0.5, 0.75, 1.0].forEach(yv => {
    const cy = toY(yv);
    ctx.beginPath();
    ctx.moveTo(pad.left, cy);
    ctx.lineTo(w - pad.right, cy);
    ctx.stroke();
    ctx.fillStyle = C.muted;
    ctx.font = '11px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(yv.toFixed(2), pad.left - 8, cy + 4);
  });

  // Decision boundary at 0.5
  ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.left, toY(0.5));
  ctx.lineTo(w - pad.right, toY(0.5));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = C.orange;
  ctx.font = '11px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('Decision Boundary (p = 0.5)', pad.left + 5, toY(0.5) - 8);

  // Title
  ctx.fillStyle = C.text;
  ctx.font = 'bold 14px Inter';
  ctx.textAlign = 'left';
  ctx.fillText(`GB Classification — Iteration ${gbClsData.iteration}`, pad.left, 25);

  // Probability curve
  ctx.strokeStyle = C.purple;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < x.length; i++) {
    const cx = toX(x[i]);
    const cy = toY(probs[i]);
    if (i === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  // Draw data points
  x.forEach((xi, i) => {
    const cx = toX(xi);
    const cyActual = toY(y[i]);
    const cyProb = toY(probs[i]);

    // Residual arrow
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, cyActual);
    ctx.lineTo(cx, cyProb);
    ctx.stroke();
    ctx.setLineDash([]);

    // Probability point
    ctx.fillStyle = C.purple;
    ctx.beginPath();
    ctx.arc(cx, cyProb, 5, 0, Math.PI * 2);
    ctx.fill();

    // Actual label point
    const color = y[i] === 1 ? C.green : C.red;
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cyActual, 8, 0, Math.PI * 2);
    if (y[i] === 1) {
      ctx.fill();
    } else {
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.fill();
      ctx.strokeStyle = C.red;
      ctx.stroke();
    }

    // Label
    ctx.fillStyle = C.text;
    ctx.font = 'bold 10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(y[i], cx, cyActual + 3.5);
  });

  // Legend
  ctx.fillStyle = C.green;
  ctx.beginPath();
  ctx.arc(w - 250, h - 15, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = C.muted;
  ctx.font = '11px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('Class 1', w - 240, h - 11);

  ctx.strokeStyle = C.red;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(w - 170, h - 15, 5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = C.muted;
  ctx.fillText('Class 0', w - 160, h - 11);

  ctx.fillStyle = C.purple;
  ctx.beginPath();
  ctx.arc(w - 95, h - 15, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = C.muted;
  ctx.fillText('Prob', w - 85, h - 11);
}

function drawSigmoid() {
  const result = getCanvasContext('sigmoid-canvas');
  if (!result) return;
  const { ctx, w, h } = result;

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const pad = { top: 30, right: 30, bottom: 35, left: 50 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const zMin = -6, zMax = 6;

  function toX(z) { return pad.left + ((z - zMin) / (zMax - zMin)) * plotW; }
  function toY(p) { return pad.top + plotH - p * plotH; }

  // Grid
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  [0, 0.5, 1].forEach(p => {
    const cy = toY(p);
    ctx.beginPath();
    ctx.moveTo(pad.left, cy);
    ctx.lineTo(w - pad.right, cy);
    ctx.stroke();
    ctx.fillStyle = C.muted;
    ctx.font = '10px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(p.toFixed(1), pad.left - 5, cy + 4);
  });

  // Sigmoid curve
  ctx.strokeStyle = C.cyan;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let z = zMin; z <= zMax; z += 0.1) {
    const p = sigmoid(z);
    const cx = toX(z);
    const cy = toY(p);
    if (z === zMin) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  // Labels
  ctx.fillStyle = C.muted;
  ctx.font = '11px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('F(x) = log-odds', w / 2, h - 5);
  ctx.save();
  ctx.translate(12, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('p = σ(F)', 0, 0);
  ctx.restore();

  // Plot current data points on sigmoid
  const { logOdds, y } = gbClsData;
  logOdds.forEach((lo, i) => {
    const clampedLO = clamp(lo, zMin, zMax);
    const p = sigmoid(lo);
    const cx = toX(clampedLO);
    const cy = toY(p);
    const color = y[i] === 1 ? C.green : C.red;

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#fff';
    ctx.font = 'bold 8px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(i + 1, cx, cy + 3);
  });

  // Title
  ctx.fillStyle = C.text;
  ctx.font = 'bold 13px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('σ(z) = 1 / (1 + e⁻ᶻ)', pad.left, 20);
}


// ═══════════════════════════════════════════════════
// SECTION 3: XGBOOST MATH CALCULATOR
// ═══════════════════════════════════════════════════

function xgbCalcUpdate() {
  const input = document.getElementById('xgb-residuals').value;
  const residuals = input.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
  if (residuals.length < 2) return;

  const splitIdx = parseInt(document.getElementById('xgb-split-idx').value);
  const lambda = parseFloat(document.getElementById('xgb-lambda').value);
  const gamma = parseFloat(document.getElementById('xgb-gamma').value);

  // Update display values
  document.getElementById('xgb-split-idx-val').textContent = splitIdx;
  document.getElementById('xgb-lambda-val').textContent = lambda;
  document.getElementById('xgb-gamma-val').textContent = gamma;

  // Update split slider max
  const splitSlider = document.getElementById('xgb-split-idx');
  splitSlider.max = residuals.length - 1;
  if (splitIdx >= residuals.length) {
    splitSlider.value = residuals.length - 1;
  }

  // For MSE: gradient g_i = -residual_i (since residual = y - ŷ, gradient = ŷ - y = -residual)
  // But we're working with residuals directly, so G = sum of gradients = -sum(residuals)
  // To keep it intuitive, let's use G = -sum(residuals), H = n (for MSE, hessian = 1 per sample)
  const gradients = residuals.map(r => -r);

  const left = gradients.slice(0, splitIdx);
  const right = gradients.slice(splitIdx);
  const all = gradients;

  const G_P = all.reduce((a, b) => a + b, 0);
  const H_P = all.length; // for MSE, hessian = 1
  const G_L = left.reduce((a, b) => a + b, 0);
  const H_L = left.length;
  const G_R = right.reduce((a, b) => a + b, 0);
  const H_R = right.length;

  const simParent = (G_P * G_P) / (H_P + lambda);
  const simLeft = (G_L * G_L) / (H_L + lambda);
  const simRight = (G_R * G_R) / (H_R + lambda);
  const gain = simLeft + simRight - simParent - gamma;

  const wParent = -G_P / (H_P + lambda);
  const wLeft = -G_L / (H_L + lambda);
  const wRight = -G_R / (H_R + lambda);

  const resultsDiv = document.getElementById('xgb-results');
  resultsDiv.innerHTML = `
    <div class="calc-result-row">
      <span class="calc-result-label">G_Parent (Σgᵢ)</span>
      <span class="calc-result-value">${G_P.toFixed(3)}</span>
    </div>
    <div class="calc-result-row">
      <span class="calc-result-label">H_Parent (n for MSE)</span>
      <span class="calc-result-value">${H_P}</span>
    </div>
    <div class="calc-result-row">
      <span class="calc-result-label">Similarity Parent = G²/(H+λ)</span>
      <span class="calc-result-value">${simParent.toFixed(3)}</span>
    </div>
    <hr style="border-color: rgba(255,255,255,0.05); margin: 0.3rem 0">
    <div class="calc-result-row">
      <span class="calc-result-label">G_Left = ${G_L.toFixed(2)}, H_Left = ${H_L}</span>
      <span class="calc-result-value" style="color:${C.blue}">Sim_L = ${simLeft.toFixed(3)}</span>
    </div>
    <div class="calc-result-row">
      <span class="calc-result-label">G_Right = ${G_R.toFixed(2)}, H_Right = ${H_R}</span>
      <span class="calc-result-value" style="color:${C.purple}">Sim_R = ${simRight.toFixed(3)}</span>
    </div>
    <hr style="border-color: rgba(255,255,255,0.05); margin: 0.3rem 0">
    <div class="calc-result-row">
      <span class="calc-result-label">🎯 GAIN = Sim_L + Sim_R − Sim_P − γ</span>
      <span class="calc-result-value ${gain > 0 ? 'gain-positive' : 'gain-negative'}">${gain.toFixed(3)} ${gain > 0 ? '✅ Split!' : '❌ No Split'}</span>
    </div>
    <hr style="border-color: rgba(255,255,255,0.05); margin: 0.3rem 0">
    <div class="calc-result-row">
      <span class="calc-result-label">w* Parent = −G/(H+λ)</span>
      <span class="calc-result-value">${wParent.toFixed(3)}</span>
    </div>
    <div class="calc-result-row">
      <span class="calc-result-label">w* Left</span>
      <span class="calc-result-value" style="color:${C.blue}">${wLeft.toFixed(3)}</span>
    </div>
    <div class="calc-result-row">
      <span class="calc-result-label">w* Right</span>
      <span class="calc-result-value" style="color:${C.purple}">${wRight.toFixed(3)}</span>
    </div>
  `;

  // Draw tree
  drawXGBTree(residuals, splitIdx, lambda, gamma, { G_P, H_P, G_L, H_L, G_R, H_R, simParent, simLeft, simRight, gain, wLeft, wRight });
}

function drawXGBTree(residuals, splitIdx, lambda, gamma, calc) {
  const result = getCanvasContext('xgb-tree-canvas');
  if (!result) return;
  const { ctx, w, h } = result;

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const cx = w / 2;

  // Parent node
  const parentY = 50;
  drawTreeNode(ctx, cx, parentY, `All Samples [${residuals.join(', ')}]`,
    `G=${calc.G_P.toFixed(1)} H=${calc.H_P} Sim=${calc.simParent.toFixed(2)}`, '#8b5cf6', 150);

  // Split condition
  ctx.fillStyle = C.orange;
  ctx.font = 'bold 12px Inter';
  ctx.textAlign = 'center';
  ctx.fillText(`Split after index ${splitIdx}`, cx, parentY + 60);

  // Left child
  const leftX = cx - 160;
  const childY = parentY + 120;
  const leftResiduals = residuals.slice(0, splitIdx);
  drawTreeNode(ctx, leftX, childY, `Left [${leftResiduals.join(', ')}]`,
    `G=${calc.G_L.toFixed(1)} H=${calc.H_L} Sim=${calc.simLeft.toFixed(2)}`, '#3b82f6', 130);
  ctx.fillStyle = C.cyan;
  ctx.font = '11px JetBrains Mono';
  ctx.fillText(`w*=${calc.wLeft.toFixed(3)}`, leftX, childY + 55);

  // Right child
  const rightX = cx + 160;
  const rightResiduals = residuals.slice(splitIdx);
  drawTreeNode(ctx, rightX, childY, `Right [${rightResiduals.join(', ')}]`,
    `G=${calc.G_R.toFixed(1)} H=${calc.H_R} Sim=${calc.simRight.toFixed(2)}`, '#8b5cf6', 130);
  ctx.fillStyle = C.cyan;
  ctx.font = '11px JetBrains Mono';
  ctx.fillText(`w*=${calc.wRight.toFixed(3)}`, rightX, childY + 55);

  // Connecting lines
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx - 40, parentY + 35);
  ctx.lineTo(leftX, childY - 20);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx + 40, parentY + 35);
  ctx.lineTo(rightX, childY - 20);
  ctx.stroke();

  // Gain result at bottom
  const gainY = childY + 85;
  const gainColor = calc.gain > 0 ? C.green : C.red;
  ctx.fillStyle = gainColor;
  ctx.font = 'bold 16px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText(`Gain = ${calc.simLeft.toFixed(2)} + ${calc.simRight.toFixed(2)} − ${calc.simParent.toFixed(2)} − ${gamma}`, cx, gainY);
  ctx.fillText(`= ${calc.gain.toFixed(3)} ${calc.gain > 0 ? '✅ SPLIT!' : '❌ NO SPLIT'}`, cx, gainY + 25);

  // Formula breakdown
  ctx.fillStyle = C.muted;
  ctx.font = '11px Inter';
  ctx.fillText('Gain = Sim_Left + Sim_Right − Sim_Parent − γ', cx, gainY + 55);
}

function drawTreeNode(ctx, x, y, line1, line2, color, width) {
  const hw = width;
  const hh = 32;

  // Glow
  ctx.shadowColor = color;
  ctx.shadowBlur = 15;

  drawRoundedRect(ctx, x - hw / 2, y - hh / 2, hw, hh * 2, 10);
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.shadowBlur = 0;

  ctx.fillStyle = C.text;
  ctx.font = '11px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText(line1, x, y + 5);

  ctx.fillStyle = C.muted;
  ctx.font = '10px JetBrains Mono';
  ctx.fillText(line2, x, y + 25);
}


// ═══════════════════════════════════════════════════
// SECTION 4: REGULARIZATION VISUALIZERS
// ═══════════════════════════════════════════════════

function updateLambdaViz() {
  const lambda = parseInt(document.getElementById('lambda-slider').value);
  document.getElementById('lambda-demo-val').textContent = `λ = ${lambda}`;

  const result = getCanvasContext('lambda-canvas');
  if (!result) return;
  const { ctx, w, h } = result;

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const pad = 30;
  const plotW = w - 2 * pad;
  const plotH = h - 2 * pad;

  // Show how leaf weights shrink with lambda
  const residuals = [-5, -3, 2, 4, 6];
  const n = residuals.length;
  const barW = plotW / (n * 2);

  // Without regularization
  const G = residuals.reduce((a, b) => a + (-b), 0);
  const H = n;

  residuals.forEach((r, i) => {
    const x = pad + i * (plotW / n) + barW / 2;

    // Without lambda (lambda=0)
    const wNoReg = r; // simplified
    // With lambda
    const wReg = -(-r) / (1 + lambda); // simplified per-sample

    const maxVal = 8;
    const barHNoReg = Math.abs(wNoReg) / maxVal * plotH * 0.4;
    const barHReg = Math.abs(wReg) / maxVal * plotH * 0.4;
    const baseY = h / 2;

    // No reg bar
    ctx.fillStyle = 'rgba(239, 68, 68, 0.4)';
    if (wNoReg >= 0) {
      ctx.fillRect(x, baseY - barHNoReg, barW * 0.8, barHNoReg);
    } else {
      ctx.fillRect(x, baseY, barW * 0.8, barHNoReg);
    }

    // With reg bar
    ctx.fillStyle = C.green;
    if (wReg >= 0) {
      ctx.fillRect(x + barW, baseY - barHReg, barW * 0.8, barHReg);
    } else {
      ctx.fillRect(x + barW, baseY, barW * 0.8, barHReg);
    }
  });

  // Center line
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, h / 2);
  ctx.lineTo(w - pad, h / 2);
  ctx.stroke();

  // Legend
  ctx.fillStyle = 'rgba(239, 68, 68, 0.6)';
  ctx.fillRect(pad, h - 20, 10, 10);
  ctx.fillStyle = C.muted;
  ctx.font = '10px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('No reg (λ=0)', pad + 15, h - 11);

  ctx.fillStyle = C.green;
  ctx.fillRect(pad + 100, h - 20, 10, 10);
  ctx.fillStyle = C.muted;
  ctx.fillText(`With λ=${lambda}`, pad + 115, h - 11);
}

function updateGammaViz() {
  const gamma = parseInt(document.getElementById('gamma-slider').value);
  document.getElementById('gamma-demo-val').textContent = `γ = ${gamma}`;

  const result = getCanvasContext('gamma-canvas');
  if (!result) return;
  const { ctx, w, h } = result;

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const pad = 30;
  const plotW = w - 2 * pad;

  // Show gains of 5 potential splits
  const gains = [18, 12, 8, 4, 1.5];
  const labels = ['Split A', 'Split B', 'Split C', 'Split D', 'Split E'];
  const maxGain = 20;
  const barH = 22;
  const gap = 8;

  gains.forEach((g, i) => {
    const y = pad + i * (barH + gap);
    const effectiveGain = g - gamma;
    const barWOrig = (g / maxGain) * plotW;
    const barWEff = Math.max(0, (effectiveGain / maxGain) * plotW);

    // Original gain (faded)
    ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
    ctx.fillRect(pad, y, barWOrig, barH);

    // Effective gain
    ctx.fillStyle = effectiveGain > 0 ? C.green : C.red;
    ctx.globalAlpha = 0.7;
    ctx.fillRect(pad, y, barWEff, barH);
    ctx.globalAlpha = 1;

    // Label
    ctx.fillStyle = C.text;
    ctx.font = '10px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(labels[i], pad - 5, y + 15);

    // Value
    ctx.fillStyle = effectiveGain > 0 ? C.green : C.red;
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'left';
    ctx.fillText(`${g} − ${gamma} = ${effectiveGain.toFixed(1)} ${effectiveGain > 0 ? '✅' : '❌'}`, pad + barWOrig + 5, y + 15);
  });

  // Gamma line
  const gammaX = pad + (gamma / maxGain) * plotW;
  if (gamma > 0 && gamma <= maxGain) {
    ctx.strokeStyle = C.orange;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(gammaX, pad - 5);
    ctx.lineTo(gammaX, pad + gains.length * (barH + gap));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.orange;
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(`γ = ${gamma}`, gammaX, pad - 10);
  }
}


// ═══════════════════════════════════════════════════
// SECTION 5: OVERFITTING VISUALIZER
// ═══════════════════════════════════════════════════

function updateOverfitViz() {
  const nEst = parseInt(document.getElementById('of-n-estimators').value);
  const lr = parseFloat(document.getElementById('of-lr').value);
  const depth = parseInt(document.getElementById('of-depth').value);
  const lambda = parseInt(document.getElementById('of-lambda').value);

  document.getElementById('of-n-val').textContent = nEst;
  document.getElementById('of-lr-val').textContent = lr.toFixed(2);
  document.getElementById('of-depth-val').textContent = depth;
  document.getElementById('of-lambda-val').textContent = lambda;

  const result = getCanvasContext('overfit-canvas');
  if (!result) return;
  const { ctx, w, h } = result;

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const pad = { top: 40, right: 30, bottom: 50, left: 60 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  // Simulate train/val curves based on parameters
  const complexity = (nEst / 100) * (lr * 5) * (depth / 3) * (1 / (1 + lambda * 0.3));
  const points = 50;
  const trainLoss = [];
  const valLoss = [];

  for (let i = 0; i < points; i++) {
    const t = (i + 1) / points;
    const iter = t * nEst;

    // Training loss always decreases
    const tl = 1.0 * Math.exp(-complexity * t * 3) + 0.01;
    trainLoss.push(tl);

    // Validation loss: decreases then increases (U-shape)
    const optimalPoint = 0.3 / (complexity * 0.5 + 0.3);
    const vlBase = 1.0 * Math.exp(-complexity * t * 2) + 0.05;
    const vlOverfit = complexity * 0.4 * Math.max(0, t - optimalPoint) ** 1.5;
    valLoss.push(vlBase + vlOverfit);
  }

  const allVals = [...trainLoss, ...valLoss];
  const yMax = Math.min(Math.max(...allVals) * 1.1, 2);
  const yMin = 0;

  function toX(i) { return pad.left + (i / (points - 1)) * plotW; }
  function toY(v) { return pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH; }

  // Grid
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const yv = yMin + (yMax - yMin) * i / 4;
    const cy = toY(yv);
    ctx.beginPath();
    ctx.moveTo(pad.left, cy);
    ctx.lineTo(w - pad.right, cy);
    ctx.stroke();
    ctx.fillStyle = C.muted;
    ctx.font = '10px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(yv.toFixed(2), pad.left - 8, cy + 4);
  }

  // Labels
  ctx.fillStyle = C.muted;
  ctx.font = '11px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Boosting Iterations →', w / 2, h - 8);

  // Training loss curve
  ctx.strokeStyle = C.blue;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  trainLoss.forEach((v, i) => {
    const cx = toX(i), cy = toY(v);
    if (i === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();

  // Validation loss curve
  ctx.strokeStyle = C.red;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  valLoss.forEach((v, i) => {
    const cx = toX(i), cy = toY(v);
    if (i === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();

  // Find optimal point (min validation loss)
  const minValIdx = valLoss.indexOf(Math.min(...valLoss));
  const optX = toX(minValIdx);
  const optY = toY(valLoss[minValIdx]);

  // Optimal vertical line
  ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  ctx.moveTo(optX, pad.top);
  ctx.lineTo(optX, h - pad.bottom);
  ctx.stroke();
  ctx.setLineDash([]);

  // Optimal point
  ctx.fillStyle = C.green;
  ctx.beginPath();
  ctx.arc(optX, optY, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = C.green;
  ctx.font = '10px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Early Stop Here!', optX, pad.top - 5);

  // Overfitting zone
  if (minValIdx < points - 5) {
    ctx.fillStyle = 'rgba(239, 68, 68, 0.06)';
    ctx.fillRect(optX, pad.top, w - pad.right - optX, plotH);
    ctx.fillStyle = 'rgba(239, 68, 68, 0.5)';
    ctx.font = '11px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('⚠️ Overfitting Zone', (optX + w - pad.right) / 2, pad.top + 20);
  }

  // Title
  ctx.fillStyle = C.text;
  ctx.font = 'bold 14px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('Train vs Validation Loss', pad.left, 25);

  // Diagnosis
  const diagEl = document.getElementById('overfit-diagnosis');
  const gap = valLoss[valLoss.length - 1] - trainLoss[trainLoss.length - 1];
  const finalTrain = trainLoss[trainLoss.length - 1];

  if (gap > 0.4) {
    diagEl.className = 'overfit-diagnosis overfit';
    diagEl.querySelector('.diagnosis-icon').textContent = '🔥';
    diagEl.querySelector('.diagnosis-text').innerHTML = `<strong style="color:${C.red}">OVERFITTING!</strong><br>Train-Val gap = ${gap.toFixed(3)}<br>💡 Fix: ↓ depth, ↓ n_estimators, ↓ LR, ↑ λ`;
  } else if (finalTrain > 0.3) {
    diagEl.className = 'overfit-diagnosis underfit';
    diagEl.querySelector('.diagnosis-icon').textContent = '😴';
    diagEl.querySelector('.diagnosis-text').innerHTML = `<strong style="color:${C.orange}">UNDERFITTING</strong><br>Both losses still high = ${finalTrain.toFixed(3)}<br>💡 Fix: ↑ depth, ↑ n_estimators, ↑ LR, ↓ λ`;
  } else {
    diagEl.className = 'overfit-diagnosis good';
    diagEl.querySelector('.diagnosis-icon').textContent = '✅';
    diagEl.querySelector('.diagnosis-text').innerHTML = `<strong style="color:${C.green}">GOOD FIT!</strong><br>Gap = ${gap.toFixed(3)}, Train = ${finalTrain.toFixed(3)}<br>🎯 Model is well-tuned!`;
  }
}

function updateBiasVar() {
  const complexity = parseInt(document.getElementById('bv-complexity').value);
  document.getElementById('bv-complexity-val').textContent = complexity;

  const result = getCanvasContext('bias-var-canvas');
  if (!result) return;
  const { ctx, w, h } = result;

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const pad = { top: 30, right: 30, bottom: 40, left: 60 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const points = 50;
  function toX(i) { return pad.left + (i / (points - 1)) * plotW; }
  function toY(v) { return pad.top + plotH - (v / 1.5) * plotH; }

  // Curves
  const bias = [], variance = [], total = [];
  for (let i = 0; i < points; i++) {
    const t = (i + 1) / points;
    const b = 1.2 * Math.exp(-3 * t);
    const v = 0.05 + 0.8 * t ** 2;
    bias.push(b);
    variance.push(v);
    total.push(b + v);
  }

  // Draw curves
  [
    { data: bias, color: '#ff6b6b', label: 'Bias²' },
    { data: variance, color: '#4ecdc4', label: 'Variance' },
    { data: total, color: '#ffd93d', label: 'Total Error' }
  ].forEach(({ data, color }) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    data.forEach((v, i) => {
      const cx = toX(i), cy = toY(v);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();
  });

  // Complexity marker
  const markerIdx = Math.floor((complexity / 10) * (points - 1));
  const markerX = toX(markerIdx);

  ctx.strokeStyle = 'rgba(255,255,255,0.4)';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  ctx.moveTo(markerX, pad.top);
  ctx.lineTo(markerX, h - pad.bottom);
  ctx.stroke();
  ctx.setLineDash([]);

  // Marker label
  const label = complexity <= 3 ? 'Underfitting' : complexity >= 8 ? 'Overfitting' : 'Sweet Spot ✅';
  const labelColor = complexity <= 3 ? C.orange : complexity >= 8 ? C.red : C.green;
  ctx.fillStyle = labelColor;
  ctx.font = 'bold 12px Inter';
  ctx.textAlign = 'center';
  ctx.fillText(label, markerX, pad.top - 8);

  // Axes
  ctx.fillStyle = C.muted;
  ctx.font = '11px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Model Complexity (depth, n_estimators, etc.) →', w / 2, h - 5);
  ctx.save();
  ctx.translate(15, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Error', 0, 0);
  ctx.restore();
}


// ═══════════════════════════════════════════════════
// SECTION 6: TREE GROWTH (Level-wise vs Leaf-wise)
// ═══════════════════════════════════════════════════

const treeGrowthState = {
  levelWise: { nodes: [{ id: 0, x: 0.5, y: 0.1, children: [], loss: 1.0, isLeaf: true }], step: 0 },
  leafWise: { nodes: [{ id: 0, x: 0.5, y: 0.1, children: [], loss: 1.0, isLeaf: true }], step: 0 },
  maxSteps: 4,
  autoTimer: null,
  autoPlaying: false
};

function treeGrowthReset() {
  if (treeGrowthState.autoTimer) clearInterval(treeGrowthState.autoTimer);
  treeGrowthState.autoPlaying = false;
  treeGrowthState.levelWise = { nodes: [{ id: 0, x: 0.5, y: 0.1, children: [], loss: 1.0, isLeaf: true, depth: 0 }], step: 0 };
  treeGrowthState.leafWise = { nodes: [{ id: 0, x: 0.5, y: 0.1, children: [], loss: 1.0, isLeaf: true, depth: 0 }], step: 0 };
  drawTreeGrowth();
  updateTreeGrowthInfo();
}

function treeGrowthStep() {
  if (treeGrowthState.levelWise.step >= treeGrowthState.maxSteps) return;

  // Level-wise: split ALL leaves at current depth
  growLevelWise();
  // Leaf-wise: split the leaf with HIGHEST loss
  growLeafWise();

  treeGrowthState.levelWise.step++;
  treeGrowthState.leafWise.step++;

  drawTreeGrowth();
  updateTreeGrowthInfo();
}

function treeGrowthAuto() {
  if (treeGrowthState.autoPlaying) {
    clearInterval(treeGrowthState.autoTimer);
    treeGrowthState.autoPlaying = false;
    return;
  }
  treeGrowthState.autoPlaying = true;
  treeGrowthState.autoTimer = setInterval(() => {
    if (treeGrowthState.levelWise.step >= treeGrowthState.maxSteps) {
      clearInterval(treeGrowthState.autoTimer);
      treeGrowthState.autoPlaying = false;
      return;
    }
    treeGrowthStep();
  }, 1200);
}

function growLevelWise() {
  const state = treeGrowthState.levelWise;
  const leaves = state.nodes.filter(n => n.isLeaf);
  const currentDepth = state.step;

  leaves.filter(n => n.depth === currentDepth).forEach(leaf => {
    leaf.isLeaf = false;
    const spread = 0.15 / (2 ** leaf.depth);
    const newDepth = leaf.depth + 1;
    const yPos = 0.1 + newDepth * 0.2;

    const left = { id: state.nodes.length, x: leaf.x - spread, y: yPos, children: [], loss: leaf.loss * (0.5 + Math.random() * 0.3), isLeaf: true, depth: newDepth };
    const right = { id: state.nodes.length + 1, x: leaf.x + spread, y: yPos, children: [], loss: leaf.loss * (0.4 + Math.random() * 0.4), isLeaf: true, depth: newDepth };

    leaf.children = [left, right];
    state.nodes.push(left, right);
  });
}

function growLeafWise() {
  const state = treeGrowthState.leafWise;
  const leaves = state.nodes.filter(n => n.isLeaf);

  // Find leaf with HIGHEST loss
  let bestLeaf = leaves[0];
  leaves.forEach(l => { if (l.loss > bestLeaf.loss) bestLeaf = l; });

  bestLeaf.isLeaf = false;
  const spread = 0.15 / (2 ** bestLeaf.depth);
  const newDepth = bestLeaf.depth + 1;
  const yPos = 0.1 + newDepth * 0.2;

  const left = { id: state.nodes.length, x: bestLeaf.x - spread, y: yPos, children: [], loss: bestLeaf.loss * (0.3 + Math.random() * 0.3), isLeaf: true, depth: newDepth };
  const right = { id: state.nodes.length + 1, x: bestLeaf.x + spread, y: yPos, children: [], loss: bestLeaf.loss * (0.3 + Math.random() * 0.2), isLeaf: true, depth: newDepth };

  bestLeaf.children = [left, right];
  state.nodes.push(left, right);
}

function drawTreeGrowth() {
  drawSingleTree('levelwise-canvas', treeGrowthState.levelWise, C.blue);
  drawSingleTree('leafwise-canvas', treeGrowthState.leafWise, C.green);
}

function drawSingleTree(canvasId, state, color) {
  const result = getCanvasContext(canvasId);
  if (!result) return;
  const { ctx, w, h } = result;

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  // Draw connections
  state.nodes.forEach(node => {
    if (node.children.length === 2) {
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(node.x * w, node.y * h + 15);
      ctx.lineTo(node.children[0].x * w, node.children[0].y * h - 15);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(node.x * w, node.y * h + 15);
      ctx.lineTo(node.children[1].x * w, node.children[1].y * h - 15);
      ctx.stroke();
    }
  });

  // Draw nodes
  state.nodes.forEach(node => {
    const nx = node.x * w;
    const ny = node.y * h;
    const radius = node.isLeaf ? 16 : 13;

    // Node circle
    if (node.isLeaf) {
      const lossIntensity = Math.min(node.loss, 1);
      const r = Math.round(lerp(16, 239, lossIntensity));
      const g = Math.round(lerp(185, 68, lossIntensity));
      const b = Math.round(lerp(129, 68, lossIntensity));
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.shadowColor = `rgb(${r},${g},${b})`;
      ctx.shadowBlur = 10;
    } else {
      ctx.fillStyle = 'rgba(255,255,255,0.1)';
      ctx.shadowBlur = 0;
    }

    ctx.beginPath();
    ctx.arc(nx, ny, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Border
    ctx.strokeStyle = node.isLeaf ? color : 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Loss label for leaves
    if (node.isLeaf) {
      ctx.fillStyle = '#fff';
      ctx.font = '9px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(node.loss.toFixed(2), nx, ny + 3);
    }
  });

  // Step label
  ctx.fillStyle = C.muted;
  ctx.font = '12px Inter';
  ctx.textAlign = 'left';
  ctx.fillText(`Step: ${state.step}`, 10, h - 10);
}

function updateTreeGrowthInfo() {
  const lwLeaves = treeGrowthState.levelWise.nodes.filter(n => n.isLeaf).length;
  const fwLeaves = treeGrowthState.leafWise.nodes.filter(n => n.isLeaf).length;
  document.getElementById('lw-leaves').textContent = lwLeaves;
  document.getElementById('fw-leaves').textContent = fwLeaves;

  const insights = [
    'Click "Grow" to see the difference!',
    'Level-wise splits ALL leaves. Leaf-wise splits only the BEST one.',
    'Notice: Leaf-wise can go DEEPER on one side — more focused!',
    'Level-wise has more leaves but balanced depth. Leaf-wise is asymmetric.',
    'Done! LightGBM (leaf-wise) focuses compute on highest-loss regions.'
  ];
  document.getElementById('tree-insight').textContent = insights[treeGrowthState.levelWise.step] || insights[4];
}

// Symmetric tree demo
function drawSymmetricDemo() {
  const result = getCanvasContext('symmetric-canvas');
  if (!result) return;
  const { ctx, w, h } = result;

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const mid = w / 2;

  // Normal tree (left side)
  ctx.fillStyle = C.text;
  ctx.font = 'bold 13px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Normal Tree', mid / 2, 25);

  // Draw normal tree with different splits
  const normalSplits = [
    { x: mid / 2, y: 55, label: 'x₁ > 5', color: C.blue },
    { x: mid / 2 - 80, y: 120, label: 'x₂ > 3', color: C.purple },
    { x: mid / 2 + 80, y: 120, label: 'x₃ > 7', color: C.orange },
  ];

  normalSplits.forEach(s => {
    drawMiniNode(ctx, s.x, s.y, s.label, s.color);
  });

  // Lines
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(mid / 2, 75); ctx.lineTo(mid / 2 - 80, 105); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(mid / 2, 75); ctx.lineTo(mid / 2 + 80, 105); ctx.stroke();

  // Normal leaves
  [mid / 2 - 120, mid / 2 - 40, mid / 2 + 40, mid / 2 + 120].forEach((lx, i) => {
    drawMiniLeaf(ctx, lx, 190, `L${i + 1}`);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.beginPath();
    const parentX = i < 2 ? mid / 2 - 80 : mid / 2 + 80;
    ctx.moveTo(parentX, 140);
    ctx.lineTo(lx, 175);
    ctx.stroke();
  });

  // Divider
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(mid, 15);
  ctx.lineTo(mid, h - 15);
  ctx.stroke();
  ctx.setLineDash([]);

  // Symmetric tree (right side)
  ctx.fillStyle = C.text;
  ctx.font = 'bold 13px Inter';
  ctx.fillText('Symmetric (Oblivious) Tree', mid + mid / 2, 25);

  const symX = mid + mid / 2;
  drawMiniNode(ctx, symX, 55, 'x₁ > 5', C.pink);
  drawMiniNode(ctx, symX - 80, 120, 'x₂ > 3', C.pink);
  drawMiniNode(ctx, symX + 80, 120, 'x₂ > 3', C.pink); // SAME split!

  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.beginPath(); ctx.moveTo(symX, 75); ctx.lineTo(symX - 80, 105); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(symX, 75); ctx.lineTo(symX + 80, 105); ctx.stroke();

  [symX - 120, symX - 40, symX + 40, symX + 120].forEach((lx, i) => {
    drawMiniLeaf(ctx, lx, 190, `L${i + 1}`);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.beginPath();
    const parentX = i < 2 ? symX - 80 : symX + 80;
    ctx.moveTo(parentX, 140);
    ctx.lineTo(lx, 175);
    ctx.stroke();
  });

  // Highlight "SAME" label
  ctx.fillStyle = C.pink;
  ctx.font = 'bold 11px Inter';
  ctx.fillText('↑ SAME split on each level! ↑', mid + mid / 2, 155);

  // Arrow
  ctx.fillStyle = C.green;
  ctx.font = '11px Inter';
  ctx.fillText('✅ Fast inference, natural regularization', mid + mid / 2, h - 30);
  ctx.fillStyle = C.muted;
  ctx.font = '11px Inter';
  ctx.fillText('↑ Different splits per node', mid / 2, h - 30);
}

function drawMiniNode(ctx, x, y, label, color) {
  drawRoundedRect(ctx, x - 40, y - 15, 80, 30, 8);
  ctx.fillStyle = 'rgba(0,0,0,0.4)';
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.font = '11px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText(label, x, y + 4);
}

function drawMiniLeaf(ctx, x, y, label) {
  ctx.fillStyle = 'rgba(16, 185, 129, 0.3)';
  ctx.beginPath();
  ctx.arc(x, y, 14, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = C.green;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = C.text;
  ctx.font = '10px Inter';
  ctx.textAlign = 'center';
  ctx.fillText(label, x, y + 4);
}


// ═══════════════════════════════════════════════════
// SECTION 7: ADABOOST ANIMATION
// ═══════════════════════════════════════════════════

const adaState = {
  x: [],
  y: [],
  labels: [],
  weights: [],
  round: 0,
  maxRounds: 6,
  stumps: [],
  alphas: [],
  autoTimer: null,
  autoPlaying: false
};

function adaInit() {
  // Generate 2D data for visualization
  const n = 20;
  adaState.x = [];
  adaState.labels = [];

  // Random seed-like generation
  for (let i = 0; i < n; i++) {
    const angle = (i / n) * Math.PI * 2;
    const r = 0.3 + (i % 3) * 0.1;
    let px, py, label;
    if (i < n / 2) {
      px = 0.3 + Math.cos(angle) * r * 0.5 + (Math.sin(i * 7) * 0.05);
      py = 0.4 + Math.sin(angle) * r * 0.5 + (Math.cos(i * 5) * 0.05);
      label = 1;
    } else {
      px = 0.65 + Math.cos(angle) * r * 0.4 + (Math.sin(i * 3) * 0.05);
      py = 0.55 + Math.sin(angle) * r * 0.4 + (Math.cos(i * 7) * 0.05);
      label = -1;
    }
    adaState.x.push([clamp(px, 0.05, 0.95), clamp(py, 0.05, 0.95)]);
    adaState.labels.push(label);
  }

  adaState.weights = Array(n).fill(1 / n);
  adaState.round = 0;
  adaState.stumps = [];
  adaState.alphas = [];

  adaUpdateUI();
  adaDraw();
  drawAlphaCurve();
}

function adaReset() {
  if (adaState.autoTimer) clearInterval(adaState.autoTimer);
  adaState.autoPlaying = false;
  adaInit();
}

function adaStep() {
  if (adaState.round >= adaState.maxRounds) return;

  const { x, labels, weights } = adaState;
  const n = x.length;

  // Simple stump: try horizontal and vertical splits
  let bestError = Infinity;
  let bestStump = null;

  // Try different split thresholds
  for (let dim = 0; dim < 2; dim++) {
    for (let thresh = 0.1; thresh <= 0.9; thresh += 0.05) {
      for (let dir of [1, -1]) {
        let error = 0;
        const preds = x.map(p => (p[dim] > thresh ? 1 : -1) * dir);
        for (let i = 0; i < n; i++) {
          if (preds[i] !== labels[i]) error += weights[i];
        }
        if (error < bestError) {
          bestError = error;
          bestStump = { dim, thresh, dir, preds };
        }
      }
    }
  }

  if (!bestStump || bestError >= 0.5) return;

  // Alpha
  const epsilon = Math.max(bestError, 1e-10);
  const alpha = 0.5 * Math.log((1 - epsilon) / epsilon);

  // Update weights
  const newWeights = weights.map((w, i) => {
    if (bestStump.preds[i] !== labels[i]) {
      return w * Math.exp(alpha);
    } else {
      return w * Math.exp(-alpha);
    }
  });

  // Normalize
  const wSum = newWeights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < n; i++) newWeights[i] /= wSum;

  adaState.weights = newWeights;
  adaState.stumps.push(bestStump);
  adaState.alphas.push(alpha);
  adaState.round++;

  adaUpdateUI();
  adaDraw();
  drawAlphaCurve();
}

function adaAuto() {
  if (adaState.autoPlaying) {
    clearInterval(adaState.autoTimer);
    adaState.autoPlaying = false;
    return;
  }
  adaState.autoPlaying = true;
  adaState.autoTimer = setInterval(() => {
    if (adaState.round >= adaState.maxRounds) {
      clearInterval(adaState.autoTimer);
      adaState.autoPlaying = false;
      return;
    }
    adaStep();
  }, 1000);
}

function adaUpdateUI() {
  document.querySelector('#ada-round .info-value').textContent = adaState.round;

  if (adaState.round > 0) {
    const lastStump = adaState.stumps[adaState.stumps.length - 1];
    const lastAlpha = adaState.alphas[adaState.alphas.length - 1];
    const error = adaState.weights.reduce((s, w, i) => {
      return s + (lastStump.preds[i] !== adaState.labels[i] ? w : 0);
    }, 0);
    document.querySelector('#ada-error .info-value').textContent = error.toFixed(4);
    document.querySelector('#ada-alpha .info-value').textContent = lastAlpha.toFixed(4);
  }
}

function adaDraw() {
  const result = getCanvasContext('ada-canvas');
  if (!result) return;
  const { ctx, w, h } = result;

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const { x, labels, weights, stumps } = adaState;
  const maxWeight = Math.max(...weights);

  // Draw stump decision boundaries
  stumps.forEach((stump, si) => {
    const alpha = adaState.alphas[si];
    const opacity = Math.min(alpha / 2, 0.4);

    if (stump.dim === 0) {
      // Vertical line
      ctx.strokeStyle = `rgba(245, 158, 11, ${opacity})`;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(stump.thresh * w, 0);
      ctx.lineTo(stump.thresh * w, h);
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      // Horizontal line
      ctx.strokeStyle = `rgba(6, 182, 212, ${opacity})`;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(0, stump.thresh * h);
      ctx.lineTo(w, stump.thresh * h);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  });

  // Draw data points with weight as size
  x.forEach((pt, i) => {
    const px = pt[0] * w;
    const py = pt[1] * h;
    const baseRadius = 6;
    const weightRadius = baseRadius + (weights[i] / maxWeight) * 20;
    const label = labels[i];
    const color = label === 1 ? C.green : C.red;

    // Weight ring
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.3;
    ctx.beginPath();
    ctx.arc(px, py, weightRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Point
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, baseRadius, 0, Math.PI * 2);
    ctx.fill();

    // Check if misclassified by latest stump
    if (stumps.length > 0) {
      const lastStump = stumps[stumps.length - 1];
      if (lastStump.preds[i] !== labels[i]) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        // Draw X mark
        ctx.beginPath();
        ctx.moveTo(px - 4, py - 4);
        ctx.lineTo(px + 4, py + 4);
        ctx.moveTo(px + 4, py - 4);
        ctx.lineTo(px - 4, py + 4);
        ctx.stroke();
      }
    }
  });

  // Title and legend
  ctx.fillStyle = C.text;
  ctx.font = 'bold 14px Inter';
  ctx.textAlign = 'left';
  ctx.fillText(`AdaBoost — Round ${adaState.round}`, 15, 25);

  ctx.fillStyle = C.muted;
  ctx.font = '11px Inter';
  ctx.fillText('Circle size = sample weight. Bigger = harder to classify.', 15, h - 15);

  // Legend
  ctx.fillStyle = C.green;
  ctx.beginPath(); ctx.arc(w - 110, 20, 5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = C.muted;
  ctx.font = '11px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('Class +1', w - 100, 24);

  ctx.fillStyle = C.red;
  ctx.beginPath(); ctx.arc(w - 110, 40, 5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = C.muted;
  ctx.fillText('Class -1', w - 100, 44);
}

function drawAlphaCurve() {
  const result = getCanvasContext('alpha-curve-canvas');
  if (!result) return;
  const { ctx, w, h } = result;

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const pad = { top: 15, right: 15, bottom: 25, left: 35 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  // Draw alpha vs error curve
  const points = 50;
  ctx.strokeStyle = C.purple;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 1; i < points; i++) {
    const eps = i / points;
    if (eps >= 0.5) continue;
    const alpha = 0.5 * Math.log((1 - eps) / eps);
    const cx = pad.left + (eps / 0.5) * plotW;
    const cy = pad.top + plotH - (alpha / 3) * plotH;
    if (i === 1) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  // Axes
  ctx.fillStyle = C.muted;
  ctx.font = '9px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('ε (error) →', w / 2, h - 3);
  ctx.textAlign = 'right';
  ctx.fillText('α', pad.left - 5, pad.top + plotH / 2);

  // Mark current alpha if exists
  if (adaState.alphas.length > 0) {
    const lastAlpha = adaState.alphas[adaState.alphas.length - 1];
    const lastStump = adaState.stumps[adaState.stumps.length - 1];
    let eps = 0;
    adaState.weights.forEach((w, i) => {
      if (lastStump.preds[i] !== adaState.labels[i]) eps += w;
    });

    if (eps < 0.5) {
      const cx = pad.left + (eps / 0.5) * plotW;
      const cy = pad.top + plotH - (lastAlpha / 3) * plotH;
      ctx.fillStyle = C.orange;
      ctx.beginPath();
      ctx.arc(cx, clamp(cy, pad.top, pad.top + plotH), 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}


// ═══════════════════════════════════════════════════
// NAVIGATION & SCROLL TRACKING
// ═══════════════════════════════════════════════════

const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('.section');

function updateActiveNav() {
  let current = '';
  sections.forEach(section => {
    const sectionTop = section.offsetTop - 100;
    if (window.scrollY >= sectionTop) {
      current = section.getAttribute('id');
    }
  });

  navLinks.forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('href') === '#' + current) {
      link.classList.add('active');
    }
  });
}

window.addEventListener('scroll', updateActiveNav);


// ═══════════════════════════════════════════════════
// INTERSECTION OBSERVER FOR LAZY INITIALIZATION
// ═══════════════════════════════════════════════════

const initialized = new Set();

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting && !initialized.has(entry.target.id)) {
      initialized.add(entry.target.id);
      switch (entry.target.id) {
        case 'gb-regression':
          gbRegInit();
          break;
        case 'gb-classification':
          gbClsInit();
          break;
        case 'xgb-math':
          xgbCalcUpdate();
          updateLambdaViz();
          updateGammaViz();
          break;
        case 'overfitting':
          updateOverfitViz();
          updateBiasVar();
          break;
        case 'tree-growth':
          treeGrowthReset();
          drawSymmetricDemo();
          break;
        case 'adaboost':
          adaInit();
          break;
      }
    }
  });
}, { threshold: 0.1 });

sections.forEach(section => {
  if (section.id !== 'hero' && section.id !== 'comparison') {
    observer.observe(section);
  }
});


// ═══════════════════════════════════════════════════
// WINDOW RESIZE HANDLER
// ═══════════════════════════════════════════════════

let resizeTimeout;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    initialized.forEach(id => {
      switch (id) {
        case 'gb-regression': gbRegDraw(); break;
        case 'gb-classification': gbClsDraw(); drawSigmoid(); break;
        case 'xgb-math': xgbCalcUpdate(); updateLambdaViz(); updateGammaViz(); break;
        case 'overfitting': updateOverfitViz(); updateBiasVar(); break;
        case 'tree-growth': drawTreeGrowth(); drawSymmetricDemo(); break;
        case 'adaboost': adaDraw(); drawAlphaCurve(); break;
      }
    });
  }, 200);
});
