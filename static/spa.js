(() => {
  const SAFE_RATIO = 0.8;
  const STEADY_RATIO = 1.1;
  const TIER_DISTRIBUTION = { 冲: 0.25, 稳: 0.45, 保: 0.3 };
  const PROBABILITY_GUIDE = [
    { min: 0.75, label: "高把握", desc: "基本属于保底区间，可优先考虑专业偏好" },
    { min: 0.6, label: "较稳妥", desc: "录取机会较高，适合作为稳妥志愿" },
    { min: 0.45, label: "可尝试", desc: "存在机会，建议与保底志愿搭配" },
    { min: 0.3, label: "冲刺项", desc: "风险偏高，建议放在前段冲刺" },
    { min: 0, label: "高风险", desc: "录取不确定性较大，谨慎填报" }
  ];

  const state = {
    datasets: {
      历史: null,
      物理: null,
      艺术: null
    },
    mockTables: {
      历史: {},
      物理: {}
    }
  };

  function normalizeCol(col) {
    return String(col ?? "").replace(/\s+/g, "");
  }

  function normalizeStream(value) {
    const v = String(value ?? "").trim().toLowerCase();
    if (["历史", "hist", "history", "1", "h"].includes(v)) return "历史";
    if (["物理", "phys", "physics", "2", "p"].includes(v)) return "物理";
    if (["艺术", "art", "arts", "3", "a"].includes(v)) return "艺术";
    return "";
  }

  function toNumber(v) {
    if (v === null || v === undefined || String(v).trim() === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  function toInt(v) {
    const n = toNumber(v);
    if (n === null) return null;
    return Math.trunc(n);
  }

  function serializeScore(value) {
    if (Math.abs(value - Math.round(value)) < 1e-6) return Math.round(value);
    return Math.round(value * 100) / 100;
  }

  function roundHalfUp(value, digits = 2) {
    const factor = 10 ** digits;
    return Math.round((value + Number.EPSILON) * factor) / factor;
  }

  function calcArtComposite(cultureScore, artScore) {
    const value = (cultureScore / 750) * 300 * 0.5 + artScore * 0.5;
    return roundHalfUp(value, 2);
  }

  function calcRatio(userRank, minRank) {
    if (minRank <= 0) return 999;
    return userRank / minRank;
  }

  function calcTier(ratio) {
    if (ratio <= SAFE_RATIO) return "保";
    if (ratio <= STEADY_RATIO) return "稳";
    return "冲";
  }

  function calcProbability(ratio) {
    let p = 0.08;
    if (ratio <= 0.65) p = 0.84;
    else if (ratio <= SAFE_RATIO) p = 0.66 + (SAFE_RATIO - ratio) * (0.18 / 0.15);
    else if (ratio <= STEADY_RATIO) p = 0.4 + (STEADY_RATIO - ratio) * (0.26 / 0.3);
    else if (ratio <= 1.3) p = 0.2 + (1.3 - ratio) * (0.2 / 0.2);
    p = Math.max(0.05, Math.min(0.9, p));
    return Math.round(p * 10000) / 10000;
  }

  function explainProbability(probability, tier, ratio) {
    for (const level of PROBABILITY_GUIDE) {
      if (probability >= level.min) {
        return `${level.label}：${level.desc}（分档${tier}，位次比${ratio.toFixed(3)}）`;
      }
    }
    return `高风险：录取不确定性较大（分档${tier}，位次比${ratio.toFixed(3)}）`;
  }

  function readWorkbook(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const data = new Uint8Array(reader.result);
          const wb = XLSX.read(data, { type: "array" });
          resolve(wb);
        } catch (err) {
          reject(err);
        }
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsArrayBuffer(file);
    });
  }

  async function readWorkbookFromPath(path) {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`无法读取文件 ${path}（HTTP ${res.status}）`);
    }
    const buffer = await res.arrayBuffer();
    const data = new Uint8Array(buffer);
    return XLSX.read(data, { type: "array" });
  }

  function firstSheetRows(workbook) {
    const name = workbook.SheetNames[0];
    const sheet = workbook.Sheets[name];
    return XLSX.utils.sheet_to_json(sheet, { defval: "" });
  }

  function parseAdmissionRows(rows) {
    const mapped = rows.map((r) => {
      const normalized = {};
      for (const k of Object.keys(r)) normalized[normalizeCol(k)] = r[k];
      return {
        院校代号: String(normalized["院校代号"] ?? "").trim(),
        院校名称: String(normalized["院校名称"] ?? "").trim(),
        专业代号: String(normalized["专业代号"] ?? "").trim(),
        专业名称: String(normalized["专业名称"] ?? "").trim(),
        最低分: toNumber(normalized["投档最低分"] ?? normalized["最低分"])
      };
    });
    const valid = mapped.filter((r) => r.院校名称 && r.专业名称 && r.最低分 !== null);
    if (!valid.length) throw new Error("录取表没有有效数据，请检查列名与内容");
    return valid;
  }

  function parseScoreTableRows(rows) {
    const scoreMap = new Map();
    let topScore = null;
    let topRank = null;
    for (const r of rows) {
      const normalized = {};
      for (const k of Object.keys(r)) normalized[normalizeCol(k)] = r[k];
      const segmentRaw = normalized["分数段"];
      const rankRaw = toInt(normalized["累计人数"]);
      if (segmentRaw === undefined || rankRaw === null) continue;
      const segment = String(segmentRaw).trim();
      if (!segment || segment.toLowerCase() === "nan") continue;
      const cleaned = segment.replace(/分/g, "").replace(/\s+/g, "");

      if (/及以上/.test(cleaned)) {
        const nums = cleaned.match(/\d+(?:\.\d+)?/g);
        if (nums?.length) {
          topScore = Number(nums[0]);
          topRank = rankRaw;
        }
        continue;
      }
      if (/^\d+(?:\.\d+)?$/.test(cleaned)) {
        scoreMap.set(Number(cleaned), rankRaw);
        continue;
      }
      const nums = cleaned.match(/\d+(?:\.\d+)?/g);
      if (nums?.length) scoreMap.set(Number(nums[nums.length - 1]), rankRaw);
    }
    if (!scoreMap.size) throw new Error("一分一段表没有有效分数段数据");
    return { scoreMap, topScore, topRank };
  }

  function buildRankMapFromAdmission(admissionRows) {
    const counts = new Map();
    for (const row of admissionRows) {
      const s = Number(row.最低分);
      counts.set(s, (counts.get(s) ?? 0) + 1);
    }
    const scores = [...counts.keys()].sort((a, b) => b - a);
    const scoreMap = new Map();
    let cum = 0;
    for (const s of scores) {
      cum += counts.get(s);
      scoreMap.set(s, cum);
    }
    return { scoreMap, topScore: scores[0], topRank: scoreMap.get(scores[0]) };
  }

  function sortedScoresDesc(scoreMap) {
    return [...scoreMap.keys()].sort((a, b) => b - a);
  }

  function scoreToRank(score, scoreMap, topScore, topRank) {
    if (topScore !== null && topRank !== null && score >= topScore) return topRank;
    const scores = sortedScoresDesc(scoreMap);
    for (const s of scores) {
      if (score >= s) return scoreMap.get(s);
    }
    return scoreMap.get(scores[scores.length - 1]);
  }

  function rankToScore(rank, scoreMap) {
    const r = Math.max(1, rank);
    const scores = sortedScoresDesc(scoreMap);
    let last = scores[scores.length - 1];
    for (const s of scores) {
      if (scoreMap.get(s) >= r) return s;
      last = s;
    }
    return last;
  }

  function convertRankByPercentile(sourceRank, sourceTotal, targetTotal) {
    if (sourceTotal <= 0 || targetTotal <= 0) throw new Error("总人数无效，无法换算");
    let ratio = sourceRank / sourceTotal;
    ratio = Math.max(1 / sourceTotal, Math.min(ratio, 1));
    const target = Math.round(ratio * targetTotal);
    return Math.max(1, Math.min(target, targetTotal));
  }

  function buildRankedAdmission(admissionRows, scoreMap, topScore, topRank) {
    return admissionRows.map((row) => ({
      ...row,
      最低位次: scoreToRank(Number(row.最低分), scoreMap, topScore, topRank)
    }));
  }

  function buildDataset(admissionRows, scorePack) {
    const ranked = buildRankedAdmission(admissionRows, scorePack.scoreMap, scorePack.topScore, scorePack.topRank);
    const totalRank = Math.max(...ranked.map((r) => Number(r.最低位次)));
    return {
      admission: ranked,
      scoreMap: scorePack.scoreMap,
      topScore: scorePack.topScore,
      topRank: scorePack.topRank,
      totalRank
    };
  }

  function buildTierTargets(topN) {
    const targets = {
      冲: Math.floor(topN * TIER_DISTRIBUTION.冲),
      稳: Math.floor(topN * TIER_DISTRIBUTION.稳),
      保: Math.floor(topN * TIER_DISTRIBUTION.保)
    };
    let remain = topN - (targets.冲 + targets.稳 + targets.保);
    const fill = ["稳", "保", "冲"];
    let idx = 0;
    while (remain > 0) {
      targets[fill[idx % fill.length]] += 1;
      remain -= 1;
      idx += 1;
    }
    return targets;
  }

  function selectByTierStrategy(rows, topN, perSchoolLimit = 2) {
    if (!rows.length) return [];
    const desired = buildTierTargets(topN);
    const sortFn = (a, b) => a.fit - b.fit || b.probability - a.probability || a.min_rank - b.min_rank;
    const tierOrder = ["冲", "稳", "保"];
    const tiers = {
      冲: rows.filter((r) => r.tier === "冲").sort(sortFn),
      稳: rows.filter((r) => r.tier === "稳").sort(sortFn),
      保: rows.filter((r) => r.tier === "保").sort(sortFn)
    };
    const all = [...rows].sort(sortFn);
    const selected = [];
    const selectedSet = new Set();
    const schoolCount = new Map();

    function pick(pool, need, enforceLimit) {
      let left = need;
      for (const row of pool) {
        if (selected.length >= topN || left <= 0) break;
        const key = row._idx;
        if (selectedSet.has(key)) continue;
        const cnt = schoolCount.get(row.school_name) ?? 0;
        if (enforceLimit && cnt >= perSchoolLimit) continue;
        selected.push(row);
        selectedSet.add(key);
        schoolCount.set(row.school_name, cnt + 1);
        left -= 1;
      }
    }

    for (const tier of tierOrder) pick(tiers[tier], desired[tier], true);
    if (selected.length < topN) pick(all, topN - selected.length, true);
    if (selected.length < topN) pick(all, topN - selected.length, false);
    return selected;
  }

  function recommend(stream, userRank, filters) {
    const ds = state.datasets[stream];
    if (!ds) throw new Error(`${stream} 数据未加载`);
    const schoolKeyword = String(filters.school_keyword ?? "").trim();
    const majorKeyword = String(filters.major_keyword ?? "").trim();
    const regionKeyword = String(filters.region_keyword ?? "").trim();
    const hasKeyword = Boolean(schoolKeyword || majorKeyword || regionKeyword);
    const topN = Math.max(1, Number(filters.top_n ?? 50));

    let rows = ds.admission.map((row, idx) => {
      const ratio = calcRatio(userRank, Number(row.最低位次));
      const tier = calcTier(ratio);
      const probability = calcProbability(ratio);
      return {
        _idx: idx,
        school_code: row.院校代号,
        school_name: row.院校名称,
        major_code: row.专业代号,
        major_name: row.专业名称,
        min_score: serializeScore(Number(row.最低分)),
        min_rank: Number(row.最低位次),
        rank_ratio: Math.round(ratio * 10000) / 10000,
        tier,
        probability,
        probability_desc: explainProbability(probability, tier, ratio),
        fit: Math.abs(ratio - 1)
      };
    });

    if (schoolKeyword) rows = rows.filter((r) => r.school_name.includes(schoolKeyword));
    if (majorKeyword) rows = rows.filter((r) => r.major_name.includes(majorKeyword));
    if (regionKeyword) rows = rows.filter((r) => r.school_name.includes(regionKeyword));

    rows.sort((a, b) => a.fit - b.fit || b.probability - a.probability || a.min_rank - b.min_rank);
    rows = hasKeyword ? rows.slice(0, topN) : selectByTierStrategy(rows, topN, 2);

    const stats = { 冲: 0, 稳: 0, 保: 0 };
    rows.forEach((r) => {
      if (stats[r.tier] !== undefined) stats[r.tier] += 1;
      delete r.fit;
      delete r._idx;
    });
    return { rows, stats };
  }

  function rankFromPredictInput(form) {
    const stream = normalizeStream(form.stream);
    if (!stream) throw new Error("科类仅支持 历史/物理/艺术");
    const rank = toInt(form.rank);
    const score = toNumber(form.score);

    if (rank && rank > 0) return { stream, userRank: rank, usedScore: null };

    if (stream === "艺术") {
      let usedScore = score;
      if (usedScore === null) {
        const culture = toNumber(form.art_culture_score);
        const art = toNumber(form.art_special_score);
        if (culture === null || art === null) throw new Error("艺术类请填写综合分，或填写文化成绩与艺考成绩");
        usedScore = calcArtComposite(culture, art);
      }
      const ds = state.datasets[stream];
      if (!ds) throw new Error("艺术数据未加载");
      const userRank = scoreToRank(usedScore, ds.scoreMap, ds.topScore, ds.topRank);
      return { stream, userRank, usedScore };
    }

    if (score !== null) {
      const ds = state.datasets[stream];
      if (!ds) throw new Error(`${stream} 数据未加载`);
      const userRank = scoreToRank(score, ds.scoreMap, ds.topScore, ds.topRank);
      return { stream, userRank, usedScore: score };
    }

    throw new Error("请至少输入分数或位次");
  }

  function rankFromMockInput(form) {
    const stream = normalizeStream(form.stream);
    if (!["历史", "物理"].includes(stream)) throw new Error("联考换算仅支持 历史/物理");
    const ds = state.datasets[stream];
    if (!ds) throw new Error(`${stream} 数据未加载`);

    const mockKey = form.mock_exam_key;
    const table = state.mockTables[stream][mockKey];
    if (!table) throw new Error("所选联考数据不存在，请重新选择联考试卷");

    const mockRank = toInt(form.mock_rank);
    const mockScore = toNumber(form.mock_score);
    let sourceRank = null;
    if (mockRank && mockRank > 0) sourceRank = mockRank;
    else if (mockScore !== null) sourceRank = scoreToRank(mockScore, table.scoreMap, table.topScore, table.topRank);
    if (sourceRank === null) throw new Error("联考换算需输入联考成绩或联考位次");

    const targetRank = convertRankByPercentile(sourceRank, table.totalRank, ds.totalRank);
    const predictedScore = rankToScore(targetRank, ds.scoreMap);
    const userRank = scoreToRank(predictedScore, ds.scoreMap, ds.topScore, ds.topRank);

    return {
      stream,
      userRank,
      usedScore: predictedScore,
      conversion: {
        mock_exam_label: table.label,
        mock_rank: sourceRank,
        mock_total: table.totalRank,
        gaokao_rank: targetRank,
        gaokao_total: ds.totalRank,
        predicted_score: serializeScore(predictedScore)
      }
    };
  }

  function renderGuide() {
    return `<div class="prob-guide">${PROBABILITY_GUIDE.map((i) => `
      <div class="guide-item">
        <p>${Math.round(i.min * 100)}%+</p><strong>${i.label}</strong><span>${i.desc}</span>
      </div>`).join("")}
    </div>`;
  }

  function renderTierButtons(total, stats) {
    return `
      <div class="tier-row">
        <button type="button" class="tier-filter active" data-tier-filter="全部" aria-pressed="true">全部 ${total}</button>
        <button type="button" class="tier-filter crash" data-tier-filter="冲" aria-pressed="false">冲 ${stats.冲}</button>
        <button type="button" class="tier-filter steady" data-tier-filter="稳" aria-pressed="false">稳 ${stats.稳}</button>
        <button type="button" class="tier-filter safe" data-tier-filter="保" aria-pressed="false">保 ${stats.保}</button>
      </div>
      <p class="tier-filter-status">当前显示：全部（${total}条）</p>
    `;
  }

  function renderTable(rows) {
    if (!rows.length) return '<p class="empty">没有符合当前条件的结果，请调整筛选条件。</p>';
    const body = rows.map((row) => `
      <tr data-tier="${row.tier}">
        <td data-label="院校代码">${row.school_code}</td>
        <td data-label="院校名称">${row.school_name}</td>
        <td data-label="专业代码">${row.major_code}</td>
        <td data-label="专业名称">${row.major_name}</td>
        <td data-label="最低分">${row.min_score}</td>
        <td data-label="最低位次">${row.min_rank}</td>
        <td data-label="位次比">${row.rank_ratio.toFixed(4)}</td>
        <td data-label="分档"><span class="tier-pill ${row.tier === "保" ? "tier-safe" : row.tier === "稳" ? "tier-steady" : "tier-crash"}">${row.tier}</span></td>
        <td data-label="概率"><span class="prob-chip" title="${row.probability_desc}">${(row.probability * 100).toFixed(1)}%</span></td>
      </tr>
    `).join("");

    return `
      <div class="table-wrap">
        <table class="recommend-table">
          <thead>
            <tr>
              <th>院校代码</th><th>院校名称</th><th>专业代码</th><th>专业名称</th>
              <th>最低分</th><th>最低位次</th><th>位次比</th><th>分档</th><th>概率</th>
            </tr>
          </thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    `;
  }

  function bindTierFilter(root) {
    const buttons = root.querySelectorAll("[data-tier-filter]");
    const rows = root.querySelectorAll("tr[data-tier]");
    const status = root.querySelector(".tier-filter-status");
    const apply = (tier) => {
      let visible = 0;
      rows.forEach((row) => {
        const matched = tier === "全部" || row.dataset.tier === tier;
        row.classList.toggle("is-row-hidden", !matched);
        if (matched) visible += 1;
      });
      buttons.forEach((btn) => {
        const active = btn.dataset.tierFilter === tier;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-pressed", active ? "true" : "false");
      });
      if (status) status.textContent = `当前显示：${tier}（${visible}条）`;
    };
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => apply(btn.dataset.tierFilter));
    });
    if (buttons.length && rows.length) apply("全部");
  }

  function renderPredictResult(data, rec) {
    const el = document.getElementById("predict-result");
    el.className = "card";
    el.innerHTML = `
      <div class="card-head"><h2>结果总览</h2></div>
      <div class="kpi-grid">
        <div class="kpi"><p class="kpi-label">科类</p><p class="kpi-value">${data.stream}</p></div>
        <div class="kpi"><p class="kpi-label">换算位次</p><p class="kpi-value">${data.userRank}</p></div>
        <div class="kpi"><p class="kpi-label">参考分数</p><p class="kpi-value">${data.usedScore === null ? "-" : serializeScore(data.usedScore)}</p></div>
        <div class="kpi"><p class="kpi-label">模式</p><p class="kpi-value">智能推荐</p></div>
        <div class="kpi"><p class="kpi-label">概率评估</p><p class="kpi-value">自动计算</p></div>
      </div>
      ${renderTierButtons(rec.rows.length, rec.stats)}
      ${renderGuide()}
      <div class="card-head" style="margin-top:12px;"><h2>推荐列表</h2><p>鼠标悬停概率可查看解释。</p></div>
      ${renderTable(rec.rows)}
    `;
    bindTierFilter(el);
  }

  function renderMockResult(data, rec) {
    const el = document.getElementById("mock-result");
    el.className = "card";
    el.innerHTML = `
      <div class="card-head"><h2>换算结果</h2></div>
      <div class="kpi-grid">
        <div class="kpi"><p class="kpi-label">科类</p><p class="kpi-value">${data.stream}</p></div>
        <div class="kpi"><p class="kpi-label">联考位次</p><p class="kpi-value">${data.conversion.mock_rank}</p></div>
        <div class="kpi"><p class="kpi-label">等效高考分</p><p class="kpi-value">${serializeScore(data.usedScore)}</p></div>
        <div class="kpi"><p class="kpi-label">等效高考位次</p><p class="kpi-value">${data.userRank}</p></div>
        <div class="kpi"><p class="kpi-label">模式</p><p class="kpi-value">联考换算</p></div>
      </div>
      <p class="conversion-note">联考试卷：${data.conversion.mock_exam_label}<br>联考位次 ${data.conversion.mock_rank}/${data.conversion.mock_total} → 高考等效位次 ${data.conversion.gaokao_rank}/${data.conversion.gaokao_total} → 预测高考分 ${data.conversion.predicted_score}</p>
      ${renderTierButtons(rec.rows.length, rec.stats)}
      ${renderGuide()}
      <div class="card-head" style="margin-top:12px;"><h2>推荐列表</h2><p>已按换算后高考位次生成。</p></div>
      ${renderTable(rec.rows)}
    `;
    bindTierFilter(el);
  }

  function showError(containerId, message) {
    const el = document.getElementById(containerId);
    el.className = "card error-card";
    el.innerHTML = `<h2>输入错误</h2><p>${message}</p>`;
  }

  function extractMockMeta(filename) {
    const stem = filename.replace(/\.xlsx?$/i, "");
    const lower = stem.toLowerCase();
    let stream = "";
    if (stem.includes("历史") || lower.includes("hist") || lower.includes("history")) stream = "历史";
    else if (stem.includes("物理") || lower.includes("phys") || lower.includes("physics")) stream = "物理";
    if (!stream) return null;
    let label = stem
      .replace(/历史类?|物理类?/g, "")
      .replace(/history|hist|physics|phys/gi, "")
      .replace(/联考|一分一段表|分数段/g, "")
      .replace(/[_\-\s]+/g, " ")
      .trim();
    if (!label) label = "联考";
    return { stream, label };
  }

  function updateMockSelect() {
    const stream = document.getElementById("mock-stream").value;
    const select = document.getElementById("mock-exam-select");
    const options = Object.entries(state.mockTables[stream] ?? {});
    select.innerHTML = options.length
      ? options.map(([key, v]) => `<option value="${key}">${v.label}</option>`).join("")
      : '<option value="">未识别到联考数据文件</option>';
    select.disabled = options.length === 0;
  }

  async function loadAllData() {
    const getFile = (id) => document.getElementById(id).files[0] ?? null;
    const getFiles = (id) => [...document.getElementById(id).files];
    const histAdmissionFile = getFile("file-hist-admission");
    const histScoreFile = getFile("file-hist-score");
    const physAdmissionFile = getFile("file-phys-admission");
    const physScoreFile = getFile("file-phys-score");
    const artAdmissionFile = getFile("file-art-admission");
    const mockFiles = getFiles("file-mock");

    if (!histAdmissionFile || !histScoreFile || !physAdmissionFile || !physScoreFile) {
      throw new Error("请至少上传历史/物理的录取表与一分一段表");
    }

    const [histAdmissionWb, histScoreWb, physAdmissionWb, physScoreWb] = await Promise.all([
      readWorkbook(histAdmissionFile),
      readWorkbook(histScoreFile),
      readWorkbook(physAdmissionFile),
      readWorkbook(physScoreFile)
    ]);

    const histAdmission = parseAdmissionRows(firstSheetRows(histAdmissionWb));
    const histScore = parseScoreTableRows(firstSheetRows(histScoreWb));
    state.datasets.历史 = buildDataset(histAdmission, histScore);

    const physAdmission = parseAdmissionRows(firstSheetRows(physAdmissionWb));
    const physScore = parseScoreTableRows(firstSheetRows(physScoreWb));
    state.datasets.物理 = buildDataset(physAdmission, physScore);

    if (artAdmissionFile) {
      const artWb = await readWorkbook(artAdmissionFile);
      const artAdmission = parseAdmissionRows(firstSheetRows(artWb));
      const artScore = buildRankMapFromAdmission(artAdmission);
      state.datasets.艺术 = buildDataset(artAdmission, artScore);
    } else {
      state.datasets.艺术 = null;
    }

    state.mockTables.历史 = {};
    state.mockTables.物理 = {};
    for (const file of mockFiles) {
      const meta = extractMockMeta(file.name);
      if (!meta) continue;
      const wb = await readWorkbook(file);
      const scorePack = parseScoreTableRows(firstSheetRows(wb));
      const totalRank = Math.max(...[...scorePack.scoreMap.values()].map((v) => Number(v)));
      const key = `${meta.stream}:${file.name}`;
      state.mockTables[meta.stream][key] = {
        key,
        label: meta.label,
        scoreMap: scorePack.scoreMap,
        topScore: scorePack.topScore,
        topRank: scorePack.topRank,
        totalRank
      };
    }
  }

  async function loadAllDataFromDefaults() {
    const REQUIRED_PATHS = {
      histAdmission: "./hist.xlsx",
      histScore: "./历史类含加分一分段表.xlsx",
      physAdmission: "./phys.xlsx",
      physScore: "./物理类含加分一分段表.xlsx"
    };
    const OPTIONAL_ART_PATHS = ["./艺术类.xlsx", "./艺术批.xlsx"];
    const OPTIONAL_MOCK_PATHS = [
      "./历史类联考一分一段表.xlsx",
      "./历史联考一分一段表.xlsx",
      "./hist_mock.xlsx",
      "./物理类联考一分一段表.xlsx",
      "./物理联考一分一段表.xlsx",
      "./phys_mock.xlsx",
      "./mock_exams/历史_一诊.xlsx",
      "./mock_exams/历史_二诊.xlsx",
      "./mock_exams/物理_一诊.xlsx",
      "./mock_exams/物理_二诊.xlsx",
      "./mock_exams/历史_康德一诊.xlsx",
      "./mock_exams/历史_九龙坡一诊.xlsx",
      "./mock_exams/物理_康德一诊.xlsx",
      "./mock_exams/物理_九龙坡一诊.xlsx"
    ];

    const [histAdmissionWb, histScoreWb, physAdmissionWb, physScoreWb] = await Promise.all([
      readWorkbookFromPath(REQUIRED_PATHS.histAdmission),
      readWorkbookFromPath(REQUIRED_PATHS.histScore),
      readWorkbookFromPath(REQUIRED_PATHS.physAdmission),
      readWorkbookFromPath(REQUIRED_PATHS.physScore)
    ]);

    const histAdmission = parseAdmissionRows(firstSheetRows(histAdmissionWb));
    const histScore = parseScoreTableRows(firstSheetRows(histScoreWb));
    state.datasets.历史 = buildDataset(histAdmission, histScore);

    const physAdmission = parseAdmissionRows(firstSheetRows(physAdmissionWb));
    const physScore = parseScoreTableRows(firstSheetRows(physScoreWb));
    state.datasets.物理 = buildDataset(physAdmission, physScore);

    state.datasets.艺术 = null;
    for (const artPath of OPTIONAL_ART_PATHS) {
      try {
        const artWb = await readWorkbookFromPath(artPath);
        const artAdmission = parseAdmissionRows(firstSheetRows(artWb));
        const artScore = buildRankMapFromAdmission(artAdmission);
        state.datasets.艺术 = buildDataset(artAdmission, artScore);
        break;
      } catch (err) {
        continue;
      }
    }

    state.mockTables.历史 = {};
    state.mockTables.物理 = {};
    for (const path of OPTIONAL_MOCK_PATHS) {
      try {
        const wb = await readWorkbookFromPath(path);
        const fileName = path.split("/").pop() || path;
        const meta = extractMockMeta(fileName);
        if (!meta) continue;
        const scorePack = parseScoreTableRows(firstSheetRows(wb));
        const totalRank = Math.max(...[...scorePack.scoreMap.values()].map((v) => Number(v)));
        const key = `${meta.stream}:${fileName}`;
        state.mockTables[meta.stream][key] = {
          key,
          label: meta.label,
          scoreMap: scorePack.scoreMap,
          topScore: scorePack.topScore,
          topRank: scorePack.topRank,
          totalRank
        };
      } catch (err) {
        continue;
      }
    }
  }

  function setupRouter() {
    const views = {
      predict: document.getElementById("view-predict"),
      mock: document.getElementById("view-mock"),
      about: document.getElementById("view-about")
    };
    const links = [...document.querySelectorAll(".tabs-link[data-route]")];
    const render = () => {
      const route = (location.hash.replace(/^#\//, "") || "predict").toLowerCase();
      const key = ["predict", "mock", "about"].includes(route) ? route : "predict";
      Object.entries(views).forEach(([name, el]) => {
        el.classList.toggle("is-hidden", name !== key);
      });
      links.forEach((a) => {
        a.classList.toggle("active", a.dataset.route === key);
      });
    };
    window.addEventListener("hashchange", render);
    render();
  }

  function setupArtForm() {
    const stream = document.getElementById("predict-stream");
    const scoreLabel = document.getElementById("predict-score-label");
    const scoreInput = document.querySelector("#predict-form input[name='score']");
    const culture = document.getElementById("art-culture");
    const special = document.getElementById("art-special");
    const composite = document.getElementById("art-composite");
    const artOnly = document.querySelectorAll(".art-only");
    const normalOnly = document.querySelectorAll(".normal-only");

    const calc = () => {
      const c = toNumber(culture.value);
      const s = toNumber(special.value);
      if (c === null || s === null) {
        composite.value = "";
        return;
      }
      const v = calcArtComposite(c, s);
      composite.value = v.toFixed(2);
      scoreInput.value = v.toFixed(2);
    };

    const refresh = () => {
      const isArt = stream.value === "艺术";
      artOnly.forEach((el) => {
        el.classList.toggle("is-hidden", !isArt);
      });
      normalOnly.forEach((el) => {
        el.classList.toggle("is-hidden", isArt);
      });
      scoreLabel.textContent = isArt ? "综合分（可选，可由下方自动换算）" : "分数（可选）";
      if (isArt) calc();
    };
    stream.addEventListener("change", refresh);
    culture.addEventListener("input", calc);
    special.addEventListener("input", calc);
    refresh();
  }

  function setupHandlers() {
    document.getElementById("btn-load-data").addEventListener("click", async () => {
      const status = document.getElementById("data-status");
      status.textContent = "状态：正在加载...";
      try {
        await loadAllData();
        updateMockSelect();
        const artLoaded = state.datasets.艺术 ? "，艺术已加载" : "，艺术未加载";
        const mockCount = Object.keys(state.mockTables.历史).length + Object.keys(state.mockTables.物理).length;
        status.textContent = `状态：加载完成（历史/物理已就绪${artLoaded}，联考文件 ${mockCount} 个）`;
      } catch (err) {
        status.textContent = `状态：加载失败（${err.message}）`;
      }
    });

    document.getElementById("btn-auto-load").addEventListener("click", async () => {
      const status = document.getElementById("data-status");
      status.textContent = "状态：正在自动拉取默认 XLS...";
      try {
        await loadAllDataFromDefaults();
        updateMockSelect();
        const artLoaded = state.datasets.艺术 ? "，艺术已加载" : "，艺术未加载";
        const mockCount = Object.keys(state.mockTables.历史).length + Object.keys(state.mockTables.物理).length;
        status.textContent = `状态：自动加载完成（历史/物理已就绪${artLoaded}，联考文件 ${mockCount} 个）`;
      } catch (err) {
        status.textContent = `状态：自动加载失败（${err.message}）`;
      }
    });

    document.getElementById("mock-stream").addEventListener("change", updateMockSelect);

    document.getElementById("predict-form").addEventListener("submit", (e) => {
      e.preventDefault();
      try {
        const form = Object.fromEntries(new FormData(e.currentTarget).entries());
        const data = rankFromPredictInput(form);
        const rec = recommend(data.stream, data.userRank, form);
        renderPredictResult(data, rec);
      } catch (err) {
        showError("predict-result", err.message);
      }
    });

    document.getElementById("mock-form").addEventListener("submit", (e) => {
      e.preventDefault();
      try {
        const form = Object.fromEntries(new FormData(e.currentTarget).entries());
        const data = rankFromMockInput(form);
        const rec = recommend(data.stream, data.userRank, form);
        renderMockResult(data, rec);
      } catch (err) {
        showError("mock-result", err.message);
      }
    });
  }

  function init() {
    setupRouter();
    setupArtForm();
    setupHandlers();
    updateMockSelect();
  }

  init();
})();
