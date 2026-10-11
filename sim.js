/*
 * In-browser port of fantasy_sim.py (auction + season Monte Carlo).
 * Runs as a Web Worker on the public site; also loads in Node for testing.
 * Function names mirror the Python ones; keep them in step when the model changes.
 * Inputs come from web/data.json (see export_web_data.py).
 */
"use strict";

const FGM = 0, FGA = 1, FTM = 2, FTA = 3, TPM = 4, PTS = 5, REB = 6, AST = 7, STL = 8, BLK = 9, TOV = 10, NC = 11;
const CATS = ["FG%", "FT%", "3PTM", "PTS", "REB", "AST", "ST", "BLK", "TO"];
const NK = CATS.length;
const WEIGHTS = {
  balanced: {},
  punt_ft: {"FT%": 0},
  punt_to: {TO: 0},
  punt_fg_to: {"FG%": 0, TO: 0},
  punt_ast_to: {AST: 0, TO: 0},
};
const STYLES = ["spread", "stars"];
const STARS_TOP_N = 12, STARS_UP = 1.20, STARS_DOWN = 0.85;

// ---------------- random numbers (seedable, so a seed reproduces a run) ----------------
function makeRng(...seeds) {
  let h = 1779033703;
  const mix = x => { h = Math.imul(h ^ (x >>> 0), 3432918353); h = (h << 13) | (h >>> 19); return h; };
  for (const s of seeds) { mix(s); mix(Math.floor(s / 4294967296)); }
  const out = () => { h = Math.imul(h ^ (h >>> 16), 2246822507); h = Math.imul(h ^ (h >>> 13), 3266489909); return (h ^= h >>> 16) >>> 0; };
  let a = out(), b = out(), c = out(), d = out(), spare = null;
  const random = () => { // sfc32
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b) | 0; a = b ^ (b >>> 9); b = (c + (c << 3)) | 0; c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0; t = (t + d) | 0; c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let i = 0; i < 12; i++) random();
  return {
    random,
    int: n => Math.floor(random() * n),
    normal() {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0; while (u === 0) u = random();
      const r = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * random();
      spare = r * Math.sin(th); return r * Math.cos(th);
    },
    binomial(n, p) { let k = 0; for (let i = 0; i < n; i++) if (random() < p) k++; return k; },
    permutation(n) {
      const a = Array.from({length: n}, (_, i) => i);
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
      return a;
    },
  };
}

function weightVec(name) {
  const w = new Float64Array(NK).fill(1);
  for (const [k, v] of Object.entries(WEIGHTS[name])) w[CATS.indexOf(k)] = v;
  return w;
}

const argsortDesc = v => Array.from(v, (_, i) => i).sort((a, b) => v[b] - v[a]);

// ---------------- values ----------------
/** z-score each category over the draftable pool, weight, convert to auction $. seasonTot: P rows of NC. */
function dollarValues(seasonTot, w, L) {
  const P = seasonTot.length, nd = Math.min(L.teams * L.roster, P);
  function zs(sel) {
    let fgm = 0, fga = 0, ftm = 0, fta = 0;
    for (let i = 0; i < P; i++) if (sel[i]) { const r = seasonTot[i]; fgm += r[FGM]; fga += r[FGA]; ftm += r[FTM]; fta += r[FTA]; }
    const fgp = fgm / fga, ftp = ftm / fta;
    const x = seasonTot.map(r => [r[FGM] - fgp * r[FGA], r[FTM] - ftp * r[FTA], r[TPM], r[PTS], r[REB], r[AST], r[STL], r[BLK], -r[TOV]]);
    const mean = new Float64Array(NK), sd = new Float64Array(NK);
    let n = 0;
    for (let i = 0; i < P; i++) if (sel[i]) { n++; for (let j = 0; j < NK; j++) mean[j] += x[i][j]; }
    for (let j = 0; j < NK; j++) mean[j] /= n;
    for (let i = 0; i < P; i++) if (sel[i]) for (let j = 0; j < NK; j++) sd[j] += (x[i][j] - mean[j]) ** 2;
    for (let j = 0; j < NK; j++) sd[j] = Math.sqrt(sd[j] / n);
    return x.map(r => r.map((v, j) => (v - mean[j]) / (sd[j] + 1e-9)));
  }
  const dot = z => z.map(r => r.reduce((a, v, j) => a + v * w[j], 0));
  let sel = new Array(P).fill(true);
  for (let it = 0; it < 2; it++) {
    const tot = dot(zs(sel));
    sel = new Array(P).fill(false);
    argsortDesc(tot).slice(0, nd).forEach(i => sel[i] = true);
  }
  const z = zs(sel), tot = dot(z);
  const order = argsortDesc(tot), rank = new Int32Array(P);
  order.forEach((p, r) => rank[p] = r);
  const repl = tot[order[nd - 1]];
  const vr = tot.map(t => Math.max(0, t - repl)), vsum = vr.reduce((a, b) => a + b, 0);
  const dollars = vr.map((v, i) => rank[i] < nd ? 1 + v / Math.max(vsum, 1e-9) * (L.teams * L.budget - nd) : 0);
  return {dollars, z, tot};
}

function styleAdjust(vals, style) {
  if (style === "spread") return vals.slice();
  const out = vals.map(v => v * STARS_DOWN);
  argsortDesc(vals).slice(0, STARS_TOP_N).forEach(i => out[i] = vals[i] * STARS_UP);
  return out;
}

// ---------------- simulation core ----------------
/** samples[((p*K + k)*W + w)*NC + c] = player p's stat totals in week w of season replicate k. */
function buildSamples(pool, teamWeeks, K, W, rng) {
  const out = new Float32Array(pool.length * K * W * NC);
  pool.forEach((pl, i) => {
    const g = teamWeeks[pl.team], L = pl.logs, n = L.length / NC;
    for (let k = 0; k < K; k++) for (let w = 0; w < W; w++) {
      const played = rng.binomial(g[w], pl.avail), base = ((i * K + k) * W + w) * NC;
      for (let j = 0; j < played; j++) {
        const r = rng.int(n) * NC;
        for (let c = 0; c < NC; c++) out[base + c] += L[r + c];
      }
    }
  });
  return out;
}

function runAuction(anchor, byAnchor, myval, rng, sigma, L) {
  const P = anchor.length, T = L.teams;
  const owner = new Int16Array(P).fill(-1), price = new Float64Array(P);
  const money = new Float64Array(T).fill(L.budget), spots = new Int32Array(T).fill(L.roster);
  const left = new Uint8Array(P).fill(1);
  let nleft = P, spotsLeft = T * L.roster, nom = rng.int(T);
  const topLeft = n => { const r = []; for (const p of byAnchor) { if (left[p]) { r.push(p); if (r.length === n) break; } } return r; };
  const vals = new Float64Array(T), maxbid = new Float64Array(T);
  while (spotsLeft > 0 && nleft > 0) {
    if (spots[nom] === 0) { nom = (nom + 1) % T; continue; }
    let p;
    if (nom === 0) { // my nomination
      if (money[0] - (spots[0] - 1) <= 2) { // $1 mode: grab my best remaining value
        let best = -Infinity;
        for (let i = 0; i < P; i++) if (left[i] && myval[i] > best) { best = myval[i]; p = i; }
      } else { // bait or drain
        const top = topLeft(30);
        let bd = -Infinity, bp = top[0];
        for (const q of top) { const d = anchor[q] - myval[q]; if (d > bd) { bd = d; bp = q; } }
        p = bd > 3 ? bp : top[0];
      }
    } else {
      const top = topLeft(8);
      p = top[rng.int(top.length)];
    }
    for (let t = 0; t < T; t++) {
      maxbid[t] = spots[t] > 0 ? money[t] - (spots[t] - 1) : -1;
      vals[t] = anchor[p] * Math.exp(rng.normal() * sigma);
    }
    vals[0] = myval[p];
    vals[nom] = Math.max(vals[nom], 1);
    let w = 0;
    for (let t = 0; t < T; t++) {
      vals[t] = Math.min(vals[t], maxbid[t]);
      if (spots[t] === 0) vals[t] = -1;
      vals[t] += rng.random() * 1e-6;
      if (vals[t] > vals[w]) w = t;
    }
    let second = -Infinity;
    for (let t = 0; t < T; t++) if (t !== w && vals[t] > second) second = vals[t];
    let pr = second >= 1 ? Math.floor(second) + 1 : 1;
    pr = Math.max(1, Math.min(pr, Math.max(1, Math.floor(vals[w])), Math.trunc(maxbid[w])));
    owner[p] = w; price[p] = pr;
    money[w] -= pr; spots[w]--; spotsLeft--;
    left[p] = 0; nleft--;
    nom = (nom + 1) % T;
  }
  return {owner, price};
}

function roundRobin(n, weeks) {
  let t = Array.from({length: n}, (_, i) => i);
  const rounds = [];
  for (let r = 0; r < n - 1; r++) {
    rounds.push(Array.from({length: n / 2}, (_, i) => [t[i], t[n - 1 - i]]));
    t = [t[0], t[n - 1], ...t.slice(1, n - 1)];
  }
  return Array.from({length: weeks}, (_, w) => rounds[w % (n - 1)]);
}

function catValues(t, o) { // t: NC stats at offset o -> NK values, TO negated so higher = better
  return [t[o + FGM] / Math.max(t[o + FGA], 1), t[o + FTM] / Math.max(t[o + FTA], 1), t[o + TPM], t[o + PTS],
          t[o + REB], t[o + AST], t[o + STL], t[o + BLK], -t[o + TOV]];
}

function playSeason(owner, samples, K, W, k, nreg, rr, rng, L) {
  const T = L.teams;
  const tot = new Float64Array(T * W * NC);
  for (let p = 0; p < owner.length; p++) {
    const t = owner[p]; if (t < 0) continue;
    const src = (p * K + k) * W * NC, dst = t * W * NC;
    for (let j = 0; j < W * NC; j++) tot[dst + j] += samples[src + j];
  }
  const cv = []; // cv[t][w] = NK values
  for (let t = 0; t < T; t++) { cv.push([]); for (let w = 0; w < W; w++) cv[t].push(catValues(tot, (t * W + w) * NC)); }
  const cmp = (a, b, w) => { let wa = 0, wb = 0; const A = cv[a][w], B = cv[b][w]; for (let j = 0; j < NK; j++) { if (A[j] > B[j]) wa++; else if (A[j] < B[j]) wb++; } return [wa, wb]; };
  const perm = rng.permutation(T);
  const W_ = new Float64Array(T), L_ = new Float64Array(T), T_ = new Float64Array(T);
  const myCats = new Float64Array(NK); let myN = 0;
  for (let w = 0; w < nreg; w++) for (const [sa, sb] of rr[w]) {
    const a = perm[sa], b = perm[sb], [wa, wb] = cmp(a, b, w), ties = NK - wa - wb;
    W_[a] += wa; L_[a] += wb; T_[a] += ties; W_[b] += wb; L_[b] += wa; T_[b] += ties;
    if (a === 0 || b === 0) {
      const me = cv[0][w], them = cv[a === 0 ? b : a][w];
      for (let j = 0; j < NK; j++) if (me[j] > them[j]) myCats[j]++;
      myN++;
    }
  }
  const pct = Array.from({length: T}, (_, t) => (W_[t] + 0.5 * T_[t]) / Math.max(W_[t] + L_[t] + T_[t], 1) + rng.random() * 1e-9);
  const order = argsortDesc(pct), rankOf = new Int32Array(T);
  order.forEach((t, i) => rankOf[t] = i);
  const seeds = order.slice(0, L.playoff_teams);
  const play = (a, b, w) => { const [wa, wb] = cmp(a, b, w); return wa !== wb ? (wa > wb ? a : b) : (rankOf[a] < rankOf[b] ? a : b); };
  const r1 = [play(seeds[2], seeds[5], nreg), play(seeds[3], seeds[4], nreg)].sort((x, y) => rankOf[x] - rankOf[y]);
  const semis = [play(seeds[0], r1[1], nreg + 1), play(seeds[1], r1[0], nreg + 1)];
  const champ = play(semis[0], semis[1], nreg + 2);
  return {cat_pct: (W_[0] + 0.5 * T_[0]) / Math.max(W_[0] + L_[0] + T_[0], 1), rank: rankOf[0] + 1,
          playoffs: rankOf[0] < L.playoff_teams, title: champ === 0, cats: myCats.map(c => c / Math.max(myN, 1))};
}

// ---------------- helpers for the report ----------------
const mean = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);
const r2 = v => v == null || !isFinite(v) ? null : Math.round(v * 100) / 100;
function percentile(sorted, q) { // numpy's default linear interpolation
  if (!sorted.length) return null;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
const pad = n => String(n).padStart(2, "0");

// ---------------- main ----------------
/**
 * opts: {auctions, replicates, sigma, pool, seed, export, label}
 * log(line, buildsDone) is called as the run progresses.
 */
function runSim(data, opts, log = () => {}) {
  const t0 = Date.now();
  const L = data.league, A = opts.auctions ?? 1000, K = opts.replicates ?? 200, sigma = opts.sigma ?? 0.3,
        seed = opts.seed ?? 7, poolN = opts.pool ?? 320;
  const rng = makeRng(seed);
  const W = data.nreg + 3, nreg = data.nreg, tw = data.team_weeks;
  let pool = data.players.map(p => {
    const logs = new Float32Array(p.g.length);
    for (let i = 0; i < p.g.length; i++) logs[i] = p.g[i] * p.scale;
    return {...p, logs};
  });
  log(`${pool.length} eligible players, ${nreg} regular-season weeks + 3 playoff weeks`, 0);

  // season projections -> keep the most relevant players for any strategy
  const season = pl => {
    const n = pl.logs.length / NC, r = new Float64Array(NC);
    for (let i = 0; i < pl.logs.length; i++) r[i % NC] += pl.logs[i];
    const games = tw[pl.team].reduce((a, b) => a + b, 0) * pl.avail;
    return {tot: Array.from(r, v => v / n * games), games};
  };
  pool.forEach(pl => Object.assign(pl, season(pl)));
  const bestRank = new Array(pool.length).fill(1e9);
  for (const name in WEIGHTS) {
    const {tot} = dollarValues(pool.map(p => p.tot), weightVec(name), L);
    argsortDesc(tot).forEach((p, r) => bestRank[p] = Math.min(bestRank[p], r));
  }
  pool = pool.filter((_, i) => bestRank[i] < poolN);
  const seasonTot = pool.map(p => p.tot), P = pool.length;
  log(`Draft pool: ${P} players`, 0);

  // anchor prices = what the room will roughly pay
  const base = dollarValues(seasonTot, weightVec("balanced"), L).dollars;
  let anchor = base.slice(), anchorSrc = "our z-score values (no Yahoo prices loaded)";
  const hits = pool.filter(p => data.yahoo && data.yahoo[p.key] != null).length;
  if (hits >= 50) {
    anchor = pool.map((p, i) => data.yahoo[p.key] ?? Math.min(base[i] * 0.5, 3));
    anchorSrc = "auction_values.csv";
  }
  const byAnchor = argsortDesc(anchor);
  log(`Opponents anchor on: ${anchorSrc}`, 0);

  log(`Simulating ${K} seasons of weekly stats per player...`, 0);
  const samples = buildSamples(pool, tw, K, W, rng);
  const rr = roundRobin(L.teams, nreg);

  const strategies = Object.keys(WEIGHTS).flatMap(n => STYLES.map(s => [n, s]));
  const nS = strategies.length;
  const prices = pool.map(() => new Float32Array(A * nS)), nPrice = new Int32Array(P);
  const results = [], per = {};
  strategies.forEach(([n, s], si) => {
    const myval = styleAdjust(dollarValues(seasonTot, weightVec(n), L).dollars, s);
    const bought = new Int32Array(P), paid = new Float64Array(P), recs = [];
    for (let i = 0; i < A; i++) {
      const arng = makeRng(seed, i); // same rooms/seasons for every strategy
      const {owner, price} = runAuction(anchor, byAnchor, myval, arng, sigma, L);
      recs.push(playSeason(owner, samples, K, W, i % K, nreg, rr, arng, L));
      for (let p = 0; p < P; p++) {
        if (owner[p] >= 0) prices[p][nPrice[p]++] = price[p];
        if (owner[p] === 0) { bought[p]++; paid[p] += price[p]; }
      }
    }
    const finish = new Array(L.teams).fill(0);
    recs.forEach(r => finish[r.rank - 1]++);
    const row = {strategy: n, style: s,
      "title_%": 100 * mean(recs.map(r => +r.title)), "playoffs_%": 100 * mean(recs.map(r => +r.playoffs)),
      avg_finish: mean(recs.map(r => r.rank)), "category_win_%": 100 * mean(recs.map(r => r.cat_pct))};
    CATS.forEach((c, j) => row[`win_${c}_%`] = 100 * mean(recs.map(r => r.cats[j])));
    results.push(row);
    per[n + ":" + s] = {myval, bought, paid, finish};
    log(`  ${n.padEnd(12)} ${s.padEnd(7)} title ${row["title_%"].toFixed(1).padStart(5)}%  playoffs ${row["playoffs_%"].toFixed(1).padStart(5)}%  ` +
        `avg finish ${row.avg_finish.toFixed(1)}   (${Math.round((Date.now() - t0) / 1000)}s)`, si + 1);
  });

  results.sort((a, b) => b["title_%"] - a["title_%"] || b["playoffs_%"] - a["playoffs_%"]);
  let best = [results[0].strategy, results[0].style];
  if (opts.export) {
    const [n, s = "spread"] = opts.export.split(":");
    if (per[n + ":" + s]) best = [n, s];
  }
  const B = per[best.join(":")];
  const zb = dollarValues(seasonTot, weightVec(best[0]), L).z;

  const kind = i => {
    const pos = String(pool[i].pos || "").toUpperCase();
    if (anchor[i] >= 8 && B.myval[i] < 0.6 * anchor[i]) return "bait";
    if (pos.includes("C")) return "big";
    if (pos && !pos.includes("F")) return "guard";
    if (pos) return "wing";
    const r = seasonTot[i];
    return r[BLK] + r[REB] / 6 > r[AST] ? "big" : "guard";
  };
  let rep = pool.map((pl, i) => {
    const sold = Array.from(prices[i].subarray(0, nPrice[i])).sort((a, b) => a - b);
    const row = {name: pl.name, team: pl.team, pos: pl.pos,
      max_bid: Math.round(B.myval[i]), room_value: Math.round(anchor[i] * 10) / 10,
      sells_median: r2(percentile(sold, 0.5)), sells_p10: r2(percentile(sold, 0.1)), sells_p90: r2(percentile(sold, 0.9)),
      "you_bought_%": Math.round(1000 * B.bought[i] / A) / 10,
      avg_you_paid: B.bought[i] ? Math.round(10 * B.paid[i] / B.bought[i]) / 10 : null,
      games_proj: Math.round(pl.games)};
    CATS.forEach((c, j) => row[`z_${c}`] = r2(zb[i][j]));
    row.type = kind(i);
    return row;
  });
  rep.sort((a, b) => b.max_bid - a.max_bid);
  const targets = rep.filter(r => r.type !== "bait" && r.max_bid >= 3).slice(0, 45);
  const fliers = rep.filter(r => r.type !== "bait" && r.max_bid >= 1 && r.max_bid <= 2 && r.room_value <= 2).slice(0, 15);
  const bait = rep.filter(r => r.type === "bait").sort((a, b) => b.room_value - a.room_value).slice(0, 15);
  rep.forEach(r => r.list = "");
  targets.forEach(r => r.list = "target"); fliers.forEach(r => r.list = "flier"); bait.forEach(r => r.list = "bait");
  const lines = [...targets.map(r => `${r.name}, ${r.pos}, ${r.type}, ${r.max_bid}`),
                 ...fliers.map(r => `${r.name}, ${r.pos}, flier, ${Math.max(1, r.max_bid)}`),
                 ...bait.map(r => `${r.name}, ${r.pos}, bait, 0`)];

  const now = new Date();
  const id = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  log(`Done in ${Math.round((Date.now() - t0) / 1000)}s. Best: ${results[0].strategy} / ${results[0].style}`, nS);
  return {
    id, label: opts.label || "",
    generated: `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}`,
    demo: false,
    data_exported: data.exported,
    settings: {auctions: A, replicates: K, sigma, seed, pool: P, teams: L.teams, budget: L.budget, roster: L.roster,
               reg_weeks: nreg, playoff_teams: L.playoff_teams},
    anchor_source: anchorSrc,
    cats: CATS,
    export: {strategy: best[0], style: best[1]},
    strategies: results.map(r => {
      const o = {};
      for (const [k, v] of Object.entries(r)) o[k] = typeof v === "number" ? r2(v) : v;
      return Object.assign(o, {punts: Object.keys(WEIGHTS[r.strategy]), finish_dist: per[r.strategy + ":" + r.style].finish});
    }),
    players: rep,
    playoff_weeks: [0, 1, 2].map(i => ({label: `Wk ${nreg + i + 1}`, dates: data.week_labels[nreg + i]})),
    playoff_schedule: Object.keys(tw).sort().map(t => ({team: t, games: tw[t].slice(nreg, nreg + 3)})),
    draft_targets: lines.join("\n"),
  };
}

// ---------------- worker / node glue ----------------
if (typeof module !== "undefined") {
  module.exports = {runSim, makeRng, dollarValues, runAuction, playSeason, roundRobin};
} else if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  let cached = null;
  self.onmessage = async e => {
    const {opts, dataUrl} = e.data;
    try {
      if (!cached) cached = await (await fetch(dataUrl)).json();
      const run = runSim(cached, opts, (line, done) => self.postMessage({type: "log", line, done}));
      self.postMessage({type: "done", run});
    } catch (err) {
      self.postMessage({type: "error", message: String(err && err.message || err)});
    }
  };
}
