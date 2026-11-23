import ForceGraph3D from '3d-force-graph';
import * as THREE from 'three';
import { LAYER_COLOR, LAYER_NAME, STAGES } from './content.js';

const BG = '#121418';
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
const RGB = Object.fromEntries(Object.entries(LAYER_COLOR).map(([k, v]) => [k, hex(v)]));

// Cámara por etapa: a qué capa mira, a qué distancia y desde qué ángulo (rad)
const TOUR = [
  { look: null, d: 430, ang: 0.0 },
  { look: 'ORN', d: 430, ang: -0.55 },
  { look: 'PN', d: 420, ang: 0.35 },
  { look: 'KC', d: 430, ang: 0.65 },
  { look: 'MBON', d: 400, ang: -0.3 },
];
// Conexiones que se animan con partículas en cada etapa
const FLOW = {
  1: [['ORN', 'PN'], ['ORN', 'LN']],
  2: [['ORN', 'PN'], ['PN', 'LN'], ['LN', 'PN']],
  3: [['PN', 'KC']],
  4: [['KC', 'MBON']],
};

const VERT = `
attribute float aSize; attribute float aAlpha; attribute float aPhase; attribute float aGlow; attribute vec3 aColor;
varying vec3 vC; varying float vA; varying float vG; uniform float uTime; uniform float uScale;
void main() {
  vC = aColor; vA = aAlpha; vG = aGlow;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float tw = 1.0 + 0.06 * sin(uTime * 2.0 + aPhase * 6.2831) * step(0.4, aAlpha);
  gl_PointSize = aSize * tw * uScale * 2.2 / -mv.z; // sprite: núcleo + halo muy suave
  gl_Position = projectionMatrix * mv;
}`;
const FRAG = `
varying vec3 vC; varying float vA; varying float vG;
void main() {
  float r = length(gl_PointCoord - vec2(0.5)) * 2.0;
  float core = smoothstep(0.34, 0.16, r);
  float halo = pow(max(0.0, 1.0 - r), 2.4) * (0.05 + 0.85 * vG);   // las neuronas activas tienen halo, las demás no
  float ring = smoothstep(0.62, 0.55, r) * smoothstep(0.42, 0.5, r) * vG * 0.55; // anillo fino alrededor de las activas
  float a = core + halo + ring;
  gl_FragColor = vec4(vC * a * vA, a * vA);
}`;

/** Visor 3D: neuronas brillantes en su posición real, pulsos de señal y cámara guiada por etapas. */
export class ViewerEngine {
  constructor(el, graph, acts) {
    this.el = el;
    this.graph = graph;
    this.acts = acts;
    const { layer, layer_idx, type, side, pos_um, n } = graph.nodes;
    this.n = n;
    this.layerOf = layer;
    this.layerNodes = {};
    for (let i = 0; i < n; i++) (this.layerNodes[layer[i]] ??= [])[layer_idx[i]] = i;
    this.visible = Object.fromEntries(graph.layers.map((l) => [l, true]));
    this.level = new Float32Array(n);
    this.target = new Float32Array(n);
    this.rate = Float32Array.from({ length: n }, () => 0.06 + 0.14 * Math.random());
    this.t0 = new Float32Array(n);      // instante en que cada neurona empieza a encenderse (onda)
    this.pend = new Uint8Array(n);      // ¿está esperando su turno?
    this.flash = new Float32Array(n);   // destello al activarse
    this._userTouched = false;
    this.idle = true;
    this.topN = 1;
    this.stage = 0;
    this.example = 0;
    this.playing = false;
    this.autoCamera = true;
    this._timers = [];
    this._pset = new Set();
    this._dirty = true;

    // centroides por capa (para dirigir la cámara)
    this.centroid = { ALL: [0, 0, 0] };
    for (const l of graph.layers) {
      const idx = this.layerNodes[l];
      this.centroid[l] = [0, 1, 2].map((a) => idx.reduce((s, i) => s + pos_um[i][a], 0) / idx.length);
    }

    this._nodes = Array.from({ length: n }, (_, i) => ({
      id: i, x: pos_um[i][0], y: pos_um[i][1], z: pos_um[i][2],
      fx: pos_um[i][0], fy: pos_um[i][1], fz: pos_um[i][2],
    }));
    this._sortEdges();

    this.fg = ForceGraph3D({ controlType: 'orbit' })(el)
      .backgroundColor(BG)
      .showNavInfo(false)
      .nodeVisibility(() => false) // los puntos los dibuja `this.points`
      .enablePointerInteraction(false)
      .linkVisibility((l) => this.visible[layer[l.source.id ?? l.source]] && this.visible[layer[l.target.id ?? l.target]])
      .linkColor((l) => this._linkColor(l))
      .linkWidth(0).linkOpacity(1)
      .linkDirectionalParticles(0)
      .cooldownTicks(0).warmupTicks(0)
      .enableNodeDrag(false)
      .width(el.clientWidth).height(el.clientHeight);

    this._buildPoints(pos_um);
    this._setLinks();

    this.fg.cameraPosition({ x: 0, y: 60, z: 430 });
    const c = this.fg.controls();
    c.autoRotate = true; c.autoRotateSpeed = 0.55; c.enableDamping = true;
    c.addEventListener('start', () => { c.autoRotate = false; this._userTouched = true; }); // el usuario toma el control

    this._ro = new ResizeObserver(() => this.fg.width(el.clientWidth).height(el.clientHeight));
    this._ro.observe(el);
    this._hover(layer, type, side);
    this._loop();
  }

  // ---- puntos brillantes (una sola malla para las 8.000 neuronas) ----
  _buildPoints(pos_um) {
    const n = this.n;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(n * 3);
    pos_um.forEach((p, i) => pos.set(p, i * 3));
    this.aColor = new Float32Array(n * 3);
    this.aSize = new Float32Array(n);
    this.aAlpha = new Float32Array(n);
    this.aGlow = new Float32Array(n);
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aColor', new THREE.BufferAttribute(this.aColor, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('aSize', new THREE.BufferAttribute(this.aSize, 1).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('aAlpha', new THREE.BufferAttribute(this.aAlpha, 1).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('aGlow', new THREE.BufferAttribute(this.aGlow, 1).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('aPhase', new THREE.BufferAttribute(Float32Array.from({ length: n }, Math.random), 1));
    geo.computeBoundingSphere();
    this.mat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uScale: { value: 620 * Math.min(window.devicePixelRatio || 1, 2) } },
      vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(geo, this.mat);
    this.points.frustumCulled = false;
    this.fg.scene().add(this.points);
    this._paint();
  }

  /** Reescribe color, tamaño y brillo de cada neurona según su activación. */
  _paint() {
    for (let i = 0; i < this.n; i++) {
      const L = this.layerOf[i];
      const base = RGB[L];
      let size, alpha, glow, mixw;
      if (!this.visible[L]) { size = 0; alpha = 0; glow = 0; mixw = 0; }
      else if (this.idle) { size = 3.4; alpha = 0.8; glow = 0; mixw = 0; }
      else {
        const a = Math.min(1, Math.sqrt(this.level[i]) + 0.7 * this.flash[i]); // activación + destello
        size = 2.4 + (L === 'KC' ? 8 : 11) * a * a;
        alpha = 0.14 + 0.86 * a;
        glow = a * a;
        mixw = 0.32 * a * a; // las más activas aclaran su color (núcleo caliente)
      }
      this.aSize[i] = size; this.aAlpha[i] = alpha; this.aGlow[i] = glow;
      for (let k = 0; k < 3; k++) this.aColor[i * 3 + k] = base[k] + (1 - base[k]) * mixw;
    }
    const g = this.points.geometry;
    g.attributes.aColor.needsUpdate = g.attributes.aSize.needsUpdate = g.attributes.aAlpha.needsUpdate = g.attributes.aGlow.needsUpdate = true;
  }

  // ---- información al pasar el ratón ----
  _hover(layer, type, side) {
    const tip = document.createElement('div');
    // estilo propio (fondo claro y texto oscuro: se lee sobre el visor oscuro)
    tip.style.cssText = 'position:absolute;z-index:3;display:none;pointer-events:none;white-space:nowrap;padding:6px 10px;border-radius:4px;'
      + 'background:#f4f6f5;color:#0e1815;font:500 12px/1.35 Inter Variable,system-ui,sans-serif;box-shadow:0 4px 14px rgba(0,0,0,.45);border-left:4px solid #888;';
    this.el.appendChild(tip);
    const ray = new THREE.Raycaster();
    ray.params.Points.threshold = 2.2;
    const dom = this.fg.renderer().domElement;
    let pending = null;
    this._onMove = (ev) => {
      pending = ev;
      if (this._hoverRaf) return;
      this._hoverRaf = requestAnimationFrame(() => {
        this._hoverRaf = 0;
        const r = dom.getBoundingClientRect();
        const m = new THREE.Vector2(((pending.clientX - r.left) / r.width) * 2 - 1, -((pending.clientY - r.top) / r.height) * 2 + 1);
        ray.setFromCamera(m, this.fg.camera());
        const hit = ray.intersectObject(this.points).find((h) => this.visible[layer[h.index]]);
        if (!hit) { tip.style.display = 'none'; return; }
        const i = hit.index;
        const SIDE = { left: 'lado izquierdo', right: 'lado derecho', center: 'centro' };
        const name = LAYER_NAME[layer[i]].split(' (')[0];
        const detail = [type[i] ? type[i].replace(/^ORN_/, 'receptor ') : 'sin tipo asignado', SIDE[side[i]]].filter(Boolean).join(' · ');
        tip.style.borderLeftColor = LAYER_COLOR[layer[i]];
        tip.innerHTML = `<div style="font-weight:700">${name}</div><div style="color:#56635e">${detail}</div>`;
        tip.style.display = 'block';
        // junto al cursor, sin salirse del visor
        const x = pending.clientX - r.left, y = pending.clientY - r.top;
        const left = x + 14 + tip.offsetWidth > r.width ? x - 14 - tip.offsetWidth : x + 14;
        const top = y + 14 + tip.offsetHeight > r.height ? y - 14 - tip.offsetHeight : y + 14;
        tip.style.left = `${Math.max(4, left)}px`; tip.style.top = `${Math.max(4, top)}px`;
      });
    };
    dom.addEventListener('pointermove', this._onMove);
    this._tip = tip; this._dom = dom;
  }

  // ---- aristas: solo las `topN` entradas más fuertes de cada neurona ----
  _sortEdges() {
    const { target, syn, n } = this.graph.edges;
    this._order = Array.from({ length: n }, (_, i) => i).sort((a, b) => target[a] - target[b] || syn[b] - syn[a]);
  }
  _linkColor(l) {
    const s = l.source.id ?? l.source;
    const c = RGB[this.layerOf[s]].map((v) => Math.round(v * 255));
    const inhib = this.graph.nodes.sign[s] < 0;
    if (this._pset.has(l)) return `rgba(${c[0]},${c[1]},${c[2]},0.6)`; // vía por la que viaja la señal
    return `rgba(${c[0]},${c[1]},${c[2]},${this.stage > 0 ? 0.02 : inhib ? 0.05 : 0.025})`;
  }
  _setLinks() {
    const { source, target } = this.graph.edges;
    const links = [];
    let count = 0, last = -1;
    for (const e of this._order) {
      if (target[e] !== last) { last = target[e]; count = 0; }
      if (count++ < this.topN) links.push({ source: source[e], target: target[e] });
    }
    this.nLinks = links.length;
    this.fg.graphData({ nodes: this._nodes, links });
    this._pset = new Set();
    if (this.stage > 0) this._particles(this.stage);
  }
  setTopN(v) { this.topN = v; this._setLinks(); return this.nLinks; }
  setLayerVisible(l, on) { this.visible[l] = on; this._dirty = true; this.fg.linkVisibility((k) => this.visible[this.layerOf[k.source.id ?? k.source]] && this.visible[this.layerOf[k.target.id ?? k.target]]); }
  setAutoCamera(on) { this.autoCamera = on; }

  // ---- pulsos de señal que viajan por las conexiones ----
  _particles(stage) {
    const pairs = FLOW[stage] ?? [];
    const cand = (this.fg.graphData().links ?? []).filter((l) => {
      const s = l.source.id ?? l.source, t = l.target.id ?? l.target;
      if (!pairs.some(([a, b]) => this.layerOf[s] === a && this.layerOf[t] === b)) return false;
      if (this.target[s] < 0.2) return false;
      return stage !== 3 || this.target[t] > 0.2;
    });
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    cand.sort(() => rnd() - 0.5);
    this._pset = new Set(cand.slice(0, 380));
    this.fg
      .linkColor((l) => this._linkColor(l))
      .linkDirectionalParticles((l) => (this._pset.has(l) ? 2 : 0))
      .linkDirectionalParticleSpeed(0.0095)
      .linkDirectionalParticleWidth(2.3)
      .linkDirectionalParticleColor((l) => LAYER_COLOR[this.layerOf[l.source.id ?? l.source]]);
  }

  // ---- cámara guiada ----
  _tour(stage) {
    const c = this.fg.controls();
    if (!this.autoCamera) return;
    const t = TOUR[stage];
    const look = this.centroid[t.look ?? 'ALL'];
    c.autoRotate = false;
    this.fg.cameraPosition(
      { x: look[0] + Math.sin(t.ang) * t.d, y: look[1] + 0.16 * t.d, z: look[2] + Math.cos(t.ang) * t.d },
      { x: look[0], y: look[1], z: look[2] }, 1700,
    );
    setTimeout(() => { if (this.stage === stage && !this._userTouched) { c.autoRotateSpeed = stage === 0 ? 0.55 : 0.3; c.autoRotate = true; } }, 1900);
  }

  // ---- bucle: suaviza la activación y repinta ----
  _loop() {
    this._alive = true;
    const tick = (now) => {
      for (let i = 0; i < this.n; i++) {
        if (this.pend[i]) {
          if (now < this.t0[i]) continue; // todavía no le toca (onda de activación)
          this.pend[i] = 0; this.flash[i] = 1; this._dirty = true;
        }
        const d = this.target[i] - this.level[i];
        if (d !== 0) {
          this.level[i] = Math.abs(d) > 0.004 ? this.level[i] + d * this.rate[i] : this.target[i];
          this._dirty = true;
        }
        if (this.flash[i] > 0.003) { this.flash[i] *= 0.955; this._dirty = true; } else if (this.flash[i] !== 0) { this.flash[i] = 0; this._dirty = true; }
      }
      if (this._dirty) { this._dirty = false; this._paint(); }
      this.mat.uniforms.uTime.value = now / 1000;
      if (this._alive) this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  destroy() {
    this._alive = false;
    cancelAnimationFrame(this._raf);
    cancelAnimationFrame(this._hoverRaf);
    this.stop();
    this._ro?.disconnect();
    this._dom?.removeEventListener('pointermove', this._onMove);
    this._tip?.remove();
    this.fg._destructor?.();
  }

  // ---- animación por etapas ----
  _norm(arr) { let m = 0; for (const v of arr) if (v > m) m = v; return m || 1; }
  _setState(stage, sub = null) {
    const ex = this.acts.examples[this.example];
    const t = new Float32Array(this.n);
    const wasIdle = this.idle;
    this.idle = stage === 0;
    const steps = ex.steps, last = steps.length - 1;
    const put = (layer, values, scale, mx) => {
      const nodes = this.layerNodes[layer];
      for (let k = 0; k < nodes.length; k++) t[nodes[k]] = Math.min(1, (values[k] / mx) * scale);
    };
    const mxOf = (layer) => Math.max(...steps.map((s) => this._norm(s[layer])));
    if (stage >= 1) put('ORN', steps[stage === 2 && sub !== null ? sub : 0].ORN, stage === 1 ? 1 : 0.6, mxOf('ORN'));
    if (stage >= 2) {
      const s = stage === 2 && sub !== null ? sub : last, f = stage === 2 ? 1 : 0.5;
      put('PN', steps[s].PN, f, mxOf('PN')); put('LN', steps[s].LN, f, mxOf('LN'));
    }
    if (stage >= 3) {
      const kc = this.layerNodes.KC, m = this._norm(ex.KC.val);
      ex.KC.idx.forEach((k, j) => { t[kc[k]] = Math.min(1, (ex.KC.val[j] / m) * (stage === 3 ? 1 : 0.7)); });
    }
    if (stage >= 4) put('MBON', ex.MBON, 1, this._norm(ex.MBON));
    // onda: las neuronas recién activadas se encienden en orden de distancia al centro de su capa
    const now = performance.now();
    const waves = sub === null || sub === 0;
    for (const L of this.graph.layers) {
      const c = this.centroid[L], nodes = this.layerNodes[L];
      let maxd = 1;
      const ds = nodes.map((i) => { const p = this.graph.nodes.pos_um[i]; return Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]); });
      for (const d of ds) if (d > maxd) maxd = d;
      nodes.forEach((i, k) => {
        if (t[i] > 0.15 && this.target[i] < 0.15 && !this.idle) {
          this.pend[i] = 1;
          this.t0[i] = now + (waves ? (ds[k] / maxd) * 1100 : 0);
        } else if (t[i] <= 0.15) this.pend[i] = 0;
      });
    }
    this.target = t;
    const changed = stage !== this.stage;
    this.stage = stage;
    this._dirty = true;
    if (this.idle !== wasIdle) this.level.fill(0);
    if (stage > 0 && (sub === null || sub === 0 || changed)) this._particles(stage);
    else if (stage === 0) { this._pset = new Set(); this.fg.linkDirectionalParticles(0); this.fg.linkColor((l) => this._linkColor(l)); }
    if (changed) this._tour(stage);
  }

  /** Cuántas neuronas de las capas de la etapa actual están activas (para el rótulo de la pantalla). */
  activeInfo() {
    const layers = { 1: ['ORN'], 2: ['PN', 'LN'], 3: ['KC'], 4: ['MBON'] }[this.stage] ?? [];
    return layers.map((L) => {
      const nodes = this.layerNodes[L];
      return { layer: L, active: nodes.filter((i) => this.target[i] > 0.2).length, total: nodes.length };
    });
  }

  setExample(i) { this.stop(); this.example = i; this.stage = -1; this._setState(0); this.onChange?.(0, null); }
  goTo(stage, sub = null) { this._setState(stage, sub); this.onChange?.(stage, sub); }
  next() { this.stop(); this.goTo(Math.min(this.stage + 1, STAGES.length - 1)); }
  prev() { this.stop(); this.goTo(Math.max(this.stage - 1, 0)); }
  reset() { this.stop(); this.goTo(0); }
  stop() { this._timers.forEach(clearTimeout); this._timers = []; this.playing = false; this.onPlay?.(false); }
  play() {
    this.stop();
    this.playing = true; this.onPlay?.(true);
    const at = (ms, fn) => this._timers.push(setTimeout(fn, ms));
    const nS = this.acts.examples[this.example].steps.length;
    let t = 0;
    this.goTo(0);
    at((t += 400), () => this.goTo(1));
    at((t += 3000), () => this.goTo(2, 0));
    for (let s = 1; s < nS; s++) at((t += 1300), () => this.goTo(2, s));
    at((t += 2000), () => this.goTo(3));
    at((t += 3200), () => this.goTo(4));
    at((t += 400), () => { this.playing = false; this.onPlay?.(false); });
  }
}
