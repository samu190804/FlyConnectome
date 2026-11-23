import { useRef, useState } from 'react';
import { SERIES_STYLE } from '../lib/content.js';

const W = 640, H = 380, M = { l: 62, r: 16, t: 14, b: 54 };
const IW = W - M.l - M.r, IH = H - M.t - M.b;

/** Gráfica de líneas con puntos, leyenda activable y tooltip. series: [{key,label,mean[],std[]}] */
export default function LineChart({ title, xs, series, xLabel, yLabel }) {
  const svgRef = useRef(null);
  const [hidden, setHidden] = useState(() => new Set());
  const [hover, setHover] = useState(null); // {i, left, top}
  const x = (i) => M.l + (xs.length === 1 ? IW / 2 : (i / (xs.length - 1)) * IW);
  const y = (v) => M.t + IH * (1 - v);
  const shown = series.filter((s) => !hidden.has(s.key));

  const onMove = (ev) => {
    const r = svgRef.current.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(xs.length - 1, Math.round(((px - M.l) / IW) * (xs.length - 1))));
    const box = ev.currentTarget.closest('[data-chart]').getBoundingClientRect();
    setHover({ i, left: ev.clientX - box.left + 14, top: ev.clientY - box.top - 10 });
  };
  const toggle = (key) => setHidden((h) => { const n = new Set(h); n.has(key) ? n.delete(key) : n.add(key); return n; });

  return (
    <div data-chart className="relative rounded border border-line bg-panel p-2.5">
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} className="block h-auto w-full">
        {[0, 0.2, 0.4, 0.6, 0.8, 1].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="chart-grid" />
            <text x={M.l - 8} y={y(v) + 4} textAnchor="end" className="chart-tick">{v.toFixed(1)}</text>
          </g>
        ))}
        {xs.map((v, i) => <text key={v} x={x(i)} y={H - M.b + 18} textAnchor="middle" className="chart-tick">{v}</text>)}
        <text x={M.l + IW / 2} y={H - 8} textAnchor="middle" className="chart-axis">{xLabel}</text>
        <text x={16} y={M.t + IH / 2} textAnchor="middle" transform={`rotate(-90 16 ${M.t + IH / 2})`} className="chart-axis">{yLabel}</text>
        <line x1={M.l} x2={W - M.r} y1={y(0.1)} y2={y(0.1)} className="chart-chance" />
        <text x={W - M.r - 4} y={y(0.1) - 5} textAnchor="end" className="chart-tick">azar</text>
        {shown.map((s) => {
          const st = SERIES_STYLE[s.key];
          return (
            <g key={s.key}>
              <polyline points={s.mean.map((v, i) => `${x(i)},${y(v)}`).join(' ')} fill="none" stroke={st.color} strokeWidth="2.2" strokeDasharray={st.dash} />
              {s.mean.map((v, i) => s.key === 'mlp'
                ? <rect key={i} x={x(i) - 3.5} y={y(v) - 3.5} width="7" height="7" fill={st.color} />
                : <circle key={i} cx={x(i)} cy={y(v)} r="3.5" fill={s.key.endsWith('_A') ? '#f4f6f5' : st.color} stroke={st.color} strokeWidth="1.6" />)}
            </g>
          );
        })}
        {hover && <line x1={x(hover.i)} x2={x(hover.i)} y1={M.t} y2={M.t + IH} stroke="#6d7b75" />}
        <rect x={M.l} y={M.t} width={IW} height={IH} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      {hover && (
        <div className="pointer-events-none absolute z-[3] whitespace-nowrap rounded-md border border-line bg-panel px-2 py-1.5 text-xs shadow-sm" style={{ left: hover.left, top: hover.top }}>
          <div className="mb-0.5 text-mute">{xLabel}: {xs[hover.i]}</div>
          {shown.map((s) => (
            <div key={s.key}>
              <i className="mr-1.5 inline-block size-2 rounded-full" style={{ background: SERIES_STYLE[s.key].color }} />
              {s.label}: <b>{s.mean[hover.i].toFixed(2)}</b> ± {s.std[hover.i].toFixed(2)}
            </div>
          ))}
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-1.5">
        {series.map((s) => {
          const st = SERIES_STYLE[s.key];
          const off = hidden.has(s.key);
          return (
            <button key={s.key} type="button" aria-pressed={!off} onClick={() => toggle(s.key)}
              className={`inline-flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded border border-line bg-panel px-2 py-0.5 text-[0.8rem] hover:bg-bg2 ${off ? 'opacity-40' : ''}`}>
              <svg width="26" height="10" aria-hidden="true" className="flex-none"><line x1="0" x2="26" y1="5" y2="5" stroke={st.color} strokeWidth="2.4" strokeDasharray={st.dash} /></svg>
              {s.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
