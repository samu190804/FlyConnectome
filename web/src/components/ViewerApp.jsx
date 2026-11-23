import { useEffect, useRef, useState } from 'react';
import { ViewerEngine } from '../lib/viewerEngine.js';
import { CLASS_ES, LAYER_COLOR, LAYER_NAME, STAGES } from '../lib/content.js';

const base = import.meta.env.BASE_URL;
const load = (f) => fetch(`${base}data/${f}`).then((r) => { if (!r.ok) throw new Error(`${f}: ${r.status}`); return r.json(); });
const dots = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, '.');
const btn = 'cursor-pointer rounded border border-line bg-panel px-2.5 py-1.5 hover:bg-bg2 disabled:cursor-default disabled:opacity-50 disabled:hover:bg-panel';

/** Isla React: visor 3D (three.js / 3d-force-graph) + panel con la historia de cada etapa. */
export default function ViewerApp() {
  const stageEl = useRef(null);
  const engine = useRef(null);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [example, setExample] = useState(0);
  const [topN, setTopN] = useState(1);
  const [linkCount, setLinkCount] = useState(0);
  const [autoCam, setAutoCam] = useState(true);
  const [info, setInfo] = useState([]);
  const [visible, setVisible] = useState(() => Object.fromEntries(Object.keys(LAYER_NAME).map((l) => [l, true])));

  useEffect(() => {
    Promise.all([load('graph.json'), load('activations.json')]).then(([graph, acts]) => setData({ graph, acts })).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!data) return;
    const v = new ViewerEngine(stageEl.current, data.graph, data.acts);
    v.onChange = (st) => { setStage(st); setInfo(v.activeInfo()); };
    v.onPlay = setPlaying;
    engine.current = v;
    setLinkCount(v.nLinks);
    v.setExample(0);
    return () => { v.destroy(); engine.current = null; };
  }, [data]);

  const ready = !!data;
  const s = STAGES[stage];
  const ex = data?.acts.examples[example];
  const order = ex ? ex.probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3) : [];

  return (
    <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
      <div className="relative h-[min(72vh,680px)] min-h-[380px] overflow-hidden rounded border border-ink/70 bg-[#121418]">
        <div ref={stageEl} role="img" aria-label="Visor 3D del circuito olfativo" className="absolute inset-0" />
        {ready && stage > 0 && info.length > 0 && (
          <div className="pointer-events-none absolute left-3 top-3 z-[2] max-w-[80%] rounded bg-black/55 px-3 py-2 text-[#eef1ef] backdrop-blur-sm">
            <div className="font-display text-sm font-semibold">{STAGES[stage].title}</div>
            {info.map(({ layer, active, total }) => (
              <div key={layer} className="mt-0.5 flex items-center gap-2 text-xs">
                <i className="inline-block size-2 rounded-full" style={{ background: LAYER_COLOR[layer] }} />
                <b>{dots(active)}</b> de {dots(total)} · {LAYER_NAME[layer].split(' (')[0]} activas ({((active / total) * 100).toFixed(0)} %)
              </div>
            ))}
          </div>
        )}
        {!ready && <div className="pointer-events-none absolute inset-0 z-[2] grid place-items-center p-5 text-center text-[#9a9486]">
          {error ? `No se pudieron cargar los datos (${error}). Ejecuta «uv run python -m flyconn.export» y recarga.` : 'Cargando el circuito…'}
        </div>}
      </div>

      <aside className="rounded border border-line bg-panel p-4">
        <label htmlFor="odor" className="mb-1 block text-sm text-mute">Olor</label>
        <select id="odor" disabled={!ready} value={example}
          onChange={(e) => { const i = Number(e.target.value); setExample(i); engine.current?.setExample(i); }}
          className="w-full rounded border border-line bg-bg px-2.5 py-1.5 disabled:opacity-50">
          {ready && data.acts.examples.map((e, i) => <option key={e.odor} value={i}>{e.odor} ({CLASS_ES[e.class]})</option>)}
        </select>

        <div className="my-2.5 flex flex-wrap gap-1.5">
          <button type="button" disabled={!ready} onClick={() => (playing ? engine.current.stop() : engine.current.play())}
            className={`${btn} min-w-[120px] flex-1 !border-ink !bg-ink !font-semibold !text-bg hover:!border-accent hover:!bg-accent`}>{playing ? '■ Detener' : '▶ Reproducir'}</button>
          <button type="button" disabled={!ready} onClick={() => engine.current.prev()} aria-label="Etapa anterior" className={btn}>◀</button>
          <button type="button" disabled={!ready} onClick={() => engine.current.next()} aria-label="Etapa siguiente" className={btn}>▶</button>
          <button type="button" disabled={!ready} onClick={() => engine.current.reset()} className={btn}>Reiniciar</button>
        </div>

        <ol className="mb-2.5 grid list-none grid-cols-2 gap-1.5 p-0">
          {STAGES.slice(1).map((st, i) => (
            <li key={st.title}>
              <button type="button" disabled={!ready} onClick={() => { engine.current.stop(); engine.current.goTo(i + 1); }}
                className={`w-full cursor-pointer rounded border px-1.5 py-1 text-left text-xs leading-snug ${stage === i + 1 ? 'border-ink bg-ink text-bg' : 'border-line bg-transparent text-mute hover:border-mute'}`}>
                {i + 1}. {st.title.replace(/^\d · /, '')}
              </button>
            </li>
          ))}
        </ol>

        <div aria-live="polite">
          <h3 className="mb-1.5 mt-1.5 text-lg font-bold">{s.title}</h3>
          <p className="my-1.5 text-[0.95rem]">{s.bio}</p>
          {s.ml && <p className="my-1.5 text-[0.9rem] italic text-accent">{s.ml}</p>}
        </div>

        {stage === 4 && ex && (
          <div className="mt-2.5 rounded bg-bg2 p-2.5 text-[0.9rem]">
            <div className={`mb-2 ${ex.pred === ex.class ? 'text-ok' : 'text-ko'}`}>
              {ex.pred === ex.class ? '✔ Acierta' : '✘ Falla'}: la red dice <b>{CLASS_ES[ex.pred]}</b>, y el olor es <b>{CLASS_ES[ex.class]}</b>.
            </div>
            {order.map(([p, i]) => (
              <div key={i} className="my-0.5 grid grid-cols-[110px_1fr_40px] items-center gap-1.5 text-[0.82rem]">
                <span>{CLASS_ES[data.acts.meta.classes[i]]}</span>
                <div className="h-2 overflow-hidden rounded bg-line"><i className="block h-full bg-ink" style={{ width: `${p * 100}%` }} /></div>
                <b>{(p * 100).toFixed(0)} %</b>
              </div>
            ))}
            <p className="mt-2 text-sm text-mute">Es un olor de prueba (no visto al entrenar). Esta red acierta alrededor de la mitad de las veces, como en las gráficas.</p>
          </div>
        )}

        <details className="mt-3">
          <summary className="cursor-pointer text-mute">Opciones de visualización</summary>
          <label htmlFor="links" className="mt-2 block text-sm text-mute">Conexiones por neurona: <b>{topN}</b> <span>({linkCount.toLocaleString('es-ES')} líneas)</span></label>
          <input id="links" type="range" min="0" max="8" step="1" value={topN} disabled={!ready} className="w-full"
            onChange={(e) => { const n = Number(e.target.value); setTopN(n); setLinkCount(engine.current.setTopN(n)); }} />
          <label className="mt-2 flex items-center gap-2 text-[0.85rem] text-ink"><input type="checkbox" checked={autoCam} onChange={(e) => { setAutoCam(e.target.checked); engine.current.setAutoCamera(e.target.checked); }} />Cámara automática (viaja entre etapas)</label>
          <div className="mt-1">
            {Object.keys(LAYER_NAME).map((l) => (
              <label key={l} className="my-1 flex items-center gap-2 text-[0.85rem] text-ink">
                <input type="checkbox" checked={visible[l]} onChange={(e) => { setVisible((v) => ({ ...v, [l]: e.target.checked })); engine.current.setLayerVisible(l, e.target.checked); }} />
                <i className="inline-block size-2.5 rounded-full" style={{ background: LAYER_COLOR[l] }} />{LAYER_NAME[l]}
              </label>
            ))}
          </div>
          <p className="mt-1 text-sm text-mute">Se dibujan las entradas más fuertes de cada neurona. Líneas rosas: conexiones que frenan; azules: que activan.</p>
        </details>
      </aside>
    </div>
  );
}
