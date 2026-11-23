import { useEffect, useState } from 'react';
import LineChart from './LineChart.jsx';
import { MODEL_CARDS, PAIR_LABEL, SERIES_STYLE } from '../lib/content.js';

const fmt = (n) => n.toLocaleString('es-ES');

/** Tarjetas de modelos, curvas comparativas y diferencias emparejadas (datos de results.json). */
export default function Results({ results: r }) {
  const series = (curve) => r.models.map((m) => {
    const keys = Object.keys(curve[m]);
    return { key: m, label: MODEL_CARDS[m][0], mean: keys.map((k) => curve[m][k].mean), std: keys.map((k) => curve[m][k].std) };
  });
  const ks = Object.keys(Object.values(r.paired_differences)[0]);
  const se = Math.sqrt(r.n_seeds);

  const [on, setOn] = useState(false);
  useEffect(() => { const t = setTimeout(() => setOn(true), 150); return () => clearTimeout(t); }, []);
  const kmax = String(r.robustness_examples_per_class);
  const board = r.models.map((m) => ({ m, acc: r.learning_curve[m][kmax].mean })).sort((a, b) => b.acc - a.acc);

  return (
    <div>
      <div className="mb-6 rounded border border-line bg-panel p-5">
        <h3 className="font-display mb-1 text-xl font-semibold">Marcador final <span className="text-sm font-normal text-mute">(con {kmax} ejemplos por clase; azar = 10 %)</span></h3>
        <div className="mt-3 space-y-2.5">
          {board.map(({ m, acc }, i) => (
            <div key={m} className="grid grid-cols-[110px_1fr_48px] items-center gap-3 text-sm sm:grid-cols-[150px_1fr_52px]">
              <span className="font-medium">{MODEL_CARDS[m][0]}</span>
              <div className="h-3 overflow-hidden rounded-sm bg-bg2">
                <div className="bar-fill h-full" style={{ width: on ? `${acc * 100}%` : '0%', background: SERIES_STYLE[m].color }} />
              </div>
              <b className="text-right text-base">{(acc * 100).toFixed(0)} %</b>
            </div>
          ))}
        </div>
        <p className="mt-3 text-sm text-mute">Las diferencias de pocos puntos entre modelos están dentro del ruido; mira la tabla de más abajo.</p>
      </div>
      <div className="mb-2 mt-4 grid grid-cols-[repeat(auto-fit,minmax(170px,1fr))] gap-2.5">
        {r.models.map((m) => (
          <div key={m} className="relative rounded border border-line bg-panel p-3 pl-4">
            <i className="absolute inset-y-2.5 left-0 w-1 rounded-sm" style={{ background: SERIES_STYLE[m].color }} />
            <h4 className="font-bold">{MODEL_CARDS[m][0]}</h4>
            <p className="mb-1.5 text-[0.85rem] text-[#26332e]">{MODEL_CARDS[m][1]}</p>
            <span className="text-sm text-mute">{fmt(r.n_params[m])} parámetros entrenables</span>
          </div>
        ))}
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <figure>
          <h3 className="font-display mb-2 text-xl font-semibold">Curva de aprendizaje</h3>
          <LineChart title="Curva de aprendizaje" xs={Object.keys(r.learning_curve[r.models[0]])} series={series(r.learning_curve)}
            xLabel="ejemplos por clase" yLabel="precisión en olores no vistos" />
          <figcaption className="mt-1.5 text-sm text-mute">Más ejemplos de entrenamiento por clase, más precisión. Media de {r.n_seeds} semillas; pasa el ratón para ver la desviación.</figcaption>
        </figure>
        <figure>
          <h3 className="font-display mb-2 text-xl font-semibold">Robustez al ruido</h3>
          <LineChart title="Robustez al ruido" xs={Object.keys(r.noise_robustness[r.models[0]]).map((k) => String(Number(k)))} series={series(r.noise_robustness)}
            xLabel="ruido en la prueba (espigas/s)" yLabel="precisión en olores no vistos" />
          <figcaption className="mt-1.5 text-sm text-mute">Redes entrenadas con {r.robustness_examples_per_class} ejemplos por clase, probadas con más ruido en la entrada.</figcaption>
        </figure>
      </div>

      <h3 className="font-display mb-2 mt-10 text-xl font-semibold">¿Quién gana a quién?</h3>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[0.9rem]">
          <thead>
            <tr>
              <th className="border-b border-line px-2.5 py-2 text-left">Comparación (diferencia de precisión)</th>
              {ks.map((k) => <th key={k} className="border-b border-line px-2.5 py-2 text-right">{k} ej.</th>)}
            </tr>
          </thead>
          <tbody>
            {Object.entries(r.paired_differences).map(([key, row]) => {
              const [name, q] = PAIR_LABEL[key];
              return (
                <tr key={key}>
                  <th scope="row" className="border-b border-line px-2.5 py-2 text-left">{name}<small className="block text-xs font-normal text-mute">{q}</small></th>
                  {ks.map((k) => {
                    const { mean, std } = row[k];
                    const strong = Math.abs(mean) > (2 * std) / se;
                    const bg = strong ? (mean > 0 ? 'bg-ok/15' : 'bg-ko/15') : '';
                    return (
                      <td key={k} className={`border-b border-line px-2.5 py-2 text-right ${bg}`}>
                        {mean > 0 ? '+' : ''}{mean.toFixed(3)}<small className="block text-xs text-mute">± {std.toFixed(2)}</small>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-sm text-mute">Diferencia media por semilla (misma partición de olores). Una celda de color indica que la diferencia supera unas 2 veces su error típico (aproximado, sin corregir por comparaciones múltiples); las demás están dentro del ruido.</p>

      <div className="mt-8 rounded border border-line border-l-[3px] border-l-accent bg-panel px-5 py-2">
        <h3 className="font-display mb-1 mt-3 text-xl font-semibold">Lo que vemos</h3>
        <ul className="list-disc pl-5">
          <li className="my-1"><b>El cableado real no mejora a uno aleatorio</b> cuando se entrena toda la red (conectoma B frente a aleatorio B: empate), y con pesos fijos queda algo por debajo.</li>
          <li className="my-1"><b>Un MLP genérico del mismo tamaño rinde igual o algo mejor</b>, sobre todo con pocos ejemplos.</li>
          <li className="my-1">Entrenar el interior de la red (B frente a A) da unos 5 puntos.</li>
          <li className="my-1">Lo único favorable al conectoma: con pesos fijos, <b>aguanta mejor el ruido</b> que el aleatorio.</li>
        </ul>
        <p className="mb-2 mt-2">Un resultado negativo es un resultado válido: en esta versión, el cableado concreto no aporta precisión extra.</p>
      </div>
    </div>
  );
}
