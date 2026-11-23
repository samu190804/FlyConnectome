// Textos de la web (español). Cifras: ver README y CITATIONS.md del repositorio.

export const CLASS_ES = {
  acid: 'ácidos', acetate_ester: 'acetatos', other_ester: 'otros ésteres', alcohol: 'alcoholes',
  aldehyde: 'aldehídos', ketone: 'cetonas', terpene: 'terpenos', aromatic: 'aromáticos',
  lactone: 'lactonas', nitrogen_sulfur: 'nitrógeno y azufre',
};

export const LAYER_NAME = {
  ORN: 'Receptores olfativos (ORN)', PN: 'Neuronas de proyección (PN)', LN: 'Neuronas locales (LN)',
  KC: 'Células de Kenyon (KC)', MBON: 'Neuronas de salida (MBON)',
};

export const LAYER_COLOR = {
  ORN: '#d99a2b', PN: '#4a90c2', LN: '#cf5b4b', KC: '#8e6bbf', MBON: '#4fa36b',
};

// Una entrada por etapa de la animación: texto biológico + analogía de machine learning
export const STAGES = [
  {
    title: 'Elige un olor',
    bio: 'Cada punto es una neurona real, colocada donde está en el cerebro de la mosca. Elige un olor y pulsa «Reproducir» para ver cómo viaja la señal por la red.',
    ml: null,
    layers: [],
  },
  {
    title: '1 · Los receptores de las antenas (ORN)',
    bio: 'Las neuronas receptoras de las antenas son la puerta de entrada del olfato. Cada una tiene un solo tipo de receptor (hay unos 50 tipos en la mosca; en este circuito, 53). Un olor no activa un solo tipo, sino varios a la vez: es un código combinatorio.',
    ml: 'Analogía de ML: es el vector de características de entrada.',
    layers: ['ORN'],
  },
  {
    title: '2 · El lóbulo antenal (PN y LN)',
    bio: 'Las receptoras del mismo tipo convergen en una misma «estación» (un glomérulo), donde pasan el mensaje a las neuronas de proyección (PN). Las neuronas locales (LN) frenan la señal para que el olor se reconozca con independencia de su intensidad. Aquí la señal rebota unos pasos antes de salir.',
    ml: 'Analogía de ML: preprocesado y normalización de la entrada.',
    layers: ['PN', 'LN'],
  },
  {
    title: '3 · Las células de Kenyon (KC)',
    bio: 'Cada célula de Kenyon recibe señal de unas pocas PN (unas 5 de media). Son miles (más de 2.000 por hemisferio) y solo una pequeña parte, alrededor del 5 %, se activa a la vez. Olores parecidos dejan «firmas» distintas.',
    ml: 'Analogía de ML: proyección aleatoria a alta dimensión con activación dispersa.',
    layers: ['KC'],
  },
  {
    title: '4 · Las neuronas de salida (MBON)',
    bio: 'Unas pocas decenas de neuronas leen las células de Kenyon activas y producen la respuesta. En la mosca real, la dopamina (recompensa o castigo) modifica las sinapsis de las KC con las MBON y así aprende a acercarse o alejarse de un olor. En nuestro modelo se entrena esa etapa, sin dopamina.',
    ml: 'Analogía de ML: un clasificador lineal entrenable. Aquí una capa final traduce la actividad de las MBON a las 10 clases de olores.',
    layers: ['MBON'],
  },
];

export const REFERENCES = [
  ['Dorkenwald et al., 2024, Nature. Conectoma completo del cerebro adulto de Drosophila (FlyWire).', 'https://doi.org/10.1038/s41586-024-07558-y'],
  ['Schlegel et al., 2024, Nature. Anotaciones y tipos celulares del conectoma de FlyWire.', 'https://doi.org/10.1038/s41586-024-07686-5'],
  ['Zheng et al., 2018, Cell. Reconstrucción de un cerebro completo de Drosophila por microscopía electrónica.', 'https://doi.org/10.1016/j.cell.2018.06.019'],
  ['Eckstein, Bates et al., 2024, Cell. Predicción del neurotransmisor de cada neurona.', 'https://doi.org/10.1016/j.cell.2024.03.016'],
  ['Yu et al., 2025, bioRxiv. Detección de sinapsis usada en las conexiones descargadas.', 'https://www.biorxiv.org/content/10.1101/2025.07.11.664377v1'],
  ['Hallem y Carlson, 2006, Cell. Respuestas de los receptores olfativos a 110 olores.', 'https://doi.org/10.1016/j.cell.2006.01.050'],
  ['Couto, Alenius y Dickson, 2005, Current Biology. Mapa de receptores a glomérulos.', 'https://doi.org/10.1016/j.cub.2005.07.034'],
  ['FlyWire Codex (descarga de datos).', 'https://codex.flywire.ai/'],
];

// [nombre, descripción, ¿la usa el proyecto?]
export const REGIONS = [
  ['Lóbulos ópticos', 'visión', false],
  ['Lóbulo antenal', 'primera estación del olfato', true],
  ['Cuerpo fungiforme', 'aprendizaje y memoria olfativa', true],
  ['Cuerno lateral', 'respuestas innatas a olores', false],
  ['Complejo central', 'orientación y navegación', false],
  ['Zona subesofágica', 'gusto', false],
];

export const MODEL_CARDS = {
  connectome_A: ['Conectoma A', 'Cableado real, pesos fijos. Solo se aprende la última etapa (como en la mosca).'],
  connectome_B: ['Conectoma B', 'Cableado real, todas las conexiones reales se entrenan.'],
  random_A: ['Aleatorio A', 'Mismas neuronas y nº de conexiones, pero al azar. Pesos fijos.'],
  random_B: ['Aleatorio B', 'Igual que el aleatorio A, con todas las conexiones entrenables.'],
  mlp: ['MLP denso', 'Red genérica: todo conectado con todo, sin estructura biológica.'],
};

export const PAIR_LABEL = {
  'connectome_B-random_B': ['Conectoma B − Aleatorio B', '¿Importa qué neurona se conecta con cuál?'],
  'connectome_A-random_A': ['Conectoma A − Aleatorio A', 'Lo mismo, con pesos fijos.'],
  'connectome_B-mlp': ['Conectoma B − MLP denso', '¿Ayuda la estructura biológica frente a una red genérica?'],
  'connectome_B-connectome_A': ['Conectoma B − Conectoma A', '¿Merece la pena entrenar el interior de la red?'],
};

export const SERIES_STYLE = {
  connectome_A: { color: '#2b6a9e', dash: '6 4' },
  connectome_B: { color: '#2b6a9e', dash: '' },
  random_A: { color: '#c0662a', dash: '6 4' },
  random_B: { color: '#c0662a', dash: '' },
  mlp: { color: '#3c8052', dash: '' },
};
