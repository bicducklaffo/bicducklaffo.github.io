const A = 3.480061091e-7;
const B = 250;
const C = 3.091619094e-6;
const D = 6.82775184551527e-5;
const E_CONST = -0.0301431777;

const STAT_NAMES = ["STR", "DEF", "SPD", "DEX"];

function gainCoefficients(happy, gym, mod) {
  const k = (A * Math.log(happy + B) + C) * gym * mod;
  const m = (D * (happy + B) + E_CONST) * gym * mod;
  return { k, m };
}

function applyEnergy(S0, energy, happy, gym, mod) {
  const { k, m } = gainCoefficients(happy, gym, mod);

  if (energy <= 0) return S0;

  if (Math.abs(k) < 1e-12) {
    return S0 + m * energy;
  }

  const shift = m / k;
  return (S0 + shift) * Math.exp(k * energy) - shift;
}

function solveEnergyForGain(S0, S1, happy, gym, mod) {
  const { k, m } = gainCoefficients(happy, gym, mod);

  if (S1 <= S0) return 0;

  if (Math.abs(k) < 1e-12) {
    if (m <= 0) return 0;
    return (S1 - S0) / m;
  }

  const shift = m / k;
  const numerator = S1 + shift;
  const denominator = S0 + shift;

  if (numerator <= 0 || denominator <= 0) return 0;

  const ratio = numerator / denominator;
  if (ratio <= 1) return 0;

  return Math.log(ratio) / k;
}

function inferSplitByInversion(before, after, gym, mod, inferenceHappy = 5000) {
  const inferredEnergies = [];

  for (let i = 0; i < 4; i++) {
    const S0 = Number(before[i] || 0);
    const S1 = Number(after[i] || 0);
    const requiredE = solveEnergyForGain(S0, S1, inferenceHappy, gym, mod);
    inferredEnergies.push(Math.max(requiredE, 0));
  }

  const total = inferredEnergies.reduce((a, b) => a + b, 0);

  if (total <= 0) {
    return {
      weights: [0, 0, 0, 0],
      inferredEnergies: [0, 0, 0, 0],
      trainedIndices: []
    };
  }

  const weights = inferredEnergies.map(e => e / total);
  const trainedIndices = inferredEnergies
    .map((e, i) => ({ e, i }))
    .filter(x => x.e > 0)
    .map(x => x.i);

  return {
    weights,
    inferredEnergies,
    trainedIndices
  };
}

function normalizeBaselines(baselines) {
  if (!Array.isArray(baselines) || baselines.length === 0) {
    throw new Error("At least one baseline is required.");
  }

  return baselines.map((b, idx) => {
    const name = (b.name || `Baseline ${idx + 1}`).trim();
    const xans = Math.max(0, Math.min(3, Number(b.xans || 0)));
    const refill = !!b.refill;
    const natural = Math.max(0, Number(b.natural || 0));
    const happy = Math.max(1, Number(b.happy || 5000));
    const ePerDay = xans * 250 + (refill ? 150 : 0) + natural;

    if (ePerDay <= 0) {
      throw new Error(`Baseline "${name}" has no energy per day.`);
    }

    return {
      name,
      xans,
      refill,
      natural,
      happy,
      ePerDay
    };
  });
}

function expectedGainForStat(S0, weight, totalDays, baseline, gym, mod) {
  const energy = totalDays * baseline.ePerDay * weight;
  const finalStat = applyEnergy(S0, energy, baseline.happy, gym, mod);
  return Math.max(finalStat - S0, 0);
}

function calculateTornEfficiency(
  before,
  after,
  hours,
  gym = 6.0,
  mod = 1.1,
  baselines = null,
  inferenceHappy = 5000
) {
  if (!Array.isArray(before) || !Array.isArray(after) || before.length !== 4 || after.length !== 4) {
    throw new Error("Before and after must each be arrays of 4 stats: [STR, DEF, SPD, DEX]");
  }

  const totalDays = Number(hours) / 24;
  if (!isFinite(totalDays) || totalDays <= 0) {
    throw new Error("Hours must be a positive number.");
  }

  const beforeNums = before.map(x => Number(x));
  const afterNums = after.map(x => Number(x));

  const normalizedBaselines = normalizeBaselines(
    baselines || [
      { name: "Non-hop", xans: 3, refill: true, natural: 600, happy: 5000 },
      { name: "Hop", xans: 3, refill: true, natural: 0, happy: 20000 }
    ]
  );

  const inference = inferSplitByInversion(beforeNums, afterNums, gym, mod, inferenceHappy);
  const split = inference.weights;

  const perStat = [];
  const combinedExpected = {};
  let combinedActual = 0;

  normalizedBaselines.forEach(b => {
    combinedExpected[b.name] = 0;
  });

  for (let i = 0; i < 4; i++) {
    const S0 = beforeNums[i];
    const S1 = afterNums[i];
    const actual = Math.max(S1 - S0, 0);

    if (actual <= 0 || split[i] <= 0) continue;

    const expectedByBaseline = {};
    const scoreByBaseline = {};
    let bestExpected = 0;

    normalizedBaselines.forEach(b => {
      const expected = expectedGainForStat(S0, split[i], totalDays, b, gym, mod);
      expectedByBaseline[b.name] = expected;
      scoreByBaseline[b.name] = expected > 0 ? actual / expected : 0;
      combinedExpected[b.name] += expected;
      if (expected > bestExpected) bestExpected = expected;
    });

    perStat.push({
      name: STAT_NAMES[i],
      actual,
      inferredEnergy: inference.inferredEnergies[i],
      weight: split[i],
      expectedByBaseline,
      scoreByBaseline,
      frontier: bestExpected > 0 ? actual / bestExpected : 0
    });

    combinedActual += actual;
  }

  const combinedByBaseline = {};
  let bestCombinedExpected = 0;

  normalizedBaselines.forEach(b => {
    const exp = combinedExpected[b.name];
    combinedByBaseline[b.name] = exp > 0 ? combinedActual / exp : 0;
    if (exp > bestCombinedExpected) bestCombinedExpected = exp;
  });

  return {
    inputs: {
      before: beforeNums,
      after: afterNums,
      hours: Number(hours),
      gym,
      mod,
      inferenceHappy
    },
    baselines: normalizedBaselines,
    inference: {
      inferenceHappy,
      inferredEnergies: inference.inferredEnergies,
      split
    },
    split,
    perStat,
    combinedByBaseline,
    frontierCombined: bestCombinedExpected > 0 ? combinedActual / bestCombinedExpected : 0
  };
}
