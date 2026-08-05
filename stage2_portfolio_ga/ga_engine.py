"""
Portfolio-Level Genetic Algorithm Engine
========================================

Genes interact because fitness = Sharpe of the combined weighted portfolio.
Changing one asset's strategy affects optimal choices for all others via
covariance and diversification.

Chromosome: (strategy_index[N], weight[N]) where weights sum to 1.
Precomputes all strategy-asset daily returns into a single numpy 3D array
upfront; fitness evaluation is then just array indexing + dot product.
"""
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional

from config import (
    N_STRATEGIES, POPULATION_SIZE, N_GENERATIONS, TOURNAMENT_SIZE,
    CROSSOVER_RATE, CROSSOVER_TYPE, STRATEGY_MUTATION_RATE,
    WEIGHT_MUTATION_RATE, WEIGHT_MUTATION_SIGMA, ELITE_COUNT,
    TRADING_DAYS_PER_YEAR,
)

log = logging.getLogger(__name__)

# Default GA parameters (mirrors config values)
DEFAULT_GA_PARAMS = {
    'population_size': POPULATION_SIZE,
    'n_generations': N_GENERATIONS,
    'tournament_size': TOURNAMENT_SIZE,
    'crossover_rate': CROSSOVER_RATE,
    'crossover_type': CROSSOVER_TYPE,
    'strategy_mutation_rate': STRATEGY_MUTATION_RATE,
    'weight_mutation_rate': WEIGHT_MUTATION_RATE,
    'weight_mutation_sigma': WEIGHT_MUTATION_SIGMA,
    'elite_count': ELITE_COUNT,
}


class Chromosome:
    __slots__ = ['strategy_genes', 'weight_genes', 'asset_names',
                 'fitness', 'portfolio_sharpe']

    def __init__(self, strategy_genes: np.ndarray, weight_genes: np.ndarray,
                 asset_names: list):
        self.strategy_genes = strategy_genes
        self.weight_genes = weight_genes
        self.asset_names = asset_names
        self.fitness = None
        self.portfolio_sharpe = None

    def copy(self):
        c = Chromosome(
            self.strategy_genes.copy(),
            self.weight_genes.copy(),
            self.asset_names,
        )
        return c


def ann_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 30:
        return -10.0
    mu = np.mean(returns)
    sigma = np.std(returns)
    if sigma == 0:
        return 0.0
    return (mu / sigma) * np.sqrt(TRADING_DAYS_PER_YEAR)


def _ann_sharpe_fast(mu, sigma):
    """Vectorised Sharpe from precomputed mean/std."""
    if sigma == 0:
        return 0.0
    return (mu / sigma) * np.sqrt(TRADING_DAYS_PER_YEAR)


class GAEngine:
    def __init__(self, asset_names: list,
                 precomputed_strat: Dict[Tuple[str, int], pd.Series],
                 precomputed_bh: Dict[str, pd.Series],
                 allowed_strategies: list = None,
                 seed: int = 42,
                 ga_params: dict = None):
        """
        precomputed_strat: {(asset, strat_idx): pd.Series of daily strategy returns}
        precomputed_bh:    {asset: pd.Series of daily B&H returns}
        allowed_strategies: list of strategy indices the GA may use (default: all)
        ga_params: dict overriding default GA hyperparameters
        """
        self.asset_names = asset_names
        self.n_genes = len(asset_names)
        self.precomputed_strat = precomputed_strat
        self.precomputed_bh = precomputed_bh
        self.allowed_strategies = (allowed_strategies if allowed_strategies is not None
                                   else list(range(N_STRATEGIES)))
        self.n_allowed = len(self.allowed_strategies)
        self.rng = np.random.RandomState(seed)

        # Merge defaults with any overrides
        self.params = dict(DEFAULT_GA_PARAMS)
        if ga_params:
            self.params.update(ga_params)

        # Build aligned B&H DataFrame for index alignment (kept for OOS compat)
        self.bh_df = pd.DataFrame(
            {a: precomputed_bh[a] for a in asset_names}
        ).fillna(0.0)

        # ── Precompute numpy 3D returns matrix for fast fitness eval ──
        # returns_matrix[strat_idx, asset_idx, day] = daily return
        common_idx = self.bh_df.index
        n_days = len(common_idx)
        max_strat = max(max(self.allowed_strategies), 0) + 1

        self._returns_matrix = np.zeros((max_strat, self.n_genes, n_days))
        for asset_i, asset in enumerate(asset_names):
            for strat_idx in self.allowed_strategies:
                key = (asset, strat_idx)
                series = precomputed_strat.get(key)
                if series is not None and len(series) > 0:
                    aligned = series.reindex(common_idx).fillna(0.0).values
                    self._returns_matrix[strat_idx, asset_i, :len(aligned)] = aligned

        self._n_days = n_days
        self._sqrt_tdy = np.sqrt(TRADING_DAYS_PER_YEAR)

    def run(self) -> Tuple[Chromosome, List[float], List[float]]:
        """Run the GA. Returns (best_chromosome, best_history, avg_history)."""
        pop_size = self.params['population_size']
        n_gen = self.params['n_generations']
        elite_count = self.params['elite_count']

        population = self._init_population()
        self._evaluate_all(population)
        population.sort(key=lambda c: c.fitness, reverse=True)

        best_history = []
        avg_history = []

        for gen in range(n_gen):
            next_gen = []

            # Elitism
            for i in range(elite_count):
                next_gen.append(population[i].copy())
                next_gen[-1].fitness = population[i].fitness
                next_gen[-1].portfolio_sharpe = population[i].portfolio_sharpe

            # Fill remaining
            while len(next_gen) < pop_size:
                pa = self._tournament_select(population)
                pb = self._tournament_select(population)
                ca, cb = self._crossover(pa, pb)
                ca = self._mutate(ca)
                cb = self._mutate(cb)
                next_gen.append(ca)
                if len(next_gen) < pop_size:
                    next_gen.append(cb)

            # Evaluate new individuals (elites already have fitness)
            self._evaluate_all(next_gen)
            next_gen.sort(key=lambda c: c.fitness, reverse=True)
            population = next_gen

            best = population[0]
            best_history.append(best.fitness)
            avg_fitness = np.mean([c.fitness for c in population])
            avg_history.append(avg_fitness)

            if (gen + 1) % 50 == 0 or gen == 0:
                log.info(f"    Gen {gen+1:>3}: fitness={best.fitness:+.4f}  "
                         f"port_sharpe={best.portfolio_sharpe:+.4f}  "
                         f"top_wt={best.weight_genes.max():.3f}")

        return population[0], best_history, avg_history

    def _init_population(self) -> List[Chromosome]:
        pop_size = self.params['population_size']
        pop = []
        for _ in range(pop_size):
            idx = self.rng.randint(0, self.n_allowed, size=self.n_genes)
            strats = np.array([self.allowed_strategies[i] for i in idx])
            weights = self.rng.dirichlet(np.ones(self.n_genes))
            pop.append(Chromosome(strats, weights, self.asset_names))
        return pop

    def _tournament_select(self, population: List[Chromosome]) -> Chromosome:
        t_size = self.params['tournament_size']
        idxs = self.rng.choice(len(population), size=t_size, replace=False)
        best = max(idxs, key=lambda i: population[i].fitness)
        return population[best]

    def _crossover(self, pa: Chromosome, pb: Chromosome) -> Tuple[Chromosome, Chromosome]:
        cx_rate = self.params['crossover_rate']
        cx_type = self.params.get('crossover_type', 'two_point')

        # Stage 1: Gate — crossover only happens with probability cx_rate
        if self.rng.random() >= cx_rate:
            # No crossover: return copies of parents unchanged
            return (pa.copy(), pb.copy())

        # Stage 2: Apply crossover operator
        ca_strat = pa.strategy_genes.copy()
        cb_strat = pb.strategy_genes.copy()
        ca_wt = pa.weight_genes.copy()
        cb_wt = pb.weight_genes.copy()

        if cx_type == 'one_point':
            # Single cut point; swap everything after it
            if self.n_genes < 2:
                return (Chromosome(ca_strat, ca_wt, self.asset_names),
                        Chromosome(cb_strat, cb_wt, self.asset_names))
            pt = self.rng.randint(1, self.n_genes)  # 1..n_genes-1
            ca_strat[pt:], cb_strat[pt:] = cb_strat[pt:].copy(), ca_strat[pt:].copy()
            ca_wt[pt:], cb_wt[pt:] = cb_wt[pt:].copy(), ca_wt[pt:].copy()

        elif cx_type == 'two_point':
            # Two cut points; swap the segment between them
            pts = sorted(self.rng.choice(self.n_genes + 1, size=2, replace=False))
            p1, p2 = pts[0], pts[1]
            ca_strat[p1:p2], cb_strat[p1:p2] = cb_strat[p1:p2].copy(), ca_strat[p1:p2].copy()
            ca_wt[p1:p2], cb_wt[p1:p2] = cb_wt[p1:p2].copy(), ca_wt[p1:p2].copy()

        elif cx_type == 'uniform':
            # Each gene swaps with 50% probability
            mask = self.rng.random(self.n_genes) < 0.5
            ca_strat[mask], cb_strat[mask] = cb_strat[mask].copy(), ca_strat[mask].copy()
            ca_wt[mask], cb_wt[mask] = cb_wt[mask].copy(), ca_wt[mask].copy()

        # Renormalise weights
        ca_wt = ca_wt / ca_wt.sum()
        cb_wt = cb_wt / cb_wt.sum()

        return (Chromosome(ca_strat, ca_wt, self.asset_names),
                Chromosome(cb_strat, cb_wt, self.asset_names))

    def _mutate(self, chromo: Chromosome) -> Chromosome:
        strat_mut_rate = self.params['strategy_mutation_rate']
        wt_mut_rate = self.params['weight_mutation_rate']
        wt_mut_sigma = self.params['weight_mutation_sigma']

        m = chromo.copy()
        m.fitness = None

        # Strategy mutation: vectorised random reset
        mask = self.rng.random(self.n_genes) < strat_mut_rate
        n_mut = mask.sum()
        if n_mut > 0:
            new_idxs = self.rng.randint(0, self.n_allowed, size=n_mut)
            m.strategy_genes[mask] = np.array([self.allowed_strategies[i] for i in new_idxs])

        # Weight mutation: vectorised gaussian perturbation
        mask = self.rng.random(self.n_genes) < wt_mut_rate
        n_mut = mask.sum()
        if n_mut > 0:
            m.weight_genes[mask] += self.rng.normal(0, wt_mut_sigma, size=n_mut)
            m.weight_genes = np.maximum(m.weight_genes, 1e-6)

        # Renormalise
        m.weight_genes = m.weight_genes / m.weight_genes.sum()
        return m

    def _evaluate_all(self, population: List[Chromosome]):
        for chromo in population:
            if chromo.fitness is not None:
                continue
            self._compute_fitness(chromo)

    def _compute_fitness(self, chromo: Chromosome):
        # Pure numpy: gather each asset's chosen strategy returns, dot with weights
        strat_genes = chromo.strategy_genes.astype(int)
        # returns_matrix[strat_idx, asset_idx, :] for each asset
        # Select the right strategy row for each asset
        asset_returns = self._returns_matrix[strat_genes, np.arange(self.n_genes), :]
        # asset_returns shape: (n_assets, n_days)

        # Portfolio daily returns = weighted sum
        portfolio_ret = chromo.weight_genes @ asset_returns  # (n_days,)

        if self._n_days < 30:
            chromo.portfolio_sharpe = -10.0
        else:
            mu = np.mean(portfolio_ret)
            sigma = np.std(portfolio_ret)
            if sigma == 0:
                chromo.portfolio_sharpe = 0.0
            else:
                chromo.portfolio_sharpe = (mu / sigma) * self._sqrt_tdy

        chromo.fitness = chromo.portfolio_sharpe
