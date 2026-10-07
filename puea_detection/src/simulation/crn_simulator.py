"""
Temporal cognitive-radio-network simulator.

Model (Chhetry & Marchang, 2021):

    Pr = Pt * r^(-alpha) * exp(a * beta),   a = ln(10) / 10,  beta ~ N(0, sigma^2)

Because 10*log10(exp(a*beta)) = beta, in the dB domain this is

    Pr_dBm = Pt_dBm - 10 * alpha * log10(r) + beta.

Each SU measures received energy = Pr + noise (linear mW), reported in dBm,
and an estimate of its own noise floor (obtained during idle periods).

One transmitter is active per sensing slot: the legitimate PU, or the
attacker when the attacker is active in that slot (PUEA is launched while
the PU is idle).

Data produced per slot is split into:

* **observable**: what the fusion center (FC) receives - ``rss_dbm`` and
  ``noise_est_dbm`` for every SU. Detectors may only read these (plus the
  static SU / PU positions).
* **ground truth**: transmitter identity, position, power, channel state,
  shadowing - used only for labels and evaluation.
* **oracle**: an independent legitimate-PU observation under the *current*
  channel state, used only to measure how far an adaptive profile is from
  the true legitimate distribution (adaptation error).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import stream_key

A_COEF = np.log(10.0) / 10.0

GEOMETRY_STREAM = "geometry"


def db_to_mw(x_db):
    return np.power(10.0, np.asarray(x_db) / 10.0)


def mw_to_db(x_mw):
    return 10.0 * np.log10(np.asarray(x_mw))


def pairwise_distance(tx_xy, su_xy, min_distance: float = 1.0):
    """Distance from transmitter(s) to every SU; tx_xy is (2,) or (T, 2)."""
    tx = np.asarray(tx_xy, dtype=float)
    d = np.linalg.norm(su_xy[None, :, :] - np.atleast_2d(tx)[:, None, :], axis=-1)
    d = np.maximum(d, min_distance)
    return d[0] if tx.ndim == 1 else d


def received_power_dbm(tx_power_dbm, distance, alpha, beta_db):
    """
    Pr = Pt * r^(-alpha) * exp(a * beta) evaluated in dBm.

    ``beta_db`` is the Gaussian shadowing variable in dB; exp(a*beta) with
    a = ln(10)/10 equals 10^(beta/10), i.e. an additive beta in dB.
    """
    return np.asarray(tx_power_dbm) - 10.0 * np.asarray(alpha) * np.log10(distance) + beta_db


@dataclass
class Geometry:
    """Static network layout (fixed for a seed)."""

    su_xy: np.ndarray  # (N, 2)
    pu_xy: np.ndarray  # (2,)
    area_size: float
    noise_offset_db: np.ndarray  # (N,) static per-SU noise-figure offset
    min_distance: float

    @property
    def n_sus(self) -> int:
        return self.su_xy.shape[0]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "su": np.arange(self.n_sus),
                "x": self.su_xy[:, 0],
                "y": self.su_xy[:, 1],
                "noise_offset_db": self.noise_offset_db,
            }
        )


@dataclass
class ChannelSchedule:
    """Per-slot legitimate channel state (length-T arrays)."""

    pu_power_dbm: np.ndarray
    alpha: np.ndarray
    shadow_sigma_db: np.ndarray
    noise_floor_dbm: np.ndarray

    @property
    def n_slots(self) -> int:
        return len(self.pu_power_dbm)


@dataclass
class StreamRandomness:
    """
    All random draws for a stream, drawn up front so that closed-loop runs of
    different detectors see *identical* PU observations and channel noise
    (paired comparison). Shadow fields are unit-variance (AR(1)-filtered).
    """

    shadow_pu: np.ndarray  # (T, N)
    shadow_att: np.ndarray  # (T, N)
    shadow_oracle: np.ndarray  # (T, N)
    noise_fluct: np.ndarray  # (T, N)
    noise_est: np.ndarray  # (T, N)
    noise_fluct_oracle: np.ndarray  # (T, N)
    noise_est_oracle: np.ndarray  # (T, N)
    attack_uniform: np.ndarray  # (T,)  decides whether the attacker is active
    jitter: np.ndarray  # (T,)  attacker power jitter
    rng: np.random.Generator = field(repr=False, default=None)  # for policy-specific draws


def _ar1(z: np.ndarray, rho: float) -> np.ndarray:
    """AR(1) filter along time with unit stationary variance."""
    if rho == 0.0:
        return z
    out = np.empty_like(z)
    out[0] = z[0]
    s = np.sqrt(1.0 - rho**2)
    for t in range(1, z.shape[0]):
        out[t] = rho * out[t - 1] + s * z[t]
    return out


class CRNSimulator:
    """Geometry + propagation. Scenario logic lives in ``scenarios.py``."""

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.seed = int(seed)
        self.geometry = self._make_geometry()

    # ------------------------------------------------------------------ setup
    def _rng(self, name: str) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, stream_key(name)]))

    def _make_geometry(self) -> Geometry:
        g = self.cfg["geometry"]
        ch = self.cfg["channel"]
        rng = self._rng(GEOMETRY_STREAM)
        n = int(g["n_sus"])
        area = float(g["area_size"])
        su_xy = rng.uniform(0.0, area, size=(n, 2))
        noise_offset = rng.normal(0.0, ch["noise_su_offset_std_db"], size=n)
        return Geometry(
            su_xy=su_xy,
            pu_xy=np.asarray(g["pu_xy"], dtype=float),
            area_size=area,
            noise_offset_db=noise_offset,
            min_distance=float(g["min_su_distance"]),
        )

    def new_randomness(self, n_slots: int, stream_name: str) -> StreamRandomness:
        rng = self._rng(stream_name)
        n = self.geometry.n_sus
        rho = float(self.cfg["channel"]["shadow_time_correlation"])

        def field_():
            return _ar1(rng.standard_normal((n_slots, n)), rho)

        return StreamRandomness(
            shadow_pu=field_(),
            shadow_att=field_(),
            shadow_oracle=field_(),
            noise_fluct=rng.standard_normal((n_slots, n)),
            noise_est=rng.standard_normal((n_slots, n)),
            noise_fluct_oracle=rng.standard_normal((n_slots, n)),
            noise_est_oracle=rng.standard_normal((n_slots, n)),
            attack_uniform=rng.uniform(size=n_slots),
            jitter=rng.standard_normal(n_slots),
            rng=rng,
        )

    # ------------------------------------------------------------ propagation
    def _measure(self, t, tx_xy, tx_power_dbm, sched: ChannelSchedule, shadow_unit, fluct, est):
        ch = self.cfg["channel"]
        geo = self.geometry
        dist = pairwise_distance(tx_xy, geo.su_xy, geo.min_distance)
        shadow_db = sched.shadow_sigma_db[t] * shadow_unit
        pr_dbm = received_power_dbm(tx_power_dbm, dist, sched.alpha[t], shadow_db)
        noise_dbm = sched.noise_floor_dbm[t] + geo.noise_offset_db + ch["noise_fluctuation_std_db"] * fluct
        rss_dbm = mw_to_db(db_to_mw(pr_dbm) + db_to_mw(noise_dbm))
        noise_est_dbm = sched.noise_floor_dbm[t] + geo.noise_offset_db + ch["noise_estimate_error_std_db"] * est
        return dist, shadow_db, rss_dbm, noise_est_dbm

    def observe_slot(self, t: int, tx_xy, tx_power_dbm: float, is_attacker: bool,
                     sched: ChannelSchedule, rnd: StreamRandomness) -> dict:
        """Simulate one sensing slot (and the evaluation-only oracle PU observation)."""
        shadow = rnd.shadow_pu[t]
        if is_attacker:
            # Shadowing of the attacker->SU links is correlated with that of the
            # PU->SU links according to the attacker's displacement from the PU
            # (Gudmundson exponential model): a transmitter close to the PU sees
            # nearly the same obstacles. d_corr = None -> independent fields.
            d_corr = self.cfg["channel"].get("shadow_decorrelation_distance")
            c = 0.0
            if d_corr:
                c = float(np.exp(-np.linalg.norm(np.asarray(tx_xy, float) - self.geometry.pu_xy) / d_corr))
            shadow = c * rnd.shadow_pu[t] + np.sqrt(1.0 - c ** 2) * rnd.shadow_att[t]
        dist, shadow_db, rss, noise_est = self._measure(
            t, tx_xy, tx_power_dbm, sched, shadow, rnd.noise_fluct[t], rnd.noise_est[t]
        )
        _, _, o_rss, o_noise_est = self._measure(
            t, self.geometry.pu_xy, sched.pu_power_dbm[t], sched,
            rnd.shadow_oracle[t], rnd.noise_fluct_oracle[t], rnd.noise_est_oracle[t],
        )
        return {
            "rss_dbm": rss,
            "noise_est_dbm": noise_est,
            "snr_db": rss - noise_est,
            "distance": dist,
            "shadow_db": shadow_db,
            "oracle_rss_dbm": o_rss,
            "oracle_noise_est_dbm": o_noise_est,
        }


# ---------------------------------------------------------------------------
# Stream container
# ---------------------------------------------------------------------------

SCALAR_COLUMNS = [
    "slot", "label", "transmitter", "tx_x", "tx_y", "tx_power_dbm",
    "pu_power_dbm", "alpha", "shadow_sigma_db", "noise_floor_dbm",
    "attack_progress", "attack_phase",
]
MATRIX_FIELDS = [
    "rss_dbm", "noise_est_dbm", "snr_db", "distance", "shadow_db",
    "oracle_rss_dbm", "oracle_noise_est_dbm",
]
#: The only per-slot fields a detector is allowed to read (plus geometry).
OBSERVABLE_FIELDS = ("rss_dbm", "noise_est_dbm")


@dataclass
class StreamData:
    """
    A realised sensing stream. ``scalars`` holds one row per slot (ground
    truth + metadata); ``matrices`` holds (T, N) per-SU arrays.
    """

    scenario: str
    scalars: pd.DataFrame
    matrices: dict

    @property
    def n_slots(self) -> int:
        return len(self.scalars)

    @property
    def labels(self) -> np.ndarray:
        return self.scalars["label"].to_numpy(dtype=int)

    def observable(self) -> dict:
        """Leakage guard: returns only FC-observable per-SU measurements."""
        return {k: self.matrices[k] for k in OBSERVABLE_FIELDS}

    def oracle(self) -> dict:
        """Evaluation-only oracle PU measurements (never passed to detectors)."""
        return {"rss_dbm": self.matrices["oracle_rss_dbm"],
                "noise_est_dbm": self.matrices["oracle_noise_est_dbm"]}

    # -------------------------------------------------------------- persist
    def to_frame(self) -> pd.DataFrame:
        parts = [self.scalars.reset_index(drop=True)]
        for name in MATRIX_FIELDS:
            m = self.matrices[name]
            cols = [f"{name}__su{i:02d}" for i in range(m.shape[1])]
            parts.append(pd.DataFrame(m, columns=cols))
        return pd.concat(parts, axis=1)

    @classmethod
    def from_frame(cls, scenario: str, df: pd.DataFrame) -> "StreamData":
        scalars = df[SCALAR_COLUMNS].copy()
        matrices = {}
        for name in MATRIX_FIELDS:
            cols = sorted(c for c in df.columns if c.startswith(f"{name}__su"))
            matrices[name] = df[cols].to_numpy(dtype=float)
        return cls(scenario=scenario, scalars=scalars, matrices=matrices)

    def save(self, path: str) -> None:
        self.to_frame().to_csv(path, index=False, float_format="%.5f", compression="gzip")

    @classmethod
    def load(cls, scenario: str, path: str) -> "StreamData":
        return cls.from_frame(scenario, pd.read_csv(path, compression="gzip"))

    @classmethod
    def from_slots(cls, scenario: str, rows: list, slot_obs: list) -> "StreamData":
        scalars = pd.DataFrame(rows, columns=SCALAR_COLUMNS)
        matrices = {k: np.vstack([o[k] for o in slot_obs]) for k in MATRIX_FIELDS}
        return cls(scenario=scenario, scalars=scalars, matrices=matrices)

    @classmethod
    def concat(cls, scenario: str, streams: list) -> "StreamData":
        scalars = pd.concat([s.scalars for s in streams], ignore_index=True)
        scalars["slot"] = np.arange(len(scalars))
        matrices = {k: np.vstack([s.matrices[k] for s in streams]) for k in MATRIX_FIELDS}
        return cls(scenario=scenario, scalars=scalars, matrices=matrices)
