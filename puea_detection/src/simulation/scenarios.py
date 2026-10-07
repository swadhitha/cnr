"""
Scenario definitions: legitimate channel drift schedules and attacker policies.

Legitimate drift changes *how the PU is observed* (power, shadowing, noise)
but never *where the PU is*. Attackers always transmit from a location other
than the PU's; they differ in how hard they try to look like the PU.

Attacker policies expose

* ``slot_params(t, sched) -> AttackSlot``: whether the attacker transmits in
  slot ``t`` and with which position / power;
* ``feedback(flagged)``: called after the detector has decided on an
  attacker slot. Only closed-loop policies (poisoning) use it. The feedback
  models something a real attacker can observe: whether the SUs vacated the
  channel (not flagged) or kept using it (flagged).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crn_simulator import (
    ChannelSchedule,
    CRNSimulator,
    StreamData,
    pairwise_distance,
)


# ---------------------------------------------------------------------------
# Legitimate channel schedule
# ---------------------------------------------------------------------------

def ramp(n_slots: int, start: int, end: int, delta: float) -> np.ndarray:
    """0 before ``start``, linear to ``delta`` at ``end``, held afterwards."""
    t = np.arange(n_slots, dtype=float)
    if end <= start:
        return np.where(t >= start, delta, 0.0)
    return delta * np.clip((t - start) / (end - start), 0.0, 1.0)


DRIFT_TARGETS = {
    "pu_power_db": "pu_power_dbm",
    "shadow_sigma_db": "shadow_sigma_db",
    "noise_floor_db": "noise_floor_dbm",
    "path_loss_exponent": "alpha",
}


def build_schedule(cfg: dict, scenario_cfg: dict, n_slots: int) -> ChannelSchedule:
    ch = cfg["channel"]
    sched = {
        "pu_power_dbm": np.full(n_slots, float(cfg["pu"]["power_dbm"])),
        "alpha": np.full(n_slots, float(ch["path_loss_exponent"])),
        "shadow_sigma_db": np.full(n_slots, float(ch["shadow_sigma_db"])),
        "noise_floor_dbm": np.full(n_slots, float(ch["noise_floor_dbm"])),
    }
    for key, spec in (scenario_cfg.get("drift") or {}).items():
        target = DRIFT_TARGETS[key]
        sched[target] = sched[target] + ramp(n_slots, int(spec["start"]), int(spec["end"]), float(spec["delta"]))
    return ChannelSchedule(**sched)


def drift_window(scenario_cfg: dict):
    """(start, end) slots spanned by all drift ramps, or None if no drift."""
    drift = scenario_cfg.get("drift") or {}
    if not drift:
        return None
    return (min(int(s["start"]) for s in drift.values()), max(int(s["end"]) for s in drift.values()))


# ---------------------------------------------------------------------------
# Attacker helpers
# ---------------------------------------------------------------------------

def matched_power_dbm(att_xy, pu_power_dbm: float, geometry, alpha: float) -> float:
    """
    Transmit power at which the attacker's *mean* received power over the SUs
    (in dB) equals the PU's. This assumes a strong attacker that knows the SU
    positions and the path-loss exponent.
    """
    d_att = pairwise_distance(att_xy, geometry.su_xy, geometry.min_distance)
    d_pu = pairwise_distance(geometry.pu_xy, geometry.su_xy, geometry.min_distance)
    return float(pu_power_dbm + 10.0 * alpha * (np.mean(np.log10(d_att)) - np.mean(np.log10(d_pu))))


def point_toward(origin, target, distance: float) -> np.ndarray:
    """Point at ``distance`` from ``origin`` in the direction of ``target``."""
    origin = np.asarray(origin, dtype=float)
    v = np.asarray(target, dtype=float) - origin
    return origin + v / np.linalg.norm(v) * distance


@dataclass
class AttackSlot:
    active: bool
    xy: np.ndarray | None = None
    power_dbm: float = np.nan
    progress: float = np.nan
    phase: str = "none"


class AttackPolicy:
    closed_loop = False

    def __init__(self, sim: CRNSimulator, spec: dict, rnd):
        self.sim, self.spec, self.rnd = sim, spec, rnd
        self.cfg = sim.cfg
        self.geometry = sim.geometry

    def is_active(self, t: int, onset: int, rate: float) -> bool:
        return t >= onset and self.rnd.attack_uniform[t] < rate

    def slot_params(self, t: int, sched: ChannelSchedule) -> AttackSlot:  # pragma: no cover
        raise NotImplementedError

    def feedback(self, flagged: bool) -> None:
        return None


class NoAttack(AttackPolicy):
    def slot_params(self, t, sched):
        return AttackSlot(active=False)


class BasicAttack(AttackPolicy):
    """Fixed location, fixed (not PU-matched) power."""

    def slot_params(self, t, sched):
        if not self.is_active(t, self.spec["onset"], self.spec["rate"]):
            return AttackSlot(active=False)
        a = self.cfg["attacker"]
        return AttackSlot(True, np.asarray(a["basic_xy"], float), float(a["basic_power_dbm"]), 0.0, "attack")


class PowerMatchingAttack(AttackPolicy):
    """Fixed location; power chosen so mean RSS over the SUs matches the PU."""

    def slot_params(self, t, sched):
        if not self.is_active(t, self.spec["onset"], self.spec["rate"]):
            return AttackSlot(active=False)
        xy = np.asarray(self.cfg["attacker"]["basic_xy"], float)
        p = matched_power_dbm(xy, sched.pu_power_dbm[t], self.geometry, sched.alpha[t])
        return AttackSlot(True, xy, p, 1.0, "attack")


class GradualAttack(AttackPolicy):
    """
    Open-loop evasion. PU-likeness q steps through ``stages``: the attacker
    moves from basic_xy toward a point ``min_distance_to_pu`` from the PU and
    its power moves from basic_power_dbm toward the matched power.
    """

    def params_at(self, q: float, pu_power: float, alpha: float):
        a = self.cfg["attacker"]
        start = np.asarray(a["basic_xy"], float)
        near = point_toward(self.geometry.pu_xy, start, self.spec["min_distance_to_pu"])
        xy = start + q * (near - start)
        power = (1 - q) * a["basic_power_dbm"] + q * matched_power_dbm(xy, pu_power, self.geometry, alpha)
        return xy, float(power)

    def slot_params(self, t, sched):
        onset = self.spec["onset"]
        if not self.is_active(t, onset, self.spec["rate"]):
            return AttackSlot(active=False)
        stages = self.spec["stages"]
        span = (sched.n_slots - onset) / len(stages)
        q = float(stages[min(int((t - onset) // span), len(stages) - 1)])
        xy, power = self.params_at(q, sched.pu_power_dbm[t], sched.alpha[t])
        return AttackSlot(True, xy, power, q, "attack")


class StealthyAttack(AttackPolicy):
    """
    Adaptive attacker close to the PU that re-matches the PU's *current*
    power every slot (tracking legitimate drift) with small jitter, and
    transmits rarely.
    """

    def slot_params(self, t, sched):
        if not self.is_active(t, self.spec["onset"], self.spec["rate"]):
            return AttackSlot(active=False)
        centre = np.full(2, self.geometry.area_size / 2.0)
        xy = point_toward(self.geometry.pu_xy, centre, self.spec["distance_from_pu"])
        p = matched_power_dbm(xy, sched.pu_power_dbm[t], self.geometry, sched.alpha[t])
        p += self.spec["power_jitter_db"] * self.rnd.jitter[t]
        return AttackSlot(True, xy, float(p), 1.0, "attack")


class PoisoningAttack(AttackPolicy):
    """
    Closed-loop boiling-frog poisoning of an adaptive PU profile.

    Progress p = 0 is the best PU mimic the attacker can produce (placed
    ``start_distance_from_pu`` from the PU, power-matched); p = 1 is its
    target operating point: (basic_xy, basic_power_dbm) by default, or a
    point ``target_distance_from_pu`` from the PU at matched power +
    ``target_power_offset_db`` if those keys are given. During the poison
    phase p rises by ``step_up`` after every slot that was *not* flagged and
    falls by ``step_down`` after every flagged slot. In the attack phase it
    transmits at the target point regardless of p, which measures whether
    poisoning made the target attack acceptable.
    """

    closed_loop = True

    def __init__(self, sim, spec, rnd):
        super().__init__(sim, spec, rnd)
        self.progress = 0.0
        a = self.cfg["attacker"]
        basic_xy = np.asarray(a["basic_xy"], float)
        if "target_distance_from_pu" in spec:
            # Target inside the anchor's resolution band, power = matched + offset.
            self.target_xy = point_toward(self.geometry.pu_xy, basic_xy, spec["target_distance_from_pu"])
            self._target_offset = float(spec["target_power_offset_db"])
        else:
            self.target_xy = basic_xy
            self._target_offset = None
        self._basic_power = float(a["basic_power_dbm"])
        self.start_xy = point_toward(self.geometry.pu_xy, self.target_xy, spec["start_distance_from_pu"])
        self._last_phase = "none"

    def target_power(self, sched, t) -> float:
        if self._target_offset is None:
            return self._basic_power
        return matched_power_dbm(self.target_xy, sched.pu_power_dbm[t], self.geometry, sched.alpha[t]) \
            + self._target_offset

    def slot_params(self, t, sched):
        s = self.spec
        if t < s["poison_start"]:
            self._last_phase = "none"
            return AttackSlot(active=False)
        if t < s["attack_start"]:
            self._last_phase = "poison"
            if self.rnd.attack_uniform[t] >= s["poison_rate"]:
                return AttackSlot(active=False)
            p = self.progress
            xy = self.start_xy + p * (self.target_xy - self.start_xy)
            mimic = matched_power_dbm(xy, sched.pu_power_dbm[t], self.geometry, sched.alpha[t])
            power = (1 - p) * mimic + p * self.target_power(sched, t)
            return AttackSlot(True, xy, float(power), p, "poison")
        self._last_phase = "attack"
        if self.rnd.attack_uniform[t] >= s["attack_rate"]:
            return AttackSlot(active=False)
        return AttackSlot(True, self.target_xy.copy(), self.target_power(sched, t), 1.0, "attack")

    def feedback(self, flagged: bool) -> None:
        if self._last_phase != "poison":
            return
        step = -self.spec["step_down"] if flagged else self.spec["step_up"]
        self.progress = float(np.clip(self.progress + step, 0.0, 1.0))


class GenericRandomAttacker(AttackPolicy):
    """Training-data attacker: every slot, random position and random power."""

    def slot_params(self, t, sched):
        st = self.cfg["streams"]
        geo = self.geometry
        rng = self.rnd.rng
        while True:
            xy = rng.uniform(0.0, geo.area_size, size=2)
            if np.linalg.norm(xy - geo.pu_xy) >= st["xgb_train_attacker_min_distance"]:
                break
        lo, hi = st["xgb_train_attacker_power_offset_db"]
        p = matched_power_dbm(xy, sched.pu_power_dbm[t], geo, sched.alpha[t]) + rng.uniform(lo, hi)
        return AttackSlot(True, xy, float(p), np.nan, "train")


POLICIES = {
    None: NoAttack,
    "basic": BasicAttack,
    "power_matching": PowerMatchingAttack,
    "gradual": GradualAttack,
    "stealthy": StealthyAttack,
    "poisoning": PoisoningAttack,
    "generic_random": GenericRandomAttacker,
}


# ---------------------------------------------------------------------------
# Stream driver
# ---------------------------------------------------------------------------

class ScenarioStream:
    """
    Slot-by-slot driver for one scenario. Open-loop scenarios can be fully
    generated with :meth:`generate`; closed-loop scenarios (poisoning) must
    be stepped by the experiment runner, which calls :meth:`feedback` after
    each attacker slot with the detector's decision.
    """

    def __init__(self, sim: CRNSimulator, name: str, scenario_cfg: dict, n_slots: int,
                 stream_name: str | None = None):
        self.sim, self.name, self.scenario_cfg = sim, name, scenario_cfg
        self.sched = build_schedule(sim.cfg, scenario_cfg, n_slots)
        self.rnd = sim.new_randomness(n_slots, stream_name or f"test/{name}")
        attack = scenario_cfg.get("attack") or {}
        self.policy = POLICIES[attack.get("type")](sim, attack, self.rnd)
        self._pending_attack = False

    @property
    def closed_loop(self) -> bool:
        return self.policy.closed_loop

    @property
    def n_slots(self) -> int:
        return self.sched.n_slots

    def slot(self, t: int):
        """Returns (scalar_row, per-SU observation dict) for slot ``t``."""
        a = self.policy.slot_params(t, self.sched)
        s = self.sched
        if a.active:
            tx_xy, tx_p, who = a.xy, a.power_dbm, "attacker"
        else:
            tx_xy, tx_p, who = self.sim.geometry.pu_xy, s.pu_power_dbm[t], "pu"
        obs = self.sim.observe_slot(t, tx_xy, tx_p, a.active, s, self.rnd)
        row = [t, int(a.active), who, float(tx_xy[0]), float(tx_xy[1]), float(tx_p),
               float(s.pu_power_dbm[t]), float(s.alpha[t]), float(s.shadow_sigma_db[t]),
               float(s.noise_floor_dbm[t]), float(a.progress), a.phase]
        self._pending_attack = a.active
        return row, obs

    def feedback(self, flagged: bool) -> None:
        """Report the detector's decision for the slot just produced."""
        if self._pending_attack:
            self.policy.feedback(bool(flagged))
        self._pending_attack = False

    def generate(self) -> StreamData:
        if self.closed_loop:
            raise RuntimeError(f"Scenario '{self.name}' is closed-loop; step it with a detector.")
        rows, obs = zip(*(self.slot(t) for t in range(self.n_slots)))
        return StreamData.from_slots(self.name, list(rows), list(obs))


def generate_xgb_training_stream(sim: CRNSimulator) -> StreamData:
    """Stable PU slots + generic random attackers (see config 'streams')."""
    st = sim.cfg["streams"]
    pu = ScenarioStream(sim, "xgb_train_pu", {}, st["n_xgb_train_pu"], "xgb_train/pu").generate()
    att_stream = ScenarioStream(sim, "xgb_train_attack", {}, st["n_xgb_train_attack"], "xgb_train/attack")
    att_stream.policy = GenericRandomAttacker(sim, {}, att_stream.rnd)
    att = att_stream.generate()
    return StreamData.concat("xgb_train", [pu, att])


def generate_calibration_stream(sim: CRNSimulator) -> StreamData:
    """PU-only stable stream used for thresholds and the initial profile."""
    n = sim.cfg["streams"]["n_calibration_slots"]
    return ScenarioStream(sim, "calibration", {}, n, "calibration").generate()
