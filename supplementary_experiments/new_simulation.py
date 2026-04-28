
from __future__ import annotations

import heapq
import itertools
import math
import argparse
import csv
from pathlib import Path
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Iterable
from collections import deque

import numpy as np


                                                                               
       
                                                                               

class GateType(Enum):
    LOCAL = auto()
    REMOTE = auto()


class GateState(Enum):
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    DONE = auto()


class NetworkType(Enum):
    MPQN = "mpqn"
    QMSN = "qmsn"
    STATIC_LINE = "static_line"
    STATIC_GRID = "static_grid"


class EventType(Enum):
    LOCAL_DONE = auto()                  
    REMOTE_DONE = auto()                 
    ENT_RDY = auto()                                           
    ENT_EXP = auto()                             


                                                                               
                       
                                                                               

@dataclass(order=True)
class SimEvent:
    time: float
    seq: int
    etype: EventType = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)


                                                                               
                                  
                                                                               

@dataclass
class Gate:
    gid: str
    gtype: GateType
    qpus: Tuple[int, ...]                                              
    data_qubits: Dict[int, List[int]]                                                            
    deps: Set[str] = field(default_factory=set)                                     

    state: GateState = GateState.PENDING
    duration: float = 0.0                                                          

                                      
    ent_id: Optional[str] = None
    ent_ready: bool = False

                                         
    est: float = 0.0
    eft: float = 0.0
    criticality: float = 0.0


@dataclass
class EntanglementRecord:
    ent_id: str
    gid: str
    i: int
    j: int
    tready: float
    ttl: float
    fidelity: float = 1.0
    occupied: bool = True
    consumed: bool = False
    expired: bool = False


@dataclass
class InflightEntRequest:
    ent_id: str
    gid: str
    i: int
    j: int
    path: Tuple[int, ...]
    base_tready: float
    start_time: float
    version: int = 0  # 新增：版本号，用于干扰取消

@dataclass
class ExperimentParams:
    t_1g: float = 1.0
    t_2g: float = 100.0
    t_retry: float = 1.0
    p_ent: float = 1.0e-3
    p_swi: float = 0.99
    ent_time_mode: str = "sample"                       
    grid_rows: int = 0
    grid_cols: int = 0
    grid_routing: str = "xy"
    t_deco: float = 2100.0
    t_c: float = 1000.0
    # t_q_init: float = 1000.0
    t_init: float = 1000.0
    # t_e: float = 1000.0
    # gamma_max_hz: float = 2_000_000.0
    p_swap: float = 0.9
    delta_default: float = 1.0
    p_e: float = 0.8
    p_c: float = 0.1
    p_t: float = 0.85
    p_d: float = 0.8

                                                         
    f_1g: float = 0.9999
    f_2g_local: float = 0.995
    f_2g_remote: float = 0.990
    f_ent_base: float = 0.95
    f_link: float = 0.995
    f_swap_fid: float = 0.98
    f_switch: float = 0.9995
    f_trans: float = 0.9999
    t_fid_decay: float = t_deco

    qmsn_reconfiguration_model: bool = False

    compute_p_ent: bool = True   # 新增：控制是否自动计算 p_ent

    def __post_init__(self):
        if self.compute_p_ent:
            self.p_ent = compute_p_ent_from_components(p_e=self.p_e, p_c=self.p_c, p_t=self.p_t, p_d=self.p_d)


@dataclass
class QPU:
    qid: int
    n_data_qubits: int
    n_mem_slots: int
    n_ifaces: int

    qbusy: Dict[int, bool] = field(init=False)
    mem_free: int = field(init=False)
    iface_free: int = field(init=False)

    def __post_init__(self):
        self.qbusy = {q: False for q in range(self.n_data_qubits)}
        self.mem_free = self.n_mem_slots
        self.iface_free = self.n_ifaces

    def data_free(self, qubits: Iterable[int]) -> bool:
        return all(not self.qbusy[q] for q in qubits)

    def mark_data(self, qubits: Iterable[int], busy: bool):
        for q in qubits:
            self.qbusy[q] = busy


                                                                               
                                                                               

class OrchestratorSim:
    def __init__(
        self,
        gates: Dict[str, Gate],
        qpus: Dict[int, QPU],
        succ: Dict[str, Set[str]],
        preds: Dict[str, List[str]],
        topo: List[str],
        network: NetworkType = NetworkType.MPQN,
        params: Optional[ExperimentParams] = None,
        rng_seed: int = 0,
        trace: bool = False,
    ):
        self.gates = gates
        self.qpus = qpus
        self.succ = succ
        self.preds = preds
        self.topo = topo
        self.network = network
        self.params = params or ExperimentParams()
        self._validate_topology_config()

                              
        self.now = 0.0
        self._seq = itertools.count()
        self._pq: List[SimEvent] = []

                            
        self.ent: Dict[str, EntanglementRecord] = {}
        self.inflight: Dict[str, InflightEntRequest] = {}

                                                     
        self.rng = np.random.default_rng(rng_seed)

                                                         
        self.ent_ttl = self.params.t_deco

        self._ent_ctr = itertools.count()

                           
        self.delta_default = self.params.delta_default

                                                                                
        self.computational_fidelity = 1.0

                                  
        self.trace = trace
        self._last_ent_choice: Optional[dict] = None


                                                                               
                   
                                                                               

    def _act(self, msg: str):
        if self.trace:
            print(f"[t={self.now:8.3f}] {msg}")

                                                                               
                         
                                                                               

    def push_event(self, t: float, etype: EventType, **payload):
        if t < self.now - 1e-12:
            raise RuntimeError(f"Scheduled event in the past: t={t}, now={self.now}, {etype}")
        heapq.heappush(self._pq, SimEvent(time=t, seq=next(self._seq), etype=etype, payload=payload))

    def pop_events_until_trigger(self) -> List[SimEvent]:
        if not self._pq:
            return []
        t = self._pq[0].time
        self.now = t

        batch: List[SimEvent] = []
        while self._pq and math.isclose(self._pq[0].time, t, rel_tol=0.0, abs_tol=1e-12):
            batch.append(heapq.heappop(self._pq))

        return batch

    def gen_ent_id(self) -> str:
        return f"e{next(self._ent_ctr)}"

    def dag_exhausted(self) -> bool:
        return all(g.state == GateState.DONE for g in self.gates.values())

                                                                               
                    
                                                                               

    def initialize(self):
        self.now = 0.0
        self._pq.clear()
        self.ent.clear()
        self.inflight.clear()
        self.computational_fidelity = 1.0

                    
        for q in self.qpus.values():
            q.__post_init__()

                     
        for g in self.gates.values():
            g.ent_id = None
            g.ent_ready = False
            g.est = 0.0
            g.eft = 0.0
            g.criticality = 0.0

                                                              
            if not g.deps and g.gtype == GateType.LOCAL:
                g.state = GateState.READY
            else:
                g.state = GateState.PENDING

        self.recompute_criticality()
        self.recompute_estimates()
        self._act("INIT done")
        self._act(f"ENT time mode = {self.params.ent_time_mode}")

                                                                               
                                                              
                                                                               

    def _clip_prob(self, p: float) -> float:
        return min(max(p, 1e-12), 1.0)

    def _draw_geometric(self, p: float) -> int:
        return int(self.rng.geometric(self._clip_prob(p)))

    def _n_qpus(self) -> int:
        return max(len(self.qpus), 1)

    def _validate_topology_config(self):
        if self.network != NetworkType.STATIC_GRID:
            return

        rows = int(self.params.grid_rows)
        cols = int(self.params.grid_cols)
        routing = str(self.params.grid_routing).lower()
        n_qpus = len(self.qpus)

        if rows <= 0 or cols <= 0:
            raise ValueError(
                "STATIC_GRID requires positive grid dimensions: "
                f"grid_rows={rows}, grid_cols={cols}."
            )
        if rows * cols != n_qpus:
            raise ValueError(
                "STATIC_GRID requires grid_rows * grid_cols == number of QPUs: "
                f"{rows}*{cols} != {n_qpus}."
            )
        if routing != "xy":
            raise ValueError(f"Unsupported grid_routing='{self.params.grid_routing}'. Only 'xy' is supported.")

    def _grid_coords(self, qid: int) -> Tuple[int, int]:
        if qid not in self.qpus:
            raise ValueError(f"QPU id {qid} not found in qpus for STATIC_GRID.")
        cols = int(self.params.grid_cols)
        if cols <= 0:
            raise ValueError(f"Invalid grid_cols={cols} for STATIC_GRID.")
        row, col = divmod(qid, cols)
        rows = int(self.params.grid_rows)
        if row < 0 or row >= rows:
            raise ValueError(f"QPU id {qid} maps outside grid bounds for {rows}x{cols}.")
        return row, col

    def _grid_qid(self, row: int, col: int) -> int:
        rows = int(self.params.grid_rows)
        cols = int(self.params.grid_cols)
        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise ValueError(f"Grid coordinate out of bounds: ({row},{col}) for {rows}x{cols}.")
        qid = row * cols + col
        if qid not in self.qpus:
            raise ValueError(f"Grid coordinate ({row},{col}) maps to missing qid={qid}.")
        return qid

    def _grid_path_xy(self, i: int, j: int) -> Tuple[int, ...]:
        if str(self.params.grid_routing).lower() != "xy":
            raise ValueError(f"Unsupported grid_routing='{self.params.grid_routing}'. Only 'xy' is supported.")

        r, c = self._grid_coords(i)
        tr, tc = self._grid_coords(j)
        path: List[int] = [i]

        step_r = 1 if tr > r else -1
        while r != tr:
            r += step_r
            path.append(self._grid_qid(r, c))

        step_c = 1 if tc > c else -1
        while c != tc:
            c += step_c
            path.append(self._grid_qid(r, c))

        return tuple(path)

    def _qmsn_stages(self) -> int:
        return max(1, int(math.ceil(math.log2(max(self._n_qpus(), 2)))))

    def _mpqn_stages(self) -> int:
        lg = math.log2(max(self._n_qpus(), 2))
        extra = int(math.ceil(math.log2(max(lg - 1.0, 1.0))))
        return max(1, int(math.ceil(lg)) + extra)

    def _network_path(self, i: int, j: int) -> Tuple[int, ...]:
        if self.network == NetworkType.STATIC_LINE:
            lo, hi = min(i, j), max(i, j)
            return tuple(range(lo, hi + 1))
        if self.network == NetworkType.STATIC_GRID:
            return self._grid_path_xy(i, j)
        return (i, j)

    def _per_attempt_success_prob(self) -> float:
        if self.network == NetworkType.QMSN:
            stages = self._qmsn_stages()
        elif self.network == NetworkType.MPQN:
            stages = self._mpqn_stages()
        else:
            stages = 0

        if stages == 0:
            return self._clip_prob(self.params.p_ent)
        return self._clip_prob(self.params.p_ent * (self.params.p_swi ** (2 * stages)))

    def sample_entanglement_time(self, i: int, j: int) -> float:
        if self.network == NetworkType.MPQN:
            p_attempt = self._per_attempt_success_prob()
            n_attempt = self._draw_geometric(p_attempt)
            dt = self.params.t_init + float(n_attempt) * self.params.t_retry
            # print(f"[MPQN] now={self.now:.3f}, dt={dt:.3f}")
            return dt

        if self.network == NetworkType.QMSN:
            p_attempt = self._per_attempt_success_prob()
            if self.params.qmsn_reconfiguration_model:
                n_attempt = self._draw_geometric(p_attempt)
                dt = self.params.t_init + float(n_attempt) * self.params.t_retry
                # print(f"[QMSN-reconfig] now={self.now:.3f}, dt={dt:.3f}")
                # print(f"[QMSN-reconfig] t_m_init={self.params.t_m_init:.1f}, "
                # f"t_retry={self.params.t_retry:.1f}, n_attempt={n_attempt}, "
                # f"dt={dt:.1f}, now={self.now:.1f}")
                return dt
            else:
                attempts_per_slot = max(1, int(self.params.t_c // self.params.t_retry))
                p_slot = self._clip_prob(1.0 - ((1.0 - p_attempt) ** attempts_per_slot))
                n_slots = self._draw_geometric(p_slot)
                slot_len = self.params.t_c + self.params.t_init
                next_slot_start = math.ceil(self.now / slot_len) * slot_len
                align_wait = max(0.0, next_slot_start - self.now)
                total_slots_time = float(n_slots) * slot_len
                total_time = align_wait + total_slots_time

                # print(f"[QMSN-slot] now={self.now:.3f}, align_wait={align_wait:.3f}, "
                # f"n_slots={n_slots}, slot_len={slot_len:.3f}, slots_time={total_slots_time:.3f}, "
                # f"total={total_time:.3f}")
                return total_time

        if self.network not in (NetworkType.STATIC_LINE, NetworkType.STATIC_GRID):
            raise ValueError(f"Unsupported network for entanglement sampling: {self.network.value}")

                                                                                   
        path = self._network_path(i, j)
        d = max(len(path) - 1, 1)

        attempts_per_slot = max(1, int(self.params.t_c // self.params.t_retry))
        p_link_slot = 1.0 - ((1.0 - self.params.p_ent) ** attempts_per_slot)
        p_slot = self._clip_prob((p_link_slot ** d) * (self.params.p_swap ** max(d - 1, 0)))
        n_slots = self._draw_geometric(p_slot)
        return float(n_slots) * self.params.t_c

    def mean_entanglement_time(self, i: Optional[int] = None, j: Optional[int] = None) -> float:
        if self.network == NetworkType.MPQN:
            p_attempt = self._per_attempt_success_prob()
            return self.params.t_init + (self.params.t_retry / p_attempt)

        if self.network == NetworkType.QMSN:
            p_attempt = self._per_attempt_success_prob()
            if self.params.qmsn_reconfiguration_model:
                return self.params.t_init + (self.params.t_retry / p_attempt)
            else:
                attempts_per_slot = max(1, int(self.params.t_c // self.params.t_retry))
                p_slot = self._clip_prob(1.0 - ((1.0 - p_attempt) ** attempts_per_slot))
                slot_len = self.params.t_c + self.params.t_init
                return (0.5 * slot_len) + (slot_len / p_slot)

        if self.network not in (NetworkType.STATIC_LINE, NetworkType.STATIC_GRID):
            raise ValueError(f"Unsupported network for entanglement mean-time: {self.network.value}")
        if i is None or j is None:
            d = 1
        else:
            path = self._network_path(i, j)
            d = max(len(path) - 1, 1)
        attempts_per_slot = max(1, int(self.params.t_c // self.params.t_retry))
        p_link_slot = 1.0 - ((1.0 - self.params.p_ent) ** attempts_per_slot)
        p_slot = self._clip_prob((p_link_slot ** d) * (self.params.p_swap ** max(d - 1, 0)))
        return self.params.t_c / p_slot

    def entanglement_time(self, i: int, j: int) -> float:
        mode = self.params.ent_time_mode.lower()
        if mode == "sample":
            return self.sample_entanglement_time(i=i, j=j)
        if mode == "mean":
            return self.mean_entanglement_time(i=i, j=j)
        raise ValueError(f"Unknown ent_time_mode: {self.params.ent_time_mode}")

    def _clip_unit(self, x: float) -> float:
        return min(max(float(x), 0.0), 1.0)

    def entanglement_ready_fidelity(self, i: int, j: int) -> float:
        F = float(self.params.f_ent_base)

        if self.network in (NetworkType.STATIC_LINE, NetworkType.STATIC_GRID):
            path = self._network_path(i, j)
            d = max(len(path) - 1, 1)
            F *= self.params.f_link ** d
            F *= self.params.f_swap_fid ** max(d - 1, 0)
            return self._clip_unit(F)

        if self.network == NetworkType.MPQN:
            n_stage = self._qmsn_stages()
        elif self.network == NetworkType.QMSN:
            n_stage = self._qmsn_stages()
        else:
            n_stage = 0

        F *= self.params.f_switch ** n_stage
        F *= self.params.f_trans ** n_stage
        return self._clip_unit(F)

    def entanglement_wait_decay(self, wait_time: float) -> float:
        t = max(float(wait_time), 0.0)
        if self.params.t_fid_decay <= 0:
            return 1.0
        return self._clip_unit(math.exp(-t / self.params.t_fid_decay))

                                                                               
                                      
                                                                               

    def recompute_criticality(self):
        for gid in reversed(self.topo):
            g = self.gates[gid]
            succ_crit = 0.0
            for sid in self.succ.get(gid, set()):
                succ_crit = max(succ_crit, self.gates[sid].criticality)
            g.criticality = g.duration + succ_crit

    def recompute_estimates(self):
        for gid in self.topo:
            g = self.gates[gid]

                                                                  
            if g.state in (GateState.RUNNING, GateState.DONE):
                continue

                                                  
            dep_ready = 0.0
            for p in self.preds.get(gid, []):
                dep_ready = max(dep_ready, self.gates[p].eft)

            g.est = dep_ready

            if g.gtype == GateType.LOCAL:
                g.eft = g.est + g.duration
            else:
                i, j = g.qpus
                extra = 0.0 if g.ent_ready else self.mean_entanglement_time(i=i, j=j)
                g.eft = g.est + extra + g.duration

    def _refresh_ready_states(self):
        for g in self.gates.values():
            if g.state != GateState.PENDING:
                continue
            if g.deps:
                continue

            if g.gtype == GateType.LOCAL:
                g.state = GateState.READY
                self._act(f"GATE {g.gid} -> READY (LOCAL deps cleared)")
            else:
                if g.ent_ready:
                    g.state = GateState.READY
                    self._act(f"GATE {g.gid} -> READY (REMOTE deps cleared + ent_ready)")

                                                                               
                                                        
                                                                               

    def can_start_ent_for_gate(self, i: int, j: int) -> bool:
        qi, qj = self.qpus[i], self.qpus[j]
        if qi.mem_free <= 0 or qj.mem_free <= 0:
            return False
        path = self._network_path(i, j)
        return all(self.qpus[qid].iface_free > 0 for qid in path)
    # def _disturb_inflight_requests(self, new_gate: Gate, new_ent_id: str):
    #     """
    #     仅在重配置模式且网络为 QMSN 时，用新请求干扰所有正在生成的请求。
    #     """
    #     if not (self.params.qmsn_reconfiguration_model and self.network == NetworkType.QMSN):
    #         return
    #     if not self.inflight:
    #         return

    #     # 记录干扰日志
    #     self._act(f"DISTURB triggered by new request {new_ent_id} for gate {new_gate.gid} "
    #             f"pair=({new_gate.qpus[0]},{new_gate.qpus[1]})")
    #     for req in self.inflight.values():
    #         req.version += 1
    #         req.base_tready += self.params.t_init
    #         new_tready = req.base_tready
    #         self.push_event(new_tready, EventType.ENT_RDY,
    #                         ent_id=req.ent_id, gid=req.gid,
    #                         tready=new_tready, i=req.i, j=req.j,
    #                         version=req.version)
    #         self._act(f"  -> disturbs existing {req.ent_id} (gate {req.gid}), "
    #                 f"new_tready={new_tready:.3f}, version={req.version}")

    def start_entanglement_for_gate(self, g: Gate):
        i, j = g.qpus
        path = self._network_path(i, j)
        ent_id = self.gen_ent_id()

        g.ent_id = ent_id
        g.ent_ready = False


        for qid in path:
            self.qpus[qid].iface_free -= 1

                                                                    
        self.qpus[i].mem_free -= 1
        self.qpus[j].mem_free -= 1

        dt = self.entanglement_time(i=i, j=j)
        tready = self.now + dt

        req = InflightEntRequest(ent_id=ent_id, gid=g.gid, i=i, j=j, path=path, start_time= self.now, base_tready=tready)
        return req, tready, dt
    
    def select_one_ent_target(self, delta: float) -> Optional[str]:
        self._last_ent_choice = None
        best_gid: Optional[str] = None
        best_key: Optional[Tuple[int, float, float, float, float]] = None
        frontier_remote_exists = any(
            (
                g.gtype == GateType.REMOTE
                and g.state not in (GateState.DONE, GateState.RUNNING)
                and not g.ent_ready
                and not (g.ent_id is not None and g.ent_id in self.inflight)
                and not g.deps
            )
            for g in self.gates.values()
        )

        debug = False
        # debug = (self.network == NetworkType.QMSN)  # 仅 QMSN 时打印，可按需修改
        if debug:
            mode = "reconfig" if self.params.qmsn_reconfiguration_model else "timeslot"
            print(f"\n[SCHED] now={self.now:.3f}, mode={mode}, frontier_exists={frontier_remote_exists}")

        for g in self.gates.values():
            if g.gtype != GateType.REMOTE:
                continue
            if g.state in (GateState.DONE, GateState.RUNNING):
                continue
            if g.ent_ready:
                continue
            if g.ent_id is not None and g.ent_id in self.inflight:
                continue
            if frontier_remote_exists and g.deps:
                if debug:
                    print(f"  SKIP {g.gid}: not frontier (deps={g.deps})")
                continue

            i, j = g.qpus
            if not self.can_start_ent_for_gate(i, j):
                if debug:
                    print(f"  SKIP {g.gid}: resource unavailable (mem_free or iface)")
                continue

            dep_ready = max((self.gates[p].eft for p in self.preds.get(g.gid, [])), default=0.0)
            mean_gen = self.mean_entanglement_time(i=i, j=j)
                                                          
            t_submit = max(0.0, dep_ready - (self.ent_ttl - g.duration))
            if self.now + 1e-12 < t_submit:
                if debug:
                    print(f"  SKIP {g.gid}: now={self.now:.3f} < t_submit={t_submit:.3f}")
                continue

                                    
                                                                                             
                                                       
            t_ent_est = self.now + mean_gen
            t_start_est = max(dep_ready, t_ent_est)
            if math.isfinite(self.ent_ttl) and t_start_est + g.duration + delta > t_ent_est + self.ent_ttl:
                if debug:
                    print(f"  SKIP {g.gid}: would expire (start={t_start_est:.3f}, ent_exp={t_ent_est+self.ent_ttl:.3f})")
                continue

                                                                          
            slack = (t_ent_est + self.ent_ttl) - (t_start_est + g.duration)                                
                                    
            # frontier_penalty = 0 
            frontier_penalty = 0 if dep_ready <= self.now + 1e-12 else 1
            key = (frontier_penalty, slack, -g.criticality, dep_ready, t_submit)

            if debug:
                print(f"  CAND {g.gid}: frontier_penalty={frontier_penalty}, slack={slack:.3f}, "
                    f"criticality={g.criticality:.3f}, dep_ready={dep_ready:.3f}, t_submit={t_submit:.3f}")

            if best_key is None or key < best_key:
                best_key = key
                best_gid = g.gid
                self._last_ent_choice = {
                    "gid": g.gid,
                    "frontier": (frontier_penalty == 0),
                    "criticality": g.criticality,
                    "slack": slack,
                    "dep_ready": dep_ready,
                    "t_submit": t_submit,
                }

        if debug:
            if best_gid is not None:
                print(f"  SELECTED: {best_gid}\n")
            else:
                print(f"  NO CANDIDATE\n")

        return best_gid
    
    # def schedule_entanglement_gen(self, delta: float = 1.0) -> int:
    #     started = 0
    #     batch_reqs = []

    #     # 清空本批次预留
    #     self._batch_reserved_ifaces.clear()
    #     self._batch_reserved_mem.clear()

    #     while True:
    #         gid = self.select_one_ent_target(delta=delta)
    #         if gid is None:
    #             break
    #         g = self.gates[gid]
    #         i, j = g.qpus
    #         path = self._network_path(i, j)

    #         # 为该门预留资源（不实际扣减）
    #         for qid in path:
    #             self._batch_reserved_ifaces.add(qid)
    #         self._batch_reserved_mem.add(i)
    #         self._batch_reserved_mem.add(j)

    #         # 暂时生成请求对象（不分配真实资源，不加入 inflight）
    #         ent_id = self.gen_ent_id()
    #         g.ent_id = ent_id
    #         g.ent_ready = False
    #         dt = self.entanglement_time(i=i, j=j)
    #         tready = self.now + dt
    #         req = InflightEntRequest(ent_id=ent_id, gid=g.gid, i=i, j=j, path=path,
    #                                 base_tready=tready, start_time=self.now)
    #         batch_reqs.append((req, tready, dt, g))
    #         started += 1

    #      # ---------- 第2阶段：干扰旧的 inflight 请求（如果有）----------
    #     # 只在重配置模式 & QMSN 网络下执行
    #     if (self.params.qmsn_reconfiguration_model and 
    #         self.network == NetworkType.QMSN and 
    #         self.inflight and 
    #         batch_reqs):
    #         # 用第一个新门的信息记录日志
    #         first_gate = batch_reqs[0][3]
    #         print(f"DISTURB triggered by new request e{first_gate.ent_id} "
    #                 f"for gate {first_gate.gid}")
    #         for req in self.inflight.values():
    #             req.version += 1
    #             req.base_tready += self.params.t_init
    #             new_tready = req.base_tready
    #             self.push_event(new_tready, EventType.ENT_RDY,
    #                             ent_id=req.ent_id, gid=req.gid,
    #                             tready=new_tready, i=req.i, j=req.j,
    #                             version=req.version)
    #             print(f"  -> disturbs existing {req.ent_id} (gate {req.gid}), "
    #                     f"new_tready={new_tready:.3f}, v={req.version}")

    #     # ---------- 第3阶段：正式提交所有新请求 ----------
    #     for req, tready, dt, gate in batch_reqs:
    #         i, j = gate.qpus
    #         path = req.path
    #         # 真实扣减资源
    #         for qid in path:
    #             self.qpus[qid].iface_free -= 1
    #         self.qpus[i].mem_free -= 1
    #         self.qpus[j].mem_free -= 1

    #         # 加入 inflight
    #         self.inflight[req.ent_id] = req

    #         # 日志与事件
    #         if self.network in (NetworkType.STATIC_LINE, NetworkType.STATIC_GRID):
    #             self._act(f"START ENT ent={req.ent_id} gate={gate.gid} "
    #                     f"pair=({i},{j}) path={path} dt={dt:.3f} -> ENT_RDY at {tready:.3f}")
    #         else:
    #             self._act(f"START ENT ent={req.ent_id} for gate={gate.gid} "
    #                     f"pair=({i},{j}) dt={dt:.3f} -> ENT_RDY at {tready:.3f}")
    #         self.push_event(tready, EventType.ENT_RDY,
    #                         ent_id=req.ent_id, gid=gate.gid,
    #                         tready=tready, i=i, j=j)

    #     if started > 0:
    #         self._act(f"ENT_SCHED started {started} entgen(s) at this time")
    #     return started

    def schedule_entanglement_gen(self, delta: float = 1.0) -> int:
        
        started = 0
        batch_reqs = []  # 暂存本轮启动的请求信息

        while True:
            gid = self.select_one_ent_target(delta=delta)
            if gid is None:
                break
            g = self.gates[gid]
            # 同一批次内，先不干扰，也不立即加入 inflight
            req, tready, dt = self.start_entanglement_for_gate(g)
            batch_reqs.append((req, tready, dt, g))
            started += 1
            self.recompute_estimates()  # 若需反映资源占用，可保留

        if (self.params.qmsn_reconfiguration_model and 
            self.network == NetworkType.QMSN and 
            self.inflight and 
            batch_reqs):   # 有旧请求存在
            for req_old in self.inflight.values():
                req_old.version += 1
                req_old.base_tready = max(req_old.base_tready, self.now) + self.params.t_init
                new_tready = req_old.base_tready
                self.push_event(new_tready, EventType.ENT_RDY,
                                ent_id=req_old.ent_id, gid=req_old.gid,
                                tready=new_tready, i=req_old.i, j=req_old.j,
                                version=req_old.version)
                self._act(f"DISTURB ent={req_old.ent_id} version={req_old.version} new_tready={new_tready:.3f}")

        # 本轮所有请求选定完毕，统一加入 inflight 并推送事件
        for req, tready, dt, g in batch_reqs:
            self.inflight[req.ent_id] = req
            if self.network in (NetworkType.STATIC_LINE, NetworkType.STATIC_GRID):
                self._act(f"START ENT ent={req.ent_id} gate={req.gid} pair=({req.i},{req.j}) path={req.path} dt={dt:.3f} -> ENT_RDY at {tready:.3f}")
            else:
                self._act(f"START ENT ent={req.ent_id} for gate={req.gid} pair=({req.i},{req.j}) dt={dt:.3f} -> ENT_RDY at {tready:.3f}")
            self.push_event(tready, EventType.ENT_RDY, ent_id=req.ent_id, gid=req.gid, tready=tready, i=req.i, j=req.j)

        # print(f"[EARLY BATCH] now={self.now:.3f}, started={started}")

        if started > 0:
            self._act(f"ENT_SCHED started {started} entgen(s) at this time")
        return started
    

    def process_events(self, events: List[SimEvent]):
        for ev in events:
            if ev.etype == EventType.ENT_RDY:
                ent_id = ev.payload["ent_id"]
                gid = ev.payload["gid"]
                tready = float(ev.payload["tready"])
                i = int(ev.payload["i"])
                j = int(ev.payload["j"])
                version = ev.payload.get("version", 0)

                req = self.inflight.get(ent_id, None)
                if req is None:
                    return
                
                if req.version != version:
                    self._act(f"ENT_RDY ent={ent_id} ignored due to version mismatch")
                    return
                
                # if self.params.qmsn_reconfiguration_model and self.network == NetworkType.QMSN:
                #     elapsed = tready - req.start_time
                #     print(f"ENT_RDY ent={ent_id} (gate {gid}) succeeded, start={req.start_time:.3f}, tready={tready:3f} elapsed={elapsed:.3f} us")
                
                # if self.network == NetworkType.MPQN:
                #     elapsed = tready - req.start_time
                #     print(f"[MPQN] ENT_RDY ent={ent_id} (gate {gid}) succeeded, "
                #         f"start={req.start_time:.3f}, tready={tready:.3f}, elapsed={elapsed:.3f} us")

                req = self.inflight.pop(ent_id) 
                self.ent[ent_id] = EntanglementRecord(
                    ent_id=ent_id,
                    gid=gid,
                    i=i,
                    j=j,
                    tready=tready,
                    ttl=self.ent_ttl,
                    fidelity=self.entanglement_ready_fidelity(i=i, j=j),
                )

                                                          
                if req is not None:
                    for qid in req.path:
                        self.qpus[qid].iface_free += 1
                else:
                    self.qpus[i].iface_free += 1
                    self.qpus[j].iface_free += 1

                                             
                g = self.gates[gid]
                g.ent_ready = True

                self._act(f"EVENT ENT_RDY ent={ent_id} gate={gid} -> ent_ready=True, iface_free++")

                                     
                if math.isfinite(self.ent_ttl):
                    self.push_event(tready + self.ent_ttl, EventType.ENT_EXP, ent_id=ent_id, gid=gid)

            elif ev.etype == EventType.ENT_EXP:
                ent_id = ev.payload["ent_id"]
                gid = ev.payload["gid"]

                rec = self.ent.get(ent_id)
                self._act(f"EVENT ENT_EXP ent={ent_id} gate={gid}")

                if rec is None or rec.consumed or rec.expired:
                    continue

                                               
                if rec.occupied:
                    self.qpus[rec.i].mem_free += 1
                    self.qpus[rec.j].mem_free += 1
                    rec.occupied = False
                    self._act(f"  -> freed memory on QPU{rec.i} and QPU{rec.j}")

                rec.expired = True

                                                                  
                g = self.gates.get(gid)
                if g is not None and g.state != GateState.DONE and g.ent_id == ent_id:
                    g.ent_id = None
                    g.ent_ready = False
                    if g.state == GateState.READY:
                        g.state = GateState.PENDING
                    self._act(f"  -> gate {gid} cleared ent binding (expired)")

            elif ev.etype == EventType.LOCAL_DONE:
                gid = ev.payload["gid"]
                g = self.gates[gid]
                g.state = GateState.DONE
                g.eft = self.now

                n_used_phys = sum(len(qubits) for qubits in g.data_qubits.values())
                gate_fid = self.params.f_1g if n_used_phys == 1 else self.params.f_2g_local
                self.computational_fidelity *= self._clip_unit(gate_fid)

                                  
                for qid, qubits in g.data_qubits.items():
                    self.qpus[qid].mark_data(qubits, busy=False)

                              
                for s in self.succ.get(gid, set()):
                    self.gates[s].deps.discard(gid)

                self._act(f"EVENT LOCAL_DONE gate={gid} -> DONE")

            elif ev.etype == EventType.REMOTE_DONE:
                gid = ev.payload["gid"]
                g = self.gates[gid]
                g.state = GateState.DONE
                g.eft = self.now

                rec: Optional[EntanglementRecord] = None
                if g.ent_id is not None:
                    rec = self.ent.get(g.ent_id)

                if rec is not None and rec.occupied and not rec.consumed and not rec.expired:
                    wait_time = max(0.0, self.now - rec.tready)
                    f_wait = self.entanglement_wait_decay(wait_time)
                    f_ent_used = self._clip_unit(rec.fidelity) * f_wait
                else:
                    f_ent_used = 0.0

                gate_fid = self._clip_unit(self.params.f_2g_remote * f_ent_used)
                self.computational_fidelity *= gate_fid

                                  
                for qid, qubits in g.data_qubits.items():
                    self.qpus[qid].mark_data(qubits, busy=False)

                self._act(f"EVENT REMOTE_DONE gate={gid} -> DONE")

                                                   
                if g.ent_id is not None:
                    if rec is not None and rec.occupied and not rec.consumed and not rec.expired:
                        self.qpus[rec.i].mem_free += 1
                        self.qpus[rec.j].mem_free += 1
                        rec.consumed = True
                        rec.occupied = False
                        self._act(f"  -> consumed ent={g.ent_id}, freed memory on QPU{rec.i},QPU{rec.j}")

                    g.ent_ready = False

                              
                for s in self.succ.get(gid, set()):
                    self.gates[s].deps.discard(gid)

            else:
                raise ValueError(f"Unhandled event type: {ev.etype}")

        self._refresh_ready_states()
        self.recompute_estimates()

                                                                               
                                                   
                                                                               

    def launch_ready_gates(self):
        for g in self.gates.values():
            if g.state != GateState.READY:
                continue

            if g.gtype == GateType.LOCAL:
                i = g.qpus[0]
                if not self.qpus[i].data_free(g.data_qubits.get(i, [])):
                    continue

                                    
                self.qpus[i].mark_data(g.data_qubits.get(i, []), busy=True)

                g.state = GateState.RUNNING
                done_t = self.now + g.duration
                g.eft = done_t

                self._act(f"START LOCAL gate={g.gid} on QPU{i} dur={g.duration:.3f} -> LOCAL_DONE at {done_t:.3f}")
                self.push_event(done_t, EventType.LOCAL_DONE, gid=g.gid)

            else:
                                                
                i, j = g.qpus
                if not g.ent_ready or g.ent_id is None:
                    continue

                                          
                if not self.qpus[i].data_free(g.data_qubits.get(i, [])):
                    continue
                if not self.qpus[j].data_free(g.data_qubits.get(j, [])):
                    continue

                                     
                rec = self.ent.get(g.ent_id)
                if rec is None or rec.expired or rec.consumed or not rec.occupied:
                    g.ent_ready = False
                    g.ent_id = None
                    g.state = GateState.PENDING
                    continue

                                                         
                if self.now + g.duration > rec.tready + rec.ttl:
                    self._act(f"SKIP REMOTE gate={g.gid}: ent would expire mid-gate, will reschedule ent")
                    g.ent_ready = False
                    g.ent_id = None
                    g.state = GateState.PENDING
                    continue

                                    
                self.qpus[i].mark_data(g.data_qubits.get(i, []), busy=True)
                self.qpus[j].mark_data(g.data_qubits.get(j, []), busy=True)

                g.state = GateState.RUNNING
                done_t = self.now + g.duration
                g.eft = done_t

                self._act(f"START REMOTE gate={g.gid} pair=({i},{j}) ent={g.ent_id} dur={g.duration:.3f} -> REMOTE_DONE at {done_t:.3f}")
                self.push_event(done_t, EventType.REMOTE_DONE, gid=g.gid)

        self.recompute_estimates()

                                                                               
              
                                                                               

    def prefetch_until_cutoff(self, delta: float = 1.0, max_steps: int = 50_000_000_000_000_000_000):
        self._act("PREFETCH start")

        remote_bound = max((g.duration for g in self.gates.values() if g.gtype == GateType.REMOTE), default=0.0)

        def oldest_ready_expiry() -> Optional[float]:
            best = None
            for rec in self.ent.values():
                if rec.expired or rec.consumed or not rec.occupied:
                    continue
                exp = rec.tready + rec.ttl
                if best is None or exp < best:
                    best = exp
            return best

        for _ in range(max_steps):
            oldest_exp = oldest_ready_expiry()
            if oldest_exp is not None and self.now >= oldest_exp - (remote_bound + delta):
                self._act("PREFETCH stop: oldest ent near expiry window")
                return

            started = self.schedule_entanglement_gen(delta=delta)
            if started == 0:
                self._act("PREFETCH stop: cannot schedule more ent right now")
                return

            batch = self.pop_events_until_trigger()
            if not batch:
                self._act("PREFETCH stop: no future events")
                return

                                                           
            ent_batch = [ev for ev in batch if ev.etype in (EventType.ENT_RDY, EventType.ENT_EXP)]
            if ent_batch:
                self.process_events(ent_batch)

        raise RuntimeError("Prefetch max_steps exceeded.")

                                                                               
                                                        
                                                                               

    def run(self, max_steps: int = 50_000_000_000_000_000_000) -> float:

       

        self._act("EXEC start")
        steps = 0

        while not self.dag_exhausted():
            steps += 1
            if steps > max_steps:
                raise RuntimeError("Max steps exceeded (deadlock or too long).")

                                                                      
            popped = []
            if self._pq:
                popped = self.pop_events_until_trigger()
                if popped:
                    self._act(f"TIME JUMP -> {self.now:.3f} (popped {len(popped)} event(s))")
                    self.process_events(popped)

                                                                                    
            progressed = True
            while progressed and not self.dag_exhausted():
                progressed = False

                before = len(self._pq)
                self.launch_ready_gates()
                if len(self._pq) != before:
                    progressed = True

                started = self.schedule_entanglement_gen(delta=self.delta_default)
                
                if started > 0:
                    progressed = True

                                                                                      
                if self._pq and math.isclose(self._pq[0].time, self.now, abs_tol=1e-12):
                    batch_now = []
                    while self._pq and math.isclose(self._pq[0].time, self.now, abs_tol=1e-12):
                        batch_now.append(heapq.heappop(self._pq))
                    if batch_now:
                        self._act(f"IMMEDIATE EVENTS at t={self.now:.3f}: {len(batch_now)}")
                        self.process_events(batch_now)
                        progressed = True

                                                        
            if not self.dag_exhausted() and not self._pq:
                self._act("EXEC stop: deadlock (no future events)")
                break

        self._act("EXEC done")
        return self.now
                                                                               

def topo_sort(nodes: List[str], preds: Dict[str, List[str]], succ: Dict[str, Set[str]]) -> List[str]:
    indeg = {n: len(preds.get(n, [])) for n in nodes}
    q = deque([n for n in nodes if indeg[n] == 0])
    topo: List[str] = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ.get(u, set()):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != len(nodes):
        raise ValueError("Cycle detected; dependency extraction produced a cycle.")
    return topo


def _parse_qasm_qubit(token: str) -> int:
    token = token.strip().rstrip(";")
    left = token.find("[")
    right = token.find("]")
    if left < 0 or right < 0:
        raise ValueError(f"Bad qubit token: {token}")
    return int(token[left + 1:right])


def _read_openqasm_ops(qasm_path: str) -> Tuple[int, List[Tuple[str, List[int]]]]:
    n_qubits: Optional[int] = None
    ops: List[Tuple[str, List[int]]] = []

    with open(qasm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("OPENQASM") or line.startswith("include") or line.startswith("creg"):
                continue

            if line.startswith("qreg"):
                left = line.find("[")
                right = line.find("]")
                n_qubits = int(line[left + 1:right])
                continue

            if line.startswith("h "):
                ops.append(("h", [_parse_qasm_qubit(line.split()[1])]))
            elif line.startswith("t "):
                ops.append(("t", [_parse_qasm_qubit(line.split()[1])]))
            elif line.startswith("tdg "):
                ops.append(("tdg", [_parse_qasm_qubit(line.split()[1])]))
            elif line.startswith("cx "):
                rest = line[3:].rstrip(";")
                a_str, b_str = rest.split(",")
                ops.append(("cx", [_parse_qasm_qubit(a_str), _parse_qasm_qubit(b_str)]))
            else:
                                      
                continue

    if n_qubits is None:
        raise ValueError("No qreg found in QASM file.")
    return n_qubits, ops


def map_logical_to_physical_sequential(n_qubits: int, qpus: Dict[int, QPU]) -> Dict[int, Tuple[int, int]]:
    mapping: Dict[int, Tuple[int, int]] = {}
    qpu_ids = sorted(qpus.keys())

    lq = 0
    for qid in qpu_ids:
        for local_idx in range(qpus[qid].n_data_qubits):
            if lq == n_qubits:
                return mapping
            mapping[lq] = (qid, local_idx)
            lq += 1

    if lq != n_qubits:
        total_cap = sum(qpus[q].n_data_qubits for q in qpu_ids)
        raise ValueError(f"Not enough physical data qubits: need {n_qubits}, have {total_cap}")

    return mapping


def map_logical_to_physical_random(
    n_qubits: int,
    qpus: Dict[int, QPU],
    rng_seed: int = 0,
) -> Dict[int, Tuple[int, int]]:
    slots: List[Tuple[int, int]] = []
    for qid in sorted(qpus.keys()):
        for local_idx in range(qpus[qid].n_data_qubits):
            slots.append((qid, local_idx))

    if len(slots) < n_qubits:
        raise ValueError(f"Not enough physical data qubits: need {n_qubits}, have {len(slots)}")

    rng = np.random.default_rng(rng_seed)
    chosen_idx = rng.choice(len(slots), size=n_qubits, replace=False)
    chosen = [slots[int(i)] for i in chosen_idx]
    rng.shuffle(chosen)

    mapping: Dict[int, Tuple[int, int]] = {}
    for lq in range(n_qubits):
        mapping[lq] = chosen[lq]
    return mapping


def build_dag(
    qasm_path: str,
    qpus: Dict[int, QPU],
    T_2Q: float = 100.0,
    T_1Q: float = 1.0,
    mapping: Optional[Dict[int, Tuple[int, int]]] = None,
    mapping_mode: str = "sequential",
    mapping_seed: int = 0,
) -> Tuple[Dict[str, Gate], Dict[str, Set[str]], Dict[str, List[str]], List[str], Dict[int, Tuple[int, int]]]:
    n_qubits, ops = _read_openqasm_ops(qasm_path)
    if mapping is None:
        if mapping_mode == "random":
            mapping = map_logical_to_physical_random(n_qubits, qpus, rng_seed=mapping_seed)
        else:
            mapping = map_logical_to_physical_sequential(n_qubits, qpus)

    gates: Dict[str, Gate] = {}
    preds: Dict[str, List[str]] = {}
    succ: Dict[str, Set[str]] = {}

    last_touch: Dict[Tuple[int, int], str] = {}

    gid_ctr = 0
    for opname, lqs in ops:
        gid = f"g{gid_ctr}"
        gid_ctr += 1

        phys = [mapping[lq] for lq in lqs]                               

                                                                
        if opname == "cx":
            (qa_qpu, qa), (qb_qpu, qb) = phys
            if qa_qpu == qb_qpu:
                                                   
                gtype = GateType.LOCAL
                qpu_tuple = (qa_qpu,)
                data_qubits = {qa_qpu: [qa, qb]}
                duration = T_1Q
            else:
                gtype = GateType.REMOTE
                i, j = (qa_qpu, qb_qpu) if qa_qpu < qb_qpu else (qb_qpu, qa_qpu)
                qpu_tuple = (i, j)
                data_qubits = {qa_qpu: [qa], qb_qpu: [qb]}
                duration = T_2Q
        else:
            (qpu_id, qphys) = phys[0]
            gtype = GateType.LOCAL
            qpu_tuple = (qpu_id,)
            data_qubits = {qpu_id: [qphys]}
            duration = T_1Q

                                      
        dep_set: Set[str] = set()
        for (qpu_id, qphys) in phys:
            prev = last_touch.get((qpu_id, qphys))
            if prev is not None:
                dep_set.add(prev)

        g = Gate(
            gid=gid,
            gtype=gtype,
            qpus=qpu_tuple,
            data_qubits=data_qubits,
            deps=set(dep_set),
            duration=duration,
        )

        gates[gid] = g
        preds[gid] = list(dep_set)
        succ.setdefault(gid, set())

        for p in dep_set:
            succ.setdefault(p, set()).add(gid)

                           
        for (qpu_id, qphys) in phys:
            last_touch[(qpu_id, qphys)] = gid

    topo = topo_sort(list(gates.keys()), preds, succ)
    return gates, succ, preds, topo, mapping
    

def build_uniform_qpus(
    n_qpus: int,
    n_data_qubits: int = 1,
    n_mem_slots: int = 1,
    n_ifaces: int = 1,
) -> Dict[int, QPU]:
    return {
        qid: QPU(qid=qid, n_data_qubits=n_data_qubits, n_mem_slots=n_mem_slots, n_ifaces=n_ifaces)
        for qid in range(n_qpus)
    }


def compute_p_ent_from_components(p_e: float, p_c: float, p_t: float, p_d: float) -> float:
    return 0.5 * ((p_e * p_c * p_t) ** 2) * p_d


def _auto_grid_dims(n_qubits: int) -> Tuple[int, int]:
    n = max(int(n_qubits), 1)
    r_min = 2 if n > 2 else 1
    r_max = int(math.ceil(math.sqrt(n))) + 1

    best_r, best_c = 1, n
    best_key = (10**9, 10**9, 10**9)                                       

    for r in range(r_min, r_max + 1):
        c = int(math.ceil(n / r))
        if c < 2 and n > 2:
            c = 2
        if r > c:
            r, c = c, r
        area = r * c
        if area < n:
            continue
        key = (abs(c - r), area - n, area)
        if key < best_key:
            best_key = key
            best_r, best_c = r, c

    if n <= 2:
        return 1, n
    return best_r, best_c


def _copy_params_with(params: ExperimentParams, **overrides) -> ExperimentParams:
    return replace(params, **overrides)


def _network_list(include_static_grid: bool = True) -> List[NetworkType]:
    nets = [NetworkType.QMSN, NetworkType.MPQN, NetworkType.STATIC_LINE]
    if include_static_grid:
        nets.append(NetworkType.STATIC_GRID)
    return nets

def _get_network_modes(net: NetworkType) -> List[Tuple[Optional[bool], str]]:
    if net == NetworkType.QMSN:
        return [(False, "timeslot"), (True, "reconfig")]
    else:
        return [(None, "N/A")]


def _params_and_nqpus_for_network(
    base_params: ExperimentParams,
    network: NetworkType,
    n_qubits: int,
    circuit_name: str,
) -> Tuple[ExperimentParams, int]:
    n_qpus = n_qubits
    if network != NetworkType.STATIC_GRID:
        return base_params, n_qpus

    if base_params.grid_rows > 0 and base_params.grid_cols > 0:
        gr, gc = int(base_params.grid_rows), int(base_params.grid_cols)
        if gr * gc < n_qubits:
            raise ValueError(
                f"STATIC_GRID size too small for circuit {circuit_name}: "
                f"grid_rows*grid_cols={gr * gc}, qubits={n_qubits}."
            )
    else:
        gr, gc = _auto_grid_dims(n_qubits)

    run_params = _copy_params_with(
        base_params,
        grid_rows=gr,
        grid_cols=gc,
        grid_routing="xy",
    )
    return run_params, gr * gc


def run_single_simulation(
    qasm_path: str,
    network: NetworkType,
    seed: int,
    params: ExperimentParams,
    n_qpus: Optional[int] = None,
    mapping_mode: str = "random",
    trace: bool = False,
    prefetch: bool = True,
) -> Tuple[Dict[str, float], OrchestratorSim, Dict[int, Tuple[int, int]]]:
    n_qubits, _ = _read_openqasm_ops(qasm_path)
    if n_qpus is None:
        n_qpus = n_qubits
    if n_qpus < n_qubits:
        raise ValueError(f"n_qpus={n_qpus} is smaller than circuit qubits={n_qubits}.")

    qpus = build_uniform_qpus(n_qpus=n_qpus, n_data_qubits=1, n_mem_slots=1, n_ifaces=1)

    gates, succ, preds, topo, mapping = build_dag(
        qasm_path=qasm_path,
        qpus=qpus,
        T_2Q=params.t_2g,
        T_1Q=params.t_1g,
        mapping_mode=mapping_mode,
        mapping_seed=seed,
    )

    sim = OrchestratorSim(
        gates=gates,
        qpus=qpus,
        succ=succ,
        preds=preds,
        topo=topo,
        network=network,
        params=params,
        rng_seed=seed,
        trace=trace,
    )

    sim.initialize()
    if prefetch:
        sim.prefetch_until_cutoff(delta=params.delta_default)
    t_exec0 = sim.now
    t_total = sim.run()

    remote_count = sum(1 for g in gates.values() if g.gtype == GateType.REMOTE)
    local_count = len(gates) - remote_count

    result = {
        "seed": float(seed),
        "n_qpus": float(n_qpus),
        "n_qubits": float(n_qubits),
        "n_gates": float(len(gates)),
        "n_local_gates": float(local_count),
        "n_remote_gates": float(remote_count),
        "prefetch_time": float(t_exec0),
        "exec_time": float(t_total - t_exec0),
        "total_time": float(t_total),
        "computational_fidelity": float(sim.computational_fidelity),
    }
    return result, sim, mapping

def run_with_repeats(
    qasm_path: str,
    network: NetworkType,
    base_seed: int,
    runs: int,
    params: ExperimentParams,
    n_qpus: Optional[int] = None,
    mapping_mode: str = "random",
    trace: bool = False,
    prefetch: bool = True,
) -> Dict[str, float]:
    """
    Run many times for the same params and return the results.
    Every time use base_seed + i as the seed.
    """
    if runs <= 0:
        raise ValueError("runs must be positive")
    
    total_times = []
    exec_times = []
    prefetch_times = []
    fidelities = []
    n_qpus_val: float = 0.0
    n_qubits_val: float = 0.0
    n_gates_val: float = 0.0
    n_local_val: float = 0.0
    n_remote_val: float = 0.0

    for i in range(runs):
        seed = base_seed + i
        res, sim, mapping = run_single_simulation(
            qasm_path = qasm_path,
            network = network,
            seed = seed,
            params = params,
            n_qpus = n_qpus,
            mapping_mode = mapping_mode,
            trace = trace,
            prefetch = prefetch,
        )
        # print(f"  Run {i+1}/{runs}: seed={seed}, total_time={res['total_time']:.3f}")
        total_times.append(res["total_time"])
        exec_times.append(res["exec_time"])
        prefetch_times.append(res["prefetch_time"])
        fidelities.append(res["computational_fidelity"])

        if i == 0:
            n_qpus_val = res["n_qpus"]
            n_qubits_val = res["n_qubits"]
            n_gates_val = res["n_gates"]
            n_local_val = res["n_local_gates"]
            n_remote_val = res["n_remote_gates"]

    def mean_std(arr):
        arr = np.array(arr)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    
    total_mean, total_std = mean_std(total_times)
    exec_mean, exec_std = mean_std(exec_times)
    prefetch_mean, prefetch_std = mean_std(prefetch_times)
    fid_mean, fid_std = mean_std(fidelities)

    # print(f"[DEBUG] run_with_repeats called with runs = {runs}")

    return {
        "seed": float(base_seed),          # 基准种子，便于追溯
        "runs": runs,
        "n_qpus": n_qpus_val,
        "n_qubits": n_qubits_val,
        "n_gates": n_gates_val,
        "n_local_gates": n_local_val,
        "n_remote_gates": n_remote_val,
        "prefetch_time_mean": prefetch_mean,
        "prefetch_time_std": prefetch_std,
        "exec_time_mean": exec_mean,
        "exec_time_std": exec_std,
        "total_time_mean": total_mean,
        "total_time_std": total_std,
        "computational_fidelity_mean": fid_mean,
        "computational_fidelity_std": fid_std,
    }


def summarize_results(rows: List[Dict[str, object]], group_keys: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for key, items in groups.items():
        # print(f"Group key: {key}, items count: {len(items)}")
        # if items:
        #     print("First item keys:", items[0].keys())
        #     print("First item values:", {k: items[0][k] for k in ('total_time_mean','total_time_std','network','mode')})
        # 提取聚合后的均值数据（如果已经是均值，则直接计算均值的均值）
        total_means = np.array([it["total_time_mean"] for it in items], dtype=float)
        exec_means = np.array([it["exec_time_mean"] for it in items], dtype=float)
        prefetch_means = np.array([it["prefetch_time_mean"] for it in items], dtype=float)

        line: Dict[str, object] = {k: v for k, v in zip(group_keys, key)}
        line["runs"] = float(len(items))
        line["total_time_grand_mean"] = float(total_means.mean())
        line["total_time_grand_std"] = float(total_means.std(ddof=1) if len(total_means) > 1 else 0.0)
        line["exec_time_grand_mean"] = float(exec_means.mean())
        line["exec_time_grand_std"] = float(exec_means.std(ddof=1) if len(exec_means) > 1 else 0.0)
        line["prefetch_time_grand_mean"] = float(prefetch_means.mean())
        line["prefetch_time_grand_std"] = float(prefetch_means.std(ddof=1) if len(prefetch_means) > 1 else 0.0)

        if "computational_fidelity_mean" in items[0]:
            fid_means = np.array([it["computational_fidelity_mean"] for it in items], dtype=float)
            line["computational_fidelity_grand_mean"] = float(fid_means.mean())
            line["computational_fidelity_grand_std"] = float(fid_means.std(ddof=1) if len(fid_means) > 1 else 0.0)
        out.append(line)
    return out


def write_csv_rows(path: str, rows: List[Dict[str, object]]):
    if not rows:
        return
    out = Path(path).resolve()
    fields: Set[str] = set()
    for r in rows:
        fields.update(r.keys())
    fieldnames = sorted(fields)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote CSV: {out}")


def run_experiment_1(
    circuits: List[str],
    seed: int,
    params: ExperimentParams,
    out_prefix: str,
    mapping_mode: str = "random",
    include_static_grid: bool = True,
    runs: int = 1,
):
    rows: List[Dict[str, object]] = []
    networks = _network_list(include_static_grid=include_static_grid)
    for qasm_path in circuits:
        name = Path(qasm_path).stem
        n_qubits, _ = _read_openqasm_ops(qasm_path)

        for net in networks:
            run_params = params
            for reconfig_enabled, mode_label in _get_network_modes(net):
                n_qpus = n_qubits
                if net == NetworkType.STATIC_GRID:
                    if params.grid_rows > 0 and params.grid_cols > 0:
                        gr, gc = int(params.grid_rows), int(params.grid_cols)
                        if gr * gc < n_qubits:
                            raise ValueError(
                                f"STATIC_GRID size too small for circuit {name}: "
                                f"grid_rows*grid_cols={gr * gc}, qubits={n_qubits}."
                            )
                    else:
                        gr, gc = _auto_grid_dims(n_qubits)

                    run_params = ExperimentParams(
                        t_1g=params.t_1g,
                        t_2g=params.t_2g,
                        t_retry=params.t_retry,
                        p_ent=params.p_ent,
                        p_swi=params.p_swi,
                        ent_time_mode=params.ent_time_mode,
                        grid_rows=gr,
                        grid_cols=gc,
                        grid_routing="xy",
                        t_deco=params.t_deco,
                        t_c=params.t_c,
                        t_init=params.t_init,
                        # t_m_init=params.t_m_init,
                        # t_e=params.t_e,
                        # gamma_max_hz=params.gamma_max_hz,
                        p_swap=params.p_swap,
                        delta_default=params.delta_default,
                        f_1g=params.f_1g,
                        f_2g_local=params.f_2g_local,
                        f_2g_remote=params.f_2g_remote,
                        f_ent_base=params.f_ent_base,
                        f_link=params.f_link,
                        f_swap_fid=params.f_swap_fid,
                        f_switch=params.f_switch,
                        f_trans=params.f_trans,
                        t_fid_decay=params.t_fid_decay,
                    )
                    n_qpus = gr * gc

                if net == NetworkType.QMSN:
                    run_params = _copy_params_with(run_params, qmsn_reconfiguration_model=reconfig_enabled)

                aggregated = run_with_repeats(
                    qasm_path=qasm_path,
                    network=net,
                    base_seed=seed,
                    runs=runs,
                    params=run_params,
                    n_qpus=n_qpus,
                    mapping_mode=mapping_mode,
                    prefetch=True,
                    trace=False,
                )
                # print(f"DEBUG aggregated: runs={aggregated['runs']}, total_time_std={aggregated['total_time_std']:.3f}")

                rows.append(
                    {
                        "experiment": "exp1",
                        "circuit": name,
                        "network": net.value,
                        "mode": mode_label,
                        **aggregated,   # 包含 total_time_mean, total_time_std, ...
                    }
                )

    for row in rows:
        print(f"{row['network']:>12} {row['mode']:>10}  total_time_std = {row['total_time_std']:.3f}")

    # summary = summarize_results(rows, group_keys=["experiment", "circuit", "network", "mode"])
    write_csv_rows(f"{out_prefix}_raw.csv", rows)
    # write_csv_rows(f"{out_prefix}_summary.csv", summary)

def run_experiment_network_reconfig_compare(
    circuits: List[str],
    seed: int,
    params: ExperimentParams,
    out_prefix: str,
    mapping_mode: str = "random",
    include_static_grid: bool = True,
    runs: int=1,
):
    rows: List[Dict[str, object]] = []
    networks = _network_list(include_static_grid=include_static_grid)

    for qasm_path in circuits:
        name = Path(qasm_path).stem
        n_qubits, _ = _read_openqasm_ops(qasm_path)

        for net in networks:
            # 确定该网络需要测试的模式列表
            if net == NetworkType.QMSN:
                reconfig_modes = [False, True]   # 两种模式
            else:
                reconfig_modes = [None]          # 仅一种，None 表示不适用

            for reconfig_enabled in reconfig_modes:
                # 准备本次运行的参数
                run_params = params
                n_qpus = n_qubits

                # 处理 STATIC_GRID 的特殊尺寸逻辑（与 exp1 一致）
                if net == NetworkType.STATIC_GRID:
                    if params.grid_rows > 0 and params.grid_cols > 0:
                        gr, gc = int(params.grid_rows), int(params.grid_cols)
                        if gr * gc < n_qubits:
                            raise ValueError(f"Grid too small for {name}")
                    else:
                        gr, gc = _auto_grid_dims(n_qubits)
                    run_params = _copy_params_with(params, grid_rows=gr, grid_cols=gc)
                    n_qpus = gr * gc

                # 如果是 QMSN，设置重配置模型开关
                if net == NetworkType.QMSN:
                    run_params = _copy_params_with(run_params, qmsn_reconfiguration_model=reconfig_enabled)

                # 运行单次仿真
                result = run_with_repeats(
                    qasm_path=qasm_path,
                    network=net,
                    base_seed=seed,
                    runs=runs,
                    params=run_params,
                    n_qpus=n_qpus,
                    mapping_mode=mapping_mode,
                    prefetch=True,
                    trace=False,
                )

                # 构建行数据，对非 QMSN 网络，reconfiguration_model 字段为空
                row = {
                    "experiment": "network_reconfig_compare",
                    "circuit": name,
                    "network": net.value,
                    **result,
                }
                if net == NetworkType.QMSN:
                    row["reconfiguration_model"] = reconfig_enabled
                else:
                    row["reconfiguration_model"] = "N/A"

                rows.append(row)

    # 汇总统计（分组键包含 network 和 reconfiguration_model）
    summary = summarize_results(
        rows,
        group_keys=["experiment", "circuit", "network", "reconfiguration_model"]
    )
    write_csv_rows(f"{out_prefix}_raw.csv", rows)
    write_csv_rows(f"{out_prefix}_summary.csv", summary)

def run_experiment_pc_sweep(
    circuits: List[str],
    seed: int,
    params: ExperimentParams,
    out_prefix: str,
    p_c_values: List[float],
    p_e: float,
    p_t: float,
    p_d: float,
    mapping_mode: str = "random",
    include_static_grid: bool = True,
    experiment_name: str = "exp_pc",
    runs: int = 1,
):
    rows: List[Dict[str, object]] = []
    networks = _network_list(include_static_grid=include_static_grid)

    for p_c in p_c_values:
        # p_ent = compute_p_ent_from_components(p_e=p_e, p_c=p_c, p_t=p_t, p_d=p_d)
        params_this_pc = _copy_params_with(params, p_c=p_c)

        for qasm_path in circuits:
            name = Path(qasm_path).stem
            n_qubits, _ = _read_openqasm_ops(qasm_path)

            for net in networks:
                run_params, n_qpus = _params_and_nqpus_for_network(
                    base_params=params_this_pc,
                    network=net,
                    n_qubits=n_qubits,
                    circuit_name=name,
                )

                result = run_with_repeats(
                    qasm_path=qasm_path,
                    network=net,
                    base_seed=seed,
                    runs=runs,
                    params=run_params,
                    n_qpus=n_qpus,
                    mapping_mode=mapping_mode,
                    prefetch=True,
                    trace=False,
                )
                rows.append(
                    {
                        "experiment": experiment_name,
                        "circuit": name,
                        "network": net.value,
                        "p_c": float(p_c),
                        "p_ent": float(run_params.p_ent),
                        "p_e": float(p_e),
                        "p_t": float(p_t),
                        "p_d": float(p_d),
                        **result,
                    }
                )

    summary = summarize_results(rows, group_keys=["experiment", "circuit", "network", "p_c", "p_ent"])
    for line in summary:
        line["p_e"] = float(p_e)
        line["p_t"] = float(p_t)
        line["p_d"] = float(p_d)

    write_csv_rows(f"{out_prefix}_raw.csv", rows)
    write_csv_rows(f"{out_prefix}_summary.csv", summary)


def run_experiment_scale(
    circuits: List[str],
    seed: int,
    params: ExperimentParams,
    out_prefix: str,
    mapping_mode: str = "random",
    include_static_grid: bool = True,
    runs: int = 1,
):
    rows: List[Dict[str, object]] = []
    networks = _network_list(include_static_grid=include_static_grid)

    for qasm_path in circuits:
        name = Path(qasm_path).stem
        n_qubits, _ = _read_openqasm_ops(qasm_path)
        n_qpus = n_qubits
        log2_n_qpus = float(math.log2(max(n_qpus, 1)))

        for net in networks:
            run_params, n_qpus_this = _params_and_nqpus_for_network(
                base_params=params,
                network=net,
                n_qubits=n_qubits,
                circuit_name=name,
            )

            result = run_with_repeats(
                qasm_path=qasm_path,
                network=net,
                base_seed=seed,
                params=run_params,
                n_qpus=n_qpus_this,
                mapping_mode=mapping_mode,
                prefetch=True,
                trace=False,
                runs = runs,
            )
            rows.append(
                {
                    "experiment": "exp_scale",
                    "circuit": name,
                    "network": net.value,
                    **result,
                    "n_qpus": float(n_qpus),
                    "n_qubits": float(n_qubits),
                    "log2_n_qpus": log2_n_qpus,
                }
            )

    summary = summarize_results(rows, group_keys=["experiment", "circuit", "network", "n_qpus", "log2_n_qpus"])
    write_csv_rows(f"{out_prefix}_raw.csv", rows)
    write_csv_rows(f"{out_prefix}_summary.csv", summary)

def run_experiment_parameter_sweep(
    circuits: List[str],
    seed: int,
    params: ExperimentParams,
    out_prefix: str,
    param_name: str,
    param_values: List[float],
    mapping_mode: str = "random",
    runs: int = 1,
    include_static_grid: bool = True,
):
    rows = []
    networks = _network_list(include_static_grid=include_static_grid)

    for qasm_path in circuits:
        name = Path(qasm_path).stem
        n_qubits, _ = _read_openqasm_ops(qasm_path)

        for net in networks:
            for reconfig_enabled, mode_label in _get_network_modes(net):
                for val in param_values:
                    # if param_name == "p_ent":
                    #     run_params = _copy_params_with(params, compute_p_ent=False)
                    run_params = _copy_params_with(params, **{param_name: val})
                    
                    # print(f"[DEBUG] p_ent = {run_params.p_ent}")
                    n_qpus = n_qubits

                    if net == NetworkType.STATIC_GRID:
                        if params.grid_rows > 0 and params.grid_cols > 0:
                            gr, gc = int(params.grid_rows), int(params.grid_cols)
                            if gr * gc < n_qubits:
                                raise ValueError(f"Grid too small for {name}")
                        else:
                            gr, gc = _auto_grid_dims(n_qubits)
                        run_params = _copy_params_with(run_params, grid_rows=gr, grid_cols=gc)
                        n_qpus = gr * gc

                    if net == NetworkType.QMSN:
                        run_params = _copy_params_with(run_params, qmsn_reconfiguration_model=reconfig_enabled)

                    aggregated = run_with_repeats(
                        qasm_path=qasm_path,
                        network=net,
                        base_seed=seed,
                        runs=runs,
                        params=run_params,
                        n_qpus=n_qpus,
                        mapping_mode=mapping_mode,
                        prefetch=True,
                        trace=False,
                    )

                    row = {
                        "experiment": f"{param_name}_sweep",
                        "circuit": name,
                        "network": net.value,
                        "mode": mode_label,      # 关键：写入模式标签
                        param_name: val,
                        **aggregated,
                    }
                    rows.append(row)

    write_csv_rows(f"{out_prefix}_raw.csv", rows)
    # summary = summarize_results(rows, group_keys=["experiment", "circuit", "network", "mode", param_name])
    # write_csv_rows(f"{out_prefix}_summary.csv", summary)

def run_experiment_fidelity_sweep(
        circuits: List[str],
        seed: int,
        params: ExperimentParams,
        out_prefix: str,
        t_deco_values: List[float],
        mapping_mode: str = "random",
        runs: int = 1,
):
    rows = []
    networks = _network_list(include_static_grid = True)

    for qasm_path in circuits:
        name = Path(qasm_path).stem
        n_qubits, _ = _read_openqasm_ops(qasm_path)

        for net in networks:
            for t_deco in t_deco_values:
                run_params = _copy_params_with(params, t_deco=t_deco)
                n_qpus = n_qubits
    
                if net == NetworkType.STATIC_GRID:
                    if params.grid_rows > 0 and params.grid_cols > 0:
                        gr, gc = int(params.grid_rows), int(params.grid_cols)
                        if gr * gc < n_qubits:
                            raise ValueError(f"Grid too small for {name}")
                    else:
                        gr, gc = _auto_grid_dims(n_qubits)
                    run_params = _copy_params_with(run_params, grid_rows=gr, grid_cols=gc)
                    n_qpus = gr * gc
    
                aggregated = run_with_repeats(
                    qasm_path=qasm_path,
                    network=net,
                    base_seed=seed,
                    runs=runs,
                    params=run_params,
                    n_qpus=n_qpus,
                    mapping_mode=mapping_mode,
                )
    
                rows.append({
                    "experiment": "fidelity_sweep",
                    "circuit": name,
                    "network": net.value,
                    "t_deco": t_deco,
                    **aggregated,
                })
    write_csv_rows(f"{out_prefix}_raw.csv", rows)
    summary = summarize_results(rows, group_keys = ["experiment", "circuit", "network", "t_deco"])
    write_csv_rows(f"{out_prefix}_summary.csv", summary)

def run_experiment_tinit_sweep(
    circuits: List[str],
    seed: int,
    params: ExperimentParams,
    out_prefix: str,
    t_init_values: List[float],
    mapping_mode: str = "random",
    runs: int = 1,
    include_static_grid: bool = False,   # 可关闭静态网络以聚焦交换网络
):
    """
    扫描不同的 t_init 值，对比 MPQN、QMSN(原时隙)、QMSN(reconfig) 的性能。
    """
    rows = []
    networks_to_test = [NetworkType.MPQN, NetworkType.QMSN]

    for qasm_path in circuits:
        name = Path(qasm_path).stem
        n_qubits, _ = _read_openqasm_ops(qasm_path)

        for net in networks_to_test:
            # 为 QMSN 准备两种模式，MPQN 只有一种（mode 列标记为 'N/A'）
            if net == NetworkType.QMSN:
                mode_list = [
                    (False, "timeslot"),
                    (True,  "reconfig")
                ]
            else:
                mode_list = [(None, "N/A")]

            for reconfig_enabled, mode_label in mode_list:
                for t_init in t_init_values:
                    # 复制参数并修改
                    run_params = _copy_params_with(params, t_init=t_init)
                    n_qpus = n_qubits

                    # 如果是 QMSN，设置重配置开关
                    if net == NetworkType.QMSN:
                        run_params = _copy_params_with(run_params, qmsn_reconfiguration_model=reconfig_enabled)

                    # 运行多次聚合
                    aggregated = run_with_repeats(
                        qasm_path=qasm_path,
                        network=net,
                        base_seed=seed,
                        runs=runs,
                        params=run_params,
                        n_qpus=n_qpus,
                        mapping_mode=mapping_mode,
                        prefetch=True,
                        trace=False,
                    )

                    row = {
                        "experiment": "tinit_sweep",
                        "circuit": name,
                        "network": net.value,
                        "mode": mode_label,
                        "t_init": t_init,
                        **aggregated,
                    }
                    rows.append(row)

    # 输出结果
    write_csv_rows(f"{out_prefix}_raw.csv", rows)
    summary = summarize_results(rows, group_keys=["experiment", "circuit", "network", "mode", "t_init"])
    write_csv_rows(f"{out_prefix}_summary.csv", summary)

def _parse_csv_paths(text: str) -> List[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def _parse_csv_floats(text: str) -> List[float]:
    vals: List[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return vals



def main():
    here = Path(__file__).resolve().parent
    default_exp1 = ",".join(
        [
            str(here / "cnt3-5_180.qasm"),
            str(here / "adr4_197.qasm"),
            str(here / "0410184_169.qasm"),
            str(here / "z4_268.qasm"),
        ]
    )

    parser = argparse.ArgumentParser(description="Distributed quantum-circuit simulator.")

    parser.add_argument("--t-init-values", type=str, default="10,100,1000", help="Comma-separated t_init values for sweep")
    parser.add_argument("--qmsn-reconfiguration", action="store_true", help="Enable QMSN reconfiguration mode with disturbance")
    parser.add_argument("--experiment", choices=["exp1", "exp_pc", "exp_scale", "exp_network_reconfig", "exp_fidelity_sweep","exp_pent", "exp_tinit", "exp_parameter_sweep"], default="exp1")

    # parser.add_argument("--experiment", choices=["exp1", "exp_pc", "exp_scale"], default="exp1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mapping-mode", choices=["random", "sequential"], default="random")
    parser.add_argument("--out-prefix", default="experiment_results")
    parser.add_argument("--exp1-circuits", default=default_exp1)
    parser.add_argument(
        "--circuits",
        default="",
        help="Comma-separated circuits for exp_pc/exp_scale (also overrides exp1 when provided).",
    )
    parser.add_argument("--p-ent-values", type=str, default="0.001,0.0054,0.01,0.05", help="Comma-separated p_ent values for sweep")
    parser.add_argument("--runs", type=int, default=3, help="Number of repeated runs per configuration (different seeds)")
                    
    parser.add_argument("--t-1g", type=float, default=1.0)
    parser.add_argument("--t-2g", type=float, default=100.0)
    parser.add_argument("--t-retry", type=float, default=0.5)
    # parser.add_argument("--p-ent", type=float, default=5.4e-3)
    parser.add_argument("--p-swi", type=float, default=0.99)
    parser.add_argument("--ent-time-mode", choices=["sample", "mean"], default="sample")
    parser.add_argument("--t-deco-us", type=float, default=2100.0)
    parser.add_argument("--t-c", type=float, default=100.0)
    # parser.add_argument("--t-q-init", type=float, default=10.0)
    parser.add_argument("--t-init", type=float, default=10.0)
    # parser.add_argument("--t-e", type=float, default=1000.0)
    # parser.add_argument("--gamma-max-hz", type=float, default=2_000_000.0)
    parser.add_argument("--p-swap", type=float, default=0.9)
    parser.add_argument("--f-switch", type=float, default=0.9995)
    parser.add_argument("--f-trans", type=float, default=0.9999)
    parser.add_argument("--t-fid-decay", type=float, default=700.0)
    parser.add_argument("--grid-rows", type=int, default=0)
    parser.add_argument("--grid-cols", type=int, default=0)
    parser.add_argument("--delta-default", type=float, default=1.0)
    parser.add_argument("--p-e", type=float, default=0.99)
    parser.add_argument("--p-t", type=float, default=0.98)
    parser.add_argument("--p-c", type=float, default=0.12)
    parser.add_argument("--p-d", type=float, default=0.8)
    parser.add_argument("--p-c-values", type=str, default="0.04,0.08,0.12,0.16,0.20")
    parser.add_argument("--t-deco-values", type=str, default="500,1000,2100,4200", help="Comma-separated t_deco values for fidelity sweep")
    parser.add_argument("--param-name", type=str, default="", help="Parameter name to sweep")
    parser.add_argument("--param-values", type=str, default="", help="Comma-separated parameter values")

    args = parser.parse_args()

    params = ExperimentParams(
        t_1g=args.t_1g,
        t_2g=args.t_2g,
        t_retry=args.t_retry,
        p_e = args.p_e,
        p_c = args.p_c,
        p_t = args.p_t,
        p_d = args.p_d,
        # p_ent=args.p_ent,
        p_swi=args.p_swi,
        ent_time_mode=args.ent_time_mode,
        t_deco=args.t_deco_us,
        t_c=args.t_c,
        t_init=args.t_init,
        # t_m_init=args.t_m_init,
        # t_e=args.t_e,
        # gamma_max_hz=args.gamma_max_hz,
        p_swap=args.p_swap,
        f_switch=args.f_switch,
        f_trans=args.f_trans,
        t_fid_decay=args.t_fid_decay,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        grid_routing="xy",
        delta_default=args.delta_default,
        qmsn_reconfiguration_model =  args.qmsn_reconfiguration
    )

    exp1_circuits = _parse_csv_paths(args.exp1_circuits)
    generic_circuits = _parse_csv_paths(args.circuits)

    if args.experiment == "exp1":
        circuits = generic_circuits if generic_circuits else exp1_circuits
        print(f"Experiment: exp1")
        print(f"  circuits: {len(circuits)} files")
        print(f"  seed: {args.seed} (base), runs: {args.runs}")
        print(f"  mapping: {args.mapping_mode}, ent_time_mode: {args.ent_time_mode}")
        print(f"  output prefix: {args.out_prefix}_exp1")
        run_experiment_1(
            circuits=circuits,
            seed=args.seed,
            params=params,
            out_prefix=args.out_prefix,
            mapping_mode=args.mapping_mode,
            runs = args.runs
        )
    elif args.experiment == "exp_pc":
        circuits = generic_circuits if generic_circuits else exp1_circuits
        p_c_values = _parse_csv_floats(args.p_c_values)
        if not p_c_values:
            raise ValueError("--p-c-values must contain at least one numeric value.")
        print(
            "Experiment = exp_pc, "
            f"seed = {args.seed}, mapping = {args.mapping_mode}, ent_time_mode = {args.ent_time_mode}, "
            f"p_e={args.p_e}, p_t={args.p_t}, p_d={args.p_d}, p_c_values={p_c_values}"
        )
        run_experiment_pc_sweep(
            circuits=circuits,
            seed=args.seed,
            params=params,
            out_prefix=args.out_prefix,
            p_c_values=p_c_values,
            p_e=args.p_e,
            p_t=args.p_t,
            p_d=args.p_d,
            mapping_mode=args.mapping_mode,
        )
    elif args.experiment == "exp_network_reconfig":
        circuits = generic_circuits if generic_circuits else exp1_circuits
        print(f"Experiment = network_reconfig_compare, seed = {args.seed}, mapping = {args.mapping_mode}")
        run_experiment_network_reconfig_compare(
            circuits=circuits,
            seed=args.seed,
            params=params,
            out_prefix=args.out_prefix,
            mapping_mode=args.mapping_mode,
        )
    elif args.experiment == "exp_fidelity_sweep":
        circuits = generic_circuits if generic_circuits else exp1_circuits
        t_deco_vals = _parse_csv_floats(args.t_deco_values)
        print(f"Experiment = fidelity_sweep, t_deco values = {t_deco_vals}")
        run_experiment_fidelity_sweep(
            circuits=circuits,
            seed=args.seed,
            params=params,
            out_prefix=args.out_prefix,
            t_deco_values=t_deco_vals,
            mapping_mode=args.mapping_mode,
            runs=args.runs,
        )
    elif args.experiment == "exp_tinit":
        circuits = generic_circuits if generic_circuits else exp1_circuits
        t_init_vals = _parse_csv_floats(args.t_init_values)
        print(f"Experiment = t_init sweep, values = {t_init_vals}")
        run_experiment_tinit_sweep(
            circuits=circuits,
            seed=args.seed,
            params=params,
            out_prefix=args.out_prefix,
            t_init_values=t_init_vals,
            mapping_mode=args.mapping_mode,
            runs=args.runs,
        )
    elif args.experiment == "exp_parameter_sweep":
        if not args.param_name:
            raise ValueError("Please provide --param-name for generic parameter sweep")
        param_values = _parse_csv_floats(args.param_values)
        circuits = generic_circuits if generic_circuits else exp1_circuits
        print(f"Experiment: parameter sweep ({args.param_name})")
        print(f"  values: {param_values}")
        print(f"  circuits: {len(circuits)} files")
        print(f"  seed: {args.seed} (base), runs: {args.runs}")
        print(f"  output prefix: {args.out_prefix}_sweep_{args.param_name}")
        run_experiment_parameter_sweep(
            circuits=circuits,
            seed=args.seed,
            params=params,
            out_prefix=args.out_prefix,
            param_name=args.param_name,
            param_values=param_values,
            mapping_mode=args.mapping_mode,
            runs=args.runs,
            include_static_grid=True,
        )
    else:
        circuits = generic_circuits if generic_circuits else exp1_circuits
        print(
            "Experiment = exp_scale, "
            f"seed = {args.seed}, mapping = {args.mapping_mode}, ent_time_mode = {args.ent_time_mode}, "
            "n_qpus is set to n_qubits per circuit."
        )
        run_experiment_scale(
            circuits=circuits,
            seed=args.seed,
            params=params,
            out_prefix=args.out_prefix,
            mapping_mode=args.mapping_mode,
        )


if __name__ == "__main__":
    main()
