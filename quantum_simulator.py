from __future__ import annotations

import csv
import heapq
import math
import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple


NETWORK_QMSN = "QMSN"
NETWORK_MPQN = "MPQN"

EVENT_ENTANGLE_START = "ENTANGLE_START"
EVENT_ENTANGLE_SUCCESS = "ENTANGLE_SUCCESS"
EVENT_ENTANGLE_FAIL = "ENTANGLE_FAIL"
EVENT_DECOHERENCE = "DECOHERENCE"
# EVENT_GATE_START is kept for API completeness but not emitted in this simulator.
# We schedule gate completion directly because local/remote gate durations are
# deterministic constants.
EVENT_GATE_START = "GATE_START"
EVENT_GATE_END = "GATE_END"
EVENT_NETWORK_CYCLE = "NETWORK_CYCLE"


@dataclass(frozen=True)
class Operation:
    name: str
    qubits: Tuple[int, ...]


@dataclass(frozen=True)
class Circuit:
    operations: Sequence[Operation]
    qubit_to_qpu: Dict[int, int]


@dataclass
class Gate:
    gate_id: int
    name: str
    qubits: Tuple[int, ...]
    is_remote: bool
    src_qpu: Optional[int]
    dst_qpu: Optional[int]
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)


@dataclass
class EntanglementRequest:
    request_id: int
    gate_id: int
    src_qpu: int
    dst_qpu: int
    earliest_start: float = 0.0
    latest_start: float = 0.0
    ready_time: float = 0.0
    retry_count: int = 0
    state: str = "new"
    order: int = 0

    # Pair/attempt lifecycle fields
    attempt_start_time: Optional[float] = None
    bell_ready_time: Optional[float] = None
    decohere_time: Optional[float] = None

    # QMSN cycle assignment metadata
    assigned_cycle_start: Optional[float] = None
    assigned_cycle_end: Optional[float] = None

    failure_reason: Optional[str] = None

    # Monotonic token used to ignore stale queued events from old attempts.
    active_token: int = 0


@dataclass(frozen=True)
class SimulationParams:
    p_e: float
    p_t: float
    p_net: float
    p_d: float
    t_retry: float
    t_decoherence: float
    t_local: float
    t_remote: float
    qmsn_period: float
    max_retries: Optional[int] = None
    phase_error_mzi: float = 0.0


@dataclass(order=True)
class Event:
    time: float
    seq: int
    kind: str = field(compare=False)
    payload: object = field(compare=False)


@dataclass
class SimulationResult:
    execution_time: float
    network: str
    n_qpus: int
    seed: int
    avg_remote_fidelity: Optional[float]
    completed_gates: int
    total_gates: int
    failed_requests: int
    expired_pairs: int
    retry_events: int
    remote_gate_count: int


@dataclass
class ExperimentSummary:
    rows: List[Dict[str, float]]
    summary_rows: List[Dict[str, float]]


class DistributedQuantumSimulator:
    """Strict event-driven simulator for QMSN / MPQN distributed circuits."""

    def __init__(
        self,
        n_qpus: int,
        circuit: Circuit,
        params: SimulationParams,
        network: str,
        seed: int,
    ) -> None:
        if network not in {NETWORK_QMSN, NETWORK_MPQN}:
            raise ValueError(f"Unknown network type: {network}")
        if n_qpus < 2:
            raise ValueError("n_qpus must be >= 2")

        self.n_qpus = n_qpus
        self.circuit = circuit
        self.params = params
        self.network = network
        self.seed = seed
        self.rng = random.Random(seed)

        (
            self.gates,
            self.request_by_gate,
            self.requests,
            self.remote_request_order,
        ) = self._preprocess(circuit)
        self.total_gates = len(self.gates)

        self.time = 0.0
        self._event_seq = 0
        self.event_queue: List[Event] = []

        self.gate_pred_remaining: Dict[int, int] = {
            g.gate_id: len(g.predecessors) for g in self.gates
        }
        self.gate_ready_time: Dict[int, float] = {g.gate_id: 0.0 for g in self.gates}
        self.gate_done_time: Dict[int, float] = {}

        # Communication resource tracking
        self.comm_busy: List[bool] = [False] * n_qpus
        self.comm_owner: List[Optional[int]] = [None] * n_qpus

        # MPQN queue
        self.mpqn_waiting: List[int] = []
        self.mpqn_waiting_set: Set[int] = set()

        # QMSN pending set and cycle reservations
        self.qmsn_pending_ids: Set[int] = set()
        self.qmsn_busy_until: List[float] = [0.0] * n_qpus

        # Global counters / terminal flags
        self.expired_pairs_count = 0
        self.retry_events_count = 0
        self.failed_request_ids: Set[int] = set()
        self.terminal_failure = False

        if self.network == NETWORK_QMSN:
            self._push_event(0.0, EVENT_NETWORK_CYCLE, None)

        for gate in self.gates:
            if self.gate_pred_remaining[gate.gate_id] == 0:
                self._schedule_gate(gate.gate_id, 0.0)

    def _push_event(self, time: float, kind: str, payload: object) -> None:
        self._event_seq += 1
        heapq.heappush(
            self.event_queue,
            Event(time=time, seq=self._event_seq, kind=kind, payload=payload),
        )

    def _advance_token(self, req: EntanglementRequest) -> int:
        req.active_token += 1
        return req.active_token

    def _request_payload(self, req: EntanglementRequest) -> Tuple[int, int]:
        return (req.request_id, req.active_token)

    def _unpack_request_payload(self, payload: object) -> Tuple[int, int]:
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise RuntimeError(f"Invalid request event payload: {payload}")
        req_id, token = payload
        return int(req_id), int(token)

    def _is_stale_request_event(self, req: EntanglementRequest, token: int) -> bool:
        return token != req.active_token

    def run(self) -> SimulationResult:
        max_events = 20_000_000
        processed_events = 0

        while len(self.gate_done_time) < self.total_gates and not self.terminal_failure:
            if not self.event_queue:
                if self.failed_request_ids:
                    break
                raise RuntimeError("Event queue exhausted before all gates completed")

            event = heapq.heappop(self.event_queue)
            self.time = event.time
            processed_events += 1
            if processed_events > max_events:
                raise RuntimeError("Exceeded max event limit, possible deadlock")

            self._process_event(event)

        avg_fidelity = None
        if self.network == NETWORK_MPQN and self._count_remote_gates() > 0:
            avg_fidelity = self._mpqn_fidelity()

        return SimulationResult(
            execution_time=max(self.gate_done_time.values()) if self.gate_done_time else self.time,
            network=self.network,
            n_qpus=self.n_qpus,
            seed=self.seed,
            avg_remote_fidelity=avg_fidelity,
            completed_gates=len(self.gate_done_time),
            total_gates=self.total_gates,
            failed_requests=len(self.failed_request_ids),
            expired_pairs=self.expired_pairs_count,
            retry_events=self.retry_events_count,
            remote_gate_count=len(self.remote_request_order),
        )

    def _process_event(self, event: Event) -> None:
        if event.kind == EVENT_GATE_END:
            self._on_gate_end(int(event.payload))
        elif event.kind == EVENT_ENTANGLE_START:
            self._on_entangle_start(event.payload)
        elif event.kind == EVENT_ENTANGLE_SUCCESS:
            self._on_entangle_success(event.payload)
        elif event.kind == EVENT_ENTANGLE_FAIL:
            self._on_entangle_fail(event.payload)
        elif event.kind == EVENT_DECOHERENCE:
            self._on_decoherence(event.payload)
        elif event.kind == EVENT_NETWORK_CYCLE:
            self._process_qmsn_cycle(self.time)
        else:
            raise RuntimeError(f"Unknown event kind: {event.kind}")

    def _count_remote_gates(self) -> int:
        return len(self.remote_request_order)

    def _gate_is_ready(self, gate_id: int) -> bool:
        return gate_id not in self.gate_done_time and self.gate_pred_remaining[gate_id] == 0

    def _clear_pair_timing_fields(self, req: EntanglementRequest) -> None:
        req.attempt_start_time = None
        req.bell_ready_time = None
        req.decohere_time = None

    def _has_live_bell_pair(self, req: EntanglementRequest, now: float) -> bool:
        if req.state != "ready":
            return False
        if req.bell_ready_time is None or req.decohere_time is None:
            return False
        return now < req.decohere_time

    def _can_start_remote_gate_with_pair(self, req: EntanglementRequest, now: float) -> bool:
        # latest_start captures whether using a pair now is still meaningful for the
        # gate timing window. decohere_time captures physical lifetime of THIS pair.
        # A late success is rejected in _on_entangle_success(); this helper only
        # evaluates already-ready pairs.
        if not self._has_live_bell_pair(req, now):
            return False
        if req.decohere_time is None:
            return False
        if now > req.latest_start:
            return False
        return (now + self.params.t_remote) <= req.decohere_time

    def _on_gate_end(self, gate_id: int) -> None:
        if gate_id in self.gate_done_time:
            return

        gate = self.gates[gate_id]
        self.gate_done_time[gate_id] = self.time

        if gate.is_remote:
            req = self.request_by_gate[gate_id]
            req.state = "completed"
            req.failure_reason = None
            self._clear_pair_timing_fields(req)
            self.qmsn_pending_ids.discard(req.request_id)
            self._release_comm_qubits(req, self.time)

        for succ in gate.successors:
            self.gate_pred_remaining[succ] -= 1
            self.gate_ready_time[succ] = max(self.gate_ready_time[succ], self.time)
            if self.gate_pred_remaining[succ] == 0:
                self._schedule_gate(succ, self.gate_ready_time[succ])

    def _schedule_gate(self, gate_id: int, t_now: float) -> None:
        if gate_id in self.gate_done_time:
            return

        gate = self.gates[gate_id]

        if not gate.is_remote:
            self._push_event(t_now + self.params.t_local, EVENT_GATE_END, gate_id)
            return

        req = self.request_by_gate[gate_id]
        if req.state == "failed":
            self.terminal_failure = True
            return

        req.earliest_start = t_now
        req.latest_start = t_now + self.params.t_decoherence - self.params.t_remote
        req.ready_time = t_now

        if self._can_start_remote_gate_with_pair(req, t_now):
            req.state = "consuming"
            self._push_event(t_now + self.params.t_remote, EVENT_GATE_END, gate_id)
            return

        self._submit_request(req, t_now)

    def _submit_request(self, req: EntanglementRequest, t_now: float) -> None:
        if req.state in {"completed", "failed", "consuming"}:
            return
        if self._can_start_remote_gate_with_pair(req, t_now):
            req.state = "consuming"
            self._push_event(t_now + self.params.t_remote, EVENT_GATE_END, req.gate_id)
            return

        if self.network == NETWORK_MPQN:
            self._submit_request_mpqn(req, t_now)
        else:
            self._submit_request_qmsn(req)

    def _can_allocate_mpqn(self, req: EntanglementRequest) -> bool:
        for qpu in (req.src_qpu, req.dst_qpu):
            owner = self.comm_owner[qpu]
            if owner is not None and owner != req.request_id:
                return False
        return True

    def _reserve_mpqn_comm_qubits(self, req: EntanglementRequest) -> bool:
        if not self._can_allocate_mpqn(req):
            return False

        for qpu in (req.src_qpu, req.dst_qpu):
            self.comm_busy[qpu] = True
            self.comm_owner[qpu] = req.request_id
        return True

    def _enqueue_mpqn(self, req_id: int) -> None:
        if req_id in self.mpqn_waiting_set:
            return
        self.mpqn_waiting.append(req_id)
        self.mpqn_waiting_set.add(req_id)

    def _launch_mpqn_attempt(self, req: EntanglementRequest, t_now: float) -> bool:
        if not self._reserve_mpqn_comm_qubits(req):
            return False

        req.state = "inflight"
        token = self._advance_token(req)
        self._push_event(t_now, EVENT_ENTANGLE_START, (req.request_id, token))
        return True

    def _submit_request_mpqn(self, req: EntanglementRequest, t_now: float) -> None:
        if req.state in {"completed", "failed", "consuming", "ready"}:
            return

        if self._launch_mpqn_attempt(req, t_now):
            return

        req.state = "queued"
        self._enqueue_mpqn(req.request_id)

    def _submit_request_qmsn(self, req: EntanglementRequest) -> None:
        if req.state in {"completed", "failed", "consuming", "ready"}:
            return

        req.state = "pending"
        self.qmsn_pending_ids.add(req.request_id)

    def _on_entangle_start(self, payload: object) -> None:
        req_id, token = self._unpack_request_payload(payload)
        req = self.requests[req_id]

        if self._is_stale_request_event(req, token):
            return
        if req.state != "inflight":
            return

        if self.network == NETWORK_MPQN:
            if not self._reserve_mpqn_comm_qubits(req):
                req.state = "queued"
                self._enqueue_mpqn(req.request_id)
                return

            req.attempt_start_time = self.time
            p_attempt = self._per_attempt_success_probability(self._stage_count_mpqn())
            if p_attempt <= 0.0:
                self._push_event(self.time, EVENT_ENTANGLE_FAIL, (req.request_id, token))
                return

            k = self._sample_geometric_trials(p_attempt)
            t_success = self.time + k * self.params.t_retry
            self._push_event(t_success, EVENT_ENTANGLE_SUCCESS, (req.request_id, token))
            return

        # QMSN attempt timing is sampled at cycle processing time. We do not emit
        # per-request ENTANGLE_START events for QMSN to keep cycle semantics explicit.

    def _on_entangle_success(self, payload: object) -> None:
        req_id, token = self._unpack_request_payload(payload)
        req = self.requests[req_id]

        if self._is_stale_request_event(req, token):
            return
        if req.state != "inflight":
            return

        # A Bell pair that succeeds after latest_start is logically useless for the
        # remote gate under the specification, even if it has not yet physically
        # decohered. In that case we discard it immediately and re-request.
        if self.time > req.latest_start:
            self._re_request(req, self.time, reason="late_success")
            return

        req.state = "ready"
        req.bell_ready_time = self.time
        req.decohere_time = self.time + self.params.t_decoherence

        self._push_event(req.decohere_time, EVENT_DECOHERENCE, (req.request_id, token))

        if self._gate_is_ready(req.gate_id) and self._can_start_remote_gate_with_pair(req, self.time):
            req.state = "consuming"
            self._push_event(self.time + self.params.t_remote, EVENT_GATE_END, req.gate_id)

    def _on_decoherence(self, payload: object) -> None:
        req_id, token = self._unpack_request_payload(payload)
        req = self.requests[req_id]

        if self._is_stale_request_event(req, token):
            return
        if req.state != "ready":
            return
        if req.decohere_time is None or self.time + 1e-15 < req.decohere_time:
            return

        req.state = "expired"
        self.expired_pairs_count += 1
        self._clear_pair_timing_fields(req)

        # MPQN releases immediately. QMSN cycle reservation is NOT released here.
        self._release_comm_qubits(req, self.time)
        self._re_request(req, self.time, reason="decoherence_expired")

    def _on_entangle_fail(self, payload: object) -> None:
        req_id, token = self._unpack_request_payload(payload)
        req = self.requests[req_id]

        if self._is_stale_request_event(req, token):
            return
        if req.state not in {"inflight", "ready", "queued", "pending"}:
            return

        self._re_request(req, self.time, reason="entanglement_attempt_failed")

    def _mark_permanent_failure(self, req: EntanglementRequest, reason: str) -> None:
        req.state = "failed"
        req.failure_reason = reason
        self._clear_pair_timing_fields(req)
        self.qmsn_pending_ids.discard(req.request_id)
        self._release_comm_qubits(req, self.time)

        if req.request_id not in self.failed_request_ids:
            self.failed_request_ids.add(req.request_id)

        # Explicit stop condition for circuits with unrecoverable remote requests.
        self.terminal_failure = True

    def _re_request(self, req: EntanglementRequest, t_now: float, reason: str) -> None:
        self._clear_pair_timing_fields(req)
        self._release_comm_qubits(req, t_now)

        req.retry_count += 1
        self.retry_events_count += 1

        if self.params.max_retries is not None and req.retry_count > self.params.max_retries:
            self._mark_permanent_failure(req, reason=f"max_retries_exceeded:{reason}")
            return

        req.failure_reason = None
        req.state = "pending"
        req.earliest_start = t_now
        req.latest_start = t_now + self.params.t_decoherence - self.params.t_remote
        req.ready_time = t_now
        self._submit_request(req, t_now)

    def _release_comm_qubits(self, req: EntanglementRequest, t_now: float) -> None:
        if self.network == NETWORK_MPQN:
            for qpu in (req.src_qpu, req.dst_qpu):
                if self.comm_owner[qpu] == req.request_id:
                    self.comm_busy[qpu] = False
                    self.comm_owner[qpu] = None

            # MPQN can immediately dispatch queued requests when resources free up.
            self._dispatch_mpqn_queue(t_now)
            return

        # QMSN cycle reservation is controlled exclusively by qmsn_busy_until / cycle end.
        # We intentionally do not release resources here.

    def _dispatch_mpqn_queue(self, t_now: float) -> None:
        if not self.mpqn_waiting:
            return

        rounds = len(self.mpqn_waiting)
        for _ in range(rounds):
            req_id = self.mpqn_waiting.pop(0)
            self.mpqn_waiting_set.discard(req_id)
            req = self.requests[req_id]

            if req.state != "queued":
                continue
            if req.state in {"completed", "failed"}:
                continue
            if self._has_live_bell_pair(req, t_now):
                continue

            if self._launch_mpqn_attempt(req, t_now):
                continue

            self._enqueue_mpqn(req_id)

    def _process_qmsn_cycle(self, t_cycle: float) -> None:
        eligible = [
            self.requests[req_id]
            for req_id in self.qmsn_pending_ids
            if self.requests[req_id].earliest_start <= t_cycle
            and self.requests[req_id].state in {"pending", "queued"}
        ]
        eligible.sort(key=lambda r: (r.earliest_start, r.order))

        selected: List[EntanglementRequest] = []
        used_qpus: Set[int] = set()
        max_pairs = self.n_qpus // 2

        for req in eligible:
            if len(selected) >= max_pairs:
                break
            if req.src_qpu in used_qpus or req.dst_qpu in used_qpus:
                continue
            if self.qmsn_busy_until[req.src_qpu] > t_cycle or self.qmsn_busy_until[req.dst_qpu] > t_cycle:
                continue
            selected.append(req)
            used_qpus.add(req.src_qpu)
            used_qpus.add(req.dst_qpu)

        stage_count = self._stage_count_qmsn()
        p_attempt = self._per_attempt_success_probability(stage_count)
        max_attempts = int(self.params.qmsn_period // self.params.t_retry)

        for req in selected:
            self.qmsn_pending_ids.discard(req.request_id)

            req.state = "inflight"
            req.assigned_cycle_start = t_cycle
            req.assigned_cycle_end = t_cycle + self.params.qmsn_period
            req.attempt_start_time = t_cycle

            self.qmsn_busy_until[req.src_qpu] = req.assigned_cycle_end
            self.qmsn_busy_until[req.dst_qpu] = req.assigned_cycle_end

            token = self._advance_token(req)
            if max_attempts <= 0 or p_attempt <= 0.0:
                req.state = "pending"
                req.attempt_start_time = None
                self.qmsn_pending_ids.add(req.request_id)
                continue

            k = self._sample_geometric_trials(p_attempt)
            if k <= max_attempts:
                t_success = t_cycle + k * self.params.t_retry
                self._push_event(t_success, EVENT_ENTANGLE_SUCCESS, (req.request_id, token))
            else:
                # No success in this cycle; reservation still lasts until cycle end.
                req.state = "pending"
                req.attempt_start_time = None
                self.qmsn_pending_ids.add(req.request_id)

        self._push_event(t_cycle + self.params.qmsn_period, EVENT_NETWORK_CYCLE, None)

    def _preprocess(
        self, circuit: Circuit
    ) -> Tuple[
        List[Gate],
        Dict[int, EntanglementRequest],
        Dict[int, EntanglementRequest],
        List[int],
    ]:
        gates: List[Gate] = []
        request_by_gate: Dict[int, EntanglementRequest] = {}
        requests: Dict[int, EntanglementRequest] = {}
        remote_request_order: List[int] = []

        last_gate_on_qubit: Dict[int, int] = {}

        for gate_id, op in enumerate(circuit.operations):
            qubits = tuple(op.qubits)
            preds: Set[int] = set()
            for qubit in qubits:
                if qubit in last_gate_on_qubit:
                    preds.add(last_gate_on_qubit[qubit])

            is_remote = False
            src_qpu = None
            dst_qpu = None
            if len(qubits) == 2:
                qpu_a = circuit.qubit_to_qpu[qubits[0]]
                qpu_b = circuit.qubit_to_qpu[qubits[1]]
                if qpu_a != qpu_b:
                    is_remote = True
                    src_qpu = qpu_a
                    dst_qpu = qpu_b

            gate = Gate(
                gate_id=gate_id,
                name=op.name,
                qubits=qubits,
                is_remote=is_remote,
                src_qpu=src_qpu,
                dst_qpu=dst_qpu,
                predecessors=sorted(preds),
            )
            gates.append(gate)

            for qubit in qubits:
                last_gate_on_qubit[qubit] = gate_id

        for gate in gates:
            for pred in gate.predecessors:
                gates[pred].successors.append(gate.gate_id)

        req_id = 0
        for gate in gates:
            if gate.is_remote:
                req = EntanglementRequest(
                    request_id=req_id,
                    gate_id=gate.gate_id,
                    src_qpu=gate.src_qpu if gate.src_qpu is not None else -1,
                    dst_qpu=gate.dst_qpu if gate.dst_qpu is not None else -1,
                    order=req_id,
                )
                request_by_gate[gate.gate_id] = req
                requests[req_id] = req
                remote_request_order.append(req_id)
                req_id += 1

        # remote_request_order preserves topological creation order from preprocessing.
        # Runtime submission is still dependency-driven by gate readiness.
        return gates, request_by_gate, requests, remote_request_order

    def _stage_count_qmsn(self) -> int:
        return int(math.ceil(math.log2(self.n_qpus)))

    def _stage_count_mpqn(self) -> int:
        qmsn_stages = self._stage_count_qmsn()
        p = max(1, qmsn_stages - 1)
        extra = int(math.ceil(math.log2(p))) if p > 1 else 0
        return qmsn_stages + extra

    def _per_attempt_success_probability(self, n_stages: int) -> float:
        p = (
            0.5
            * (self.params.p_e**2)
            * (self.params.p_t**2)
            * (self.params.p_net**n_stages)
            * self.params.p_d
        )
        return min(max(p, 0.0), 1.0)

    def _sample_geometric_trials(self, p: float) -> int:
        if p <= 0.0:
            return 10**18
        if p >= 1.0:
            return 1

        u = self.rng.random()
        if u >= 1.0:
            u = math.nextafter(1.0, 0.0)
        if u <= 0.0:
            u = math.nextafter(0.0, 1.0)

        return int(math.floor(math.log(1.0 - u) / math.log(1.0 - p)) + 1)

    def _mpqn_fidelity(self) -> float:
        eps = self.params.phase_error_mzi
        qmsn_stages = self._stage_count_qmsn()
        planes = max(1, qmsn_stages - 1)
        n1 = int(math.ceil(math.log2(planes))) if planes > 1 else 0
        n2 = float(planes)
        fidelity = 1.0 - ((2.0 * n1 + n2) / 2.0) * (eps**2)
        return max(0.0, min(1.0, fidelity))


def generate_ghz_family_circuit(n_qpus: int) -> Circuit:
    if n_qpus < 2:
        raise ValueError("GHZ family requires n_qpus >= 2")

    qubit_to_qpu = {q: q for q in range(n_qpus)}
    operations: List[Operation] = [Operation(name="h", qubits=(0,))]
    for q in range(1, n_qpus):
        operations.append(Operation(name="cx", qubits=(0, q)))

    return Circuit(operations=operations, qubit_to_qpu=qubit_to_qpu)


def build_circuit_from_qasm(
    qasm_path: Path | str,
    n_qpus: int,
    mapping: str = "round_robin",
) -> Circuit:
    """
    Parse a subset of OpenQASM 2.0 and build a Circuit.
    Supported: qreg declarations and gate instructions on qreg qubits.
    Ignored: creg, barrier, measure, reset.
    """
    if n_qpus < 2:
        raise ValueError("n_qpus must be >= 2")

    mapping_mode = mapping.lower().strip()
    if mapping_mode not in {"round_robin", "block", "one_to_one"}:
        raise ValueError(f"Unsupported mapping mode: {mapping}")

    qasm_file = Path(qasm_path)
    text = qasm_file.read_text(encoding="utf-8")

    qreg_base: Dict[str, int] = {}
    qreg_size: Dict[str, int] = {}
    total_qubits = 0
    operations: List[Operation] = []

    qreg_re = re.compile(r"^qreg\s+([A-Za-z_]\w*)\[(\d+)\]\s*;$")
    gate_re = re.compile(r"^([A-Za-z_]\w*)\s*(\([^;]*\))?\s+(.+)\s*;$")
    qubit_re = re.compile(r"^([A-Za-z_]\w*)\[(\d+)\]$")

    for raw_line in text.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue

        lower = line.lower()
        if lower.startswith("openqasm") or lower.startswith("include"):
            continue

        qreg_match = qreg_re.match(line)
        if qreg_match:
            reg_name = qreg_match.group(1)
            reg_len = int(qreg_match.group(2))
            if reg_name in qreg_base:
                raise ValueError(f"Duplicate qreg: {reg_name}")
            qreg_base[reg_name] = total_qubits
            qreg_size[reg_name] = reg_len
            total_qubits += reg_len
            continue

        if (
            lower.startswith("creg ")
            or lower.startswith("barrier ")
            or lower.startswith("measure ")
            or lower.startswith("reset ")
        ):
            continue

        gate_match = gate_re.match(line)
        if not gate_match:
            continue

        gate_name = gate_match.group(1).lower()
        operand_str = gate_match.group(3)
        operands = [item.strip() for item in operand_str.split(",")]

        gate_qubits: List[int] = []
        for operand in operands:
            qb = qubit_re.match(operand)
            if qb is None:
                continue
            reg = qb.group(1)
            idx = int(qb.group(2))
            if reg not in qreg_base:
                raise ValueError(f"Unknown qreg in gate operand: {operand}")
            if idx < 0 or idx >= qreg_size[reg]:
                raise ValueError(f"Qubit index out of range: {operand}")
            gate_qubits.append(qreg_base[reg] + idx)

        if gate_qubits:
            operations.append(Operation(name=gate_name, qubits=tuple(gate_qubits)))

    if total_qubits == 0:
        raise ValueError(f"No qreg found in QASM file: {qasm_file}")

    if mapping_mode == "one_to_one" and n_qpus < total_qubits:
        raise ValueError(
            f"one_to_one mapping requires n_qpus >= qubits ({n_qpus} < {total_qubits})"
        )

    qubit_to_qpu: Dict[int, int] = {}
    if mapping_mode == "round_robin":
        for q in range(total_qubits):
            qubit_to_qpu[q] = q % n_qpus
    elif mapping_mode == "block":
        block_size = int(math.ceil(total_qubits / n_qpus))
        for q in range(total_qubits):
            qubit_to_qpu[q] = min(n_qpus - 1, q // block_size)
    else:
        for q in range(total_qubits):
            qubit_to_qpu[q] = q

    return Circuit(operations=operations, qubit_to_qpu=qubit_to_qpu)


def run_execution_vs_n(
    n_values: Sequence[int],
    seeds: Sequence[int],
    base_params: SimulationParams,
    circuit_builder: Optional[Callable[[int], Circuit]] = None,
) -> ExperimentSummary:
    rows: List[Dict[str, float]] = []
    builder = circuit_builder or generate_ghz_family_circuit

    for n_qpus in n_values:
        circuit = builder(n_qpus)
        for seed in seeds:
            for network in (NETWORK_QMSN, NETWORK_MPQN):
                sim = DistributedQuantumSimulator(
                    n_qpus=n_qpus,
                    circuit=circuit,
                    params=base_params,
                    network=network,
                    seed=seed,
                )
                result = sim.run()
                rows.append(
                    {
                        "experiment": 1.0,
                        "network": network,
                        "n_qpus": float(n_qpus),
                        "seed": float(seed),
                        "execution_time": result.execution_time,
                        "avg_remote_fidelity": (
                            result.avg_remote_fidelity
                            if result.avg_remote_fidelity is not None
                            else float("nan")
                        ),
                        "failed_requests": float(result.failed_requests),
                        "expired_pairs": float(result.expired_pairs),
                        "retry_events": float(result.retry_events),
                        "remote_gate_count": float(result.remote_gate_count),
                    }
                )

    summary = _aggregate_mean_std(
        rows,
        key_fields=("network", "n_qpus"),
        value_field="execution_time",
    )
    return ExperimentSummary(rows=rows, summary_rows=summary)


def run_execution_vs_insertion_loss(
    n_qpus_fixed: int,
    loss_stage_db_values: Sequence[float],
    seeds: Sequence[int],
    base_params: SimulationParams,
    circuit_builder: Optional[Callable[[int], Circuit]] = None,
) -> ExperimentSummary:
    rows: List[Dict[str, float]] = []
    builder = circuit_builder or generate_ghz_family_circuit
    circuit = builder(n_qpus_fixed)

    for loss_db in loss_stage_db_values:
        p_net = 10 ** (-loss_db / 10.0)
        params = replace(base_params, p_net=p_net)

        for seed in seeds:
            for network in (NETWORK_QMSN, NETWORK_MPQN):
                sim = DistributedQuantumSimulator(
                    n_qpus=n_qpus_fixed,
                    circuit=circuit,
                    params=params,
                    network=network,
                    seed=seed,
                )
                result = sim.run()
                rows.append(
                    {
                        "experiment": 2.0,
                        "network": network,
                        "n_qpus": float(n_qpus_fixed),
                        "seed": float(seed),
                        "loss_stage_db": loss_db,
                        "p_net": p_net,
                        "execution_time": result.execution_time,
                        "avg_remote_fidelity": (
                            result.avg_remote_fidelity
                            if result.avg_remote_fidelity is not None
                            else float("nan")
                        ),
                        "failed_requests": float(result.failed_requests),
                        "expired_pairs": float(result.expired_pairs),
                        "retry_events": float(result.retry_events),
                        "remote_gate_count": float(result.remote_gate_count),
                    }
                )

    summary = _aggregate_mean_std(
        rows,
        key_fields=("network", "loss_stage_db"),
        value_field="execution_time",
    )
    return ExperimentSummary(rows=rows, summary_rows=summary)


def write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError("No rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_mean_std(
    rows: Sequence[Dict[str, float]],
    key_fields: Tuple[str, ...],
    value_field: str,
) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[object, ...], List[float]] = {}

    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped.setdefault(key, []).append(float(row[value_field]))

    summary_rows: List[Dict[str, float]] = []
    for key, values in sorted(grouped.items(), key=lambda item: item[0]):
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0

        row: Dict[str, float] = {}
        for idx, field in enumerate(key_fields):
            row[field] = float(key[idx]) if isinstance(key[idx], (int, float)) else key[idx]  # type: ignore[assignment]
        row["mean_execution_time"] = mean
        row["std_execution_time"] = std
        row["samples"] = float(len(values))
        summary_rows.append(row)

    return summary_rows


# -------------------------
# Minimal self-checks (E1-E4)
# -------------------------

def _make_one_remote_circuit() -> Circuit:
    return Circuit(
        operations=[Operation(name="cx", qubits=(0, 1))],
        qubit_to_qpu={0: 0, 1: 1},
    )


def _self_check_e1_mpqn_decoherence() -> None:
    circuit = _make_one_remote_circuit()
    params = SimulationParams(
        p_e=1.0,
        p_t=1.0,
        p_net=1.0,
        p_d=1.0,
        t_retry=1.0,
        t_decoherence=2.0,
        t_local=0.1,
        t_remote=0.5,
        qmsn_period=10.0,
        max_retries=0,
    )
    sim = DistributedQuantumSimulator(2, circuit, params, NETWORK_MPQN, seed=1)
    # Force the gate to be "not ready" at success time so the fresh pair stays in
    # READY state and must eventually expire via DECOHERENCE.
    sim.gate_pred_remaining[0] = 1

    saw_success = False
    saw_deco_event = False
    while sim.event_queue:
        ev = heapq.heappop(sim.event_queue)
        sim.time = ev.time
        sim._process_event(ev)
        if ev.kind == EVENT_ENTANGLE_SUCCESS:
            saw_success = True
            saw_deco_event = any(e.kind == EVENT_DECOHERENCE for e in sim.event_queue)
            break

    assert saw_success, "E1: expected ENTANGLE_SUCCESS"
    assert saw_deco_event, "E1: expected DECOHERENCE event to be scheduled"

    result = sim.run()
    assert result.expired_pairs >= 1, "E1: expected at least one expired pair"
    assert result.failed_requests >= 1, "E1: expected permanent failure after retry limit"


def _self_check_e2_qmsn_no_same_cycle_reuse() -> None:
    # Two independent remote requests sharing QPU 0, both ready at t=0.
    circuit = Circuit(
        operations=[
            Operation(name="cx", qubits=(0, 2)),
            Operation(name="cx", qubits=(1, 3)),
        ],
        qubit_to_qpu={0: 0, 1: 0, 2: 1, 3: 2},
    )
    params = SimulationParams(
        p_e=1.0,
        p_t=1.0,
        p_net=1.0,
        p_d=1.0,
        t_retry=1.0,
        t_decoherence=100.0,
        t_local=0.1,
        t_remote=0.5,
        qmsn_period=10.0,
        max_retries=5,
    )
    sim = DistributedQuantumSimulator(4, circuit, params, NETWORK_QMSN, seed=2)
    result = sim.run()
    assert result.failed_requests == 0, "E2: unexpected permanent failure"

    # Gate 1 cannot complete in cycle 0 if gate 0 consumed shared comm pair first.
    assert sim.gate_done_time[1] >= params.qmsn_period + params.t_retry + params.t_remote - 1e-12, (
        "E2: request reused shared QPU within same cycle"
    )


def _self_check_e3_retry_limit_failure() -> None:
    circuit = _make_one_remote_circuit()
    params = SimulationParams(
        p_e=0.0,
        p_t=1.0,
        p_net=1.0,
        p_d=1.0,
        t_retry=1.0,
        t_decoherence=1.0,
        t_local=0.1,
        t_remote=0.2,
        qmsn_period=10.0,
        max_retries=1,
    )
    sim = DistributedQuantumSimulator(2, circuit, params, NETWORK_MPQN, seed=3)
    result = sim.run()
    assert result.failed_requests == 1, "E3: expected permanent failed request"
    assert sim.terminal_failure, "E3: expected explicit terminal failure flag"


def _self_check_e4_relation1_scaling() -> None:
    c4 = generate_ghz_family_circuit(4)
    c8 = generate_ghz_family_circuit(8)

    def remote_count(c: Circuit) -> int:
        return sum(
            1
            for op in c.operations
            if len(op.qubits) == 2
            and c.qubit_to_qpu[op.qubits[0]] != c.qubit_to_qpu[op.qubits[1]]
        )

    assert remote_count(c8) > remote_count(c4), "E4: remote gate count should scale with N"


def _self_check_late_success_immediate_retry() -> None:
    circuit = _make_one_remote_circuit()
    params = SimulationParams(
        p_e=1.0,
        p_t=1.0,
        p_net=1.0,
        p_d=1.0,
        t_retry=2.0,
        t_decoherence=1.0,
        t_local=0.1,
        t_remote=0.6,
        qmsn_period=10.0,
        max_retries=3,
    )
    sim = DistributedQuantumSimulator(2, circuit, params, NETWORK_MPQN, seed=11)
    req = sim.requests[0]

    saw_first_success = False
    while sim.event_queue and not saw_first_success:
        ev = heapq.heappop(sim.event_queue)
        sim.time = ev.time
        if ev.kind == EVENT_ENTANGLE_SUCCESS:
            # First success occurs at t=2 while latest_start = 0.4, so this is late.
            assert sim.time > req.latest_start
            sim._process_event(ev)
            saw_first_success = True
        else:
            sim._process_event(ev)

    assert saw_first_success, "late-success self-check: expected ENTANGLE_SUCCESS event"
    assert req.state != "ready", "late-success self-check: late success must not produce ready pair"
    assert req.retry_count >= 1, "late-success self-check: late success must trigger immediate retry"
    assert all(e.kind != EVENT_DECOHERENCE for e in sim.event_queue), (
        "late-success self-check: no DECOHERENCE should be scheduled for late success"
    )


def run_self_checks() -> None:
    _self_check_e1_mpqn_decoherence()
    _self_check_e2_qmsn_no_same_cycle_reuse()
    _self_check_e3_retry_limit_failure()
    _self_check_e4_relation1_scaling()
    _self_check_late_success_immediate_retry()


if __name__ == "__main__":
    run_self_checks()
    print("All simulator self-checks passed.")
