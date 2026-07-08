"""
MTDRL-inspired SFCR embedding algorithm.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import copy
import random
from typing import Any, Deque, Optional, cast

import networkx as nx
import numpy as np
import tensorflow as tf
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology

from calibrate.demand_predictor import DemandPredictor
from constants.topology import SERVER, SFCC


@dataclass
class MTDRLConfig:
    """
    Hyperparameters for MTDRL training/inference.
    """

    learningRate: float = 1e-3
    gamma: float = 0.95
    lambdaTrace: float = 0.9
    epsilon: float = 1.0
    epsilonMin: float = 0.05
    epsilonDecay: float = 0.995
    replayCapacity: int = 5000
    batchSize: int = 64
    targetUpdateSteps: int = 200
    trainingEpisodes: int = 150
    historyLength: int = 4
    lstmUnits: int = 128
    sharedUnits: int = 128
    taskUnits: int = 64
    congestionWeight: float = 0.35
    delayWeight: float = 0.35
    bandwidthWeight: float = 0.30
    congestionRewardBlend: float = 0.50
    eligibilityClip: float = 10.0
    positionReward: float = 0.10
    invalidActionPenalty: float = 1.0
    invalidPathPenalty: float = 1.0
    routeDelayWeight: float = 0.55
    routeBandwidthWeight: float = 0.45
    defaultBandwidthDemand: float = 1.0
    defaultLatencyBudget: float = 200.0
    defaultIngressTraffic: float = 1.0


@dataclass
class Transition:
    """
    Replay transition.
    """

    state: np.ndarray
    action: int
    reward: float
    taskRewards: np.ndarray
    nextState: np.ndarray
    done: bool
    nextMask: np.ndarray


class MTDRLQNetwork(tf.keras.Model):
    """
    LSTM + dueling + multi-task Q-network.
    """

    def __init__(
        self,
        stateDim: int,
        actionDim: int,
        lstmUnits: int,
        sharedUnits: int,
        taskUnits: int,
    ) -> None:
        super().__init__()
        self._actionDim = actionDim
        self._tasks = ("congestion", "delay", "bandwidth")
        self._lstm = tf.keras.layers.LSTM(lstmUnits, return_sequences=False)
        self._shared = tf.keras.layers.Dense(sharedUnits, activation="relu")
        self._taskLayers: dict[str, tf.keras.layers.Layer] = {}
        self._valueHeads: dict[str, tf.keras.layers.Layer] = {}
        self._advHeads: dict[str, tf.keras.layers.Layer] = {}

        for task in self._tasks:
            self._taskLayers[task] = tf.keras.layers.Dense(taskUnits, activation="relu")
            self._valueHeads[task] = tf.keras.layers.Dense(1, activation=None)
            self._advHeads[task] = tf.keras.layers.Dense(actionDim, activation=None)

        # Build graph once
        self(tf.zeros((1, 1, stateDim), dtype=tf.float32), training=False)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
        mask: Any = None,
    ) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        del mask
        trainingFlag: bool = bool(training) if training is not None else False
        x = self._lstm(inputs, training=trainingFlag)
        x = self._shared(x, training=trainingFlag)
        taskQs: dict[str, tf.Tensor] = {}
        for task in self._tasks:
            t = self._taskLayers[task](x, training=trainingFlag)
            value = self._valueHeads[task](t, training=trainingFlag)
            adv = self._advHeads[task](t, training=trainingFlag)
            taskQs[task] = value + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))

        aggregatedQ = (taskQs["congestion"] + taskQs["delay"] + taskQs["bandwidth"]) / 3.0
        return taskQs, aggregatedQ


class MTDRLSFCREmbedder:
    """
    Multi-task DRL embedder for SFCRs.
    """

    def __init__(
        self,
        topology: Topology,
        vnfCatalog: list[str],
        config: Optional[MTDRLConfig] = None,
        seed: int = 42,
    ) -> None:
        self._topology = topology
        self._config = config if config is not None else MTDRLConfig()
        self._hosts = copy.deepcopy(topology["hosts"])
        self._hostIds = [host["id"] for host in self._hosts]
        self._vnfCatalog = vnfCatalog
        self._vnfToIndex = {vnf: idx for idx, vnf in enumerate(vnfCatalog)}
        self._random = random.Random(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self._demandPredictor = DemandPredictor()

        self._graph = nx.Graph()
        self._linkCapacities: dict[str, float] = {}
        self._linkDelays: dict[str, float] = {}
        self._buildGraph()

        self._maxDelay = max(self._linkDelays.values()) if len(self._linkDelays) > 0 else 1.0
        self._stateDim = self._computeStateDim()
        self._actionDim = len(self._hosts)
        self._online = MTDRLQNetwork(
            self._stateDim,
            self._actionDim,
            self._config.lstmUnits,
            self._config.sharedUnits,
            self._config.taskUnits,
        )
        self._target = MTDRLQNetwork(
            self._stateDim,
            self._actionDim,
            self._config.lstmUnits,
            self._config.sharedUnits,
            self._config.taskUnits,
        )
        self._target.set_weights(self._online.get_weights())
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._config.learningRate)
        self._replay: Deque[Transition] = deque(maxlen=self._config.replayCapacity)
        self._steps = 0

    def embed(
        self,
        sfcrs: list[SFCRequest],
        ingressTrafficMap: Optional[dict[str, float]] = None,
    ) -> tuple[list[EmbeddingGraph], list[SFCRequest]]:
        """
        Train then embed SFCRs.
        """

        if len(sfcrs) == 0:
            return [], []

        self._validateRequests(sfcrs)
        self._train(sfcrs, ingressTrafficMap)
        accepted: list[EmbeddingGraph] = []
        failed: list[SFCRequest] = []
        hostResidual = {
            host["id"]: {
                "cpu": float(host["cpu"]),
                "memory": float(host["memory"]),
            }
            for host in self._hosts
        }
        linkResidual = {k: float(v) for k, v in self._linkCapacities.items()}
        for sfcr in sfcrs:
            embedding, hostResidual, linkResidual = self._runEpisode(
                sfcr,
                training=False,
                epsilon=0.0,
                ingressTrafficMap=ingressTrafficMap,
                hostResidual=hostResidual,
                linkResidual=linkResidual,
            )
            if embedding is None:
                failed.append(sfcr)
            else:
                accepted.append(embedding)

        return accepted, failed

    def _train(
        self,
        sfcrs: list[SFCRequest],
        ingressTrafficMap: Optional[dict[str, float]],
    ) -> None:
        epsilon: float = self._config.epsilon
        for _ in range(self._config.trainingEpisodes):
            shuffled = sfcrs[:]
            self._random.shuffle(shuffled)
            for sfcr in shuffled:
                self._runEpisode(
                    sfcr,
                    training=True,
                    epsilon=epsilon,
                    ingressTrafficMap=ingressTrafficMap,
                )
            epsilon = max(self._config.epsilonMin, epsilon * self._config.epsilonDecay)

    def _runEpisode(
        self,
        sfcr: SFCRequest,
        training: bool,
        epsilon: float,
        ingressTrafficMap: Optional[dict[str, float]],
        hostResidual: Optional[dict[str, dict[str, float]]] = None,
        linkResidual: Optional[dict[str, float]] = None,
    ) -> tuple[Optional[EmbeddingGraph], dict[str, dict[str, float]], dict[str, float]]:
        initialHostResidual = (
            hostResidual
            if hostResidual is not None
            else {
                host["id"]: {
                    "cpu": float(host["cpu"]),
                    "memory": float(host["memory"]),
                }
                for host in self._hosts
            }
        )
        initialLinkResidual = (
            linkResidual if linkResidual is not None else {k: float(v) for k, v in self._linkCapacities.items()}
        )
        hostResidual = copy.deepcopy(initialHostResidual)
        linkResidual = dict(initialLinkResidual)
        orderedVnfs = self._getOrderedVNFs(sfcr)
        if len(orderedVnfs) == 0:
            return None, initialHostResidual, initialLinkResidual

        latencyBudget = float(sfcr.get("latency", self._config.defaultLatencyBudget))
        bandwidthDemand = float(sfcr.get("bandwidthDemand", self._config.defaultBandwidthDemand))
        ingressTraffic = self._getIngressTraffic(sfcr, ingressTrafficMap)
        lastNode: str = SFCC
        selectedHosts: list[str] = []
        selectedPaths: list[list[str]] = []
        stateWindow: Deque[np.ndarray] = deque(maxlen=self._config.historyLength)
        # Paper-style eligibility trace over actions (hosts).
        eligibilityTrace: np.ndarray = np.zeros((self._actionDim,), dtype=np.float32)
        episodeFailed: bool = False

        for step, vnf in enumerate(orderedVnfs):
            demand = self._getDemand(vnf, ingressTraffic, step + 1)
            state = self._buildState(
                sfcr,
                orderedVnfs,
                step,
                lastNode,
                hostResidual,
                linkResidual,
                latencyBudget,
            )
            stateWindow.append(state)
            stateSeq = self._makeSequence(stateWindow)
            validActions = self._validActions(demand, hostResidual)

            if len(validActions) == 0:
                if training:
                    self._pushInvalidTransition(
                        stateSeq,
                        -self._config.invalidActionPenalty,
                    )
                episodeFailed = True
                break

            action = self._selectAction(stateSeq, validActions, epsilon)
            selectedHost = self._hostIds[action]
            lambdaTrace = float(min(1.0, max(0.0, self._config.lambdaTrace)))
            eligibilityTrace *= lambdaTrace
            eligibilityTrace[action] = min(
                self._config.eligibilityClip,
                float(eligibilityTrace[action] + 1.0),
            )
            pathData = self._findPath(lastNode, selectedHost, bandwidthDemand, linkResidual)
            if pathData is None:
                if training:
                    self._pushInvalidTransition(
                        stateSeq,
                        -self._config.invalidPathPenalty,
                        action=action,
                    )
                episodeFailed = True
                break

            path, pathDelay = pathData
            hostResidual[selectedHost]["cpu"] -= demand[0]
            hostResidual[selectedHost]["memory"] -= demand[1]
            if hostResidual[selectedHost]["cpu"] < 0 or hostResidual[selectedHost]["memory"] < 0:
                if training:
                    self._pushInvalidTransition(
                        stateSeq,
                        -self._config.invalidActionPenalty,
                    )
                episodeFailed = True
                break

            self._consumeBandwidth(path, bandwidthDemand, linkResidual)
            selectedHosts.append(selectedHost)
            selectedPaths.append(path)

            # Use cumulative path metrics so factors approximate full-SFC quantities.
            cumulativeDelay = 0.0
            for selectedPath in selectedPaths:
                cumulativeDelay += self._pathDelay(selectedPath)
            congestionReward = self._congestionReward(selectedHost, hostResidual)
            delayReward = self._delayFactor(cumulativeDelay, latencyBudget)
            bandwidthReward = self._bandwidthFactor(selectedPaths, linkResidual)
            # Position factor eta used as reward shaping in the paper.
            positionShape = float(step + 1) / float(len(orderedVnfs))
            # R[a_t] = r_s * r_l * (xi * r_d * egb[a_t] + zeta * r_b)
            reward = (
                positionShape
                * congestionReward
                * (
                    self._config.delayWeight * delayReward * float(eligibilityTrace[action])
                    + self._config.bandwidthWeight * bandwidthReward
                )
                + self._config.positionReward * positionShape
            )
            taskRewards = np.array(
                [
                    self._config.congestionWeight * congestionReward,
                    self._config.delayWeight * delayReward * float(eligibilityTrace[action]),
                    self._config.bandwidthWeight * bandwidthReward,
                ],
                dtype=np.float32,
            )

            done = step == (len(orderedVnfs) - 1)
            if done:
                nextStateSeq = np.zeros_like(stateSeq)
                nextMask = np.zeros((self._actionDim,), dtype=np.float32)
            else:
                nextState = self._buildState(
                    sfcr,
                    orderedVnfs,
                    step + 1,
                    selectedHost,
                    hostResidual,
                    linkResidual,
                    latencyBudget,
                )
                nextWindow: Deque[np.ndarray] = deque(stateWindow, maxlen=self._config.historyLength)
                nextWindow.append(nextState)
                nextStateSeq = self._makeSequence(nextWindow)
                nextDemand = self._getDemand(orderedVnfs[step + 1], ingressTraffic, step + 2)
                nextMask = np.zeros((self._actionDim,), dtype=np.float32)
                for validAction in self._validActions(nextDemand, hostResidual):
                    nextMask[validAction] = 1.0

            if training:
                self._replay.append(
                    Transition(
                        state=stateSeq,
                        action=action,
                        reward=reward,
                        taskRewards=taskRewards,
                        nextState=nextStateSeq,
                        done=done,
                        nextMask=nextMask,
                    )
                )
                self._optimize()

            lastNode = selectedHost

        if episodeFailed or len(selectedHosts) == 0:
            return None, initialHostResidual, initialLinkResidual

        tailPathData = self._findPath(lastNode, SERVER, bandwidthDemand, linkResidual)
        if tailPathData is None:
            return None, initialHostResidual, initialLinkResidual
        tailPath = tailPathData[0]
        self._consumeBandwidth(tailPath, bandwidthDemand, linkResidual)
        selectedPaths.append(tailPath)

        embedding = self._buildEmbeddingGraph(sfcr, orderedVnfs, selectedHosts, selectedPaths)
        return embedding, hostResidual, linkResidual

    def _pushInvalidTransition(self, stateSeq: np.ndarray, reward: float, action: Optional[int] = None) -> None:
        nextMask = np.zeros((self._actionDim,), dtype=np.float32)
        chosenAction = 0 if action is None else int(max(0, min(self._actionDim - 1, action)))
        self._replay.append(
            Transition(
                state=stateSeq,
                action=chosenAction,
                reward=reward,
                taskRewards=np.array([reward, reward, reward], dtype=np.float32),
                nextState=np.zeros_like(stateSeq),
                done=True,
                nextMask=nextMask,
            )
        )
        self._optimize()

    def _optimize(self) -> None:
        if len(self._replay) < self._config.batchSize:
            return

        batch = self._random.sample(list(self._replay), self._config.batchSize)
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int32)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        taskRewards = np.array([t.taskRewards for t in batch], dtype=np.float32)
        nextStates = np.array([t.nextState for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        nextMasks = np.array([t.nextMask for t in batch], dtype=np.float32)

        nextTaskQs, nextQ = self._target(tf.convert_to_tensor(nextStates), training=False)
        nextQNP = nextQ.numpy()
        nextTaskNP = {k: v.numpy() for k, v in nextTaskQs.items()}
        bootstrap = self._maskedMax(nextQNP, nextMasks)
        bootstrapCong = self._maskedMax(nextTaskNP["congestion"], nextMasks)
        bootstrapDelay = self._maskedMax(nextTaskNP["delay"], nextMasks)
        bootstrapBw = self._maskedMax(nextTaskNP["bandwidth"], nextMasks)

        mainTargets = rewards + (1.0 - dones) * self._config.gamma * bootstrap
        taskTargets = np.column_stack(
            (
                taskRewards[:, 0] + (1.0 - dones) * self._config.gamma * bootstrapCong,
                taskRewards[:, 1] + (1.0 - dones) * self._config.gamma * bootstrapDelay,
                taskRewards[:, 2] + (1.0 - dones) * self._config.gamma * bootstrapBw,
            )
        ).astype(np.float32)

        actionIndices = tf.stack(
            [
                tf.range(self._config.batchSize, dtype=tf.int32),
                tf.convert_to_tensor(actions, dtype=tf.int32),
            ],
            axis=1,
        )
        with tf.GradientTape() as tape:
            taskQs, q = self._online(tf.convert_to_tensor(states), training=True)
            chosenMain = tf.gather_nd(q, actionIndices)
            loss = tf.reduce_mean(tf.square(tf.convert_to_tensor(mainTargets) - chosenMain))

            chosenCong = tf.gather_nd(taskQs["congestion"], actionIndices)
            chosenDelay = tf.gather_nd(taskQs["delay"], actionIndices)
            chosenBw = tf.gather_nd(taskQs["bandwidth"], actionIndices)
            loss += 0.5 * tf.reduce_mean(
                tf.square(tf.convert_to_tensor(taskTargets[:, 0]) - chosenCong)
                + tf.square(tf.convert_to_tensor(taskTargets[:, 1]) - chosenDelay)
                + tf.square(tf.convert_to_tensor(taskTargets[:, 2]) - chosenBw)
            )

        gradients = tape.gradient(loss, self._online.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._online.trainable_variables))

        self._steps += 1
        if self._steps % self._config.targetUpdateSteps == 0:
            self._target.set_weights(self._online.get_weights())

    @staticmethod
    def _maskedMax(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
        output = np.zeros((values.shape[0],), dtype=np.float32)
        for index in range(values.shape[0]):
            valid = np.where(masks[index] > 0)[0]
            if len(valid) == 0:
                output[index] = 0.0
            else:
                output[index] = float(np.max(values[index, valid]))
        return output

    def _selectAction(self, stateSeq: np.ndarray, validActions: list[int], epsilon: float) -> int:
        if self._random.random() < epsilon:
            return self._random.choice(validActions)

        _, q = self._online(tf.convert_to_tensor(stateSeq[np.newaxis, ...], dtype=tf.float32), training=False)
        qValues = q.numpy()[0]
        bestAction = validActions[0]
        bestScore = qValues[bestAction]
        for action in validActions[1:]:
            if qValues[action] > bestScore:
                bestAction = action
                bestScore = qValues[action]
        return int(bestAction)

    def _computeStateDim(self) -> int:
        hostFeatures = len(self._hosts) * 6
        vnfFeatures = len(self._vnfCatalog)
        globalFeatures = 4
        return hostFeatures + vnfFeatures + globalFeatures

    def _buildState(
        self,
        sfcr: SFCRequest,
        orderedVnfs: list[str],
        step: int,
        lastNode: str,
        hostResidual: dict[str, dict[str, float]],
        linkResidual: dict[str, float],
        latencyBudget: float,
    ) -> np.ndarray:
        features: list[float] = []
        for host in self._hosts:
            hostId = host["id"]
            totalCPU = max(1e-6, float(host["cpu"]))
            totalMemory = max(1e-6, float(host["memory"]))
            cpuLeft = hostResidual[hostId]["cpu"]
            memLeft = hostResidual[hostId]["memory"]
            cpuUtil = 1.0 - (cpuLeft / totalCPU)
            memUtil = 1.0 - (memLeft / totalMemory)
            route = self._findPath(lastNode, hostId, self._config.defaultBandwidthDemand, linkResidual)
            delayToPreviousNorm = (route[1] / max(1.0, latencyBudget)) if route is not None else 1.0
            bandwidthToPrevious = (
                self._pathResidualBandwidthRatio(route[0], linkResidual) if route is not None else 0.0
            )
            features.extend(
                [
                    cpuLeft / totalCPU,
                    memLeft / totalMemory,
                    cpuUtil,
                    memUtil,
                    delayToPreviousNorm,
                    bandwidthToPrevious,
                ]
            )

        vnfVector = [0.0] * len(self._vnfCatalog)
        currentVnf = orderedVnfs[step] if step < len(orderedVnfs) else None
        if currentVnf in self._vnfToIndex:
            vnfVector[self._vnfToIndex[currentVnf]] = 1.0
        features.extend(vnfVector)

        ingressForState = float(self._config.defaultIngressTraffic)
        for key in ("ingressTraffic", "ingress_traffic", "reqps", "target"):
            if key in sfcr and sfcr[key] is not None:
                candidateIngress = float(sfcr[key])
                if candidateIngress > 0:
                    ingressForState = candidateIngress
                    break

        demandCpuNorm = 0.0
        demandMemoryNorm = 0.0
        if currentVnf is not None:
            demandCpu, demandMemory = self._getDemand(currentVnf, ingressForState, step + 1)
            maxHostCPU = max([max(1e-6, float(host["cpu"])) for host in self._hosts], default=1.0)
            maxHostMemory = max([max(1e-6, float(host["memory"])) for host in self._hosts], default=1.0)
            demandCpuNorm = demandCpu / maxHostCPU
            demandMemoryNorm = demandMemory / maxHostMemory

        features.extend(
            [
                float(step + 1) / float(len(orderedVnfs)),
                latencyBudget / max(1.0, self._config.defaultLatencyBudget),
                demandCpuNorm,
                demandMemoryNorm,
            ]
        )
        return np.array(features, dtype=np.float32)

    def _pathResidualBandwidthRatio(self, path: list[str], linkResidual: dict[str, float]) -> float:
        if len(path) < 2:
            return 1.0

        ratios: list[float] = []
        for index in range(len(path) - 1):
            key = self._edgeKey(path[index], path[index + 1])
            capacity = self._linkCapacities.get(key, 0.0)
            if capacity <= 0:
                continue
            ratios.append(linkResidual.get(key, 0.0) / capacity)

        if len(ratios) == 0:
            return 0.0
        return float(min(ratios))

    def _makeSequence(self, window: Deque[np.ndarray]) -> np.ndarray:
        sequence = np.zeros((self._config.historyLength, self._stateDim), dtype=np.float32)
        start = self._config.historyLength - len(window)
        for idx, value in enumerate(window):
            sequence[start + idx, :] = value
        return sequence

    def _validActions(self, demand: tuple[float, float], hostResidual: dict[str, dict[str, float]]) -> list[int]:
        valid: list[int] = []
        for index, hostId in enumerate(self._hostIds):
            if hostResidual[hostId]["cpu"] >= demand[0] and hostResidual[hostId]["memory"] >= demand[1]:
                valid.append(index)
        return valid

    def _congestionReward(self, hostId: str, hostResidual: dict[str, dict[str, float]]) -> float:
        host = [h for h in self._hosts if h["id"] == hostId][0]
        cpuUtil = 1.0 - (hostResidual[hostId]["cpu"] / max(1e-6, float(host["cpu"])))
        memUtil = 1.0 - (hostResidual[hostId]["memory"] / max(1e-6, float(host["memory"])))
        localReward = max(0.0, 1.0 - max(cpuUtil, memUtil))
        globalReward = self._globalCongestionReward(hostResidual)
        blend = float(min(1.0, max(0.0, self._config.congestionRewardBlend)))
        return blend * globalReward + (1.0 - blend) * localReward

    @staticmethod
    def _delayFactor(pathDelay: float, latencyBudget: float) -> float:
        normalizedDelay = pathDelay / max(1e-6, latencyBudget)
        return float(np.exp(-normalizedDelay))

    def _bandwidthFactor(self, paths: list[list[str]], linkResidual: dict[str, float]) -> float:
        if len(paths) == 0:
            return 1.0
        ratios: list[float] = []
        for path in paths:
            if len(path) < 2:
                continue
            for index in range(len(path) - 1):
                key = self._edgeKey(path[index], path[index + 1])
                cap = self._linkCapacities.get(key, 0.0)
                if cap <= 0:
                    continue
                ratios.append(linkResidual[key] / cap)
        if len(ratios) == 0:
            return 0.0
        bandwidthCost = 1.0 - float(np.mean(ratios))
        return float(np.exp(-max(0.0, bandwidthCost)))

    def _globalCongestionReward(self, hostResidual: dict[str, dict[str, float]]) -> float:
        utilization: list[float] = []
        for host in self._hosts:
            hostId = host["id"]
            totalCPU = max(1e-6, float(host["cpu"]))
            totalMemory = max(1e-6, float(host["memory"]))
            cpuUtil = 1.0 - (hostResidual[hostId]["cpu"] / totalCPU)
            memUtil = 1.0 - (hostResidual[hostId]["memory"] / totalMemory)
            utilization.append(max(cpuUtil, memUtil))
        if len(utilization) == 0:
            return 1.0
        # Lower utilization variance reflects better load balancing.
        variance = float(np.var(np.array(utilization, dtype=np.float32)))
        return float(np.exp(-variance))

    def _pathDelay(self, path: list[str]) -> float:
        totalDelay = 0.0
        for index in range(len(path) - 1):
            key = self._edgeKey(path[index], path[index + 1])
            totalDelay += self._linkDelays.get(key, 0.0)
        return totalDelay

    def _findPath(
        self,
        source: str,
        destination: str,
        bandwidthDemand: float,
        linkResidual: dict[str, float],
    ) -> Optional[tuple[list[str], float]]:
        if source == destination:
            return [source], 0.0

        def weightFn(nodeA: str, nodeB: str, edgeData: dict[str, float]) -> float:
            key = self._edgeKey(nodeA, nodeB)
            capacity = float(edgeData.get("bandwidth", 0.0))
            residual = linkResidual.get(key, 0.0)
            if residual < bandwidthDemand:
                return float("inf")

            delay = float(edgeData.get("delay", 0.0) or 0.0)
            delayNorm = delay / max(1.0, self._maxDelay)
            bwPressure = 1.0 - (residual / max(1e-6, capacity))
            return (
                self._config.routeDelayWeight * delayNorm
                + self._config.routeBandwidthWeight * bwPressure
                + 1e-6
            )

        try:
            path = nx.shortest_path(
                self._graph,
                source=source,
                target=destination,
                weight=weightFn,
                method="dijkstra",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        totalDelay = 0.0
        for index in range(len(path) - 1):
            key = self._edgeKey(path[index], path[index + 1])
            if linkResidual.get(key, 0.0) < bandwidthDemand:
                return None
            totalDelay += self._linkDelays.get(key, 0.0)

        return path, totalDelay

    def _consumeBandwidth(self, path: list[str], bandwidthDemand: float, linkResidual: dict[str, float]) -> None:
        for index in range(len(path) - 1):
            key = self._edgeKey(path[index], path[index + 1])
            linkResidual[key] = max(0.0, linkResidual[key] - bandwidthDemand)

    def _getDemand(self, vnf: str, ingressTraffic: float, depth: int) -> tuple[float, float]:
        divisor = 2 ** max(0, depth - 1)
        effectiveReqps = max(1, int(round(float(ingressTraffic) / float(divisor))))
        demands = self._demandPredictor.getResourceDemandsOfVNF(vnf, effectiveReqps)
        return (
            max(0.0, float(demands["cpu"])),
            max(0.0, float(demands["memory"])),
        )

    def _getIngressTraffic(
        self,
        sfcr: SFCRequest,
        ingressTrafficMap: Optional[dict[str, float]],
    ) -> float:
        for key in ("ingressTraffic", "ingress_traffic", "reqps", "target"):
            if key in sfcr and sfcr[key] is not None:
                ingress = float(sfcr[key])
                if ingress > 0:
                    return ingress
        sfcrID = sfcr["sfcrID"]
        if ingressTrafficMap is not None and sfcrID in ingressTrafficMap:
            ingressFromMap = float(ingressTrafficMap[sfcrID])
            if ingressFromMap > 0:
                return ingressFromMap
        if self._config.defaultIngressTraffic > 0:
            return float(self._config.defaultIngressTraffic)
        raise ValueError(
            f"SFCR {sfcr.get('sfcrID', '<unknown>')} is missing ingress traffic. "
            "Set one of: ingressTraffic, ingress_traffic, reqps, target."
        )

    @staticmethod
    def _validateRequests(sfcrs: list[SFCRequest]) -> None:
        for sfcr in sfcrs:
            if "sfcrID" not in sfcr or not isinstance(sfcr["sfcrID"], str):
                raise ValueError("Every SFCR must include string field `sfcrID`.")
            if "vnfs" not in sfcr or not isinstance(sfcr["vnfs"], list) or len(sfcr["vnfs"]) == 0:
                raise ValueError(f"SFCR {sfcr.get('sfcrID', '<unknown>')} must include non-empty `vnfs` list.")
            if "strictOrder" in sfcr and not isinstance(sfcr["strictOrder"], list):
                raise ValueError(f"SFCR {sfcr['sfcrID']} field `strictOrder` must be a list when provided.")

    def _getOrderedVNFs(self, sfcr: SFCRequest) -> list[str]:
        vnfs = list(sfcr["vnfs"])
        strictOrder = list(sfcr.get("strictOrder", []))
        if len(strictOrder) == 0:
            return vnfs

        lastIndex: int = -1
        for vnf in strictOrder:
            if vnf not in vnfs:
                continue
            index = vnfs.index(vnf)
            if index < lastIndex:
                vnfs.remove(vnf)
                vnfs.insert(lastIndex, vnf)
                index = lastIndex
            lastIndex = index
        return vnfs

    def _buildEmbeddingGraph(
        self,
        sfcr: SFCRequest,
        orderedVnfs: list[str],
        selectedHosts: list[str],
        selectedPaths: list[list[str]],
    ) -> EmbeddingGraph:
        chainRoot: dict[str, Any] = {
            "host": {"id": selectedHosts[0]},
            "vnf": {"id": orderedVnfs[0]},
        }
        cursor: dict[str, Any] = chainRoot
        for index in range(1, len(orderedVnfs)):
            nextNode: dict[str, Any] = {
                "host": {"id": selectedHosts[index]},
                "vnf": {"id": orderedVnfs[index]},
            }
            cursor["next"] = nextNode
            cursor = nextNode
        cursor["next"] = {"host": {"id": SERVER}, "next": TERMINAL}

        links: list[dict[str, Any]] = []
        for path in selectedPaths:
            links.append(
                {
                    "source": {"id": path[0]},
                    "destination": {"id": path[-1]},
                    "links": path[1:-1],
                }
            )

        return cast(
            EmbeddingGraph,
            {
                "sfcID": sfcr["sfcrID"],
                "sfcrID": sfcr["sfcrID"],
                "vnfs": chainRoot,
                "links": links,
            },
        )

    def _buildGraph(self) -> None:
        for link in self._topology["links"]:
            source = link["source"]
            destination = link["destination"]
            bandwidth = float(link.get("bandwidth", 0.0) or 0.0)
            delay = float(link.get("delay", 0.0) or 0.0)
            self._graph.add_edge(source, destination, bandwidth=bandwidth, delay=delay)
            key = self._edgeKey(source, destination)
            self._linkCapacities[key] = bandwidth
            self._linkDelays[key] = delay

    @staticmethod
    def _edgeKey(nodeA: str, nodeB: str) -> str:
        return f"{nodeA}-{nodeB}" if nodeA < nodeB else f"{nodeB}-{nodeA}"
