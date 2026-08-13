import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


def OUT_WARNING():
    logger.warning("Agent trying to move into invalid space; repositioning")


class Agent:
    def __init__(
        self,
        targeted_landmark,
        movements,
        scale_keys,
        brain=None,
        environement=None,
        FOV=(32, 32, 32),
        start_pos_radius=20,
        shortmem_size=10,
        speed_per_scale=(2, 1),
        focus_radius=4,
        verbose=False,
    ) -> None:
        self.focus_radius = focus_radius
        self.target = targeted_landmark
        self.scale_keys = scale_keys
        self.environement = environement
        self.scale_state = 0
        self.start_pos_radius = start_pos_radius
        self.start_position = np.array([0, 0, 0], dtype=np.int16)
        self.position = np.array([0, 0, 0], dtype=np.int16)
        self.FOV = np.array(FOV, dtype=np.int16)

        self.movement_matrix = movements["mat"]
        self.movement_id = movements["id"]

        self.brain = brain
        self.shortmem_size = shortmem_size
        self.verbose = verbose
        self.search_atempt = 0
        self.speed_per_scale = list(speed_per_scale)
        self.speed = self.speed_per_scale[0]

    def SetEnvironment(self, environement):
        self.environement = environement
        position_mem = []
        position_shortmem = []
        for i in range(environement.scale_nbr):
            position_mem.append([])
            position_shortmem.append(deque(maxlen=self.shortmem_size))
        self.position_mem = position_mem
        self.position_shortmem = position_shortmem

    def SetBrain(self, brain):
        self.brain = brain

    def ClearShortMem(self):
        for mem in self.position_shortmem:
            mem.clear()

    def SetEnvironement(self, environement):
        from .compat import deprecate

        deprecate("Agent.SetEnvironement", "Agent.SetEnvironment")
        return self.SetEnvironment(environement)

    def GoToScale(self, scale=0):
        self.position = (
            self.position
            * (
                self.environement.GetSpacing(self.scale_keys[self.scale_state])
                / self.environement.GetSpacing(self.scale_keys[scale])
            )
        ).astype(np.int16)
        self.scale_state = scale
        self.search_atempt = 0
        self.speed = self.speed_per_scale[scale]

    def SetPosAtCenter(self):
        self.position = (
            self.environement.GetSize(self.scale_keys[self.scale_state]) // 2
        ).astype(np.int16)

    def SetRandomPos(self):
        if self.scale_state == 0:
            rand_coord = np.random.randint(
                1,
                self.environement.GetSize(self.scale_keys[self.scale_state]),
                dtype=np.int16,
            )
            self.start_position = rand_coord
        else:
            rand_coord = (
                np.random.randint([1, 1, 1], self.start_pos_radius * 2)
                - self.start_pos_radius
            )
            rand_coord = self.start_position + rand_coord
            rand_coord = np.where(rand_coord < 0, 0, rand_coord)
            rand_coord = rand_coord.astype(np.int16)
        self.position = rand_coord

    def GetState(self):
        return self.environement.GetZone(
            self.scale_keys[self.scale_state], self.position, self.FOV
        )

    def UpScale(self):
        scale_changed = False
        if self.scale_state < self.environement.scale_nbr - 1:
            self.GoToScale(self.scale_state + 1)
            scale_changed = True
            self.start_position = self.position
        return scale_changed

    def PredictAction(self):
        return self.brain.Predict(self.scale_state, self.GetState())

    def Move(self, movement_idx):
        new_pos = self.position + self.movement_matrix[movement_idx] * self.speed
        if (
            new_pos.all() > 0
            and (
                new_pos < self.environement.GetSize(self.scale_keys[self.scale_state])
            ).all()
        ):
            self.position = new_pos
        else:
            OUT_WARNING()
            self.ClearShortMem()
            self.SetRandomPos()
            self.search_atempt += 1

    def SavePos(self):
        self.position_mem[self.scale_state].append(self.position)
        self.position_shortmem[self.scale_state].append(self.position)

    def Focus(self, start_pos):
        explore_pos = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.int16,
        )
        final_pos = np.array([0, 0, 0], dtype=np.float64)
        for pos in explore_pos:
            found = False
            self.position_shortmem[self.scale_state].clear()
            self.position = start_pos + self.focus_radius * pos
            while not found:
                action = self.PredictAction()
                self.Move(action)
                if self.Visited():
                    found = True
                self.SavePos()
            final_pos += self.position
        return final_pos / len(explore_pos)

    def Search(self):
        if self.verbose:
            logger.info("Searching landmark: %s", self.target)
        self.GoToScale()
        self.SetPosAtCenter()
        self.SavePos()
        found = False
        tot_step = 0
        while not found:
            tot_step += 1
            action = self.PredictAction()
            self.Move(action)
            if self.Visited():
                found = True
            self.SavePos()
            if found:
                if self.verbose:
                    logger.info(
                        "Landmark found at scale %s, pos=%s",
                        self.scale_state,
                        self.position,
                    )
                scale_changed = self.UpScale()
                found = not scale_changed
            if self.search_atempt > 2:
                logger.warning("%s: landmark not found", self.target)
                self.search_atempt = 0
                return -1

        final_pos = self.Focus(self.position)
        if self.verbose:
            logger.info("Focus result for %s: %s", self.target, final_pos)
        self.environement.AddPredictedLandmark(self.target, final_pos)
        return tot_step

    def Visited(self):
        for previous_pos in self.position_shortmem[self.scale_state]:
            if np.array_equal(self.position, previous_pos):
                return True
        return False
