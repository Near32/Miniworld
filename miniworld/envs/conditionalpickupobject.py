import copy
from typing import Optional, Tuple

import numpy as np

from gymnasium import spaces, utils
from gymnasium.core import ObsType

from miniworld.entity import COLOR_NAMES, Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

charset = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")


class ConditionalPickUpObject(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Room with multiple objects. The agent collects +1 reward for picking up
    the object mentioned in the mission, and -1 if it picks up anything else.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move_back                   |
    | 4   | pickup                      |
    | 5   | drop                        |

    ## Observation Space

    The observation space is a dictionary with the following entries:
        - image:
            an `ndarray` with shape `(obs_height, obs_width, 3)`
            representing a RGB image of what the agents sees.
        - mission:
            a string of the form 'pick up the COLOR TYPE' where
            COLOR and TYPE are a visible entity's color and type.

    ## Rewards:

    +1 when agent picks up correct object, -1 if pickup up wrong object.

    ## Arguments

    ```python
    ConditionalPickUpObject(size=12, num_objs=5, cam_pitch=-30)
    ```

    `size`: size of world

    `num_objs`: number of objects

    `cam_pitch`: pitch offset of the camera in degrees.

    """

    def __init__(
        self, 
        num_rows=4,
        num_cols=4,
        room_size=12, 
        num_objs=10, 
        cam_pitch=-30, 
        world_type='room',
        use_penalty=False,
        terminate_on_completion=True,
        **kwargs,
    ):
        assert room_size >= 2
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.world_type = world_type
        self.gap_size = 0.25
        self.num_objs = num_objs
        self.cam_pitch = cam_pitch
        self.use_penalty = use_penalty
        self.terminate_on_completion = terminate_on_completion

        MiniWorldEnv.__init__(
            self, 
            max_episode_steps=num_rows * num_cols * 24, #400, 
            **kwargs)
        utils.EzPickle.__init__(
            self, 
            num_rows,
            num_cols,
            room_size, 
            num_objs, 
            cam_pitch, 
            world_type,
            use_penalty,
            terminate_on_completion,
            **kwargs,
        )

        image_observation_space = self.observation_space
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "mission": spaces.Text(max_length=100, charset=charset),
            }
        )

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.drop + 1)

    def _gen_maze(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):
                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex="brick_wall",
                    no_ceiling=True,
                    # floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            orders = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            assert 4 <= len(orders)
            neighbors = []

            while len(neighbors) < 4:
                elem = orders[self.np_random.choice(len(orders))]
                orders.remove(elem)
                neighbors.append(elem)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif dj == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0) 

    def _gen_world(self):
        if self.world_type == 'maze':
            self._gen_maze()
        else:
            self.add_rect_room(
                min_x=0,
                max_x=self.room_size,
                min_z=0,
                max_z=self.room_size,
                wall_tex="brick_wall",
                floor_tex="asphalt",
                no_ceiling=True,
            )

        obj_types = [Ball, Box, Key]
        colorlist = list(COLOR_NAMES)

        obj_list = []
        while len(obj_list) < self.num_objs:
            obj_type = obj_types[self.np_random.choice(len(obj_types))]
            color = colorlist[self.np_random.choice(len(colorlist))]

            obj_descr = (obj_type, color)
            if obj_descr in obj_list:
                continue

            obj_list.append(obj_descr)

            if obj_type == Box:
                self.place_entity(Box(color=color, size=0.9))
            if obj_type == Ball:
                self.place_entity(Ball(color=color, size=0.9))
            if obj_type == Key:
                self.place_entity(Key(color=color))

        self.place_agent()
        self.agent.cam_pitch = self.cam_pitch

        # Create mission:
        obj = self.np_random.choice(obj_list)
        obj_color = obj[1]
        obj_type = obj[0].__name__.lower()
        self.mission = f"pick up the {obj_color} {obj_type}"

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[ObsType, dict]:
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """
        obs, info = super().reset(seed=seed)
        info['success'] = False 

        output_d = {
            "image": obs,
            "mission": copy.deepcopy(self.mission),
        }

        return output_d, info

    def step(self, action):
        self.agent.cam_pitch = self.cam_pitch
        obs, reward, termination, truncation, info = super().step(action)
        info['success'] = False

        if self.agent.carrying:
            obj_type = type(self.agent.carrying).__name__.lower()
            obj_color = getattr(self.agent.carrying, "color", None)
            if obj_color in self.mission and obj_type in self.mission:
                reward = 1
                info['success'] = True
            elif self.use_penalty:
                reward = -1
            else:
                reward = 0

            if self.terminate_on_completion \
            and reward > 0:
                termination = True

        output_d = {
            "image": obs,
            "mission": copy.deepcopy(self.mission),
        }

        return output_d, reward, termination, truncation, info


# Parameters for larger movement steps, fast stepping
fast_params = DEFAULT_PARAMS.no_random()
fast_params.set("forward_step", 0.7)
fast_params.set("turn_step", 15)

class ConditionalPickUpObjectFast(ConditionalPickUpObject):
    def __init__(
        self, 
        num_rows=3,
        num_cols=3,
        room_size=12, 
        num_objs=5, 
        cam_pitch=-30, 
        params=fast_params,
        domain_rand=False,
        collision=True,
        terminate_on_completion=True,
        **kwargs,
    ):
        ConditionalPickUpObject.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size, 
            num_objs=num_objs,
            cam_pitch=cam_pitch,
            params=params,
            domain_rand=domain_rand,
            terminate_on_completion=terminate_on_completion,
            **kwargs,
        )
        
        # Enable going through objects in order to not impair exploration.
        if not collision: 
            self.collision_entity_types = []
        else:
            self.collision_entity_types = [
                'ball',
                'box',
                'key',
                'meshent',
            ]


class ConditionalPickUpObjectFast2x2(ConditionalPickUpObject):
    def __init__(
        self, 
        num_rows=4,
        num_cols=4,
        room_size=12, 
        num_objs=5, 
        cam_pitch=-25, 
        params=fast_params,
        domain_rand=False,
        collision=True,
        terminate_on_completion=True,
        **kwargs,
    ):
        ConditionalPickUpObject.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size, 
            num_objs=num_objs,
            cam_pitch=cam_pitch,
            params=params,
            domain_rand=domain_rand,
            terminate_on_completion=terminate_on_completion,
            **kwargs,
        )
	
        # Enable going through objects in order to not impair exploration.
        if not collision: 
            self.collision_entity_types = []
        else:
            self.collision_entity_types = [
                'ball',
                'box',
                'key',
                'meshent',
            ]

 
class MazeConditionalPickUpObjectFast(ConditionalPickUpObjectFast):
    def __init__(
        self, 
        num_rows=4,
        num_cols=4,
        room_size=4, 
        num_objs=10, 
        cam_pitch=-25, 
        terminate_on_completion=True,
        **kwargs,
    ):
        ConditionalPickUpObjectFast.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size, 
            num_objs=num_objs,
            cam_pitch=cam_pitch,
            world_type='maze',
            use_penalty=False,
            terminate_on_completion=terminate_on_completion,
            **kwargs,
        )
        
        # Enable going through objects in order to not impair exploration.
        self.collision_entity_types = []
        """
        [
            'ball',
            'box',
            'key',
            'meshent',
        ]
        """

class MazeConditionalPickUpObjectFast2x2(ConditionalPickUpObjectFast):
    def __init__(
        self, 
        num_rows=4,
        num_cols=4,
        room_size=2, 
        num_objs=10, 
        cam_pitch=-25, 
        terminate_on_completion=True,
        **kwargs,
    ):
        ConditionalPickUpObjectFast.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size, 
            num_objs=num_objs,
            cam_pitch=cam_pitch,
            world_type='maze',
            use_penalty=False,
            terminate_on_completion=terminate_on_completion,
            **kwargs,
        )
        
        # Enable going through objects in order to not impair exploration.
        self.collision_entity_types = []
        """
        [
            'ball',
            'box',
            'key',
            'meshent',
        ]
        """

class MazeConditionalPickUpObjectFast3x3(ConditionalPickUpObjectFast):
    def __init__(
        self, 
        num_rows=4,
        num_cols=4,
        room_size=3, 
        num_objs=10, 
        cam_pitch=-25, 
        terminate_on_completion=True,
        **kwargs,
    ):
        ConditionalPickUpObjectFast.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size, 
            num_objs=num_objs,
            cam_pitch=cam_pitch,
            world_type='maze',
            use_penalty=False,
            terminate_on_completion=terminate_on_completion,
            **kwargs,
        )
        
        # Enable going through objects in order to not impair exploration.
        self.collision_entity_types = []
        """
        [
            'ball',
            'box',
            'key',
            'meshent',
        ]
        """


class MazeConditionalPickUpFurthestObject(ConditionalPickUpObjectFast):
    def __init__(
        self, 
        num_rows=6,
        num_cols=6,
        room_size=3, 
        num_objs=10, 
        cam_pitch=-25, 
        terminate_on_completion=True,
        **kwargs,
    ):
        ConditionalPickUpObjectFast.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size, 
            num_objs=num_objs,
            cam_pitch=cam_pitch,
            world_type='maze',
            use_penalty=False,
            terminate_on_completion=terminate_on_completion,
            **kwargs,
        )
        
        # Enable going through objects in order to not impair exploration.
        self.collision_entity_types = []
        """
        [
            'ball',
            'box',
            'key',
            'meshent',
        ]
        """
    def compute_distance(self, entity):
        dist = 0
        for x,y in zip(self.agent.pos, entity[0].pos):
            dist += np.square(x-y)
        dist = np.sqrt(dist)
        return dist

    def _gen_world(self):
        '''
        The Mission is to pick up the furthest object, instead of a randomly chosen one.
        '''
        if self.world_type == 'maze':
            self._gen_maze()
        else:
            self.add_rect_room(
                min_x=0,
                max_x=self.room_size,
                min_z=0,
                max_z=self.room_size,
                wall_tex="brick_wall",
                floor_tex="asphalt",
                no_ceiling=True,
            )

        obj_types = [Ball, Box, Key]
        colorlist = list(COLOR_NAMES)

        obj_list = []
        ent_list = []
        while len(obj_list) < self.num_objs:
            obj_type = obj_types[self.np_random.choice(len(obj_types))]
            color = colorlist[self.np_random.choice(len(colorlist))]

            obj_descr = [obj_type, color]
            if obj_descr in obj_list:
                continue
            
            obj_list.append(obj_descr)


            if obj_type == Box:
                self.place_entity(Box(color=color, size=0.9))
            if obj_type == Ball:
                self.place_entity(Ball(color=color, size=0.9))
            if obj_type == Key:
                self.place_entity(Key(color=color))
            
            ent_descr = [self.entities[-1]] + obj_descr
            ent_list.append(ent_descr)

        self.place_agent()
        self.agent.cam_pitch = self.cam_pitch

        # Create mission:
        sorted_ent_list = sorted(
            ent_list,
            key=self.compute_distance,
        )
        obj = sorted_ent_list[-1][1:]

        obj_color = obj[1]
        obj_type = obj[0].__name__.lower()
        self.mission = f"pick up the {obj_color} {obj_type}"

      
