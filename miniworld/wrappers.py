import gymnasium as gym
import numpy as np
import copy
import math 

import pyglet
from pyglet.gl import *
import ctypes


class PyTorchObsWrapper(gym.ObservationWrapper):
    """
    Transpose the observation image tensors for PyTorch
    """

    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 1, 0)


class GreyscaleWrapper(gym.ObservationWrapper):
    """
    Convert image obserations from RGB to greyscale
    """

    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[0], obs_shape[1], 1],
            dtype=self.observation_space.dtype,
        )

    def observation(self, obs):
        obs = 0.30 * obs[:, :, 0] + 0.59 * obs[:, :, 1] + 0.11 * obs[:, :, 2]

        return np.expand_dims(obs, axis=2)


from miniworld.miniworld import DEFAULT_WALL_HEIGHT

class WallEntity(object):
    def __init__(self, wall_seg):
        self.s_p0 = wall_seg[0]
        self.s_p1 = wall_seg[1]


class SymbolicImageEntityVisibilityOracleWrapper(gym.Wrapper):
    """
    Adds to the info dictionnary an entry `'symbolic_image'`,
    whose value is a (7x7x3) matrix of entities currently visible on screen, 
    following the OBJECT/COLOR_TO_IDX dictionnary from MiniGrid.
    The set of entity types to consider and the amount of visibility
    for the entity to qualify are parameterized by the following:
        - :param relevant_entity_types: List[Str] where each string is
        an element that must be found into the name of the type of 
        entities to consider.
    """

    def __init__(
        self, 
        env, 
        relevant_entity_types=['Box','Key','Ball'], 
        as_obs=False, 
    ):
        assert isinstance(relevant_entity_types, list) and len(relevant_entity_types)
        super().__init__(env)
        self.relevant_entity_types = [w.lower() for w in relevant_entity_types]
        self.as_obs = as_obs
        
        n_objects = getattr(env, 'n_object', None)
        if n_objects is None:
            n_objects = getattr(env, 'num_objs', None)
        if n_objects is None:   
            n_objects = 5

        self.observation_space = copy.deepcopy(env.observation_space)
        if self.as_obs:
            if not isinstance(self.observation_space, gym.spaces.Dict):
                obs_space = gym.spaces.Dict({
                    "image": self.observation_space,
                })
                self.observation_space = obs_space

            self.observation_space.spaces["symbolic_image"] = gym.spaces.Box(
                low=0,
                high=10,
                shape=(7,7,3),
                dtype="uint8",
            )
        
    def get_symbolic_image(self):
        OBJECT_TO_IDX = {
            "unseen": 0,
            "empty": 1,
            "wall": 2,
            "floor": 3,
            "door": 4,
            "key": 5,
            "ball": 6,
            "box": 7,
            "goal": 8,
            "lava": 9,
            "agent": 10,
        }

        COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}

        visible_ents = self.get_visible_ents()
        visible_objects = [
            [getattr(ent,'color', ''), type(ent).__name__.lower()] 
            for ent in visible_ents
        ]
        # filtering :
        visible_objects = [t for t in visible_objects if t[1] in self.relevant_entity_types]

        symbolic_image = np.zeros((7,7,3))
        idx = 0
        for color,shape in visible_objects:
            i = idx // 7
            j = idx % 7
            symbolic_image[i,j,0] = OBJECT_TO_IDX[shape]
            symbolic_image[i,j,1] = COLOR_TO_IDX[color]
            idx += 1

        return symbolic_image

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        symbolic_image = self.get_symbolic_image()
        if self.as_obs:
            obs['symbolic_image'] = symbolic_image
        else:
            info['symbolic_image'] = symbolic_image

        return obs, info

    def step(self, action, **kwargs):
        next_obs, reward, termination, truncation, info = self.env.step(action, **kwargs)
        
        symbolic_image = self.get_symbolic_image()
        if self.as_obs:
            next_obs['symbolic_image'] = symbolic_image
        else:
            info['symbolic_image'] = symbolic_image

        return next_obs, reward, termination, truncation, info


class EntityVisibilityOracleWrapper(gym.Wrapper):
    """
    Adds to the info dictionnary an entry `'visible_entities'`,
    whose value is a List of entities currently visible on screen, 
    ordered from left to right.
    The set of entity types to consider and the amount of visibility
    for the entity to qualify are parameterized by the following:
        - :param relevant_entity_types: List[Str] where each string is
        an element that must be found into the name of the type of 
        entities to consider.
        - :param qualifying_area_ratio: float in the range ]0.0;1.0]
        that specifies the minimal percentage of screen-space area of
        the entity that must be visible for the entity to qualify in 
        the returned list. E.g. 0.1 means that all entities that are 
        at least 10% visible on the screen will qualify.
        - :param qualifying_screen_ratio: float in the range ]0.0;1.0]
        that specifies the minimal percentage of screen-space that the
        visible area of the entity that must occupy for the entity to 
        qualify in the returned list. E.g. 0.01 means that all entities 
        that are at least occupying an area of 1% of the whole screen 
        will qualify.
    """

    def __init__(
        self, 
        env, 
        relevant_entity_types=['Box','Key','Ball'], 
        qualifying_area_ratio=0.15, 
        qualifying_screen_ratio=0.025, 
        as_obs=False, 
        with_top_view=False,
        verbose=False,
    ):
        assert isinstance(relevant_entity_types, list) and len(relevant_entity_types)
        assert isinstance(qualifying_area_ratio, float) and 0 < qualifying_area_ratio <= 1.0
        assert isinstance(qualifying_screen_ratio, float) and 0 < qualifying_screen_ratio <= 1.0
        super().__init__(env)
        self.relevant_entity_types = relevant_entity_types
        self.qualifying_area_ratio = qualifying_area_ratio
        self.qualifying_screen_ratio = qualifying_screen_ratio
        self.as_obs = as_obs
        self.with_top_view = with_top_view
        self.verbose = verbose
        
        n_objects = getattr(env, 'n_object', None)
        if n_objects is None:
            n_objects = getattr(env, 'num_objs', None)
        if n_objects is None:   
            n_objects = 5
        self.max_sentence_length = n_objects * 3

        self.observation_space = copy.deepcopy(env.observation_space)
        if self.as_obs:
            if not isinstance(self.observation_space, gym.spaces.Dict):
                obs_space = gym.spaces.Dict({
                    "image": self.observation_space,
                })
                self.observation_space = obs_space

            self.observation_space.spaces["visible_entities"] = gym.spaces.MultiDiscrete([100]*self.max_sentence_length)
        
        if self.with_top_view:
            if not isinstance(self.observation_space, gym.spaces.Dict):
                obs_space = gym.spaces.Dict({
                    "image": self.observation_space,
                })
                self.observation_space = obs_space
            self.observation_space.spaces["top_view"] = copy.deepcopy(self.observation_space.spaces["image"])
            self.observation_space.spaces["agent_pos_in_top_view"] = gym.spaces.Box(
                low=-math.inf, 
                high=-math.inf, 
                shape=(3, 4), 
                dtype=np.float32,
            )

    def _filter_entities(self, entities):
        fentities = []
        for ent in entities:
            relevant = False
            for rname in self.relevant_entity_types:
                if rname in type(ent).__name__:
                    relevant = True
                    break
            if relevant:
                fentities.append(ent)

        return fentities

    def _Entity2BoundingBoxMesh(self, entity):
        """
        Create PyOpenGL-like mesh's vertices object for a box 
        which would fit inside a circle of radius equal
        to the radius attribute of the :param entity:,
        which is an Entity-inheriting class instance.
        """
        # Calculate the half-height of the box as the entity's radius
        half_height = half_width = entity.radius
        
        # Calculate the positions of the box vertices based on entity.pos
        vertices = np.array([
            # Front face
            [entity.pos[0] - half_width, entity.pos[1] - half_height, entity.pos[2] - half_height],  # 0
            [entity.pos[0] - half_width, entity.pos[1] + half_height, entity.pos[2] - half_height],  # 1
            [entity.pos[0] + half_width, entity.pos[1] + half_height, entity.pos[2] - half_height],  # 2
            [entity.pos[0] + half_width, entity.pos[1] - half_height, entity.pos[2] - half_height],  # 3
            # Back face
            [entity.pos[0] - half_width, entity.pos[1] - half_height, entity.pos[2] + half_height],  # 4
            [entity.pos[0] - half_width, entity.pos[1] + half_height, entity.pos[2] + half_height],  # 5
            [entity.pos[0] + half_width, entity.pos[1] + half_height, entity.pos[2] + half_height],  # 6
            [entity.pos[0] + half_width, entity.pos[1] - half_height, entity.pos[2] + half_height],  # 7
            ],
            dtype=np.float32,
        )
        
        center = entity.pos

        return {'vertices':vertices, 'center':center}
 
    def _Wall2BoundingBoxMesh(self, wall):
        """
        Create PyOpenGL-like mesh's vertices object for a wall. 
        """
        s_p0 = wall.s_p0
        s_p1 = wall.s_p1
        min_y = 0
        global DEFAULT_WALL_HEIGHT
        max_y = DEFAULT_WALL_HEIGHT
        Y_VEC = np.array([0,1,0])
        
        wall_verts = []
        wall_verts.append(s_p0 + min_y * Y_VEC)
        wall_verts.append(s_p0 + max_y * Y_VEC)
        wall_verts.append(s_p1 + max_y * Y_VEC)
        wall_verts.append(s_p1 + min_y * Y_VEC) 
        
        # Calculate the positions of the box vertices based on entity.pos
        vertices = np.array(wall_verts, dtype=np.float32).reshape((4,3))
        center = s_p0+(s_p1-s_p0)/2+(max_y/2)*Y_VEC

        return {'vertices':vertices, 'center':center}
 
    def get_depth_buffer(self, width, height):
        self.shadow_window.switch_to()
        #self.shadow_window.context.set_current()
        self.unwrapped.obs_fb.bind()
        
        depth_map = self.unwrapped.obs_fb.get_depth_map(0.04, 100.0)
        
        '''
        # Read the depth buffer into a numpy array
        depth_buffer = (ctypes.c_float * (width*height))()
        #depth_buffer = np.zeros((height, width), dtype=np.float32)
        glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth_buffer)
        depth_buffer = np.frombuffer(depth_buffer, dtype=np.float32).reshape((height, width))
        
        # Convert the depth buffer from [0, 1] to world coordinates
        near = glGetFloatv(GL_DEPTH_RANGE)[0]
        far = glGetFloatv(GL_DEPTH_RANGE)[1]
        depth_buffer = (2.0 * depth_buffer - 1.0) * far / (far - near)
        '''

        return depth_buffer

    def depr_get_visible_entities(self):
        self.shadow_window.switch_to()
        self.unwrapped.obs_fb.bind()
        
        #entities = self.unwrapped.entities
        # Entities that ought to be considered, because 
        # not occluded by walls...
        # but still possibly occluded by each other...
        entities = self.get_visible_ents()
        filtered_entities = self._filter_entities(entities)
        meshes = [self._Entity2BoundingBoxMesh(fent) for fent in filtered_entities]
        '''
        # Add walls:
        wall_segs = self.unwrapped.wall_segs
        walls = [WallEntity(wall_seg) for wall_seg in wall_segs]
        filtered_entities += walls
        meshes += [self._Wall2BoundingBoxMesh(wall) for wall in walls]
        '''

        visible_objects = []
        
        # Retrieve projection matrix, view matrix, and viewport dimensions from OpenGL
        projection_matrix = (ctypes.c_float *16)()
        view_matrix = (ctypes.c_float * 16)()
        viewport = (ctypes.c_int * 4)()
        
        glGetIntegerv(GL_VIEWPORT, viewport)
        viewport = np.array(viewport, dtype=np.int32)
        screen_width = viewport[2]
        screen_height = viewport[3]
        
        glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix)
        glGetFloatv(GL_MODELVIEW_MATRIX, view_matrix)
        projection_matrix = np.array(projection_matrix, dtype=np.float32).reshape((4,4)).T
        view_matrix = np.array(view_matrix, dtype=np.float32).reshape((4,4)).T
        
        '''
        depth_buffer = self.get_depth_buffer(
            width=screen_width,
            height=screen_height,
        )
        '''

        # Loop through each mesh in the list
        for midx, mesh in enumerate(meshes):
            vertices = mesh['vertices']
            center = mesh['center']

            # Create Model Matrix of the mesh:
            model_matrix = np.identity(4)
            
            # Transform mesh vertices to NDC space
            vertices_homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
            vertices_ndc = (projection_matrix @ view_matrix @ model_matrix @ vertices_homogeneous.T).T
            
            mean_depth = vertices_ndc[:,2].mean()
            in_front = (mean_depth > 0.0)
            # Remove vertices that are not in front of the camera:
            in_front_indices = [idx for idx, vertex in enumerate(vertices_ndc) if vertex[2] > 0.0]
            not_in_front_indices = [idx for idx, vertex in enumerate(vertices_ndc) if vertex[2] < 0.0]
            if len(in_front_indices)==0:    continue
            #vertices_ndc = vertices_ndc[in_front_indices]
            vertices_ndc[not_in_front_indices, 2] = 0.0

            vertices_ndc /= vertices_ndc[:, 3].reshape(-1, 1)
            
            # Transform NDC vertices to screen space
            # Taking into account OpenGL inverted Y-axis... (going upward while screen Y axises are usually meant to go downward) :
            vertices_screen = np.array([(vertices_ndc[:,0]+1)*0.5*screen_width,
                (1-(vertices_ndc[:,1]+1)*0.5)*screen_height]).T
            
            # Compute screen-space bounding box coordinates in pixel space
            min_x = int(min(vertices_screen[:, 0]))
            max_x = int(max(vertices_screen[:, 0]))
            min_y = int(min(vertices_screen[:, 1]))
            max_y = int(max(vertices_screen[:, 1]))
            
            # Compute screen-space bounding box area
            screen_ent_width = np.abs(max_x - min_x)
            screen_ent_height = np.abs(max_y - min_y)
            screen_ent_area = screen_ent_width * screen_ent_height
            
            # Compute area of the bounding box that is inside the screen
            v_max_x = max(0,min(max_x, screen_width))
            v_min_x = max(0,min(min_x, screen_width))
            visible_ent_width = v_max_x - v_min_x
            v_max_y = max(0,min(max_y, screen_height))
            v_min_y = max(0,min(min_y, screen_height))
            visible_ent_height = v_max_y - v_min_y
            visible_ent_area = visible_ent_width * visible_ent_height

            # Compute the ratio of visible area to total area
            visible_percentage = visible_ent_area / screen_ent_area * 100
            screen_ratio = visible_ent_area / (screen_width*screen_height) * 100.0

            
            if self.verbose:
                print(f"Entity {filtered_entities[midx]} is {'in front' if in_front else 'not'} visible : {visible_percentage} %.")
                print(f"X : {min_x} to {max_x} / Y : {min_y} to {max_y}")
                print(f"Area : {screen_ent_area} px-sq : {screen_ratio}% of the screen") 
            
            wide_enough = (screen_ent_width > 0.05*screen_width)
            # Tricky situation when vertices are not all in front:
            on_side = (screen_ent_width > 2.5*screen_width)
            if on_side:
                if self.verbose:    
                    print(f"Entity {filtered_entities[midx]} is on the side!")
                continue
            if not wide_enough: 
                if self.verbose:
                    print(f"Entity {filtered_entities[midx]} is not wide enough!")
                    print(f"{screen_ent_width} < {0.05*screen_width} ...")
                continue

            
            if (visible_percentage >= (self.qualifying_area_ratio*100.0)) \
            or (screen_ratio >= (self.qualifying_screen_ratio*100.0)):
                # Object is sufficiently visible, then add it to visible objects list
                '''
                area_aligned_depths = {
                    (x,y):depth_buffer[x,y]
                    for x in range(v_min_x, v_max_x)
                    for y in range(v_min_y, v_max_y)
                }
                '''
                visible_objects.append(
                    {
                        'entity':filtered_entities[midx], 
                        'min_x':min_x,
                        'max_x':max_x,
                        'min_y':min_y,
                        'max_y':max_y,
                        'screen_ratio': screen_ratio,
                        'screen_area': screen_ent_area,
                        'depth':mean_depth,
                        'area':set([
                            (x,y) 
                            for x in range(v_min_x, v_max_x)
                            for y in range(v_min_y, v_max_y)
                            #if min_depth <= depth_buffer[x,y] <= max_depth
                        ]),
                    },
                )
        
        # Account for occlusions:
        non_occl_visible_objects = []
        for vidx, vent in enumerate(visible_objects):
            # No need to compute it for walls, we filter them out.
            #if isinstance(vent['entity'], WallEntity):  continue
            init_visible_area = len(vent['area'])
            for oidx, oent in enumerate(visible_objects):
                if oidx == vidx:    continue
                '''
                if isinstance(oent['entity'], WallEntity):
                    #Check whether there is screen horizontal overlap:
                    on_left = vent['max_x'] < oent['min_x']
                    on_right = vent['min_x'] > oent['max_x']
                    if on_left or on_right: continue
                    # From here on, we know there is overlap.
                    # We need to identify how much:
                    for (x,y) in vent['area']:
                        pass
                else:
                '''
                if oent['depth'] > vent['depth']:   continue
                vent['area'] = vent['area'] - oent['area']
            non_occl_visible_area = len(vent['area'])
            non_occl_visible_percentage = non_occl_visible_area / vent['screen_area'] * 100.0

            if non_occl_visible_percentage >= (self.qualifying_area_ratio*100.0):
                # The object is sufficiently visible, despite occlusions.
                non_occl_visible_objects.append(vent)
            elif init_visible_area == non_occl_visible_area \
            and vent['screen_ratio'] >= (self.qualifying_screen_ratio*100.0):
                # The object is very close from the agent, and no other object occludes it.
                non_occl_visible_objects.append(vent)

            if self.verbose:
                print(f"Entity {vent['entity']} is visible : {non_occl_visible_percentage} %.")
                print(f"X : {vent['min_x']} to {vent['max_x']} / Y : {vent['min_y']} to {vent['max_y']}")
                print(f"Area : {vent['screen_area']} px-sq : {vent['screen_ratio']}% of the screen") 
             
        # Sort visible objects from left to right based on their screen-space 
        # bounding box's right-most point:
        non_occl_visible_objects.sort(key=lambda obj: obj['max_x'] * screen_width)
        
        # Filtering for entities only :
        visible_objects = [obj['entity'] for obj in non_occl_visible_objects]
        visible_objects = [f"{getattr(ent, 'color', '')} {type(ent).__name__.lower()}" for ent in visible_objects]
        
        #visible_objects = ', '.join(visible_objects)
        visible_objects = ' '.join(visible_objects)

        if visible_objects == '':
            visible_objects = 'EoS'
        else:
            visible_objects += ' EoS'

        return visible_objects
    
    def get_visible_entities(self):
        # Filtering for entities only :
        visible_objects = self.get_visible_ents()
        visible_objects = [f"{getattr(ent, 'color', '')} {type(ent).__name__.lower()}" for ent in visible_objects]
        
        #visible_objects = ', '.join(visible_objects)
        visible_objects = ' '.join(visible_objects)

        if visible_objects == '':
            visible_objects = 'EoS'
        else:
            visible_objects += ' EoS'

        return visible_objects
    
    def get_agent_pos_in_top_view(self):
        frame_buffer, \
        min_x, max_x, min_z, max_z = self.setup_top_view()

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(min_x, max_x, -max_z, -min_z, -100, 100.0)

        # Setup the camera
        # Y maps to +Z, Z maps to +Y
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        m = [
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            -1,
            0,
            0,
            0,
            0,
            0,
            1,
        ]
        glLoadMatrixf((GLfloat * len(m))(*m))

        # Retrieve projection matrix, view matrix, and viewport dimensions from OpenGL
        projection_matrix = (ctypes.c_float *16)()
        view_matrix = (ctypes.c_float * 16)()
        viewport = (ctypes.c_int * 4)()
        
        glGetIntegerv(GL_VIEWPORT, viewport)
        viewport = np.array(viewport, dtype=np.int32)
        screen_width = viewport[2]
        screen_height = viewport[3]
        
        glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix)
        glGetFloatv(GL_MODELVIEW_MATRIX, view_matrix)
        projection_matrix = np.array(projection_matrix, dtype=np.float32).reshape((4,4)).T
        view_matrix = np.array(view_matrix, dtype=np.float32).reshape((4,4)).T
        
        Y_VEC = np.array([0, 1, 0])
        p = self.unwrapped.agent.pos + Y_VEC * self.unwrapped.agent.height
        dv = self.unwrapped.agent.dir_vec * self.unwrapped.agent.radius
        rv = self.unwrapped.agent.right_vec * self.unwrapped.agent.radius
        
        p0 = p #+ dv
        p1 = p + 2.5*(dv + rv) #+ 0.75 * (rv - dv)
        p2 = p + 2.5*(dv - rv) #+ 0.75 * (-rv - dv) 
        
        agent_pos = np.stack([
            p0,
            p1,
            p2],
            axis=0,
        )
        
        vertices = agent_pos
        # Create Model Matrix of the mesh:
        model_matrix = np.identity(4)
        
        # Transform mesh vertices to NDC space
        vertices_homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
        vertices_ndc = (projection_matrix @ view_matrix @ model_matrix @ vertices_homogeneous.T).T
        vertices_ndc /= vertices_ndc[:, 3].reshape(-1, 1)
        
        # Transform NDC vertices to screen space
        # Taking into account OpenGL inverted Y-axis... (going upward while screen Y axises are usually meant to go downward) :
        vertices_screen = np.array([
            (vertices_ndc[:,0]+1)*0.5*screen_width,
            (1-(vertices_ndc[:,1]+1)*0.5)*screen_height,
        ]).T
        
        agent_pos_in_top_view = vertices_screen
        return agent_pos_in_top_view
    
    def setup_top_view(self):
        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()
        # Bind the frame buffer before rendering into it
        frame_buffer = self.unwrapped.vis_fb #obs_fb
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.unwrapped.sky_color, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Scene extents to render
        min_x = self.unwrapped.min_x - 1
        max_x = self.unwrapped.max_x + 1
        min_z = self.unwrapped.min_z - 1
        max_z = self.unwrapped.max_z + 1

        width = max_x - min_x
        height = max_z - min_z
        aspect = width / height
        fb_aspect = frame_buffer.width / frame_buffer.height

        # Adjust the aspect extents to match the frame buffer aspect
        if aspect > fb_aspect:
            # Want to add to denom, add to height
            new_h = width / fb_aspect
            h_diff = new_h - height
            min_z -= h_diff / 2
            max_z += h_diff / 2
        elif aspect < fb_aspect:
            # Want to add to num, add to width
            new_w = height * fb_aspect
            w_diff = new_w - width
            min_x -= w_diff / 2
            max_x += w_diff / 2

        return frame_buffer, min_x, max_x, min_z, max_z
    
    def get_top_view(self):
        """
        Render a top view of the whole map (from above),
        without the agent visible.
        """
        
        frame_buffer, \
        min_x, max_x, min_z, max_z = self.setup_top_view()

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(min_x, max_x, -max_z, -min_z, -100, 100.0)

        # Setup the camera
        # Y maps to +Z, Z maps to +Y
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        m = [
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            -1,
            0,
            0,
            0,
            0,
            0,
            1,
        ]
        glLoadMatrixf((GLfloat * len(m))(*m))

        top_view_image = self.unwrapped._render_world(frame_buffer, render_agent=False)
        
        return top_view_image

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        visible_entities = self.get_visible_entities() 
        if self.as_obs:
            obs['visible_entities'] = visible_entities
        else:
            info['visible_entities'] = visible_entities 
        
        if self.with_top_view:
            self.top_view = self.get_top_view()
            obs['top_view'] = [copy.deepcopy(self.top_view)]
            self.agent_pos_in_top_view = self.get_agent_pos_in_top_view()
            obs['agent_pos_in_top_view'] = [copy.deepcopy(self.agent_pos_in_top_view)]
        
        return obs, info

    def step(self, action, **kwargs):
        next_obs, reward, termination, truncation, info = self.env.step(action, **kwargs)
        
        visible_entities = self.get_visible_entities() 
        if self.as_obs:
            next_obs['visible_entities'] = visible_entities
        else:
            info['visible_entities'] = visible_entities 

        if self.with_top_view:
            self.top_view = self.get_top_view()
            next_obs['top_view'] = [copy.deepcopy(self.top_view)]
            self.agent_pos_in_top_view = self.get_agent_pos_in_top_view()
            next_obs['agent_pos_in_top_view'] = [copy.deepcopy(self.agent_pos_in_top_view)]

        return next_obs, reward, termination, truncation, info
