import gymnasium as gym
import numpy as np

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
    """

    def __init__(self, env, relevant_entity_types=['Box','Key','Ball'], qualifying_area_ratio=0.1):
        assert isinstance(relevant_entity_types, list) and len(relevant_entity_types)
        assert isinstance(qualifying_area_ratio, float) and 0 < qualifying_area_ratio <= 1.0
        super().__init__(env)
        self.relevant_entity_types = relevant_entity_types
        self.qualifying_area_ratio = qualifying_area_ratio
    
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
 
    def get_visible_entities(self):
        #obs = self.unwrapped.render_obs()
        entities = self.unwrapped.entities
        filtered_entities = self._filter_entities(entities)
        meshes = [self._Entity2BoundingBoxMesh(fent) for fent in filtered_entities]

        visible_objects = []
        
        # Retrieve projection matrix, view matrix, and viewport dimensions from OpenGL
        projection_matrix = (ctypes.c_float *16)() #np.zeros((4, 4), dtype=np.float32)
        view_matrix = (ctypes.c_float * 16)() #np.zeros((4, 4), dtype=np.float32)
        viewport = (ctypes.c_int * 4)()
        glGetIntegerv(GL_VIEWPORT, viewport)
        viewport = np.array(viewport, dtype=np.int32)
        screen_width = viewport[2]
        screen_height = viewport[3]
        
        self.shadow_window.switch_to()
        #self.unwrapped.obs_fb.bind()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.unwrapped.agent.cam_fov_y,
            screen_width/float(screen_height),
            0.04,
            100.0,
        )

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            *self.unwrapped.agent.cam_pos,
            *(self.unwrapped.agent.cam_pos+self.unwrapped.agent.cam_dir),
            0,
            1.0,
            0.0,
        )

        glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix)
        glGetFloatv(GL_MODELVIEW_MATRIX, view_matrix)
        projection_matrix = np.array(projection_matrix, dtype=np.float32).reshape((4,4))
        view_matrix = np.array(view_matrix, dtype=np.float32).reshape((4,4))
        
        print(projection_matrix)
        print(f" Pos : {self.unwrapped.agent.cam_pos}")
        print(f" Dir : {self.unwrapped.agent.cam_dir}")
        print(view_matrix)

        # Loop through each mesh in the list
        for midx, mesh in enumerate(meshes):
            print(viewport)
            print(f" Ent Pos: {filtered_entities[midx].pos}")

            vertices = mesh['vertices']
            center = mesh['center']

            # Create Model Matrix of the mesh:
            model_matrix = np.identity(4)
            # Set the translation component of the model matrix
            #model_matrix[:3, 3] = center
            # no need, the vertices are already in place...
            # Set the scaling component of the model matrix
            model_matrix[0, 0] = 1.0 #scale[0]
            model_matrix[1, 1] = 1.0 #scale[1]
            model_matrix[2, 2] = 1.0 #scale[2] 
            
            # Transform mesh vertices to NDC space
            vertices_homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
            #vertices_ndc = (projection_matrix @ view_matrix @ model_matrix @ vertices_homogeneous.T).T
            print(f" Model : {model_matrix}")
            vertices_ndc = ( model_matrix @ vertices_homogeneous.T).T
            print(f" Model @ NDC : {vertices_ndc.mean(axis=0)}")
            print(f" View : {view_matrix}")
            vertices_ndc = (view_matrix @ vertices_ndc.T).T
            print(f" View @ Model @ NDC : {vertices_ndc.mean(axis=0)}")
            print(f" Proj : {projection_matrix}")
            vertices_ndc = (projection_matrix @ view_matrix @ model_matrix @ vertices_homogeneous.T).T
            print(f" Proj @ View @ Model @ NDC : {vertices_ndc.mean(axis=0)}")
            vertices_ndc /= vertices_ndc[:, 3].reshape(-1, 1)
            
            print(f" Homo : {vertices_homogeneous.mean(axis=0)}")
            print(f" (norm.) NDC : {vertices_ndc.mean(axis=0)}")

            # Transform NDC vertices to screen space
            #vertices_screen = (vertices_ndc[:, :2] + 1) * 0.5 * np.array([screen_width, screen_height])
            # Taking into account OpenGL inverted Y-axis... (going upward while screen Y axises are usually meant to go downward) :
            vertices_screen = np.array([(vertices_ndc[:,0]+1)*0.5*screen_width,
                (1-(vertices_ndc[:,1]+1)*0.5)*screen_height]).T
            
            print(f" Screen : {vertices_screen.mean(axis=0)}")

            # Compute screen-space bounding box coordinates in pixel space
            min_x = int(min(vertices_screen[:, 0]))
            max_x = int(max(vertices_screen[:, 0]))
            min_y = int(min(vertices_screen[:, 1]))
            max_y = int(max(vertices_screen[:, 1]))
            print(f"X : {min_x} to {max_x} / Y : {min_y} to {max_y}") 
            
            # Compute screen-space bounding box area
            screen_ent_width = max_x - min_x
            screen_ent_height = max_y - min_y
            screen_ent_area = screen_ent_width * screen_ent_height

            # Compute area of the bounding box that is inside the screen
            visible_ent_width = max(min_x + screen_width, screen_width) - max(min_x, 0)
            visible_ent_height = max(min_y + screen_height, screen_height) - max(min_y, 0)
            visible_ent_area = visible_ent_width * visible_ent_height

            # Compute the ratio of visible area to total area
            visible_percentage = visible_ent_area / screen_ent_area * 100
            
            print(f"Entity {filtered_entities[midx]} is visible at {visible_percentage} % with max_x being at {max_x}/{screen_width}.")

            # Check if visible percentage exceeds 10%
            if visible_percentage >= 10.0:
                # Object is at least 10% visible, add to visible objects list
                visible_objects.append(
                    {
                        'entity':filtered_entities[midx], 
                        'max_x':max_x,
                    },
                )

        # Sort visible objects from left to right based on their screen-space 
        # bounding box's right-most point:
        visible_objects.sort(key=lambda obj: obj['max_x'] * screen_width)
        
        # Filtering for entities only :
        visible_objects = [obj['entity'] for obj in visible_objects]

        return visible_objects
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        info['visible_entities'] = self.get_visible_entities()

        return obs, info

    def step(self, action, **kwargs):
        next_obs, reward, termination, truncation, info = self.env.step(action, **kwargs)

        info['visible_entities'] = self.get_visible_entities()

        return next_obs, reward, termination, truncation, info
