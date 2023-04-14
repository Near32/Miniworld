import gymnasium as gym
import numpy as np

import pyglet
from pyglet.gl import *


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
        vertices = [
            # Front face
            [entity.pos[0] - half_width, entity.pos[1] - half_height, 0],  # 0
            [entity.pos[0] - half_width, entity.pos[1] + half_height, 0],  # 1
            [entity.pos[0] + half_width, entity.pos[1] + half_height, 0],  # 2
            [entity.pos[0] + half_width, entity.pos[1] - half_height, 0],  # 3
            # Back face
            [entity.pos[0] - half_width, entity.pos[1] - half_height, 2 * entity.height],  # 4
            [entity.pos[0] - half_width, entity.pos[1] + half_height, 2 * entity.height],  # 5
            [entity.pos[0] + half_width, entity.pos[1] + half_height, 2 * entity.height],  # 6
            [entity.pos[0] + half_width, entity.pos[1] - half_height, 2 * entity.height],  # 7
        ]
        
        center = entity.pos

        return {'verticies':vertices, 'center':center}
 
    def get_visible_entities(self):
    	entities = copy.deepcopy(self.unwrapped.entities)
        filtered_entities = self._filter_entities(entities)
	meshes = [self._Entity2BoundingBoxMesh(fent) for fent in filtered_entities]

	visible_objects = []
        
        # Retrieve projection matrix, view matrix, and viewport dimensions from OpenGL
        projection_matrix = np.zeros((4, 4), dtype=np.float32)
        view_matrix = np.zeros((4, 4), dtype=np.float32)
        viewport = glGetIntegerv(GL_VIEWPORT)
        screen_width = viewport[2]
        screen_height = viewport[3]
        
        glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix)
        glGetFloatv(GL_MODELVIEW_MATRIX, view_matrix)
        
        # Loop through each mesh in the list
        for midx, mesh in enumerate(meshes):
            vertices = mesh['verticies']
            center = mesh['center']

            # Create Model Matrix of the mesh:
            model_matrix = np.identity(4)
            # Set the translation component of the model matrix
            model_matrix[:3, 3] = center
            # Set the scaling component of the model matrix
            model_matrix[0, 0] = 1.0 #scale[0]
            model_matrix[1, 1] = 1.0 #scale[1]
            model_matrix[2, 2] = 1.0 #scale[2] 
            
            # Transform mesh vertices to NDC space
            vertices_homogeneous = np.concatenate([vertices, np.ones((mesh.num_vertices, 1))], axis=1)
            vertices_ndc = (projection_matrix @ view_matrix @ model_matrix @ vertices_homogeneous.T).T
            vertices_ndc /= vertices_ndc[:, 3].reshape(-1, 1)

            # Transform NDC vertices to screen space
            vertices_screen = (vertices_ndc[:, :2] + 1) * 0.5 * np.array([screen_width, screen_height])

            # Compute screen-space bounding box coordinates in pixel space
            min_x = int(min(vertices_screen[:, 0]))
            max_x = int(max(vertices_screen[:, 0]))
            min_y = int(min(vertices_screen[:, 1]))
            max_y = int(max(vertices_screen[:, 1]))
     
            # Compute screen-space bounding box area
            screen_width = max_x - min_x
            screen_height = max_y - min_y
            screen_area = screen_width * screen_height

            # Compute area of the bounding box that is inside the screen
            visible_width = max(min_x + screen_width, screen_width) - max(min_x, 0)
            visible_height = max(min_y + screen_height, screen_height) - max(min_y, 0)
            visible_area = visible_width * visible_height

            # Compute the ratio of visible area to total area
            visible_percentage = visible_area / screen_area * 100

            # Check if visible percentage exceeds 10%
            if visible_percentage >= 10.0:
                # Object is at least 10% visible, add to visible objects list
                visible_objects.append({'entity':filtered_entities[midx], 'mesh':mesh})

        # Sort visible objects from left to right based on their screen-space 
        # bounding box's right-most point:
        visible_objects.sort(key=lambda obj: obj['mesh'].max_x * screen_width)
        
        # Filtering for entities only :
        visible_objects = [obj['entity'] for obj in visible_objects]

        return visible_objects
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        info['visible_entities'] = self.get_visible_entities()

        return obs, info

    def step(self, action, **kwargs):
        next_obs, reward, truncated, done, info = self.env.step(action, **kwargs)

        info['visible_entities'] = self.get_visible_entities()

        return next_obs, reward, truncated, done, info
