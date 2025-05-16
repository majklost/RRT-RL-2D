import pymunk
from typing import TYPE_CHECKING
from pymunk.pygame_util import DrawOptions

from dataclasses import dataclass
from typing import TypedDict

from ..assets.multibody import MultiBody
from ..assets.singlebody import SingleBody
if TYPE_CHECKING:
    from ..rendering.base_renderer import BaseRenderer
from .standard_config import STANDARD_CONFIG
from .collision_data import CollisionData


class Simulator:
    """
    A wrapper around pymunk space, handling the simulation, collision registration, etc...
    """

    def __init__(self, movable_objects, fixed_objects, config: dict | None = STANDARD_CONFIG):
        self._space = pymunk.Space()
        self.renderer = None
        self.movable_objects = movable_objects
        self.fixed_objects = fixed_objects
        self._steps = 0

        if config is None:
            config = STANDARD_CONFIG

        self._process_config(config)
        self._space.collision_slop = .01
        self._add_objects_to_space()
        self._collision_handling()

    def _process_config(self, config):
        self._fps = config['fps']
        self._gravity = config['gravity']
        self._damping = config['damping']
        self._space.gravity = self._gravity
        self._space.damping = self._damping

    @property
    def damping(self):
        return self._space.damping

    @property
    def fps(self) -> int:
        return self._fps

    def export(self):
        export_space = {
            "iterations": self._space.iterations, "gravity": self._space.gravity,
            "damping": self._space.damping, "idle_speed_threshold": self._space.idle_speed_threshold,
            "sleep_time_threshold": self._space.sleep_time_threshold,
            "collision_slop": self._space.collision_slop, "collision_bias": self._space.collision_bias,
            "collision_persistence": self._space.collision_persistence,
            "bodies": [b.copy() for b in self._space.bodies]
        }
        return PMExport(space=export_space, steps=self._steps)

    def import_from(self, export: 'PMExport'):
        self._space.iterations = export.space['iterations']
        self._space.gravity = export.space['gravity']
        self._space.damping = export.space['damping']
        self._space.idle_speed_threshold = export.space['idle_speed_threshold']
        self._space.sleep_time_threshold = export.space['sleep_time_threshold']
        self._space.collision_slop = export.space['collision_slop']
        self._space.collision_bias = export.space['collision_bias']
        self._space.collision_persistence = export.space['collision_persistence']
        self._steps = export.steps
        for i in range(len(export.space['bodies'])):
            self._space.bodies[i].position = export.space['bodies'][i].position
            self._space.bodies[i].velocity = export.space['bodies'][i].velocity
            self._space.bodies[i].angle = export.space['bodies'][i].angle
            self._space.bodies[i].angular_velocity = export.space['bodies'][i].angular_velocity
            self._space.bodies[i].force = export.space['bodies'][i].force
            self._space.bodies[i].torque = export.space['bodies'][i].torque

        for b in self.movable_objects:
            b.collision_clear()
        for b in self.fixed_objects:
            b.collision_clear()
        self._collect_objects()

    def _add_objects_to_space(self):
        for i, obj in enumerate(self.fixed_objects):
            obj.set_ID((i,), moveable=False)
            obj.add_to_space(self._space)
        for i, obj in enumerate(self.movable_objects):
            obj.set_ID((i,), moveable=True)
            obj.add_to_space(self._space)

    def _collision_handling(self):
        handler = self._space.add_default_collision_handler()
        def begin_fnc(a, s, d): return self._begin_collision(a, s, d)
        def sep_fnc(a, s, d): return self._end_collision(a, s, d)

        handler.post_solve = begin_fnc
        handler.separate = sep_fnc

    def _collect_objects(self):
        for b in self._space.bodies:
            if hasattr(b, 'moveId'):
                cid = b.moveId
                self.movable_objects[cid[0]].link_body(b, cid[1:])

            else:
                cid = b.fixedId
                self.fixed_objects[cid[0]].link_body(b, cid[1:])

    def attach_renderer(self, renderer: 'BaseRenderer'):
        """
        Register a renderer that will be used to render the simulator
        """
        self.renderer = renderer

    def step(self):
        """
        Step the simulator by one step
        :return: True if the simulator is still running, False otherwise
        """
        self._step()
        self._steps += 1
        if self.renderer is not None:
            self.renderer.render(self)

    @property
    def steps(self):
        return self._steps

    def _step(self):
        self._space.step(1 / self._fps)

    def draw_on(self, dops: DrawOptions):
        """
        Draw the simulator on the screen wrapped by the DrawOptions object
        :param dops: DrawOptions object
        """
        self._space.debug_draw(dops)

    def shape_query(self, shape):
        return self._space.shape_query(shape)

    def _begin_collision(self, arbiter: pymunk.Arbiter, space, data):
        b1 = arbiter.shapes[0]
        b2 = arbiter.shapes[1]
        # print("Collision detected")

        cid1, movable1 = self._get_id(b1.body)
        cid2, movable2 = self._get_id(b2.body)
        # same_object = movable1 == movable2 and cid1[0] == cid2[0]
        o1 = self._get_object(cid1, movable1)
        o2 = self._get_object(cid2, movable2)

        if o1.track_colisions:
            data1 = CollisionData(arbiter.normal, b2, o2, cid1, cid2)
            o1.collision_start(data1)
        if o2.track_colisions:
            data2 = CollisionData(-arbiter.normal, b1, o1, cid2, cid1)
            o2.collision_start(data2)

        # o1.color = (255,0,0,0)
        # o2.color = (0,255,0,0)

        return True

    @staticmethod
    def _get_id(body):
        if hasattr(body, 'moveId'):
            cid = body.moveId
            return cid, True

        elif hasattr(body, 'fixedId'):
            cid = body.fixedId
            return cid, False

        raise ValueError("Object with no ID")

    def _get_object(self, cid, movable):
        if movable:
            return self.movable_objects[cid[0]]
        else:
            return self.fixed_objects[cid[0]]

    def _end_collision(self, arbiter, space, data):
        b1 = arbiter.shapes[0]
        b2 = arbiter.shapes[1]
        # print("Collision ended")

        cid1, movable1 = self._get_id(b1.body)
        cid2, movable2 = self._get_id(b2.body)
        # same_object = movable1 == movable2 and cid1[0] == cid2[0]

        o1 = self._get_object(cid1, movable1)
        o2 = self._get_object(cid2, movable2)
        if o1.track_colisions:
            data1 = CollisionData(arbiter.normal, b2, o2, cid1, cid2)
            o1.collision_end(data1)
        if o2.track_colisions:
            data2 = CollisionData(-arbiter.normal, b1, o1, cid2, cid1)
            o2.collision_end(data2)


@dataclass
class PMExport:
    space: dict
    steps: int
