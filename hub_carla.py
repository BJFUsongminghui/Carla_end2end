import glob
import os
import sys

try:
    sys.path.append(glob.glob('../CARLA95/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
import numpy as np
import cv2
import time
import random
import math
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from collections import deque




import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name



# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame_number)
# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================




# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame_number, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

            def distance(l): return math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)















actions = [[0., 0., 0.], [0., -0.1, 0], [0., -0.35, 0], [0., -0.5, 0], [0., -0.25, 0], [0., 0.25, 0], [0., 0.5, 0],
           [0., 0.35, 0], [0., 0.1, 0],
           [1., 0., 0], [1., -0.1, 0], [1., -0.35, 0], [1., -0.5, 0], [1., -0.25, 0], [1., 0.25, 0], [1., 0.5, 0],
           [1., 0.35, 0], [1., 0.1, 0],
           [0.75, 0., 0], [0.75, -0.1, 0.0], [0.75, -0.35, 0.0], [0.75, -0.5, 0.0], [0.75, -0.25, 0.0],
           [0.75, 0.25, 0.0],
           [0.75, 0.5, 0.0], [0.75, 0.35, 0.0], [0.75, 0.1, 0.0],
           [0.5, 0., 0], [0.5, -0.1, 0.0], [0.5, -0.35, 0.0], [0.5, -0.5, 0.0], [0.5, -0.25, 0.0], [0.5, 0.25, 0.0],
           [0.5, 0.5, 0.0], [0.5, 0.35, 0.0], [0.5, 0.1, 0.0],
           [0.25, 0., 0], [0.25, -0.1, 0], [0.25, -0.35, 0], [0.25, -0.5, 0], [0.25, -0.25, 0], [0.25, 0.25, 0],
           [0.25, 0.5, 0], [0.25, 0.35, 0], [0.25, 0.1, 0],
           [0., 0., 0.15], [0., 0., 0.25], [0., 0., 0.35], [0., 0., 0.5], [0., 0., 0.75], [0., 0., 1.0]]
REPLAY_MEMORY_SIZE = 5_000

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 600
points = [[carla.Transform(carla.Location(x=-6.44617, y=42.1938, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0)),
           carla.Transform(carla.Location(x=-6.44617, y=180.1938, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=-6.44617, y=90.1938, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0)),
           carla.Transform(carla.Location(x=-6.44617, y=180.1938, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=-6.44617, y=140.1938, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0)),
           carla.Transform(carla.Location(x=-6.44617, y=180.1938, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=5.80, y=179, z=1.8431),
                           carla.Rotation(pitch=0, yaw=-90, roll=0)),
           carla.Transform(carla.Location(x=5.8, y=69.9, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=5.80, y=119, z=1.8431),
                           carla.Rotation(pitch=0, yaw=-90, roll=0)),
           carla.Transform(carla.Location(x=5.8, y=69.9, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=5.80, y=99.9, z=1.8431),
                           carla.Rotation(pitch=0, yaw=-90, roll=0)),
           carla.Transform(carla.Location(x=5.8, y=69.9, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=4.10, y=-43.1, z=1.8431),
                           carla.Rotation(pitch=0, yaw=-90, roll=0)),
           carla.Transform(carla.Location(x=7.3, y=-160.9, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=5.10, y=-93.1, z=1.8431),
                           carla.Rotation(pitch=0, yaw=-90, roll=0)),
           carla.Transform(carla.Location(x=7.3, y=-160.9, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=6.10, y=-120.1, z=1.8431),
                           carla.Rotation(pitch=0, yaw=-90, roll=0)),
           carla.Transform(carla.Location(x=7.3, y=-160.9, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=161.9, y=58.9, z=1.8431),
                           carla.Rotation(pitch=0, yaw=180, roll=0)),
           carla.Transform(carla.Location(x=107.3, y=59, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=-20, y=204, z=1.8431),
                           carla.Rotation(pitch=0, yaw=0, roll=0)),
           carla.Transform(carla.Location(x=155.6, y=204.1, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=86.2, y=7.8, z=1.8431),
                           carla.Rotation(pitch=0, yaw=0, roll=0)),
           carla.Transform(carla.Location(x=189.3, y=9.3, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))],
          [carla.Transform(carla.Location(x=-10.1, y=42.6, z=1.8431),
                           carla.Rotation(pitch=0, yaw=90, roll=0)),
           carla.Transform(carla.Location(x=-9.7, y=142.0, z=1.8431),
                           carla.Rotation(pitch=0, yaw=92.0042, roll=0))]
          ]


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    action_space_size = len(actions)

    def __init__(self):
        self.arrive_target_location = False
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(2.0)
        self.display = pygame.display.set_mode(
            (self.im_width, self.im_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.hud = HUD(self.im_width, self.im_height)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.actions = actions
        self.map = self.world.get_map()

        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None

    def reset(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        self.direction = 0
        self.collision_hist = []
        self.actor_list = []
        index = random.choice([0, 1, 2,3,4,5,6])
        self.transform = points[index][0]  # 起点
        self.player = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.player)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.player)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.player.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, self.transform, attach_to=self.player)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.target_location = points[index][1]

        while self.front_camera is None:
            time.sleep(0.01)
        self.arrive_target_location = False
        self.episode_start = time.time()
        self.player.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        current_t = self.transform
        v = self.player.get_velocity()
        measurements = self._get_measurements(current_t, self.target_location, v)
        self.collision_sensor=CollisionSensor(self.player,self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        return self.front_camera, measurements, self.direction

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        i3 = i3.reshape((3, self.im_height, self.im_width))
        self.front_camera = i3

    def step(self, action):

        throttle = self.actions[action][0]
        steer = self.actions[action][1] * self.STEER_AMT
        brake = self.actions[action][2]

        self.player.apply_control(carla.VehicleControl(throttle=throttle,
                                                        steer=steer,
                                                        brake=brake))

        v = self.player.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        current_t = self.player.get_transform()

        measurements = self._get_measurements(current_t, self.target_location, v)

        # Distance towards goal (in km)
        d_x = current_t.location.x
        d_y = current_t.location.y
        d_z = current_t.location.z
        player_location = np.array([d_x, d_y, d_z])
        goal_location = np.array([self.target_location.location.x,
                                  self.target_location.location.y,
                                  self.target_location.location.z])
        d = np.linalg.norm(player_location - goal_location)
        if d <= 3:
            done = True
            self.arrive_target_location = True
        elif len(self.collision_hist) != 0:
            done = True

        elif kmh < 1 or kmh > 90:
            done = False
        else:
            done = False

        if time.time() - self.episode_start > SECONDS_PER_EPISODE:
            done = True

        reward=1
        return self.front_camera, reward, done, measurements, self.direction

    def get_planner_command(self, current_point, end_point):
        current_location = current_point.location
        end_location = end_point.location
        global_Dao = GlobalRoutePlannerDAO(self.map)
        global_planner = GlobalRoutePlanner(global_Dao)
        global_planner.setup()
        commands = global_planner.abstract_route_plan(current_location, end_location)
        direction = commands[0].value
        return direction

    def _get_measurements(self, current_point, end_point, v):

        current_location = current_point.location
        end_location = end_point.location
        target_location_rel_x, target_location_rel_y = self.get_relative_location_target(current_location.x,
                                                                                         current_location.y,
                                                                                         current_point.rotation.yaw,
                                                                                         end_location.x, end_location.y)
        direction = self.get_planner_command(current_point, end_point)
        self.direction = direction
        target_rel_norm = np.linalg.norm(np.array([target_location_rel_x.item(), target_location_rel_y.item()]))
        target_rel_x_unit = target_location_rel_x.item() / target_rel_norm
        target_rel_y_unit = target_location_rel_y.item() / target_rel_norm
        acceleration = self.player.get_acceleration()
        direction_list = [0., 0., 0., 0., 0., 0., 0.]
        if direction == -1:
            direction_list[0] = 1.
        else:
            direction_list[direction] = 1.

        measurements = np.array([
            target_rel_y_unit,
            target_rel_x_unit,
            current_point.location.x / 200.0,
            current_point.location.y / 200.0,
            current_point.rotation.pitch / 180.0,
            current_point.rotation.yaw / 180.0,
            current_point.rotation.roll / 180.0,
            end_point.location.x / 200.0,
            end_point.location.y / 200.0,
            v.x * 3.6 / 10,
            v.y * 3.6 / 10,
            acceleration.x,
            acceleration.y])
        for d in direction_list:
            measurements = np.append(measurements, d)
        return measurements

    def get_relative_location_target(self, loc_x, loc_y, loc_yaw, target_x, target_y):
        veh_yaw = loc_yaw * np.pi / 180
        veh_dir_world = np.array([np.cos(veh_yaw), np.sin(veh_yaw)])
        veh_loc_world = np.array([loc_x, loc_y])
        target_loc_world = np.array([target_x, target_y])
        d_world = target_loc_world - veh_loc_world
        dot = np.dot(veh_dir_world, d_world)
        det = veh_dir_world[0] * d_world[1] - d_world[0] * veh_dir_world[1]
        rel_angle = np.arctan2(det, dot)
        target_location_rel_x = np.linalg.norm(d_world) * np.cos(rel_angle)
        target_location_rel_y = np.linalg.norm(d_world) * np.sin(rel_angle)

        return target_location_rel_x, target_location_rel_y

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()