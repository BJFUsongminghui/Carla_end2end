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
import numpy as np
import cv2
import time
import random
import math
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from collections import deque

actions = [[0., 0., 0.], [0., -0.1, 0], [0., -0.35, 0], [0., -0.25, 0], [0., 0.25, 0], [0., 0.5, 0],
           [0., 0.35, 0], [0., 0.1, 0],
           [1., 0., 0], [1., -0.1, 0], [1., -0.35, 0], [1., -0.25, 0], [1., 0.25, 0], [1., 0.5, 0],
           [1., 0.35, 0], [1., 0.1, 0],
           [0.75, 0., 0], [0.75, -0.1, 0.0], [0.75, -0.35, 0.0], [0.75, -0.25, 0.0],[0.75, 0.25, 0.0],
           [0.75, 0.5, 0.0], [0.75, 0.35, 0.0], [0.75, 0.1, 0.0],
           [0.5, 0., 0], [0.5, -0.1, 0.0], [0.5, -0.35, 0.0], [0.5, -0.25, 0.0], [0.5, 0.25, 0.0],
           [0.5, 0.5, 0.0], [0.5, 0.35, 0.0], [0.5, 0.1, 0.0],
           [0.25, 0., 0], [0.25, -0.1, 0], [0.25, -0.35, 0], [0.25, -0.25, 0], [0.25, 0.25, 0],
           [0.25, 0.5, 0], [0.25, 0.35, 0], [0.25, 0.1, 0],
           [0., 0., 0.15], [0., 0., 0.25], [0., 0., 0.35], [0., 0., 0.5], [0., 0., 0.75], [0., 0., 1.0]]
REPLAY_MEMORY_SIZE = 5_000

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 600
points = [[carla.Transform(carla.Location(x=220.093, y=9.80837, z=1.8431), carla.Rotation(pitch=0, yaw=0.855804, roll=0)),
           carla.Transform(carla.Location(x=231.271, y=14.2655, z=1.8431), carla.Rotation(pitch=0, yaw=91.3932, roll=0))],
          [carla.Transform(carla.Location(x=220.146, y=62.6845, z=1.8431), carla.Rotation(pitch=0, yaw=0.855804, roll=0)),
           carla.Transform(carla.Location(x=229.974, y=79.3, z=1.8431), carla.Rotation(pitch=0, yaw=91.3932, roll=0))],
          [carla.Transform(carla.Location(x=13.3843, y=193.564, z=1.8431), carla.Rotation(pitch=0, yaw=179.857, roll=0)),
           carla.Transform(carla.Location(x=5.8542, y=189.193, z=1.8431), carla.Rotation(pitch=0, yaw=-90.3625, roll=0))]
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
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.actions = actions
        self.map = self.world.get_map()
        self.lierror = False

    def reset(self):
        self.lierror = False
        self.direction = 0
        self.collision_hist = []
        self.actor_list = []
        index = random.randint(0, len(points) - 1)
        self.transform = points[index][0]  # 起点
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, self.transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # sensor.other.lane_invasion
        linesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.linesensor = self.world.spawn_actor(linesensor, self.transform, attach_to=self.vehicle)
        self.actor_list.append(self.linesensor)
        self.linesensor.listen(lambda event: self.lane_invasion(event))

        self.target_location = points[index][1]

        while self.front_camera is None:
            time.sleep(0.01)
        self.arrive_target_location = False
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        current_t = self.transform

        d_x = current_t.location.x
        d_y = current_t.location.y
        d_z = current_t.location.z
        player_location = np.array([d_x, d_y, d_z])
        goal_location = np.array([self.target_location.location.x,
                                  self.target_location.location.y,
                                  self.target_location.location.z])
        self.d = np.linalg.norm(player_location - goal_location)
        v = self.vehicle.get_velocity()
        measurements = self._get_measurements(current_t, self.target_location, v)

        return self.front_camera, measurements, self.direction

    def collision_data(self, event):
        self.collision_hist.append(event)

    def lane_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        brokensolid = "'BrokenSolid'"
        solid = "'Solid'"
        solid2 = "'SolidSolid'"
        if solid in text or solid2 in text or brokensolid in text:
            self.lierror = True

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

        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle,
                                                        steer=steer,
                                                        brake=brake))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        current_t = self.vehicle.get_transform()

        measurements = self._get_measurements(current_t, self.target_location, v)

        # Distance towards goal (in m)
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
            reward = 1000
            self.arrive_target_location = True
        elif len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 1 or kmh > 90:
            done = False
            reward = -1
        elif self.lierror:
            done = False
            self.lierror = False
            reward = -2
        else:
            done = False
            reward = 1

        if time.time() - self.episode_start > SECONDS_PER_EPISODE:
            done = True
            reward = -200

        if self.d - d > 0:
            reward +=1
        self.d = d

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
        acceleration = self.vehicle.get_acceleration()
        direction_list = [0., 0., 0., 0., 0., 0., 0.]
        if direction == -1:
            direction_list[0] = 1.
        else:
            direction_list[direction] = 1.

        measurements = np.array([
            target_rel_y_unit,
            target_rel_x_unit,
            current_point.rotation.pitch / 180.0,
            current_point.rotation.yaw / 180.0,
            current_point.rotation.roll / 180.0,
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
