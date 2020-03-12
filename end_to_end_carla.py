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
from collections import deque

Setting_ACTIONS = ['forward','left','right', 'forward_left', 'forward_right', 'brake', 'brake_left', 'brake_right']


class ACTIONS:
    forward = 0
    left = 1
    right = 2
    forward_left = 3
    forward_right = 4
    brake = 5
    brake_left = 6
    brake_right = 7
    no_action = 8


ACTION_CONTROL = {
    0: [1, 0, 0],
    1: [0, 0, -1],
    2: [0, 0, 1],
    3: [1, 0, -1],
    4: [1, 0, 1],
    5: [0, 1, 0],
    6: [0, 1, -1],
    7: [0, 1, 1],
    8: None,
}

ACTIONS_NAMES = {
    0: 'forward',
    1: 'left',
    2: 'right',
    3: 'forward_left',
    4: 'forward_right',
    5: 'brake',
    6: 'brake_left',
    7: 'brake_right',
    8: 'no_action',
}

REPLAY_MEMORY_SIZE = 5_000

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 600


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    action_space_size = len(Setting_ACTIONS)

    def __init__(self,town='/Game/Carla/Maps/Town03'):
        self.arrive_target_location = False
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(2.0)
        self.client.load_world(town)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.actions = [getattr(ACTIONS, action) for action in Setting_ACTIONS]
        self.spawn_points = self.world.get_map().get_spawn_points()[:10]  # some points to begin
        self.end_points=self.world.get_map().get_spawn_points()[-10:]
    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.spawn_points)  # 起点
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
        self.target_location = random.choice(self.end_points)

        while self.front_camera is None:
            time.sleep(0.01)
        self.arrive_target_location = False
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

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

        if self.actions[action] != ACTIONS.no_action:
            self.vehicle.apply_control(carla.VehicleControl(throttle=ACTION_CONTROL[self.actions[action]][0],
                                                            steer=ACTION_CONTROL[self.actions[action]][
                                                                      2] * self.STEER_AMT,
                                                            brake=ACTION_CONTROL[self.actions[action]][1]))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        measurements = self.vehicle.get_transform()
        # Distance towards goal (in km)
        d_x = measurements.location.x
        d_y = measurements.location.y
        d_z = measurements.location.z
        player_location = np.array([d_x, d_y, d_z])
        goal_location = np.array([self.target_location.location.x,
                                  self.target_location.location.y,
                                  self.target_location.location.z])
        d = np.linalg.norm(player_location - goal_location) / 1000
        if d <= 0.001:
            done = True
            reward = 1000
            self.arrive_target_location = True
        elif len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 10:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if time.time()-self.episode_start>SECONDS_PER_EPISODE:
            done=True

        return self.front_camera, reward, done, None
