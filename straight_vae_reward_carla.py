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

import time
import random
import math
import points as P
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

actions = P.actions
REPLAY_MEMORY_SIZE = 5_000

IM_WIDTH = 200
IM_HEIGHT = 88
SECONDS_PER_EPISODE = 600
points = P.straight_points


class CarEnv:
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    action_space_size = len(actions)

    def __init__(self):
        self.arrive_target_location = False
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(2.0)
        self.client.load_world('Town01')
        self.world = self.client.get_world()
        self.debug = self.world.debug
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.actions = actions
        self.map = self.world.get_map()
        self.lierror = 0

    def reset(self):
        self.lierror = 0
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
        transform = carla.Transform(carla.Location(x=-0.5, z=2.8), carla.Rotation(pitch=-15))
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
        kmh = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        center_ = self.map.get_waypoint(current_t.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        measurements = self._get_measurements(current_t, self.target_location, kmh, center_.transform)

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
            self.lierror = -200
        else:
            self.lierror = -100

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]

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
        kmh = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        current_t = self.vehicle.get_transform()

        center_ = self.map.get_waypoint(current_t.location, project_to_road=True, lane_type=carla.LaneType.Driving)

        measurements = self._get_measurements(current_t, self.target_location, kmh, center_.transform)

        d = self.get_distance(current_t, self.target_location)
        if time.time() - self.episode_start > SECONDS_PER_EPISODE:
            done = True
            reward = -200
        elif d <= 3:
            done = True
            reward = 1000
            self.arrive_target_location = True
        elif len(self.collision_hist) != 0:
            done = True
            reward = -100
        elif self.lierror != 0:
            done = True
            reward = self.lierror
            self.lierror = 0
        else:
            done = False
            speed_reward = self.speed_reward(kmh)
            self.debug.draw_point(center_.transform.location)
            # center_next = center_.next(10)
            # if len(center_next):
            #   self.debug.draw_point(center_next[0].transform.location)
            if center_.is_intersection:
                a_reward = 0
                yaw_reward = 0
                d_reward = (self.d - d)

            else:
                a_reward = -1 * self.get_distance(current_t, center_.transform) / 5
                yaw_reward = self.get_yaw(current_t, center_.transform)
                d_reward = 0
            # 角度的奖励值

            reward = speed_reward * yaw_reward + a_reward + d_reward

            # print(reward)
        #reward = reward / 10.0
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

    def _get_measurements(self, current_point, end_point, kmh, center_):

        direction = self.get_planner_command(current_point, end_point)
        self.direction = direction
        '''
        current_location = current_point.location
        end_location = end_point.location
        target_location_rel_x, target_location_rel_y = self.get_relative_location_target(current_location.x,
                                                                                         current_location.y,
                                                                                         current_point.rotation.yaw,
                                                                                         end_location.x, end_location.y)
        
        # target_rel_norm = np.linalg.norm(np.array([target_location_rel_x.item(), target_location_rel_y.item()]))
        # target_rel_x_unit = target_location_rel_x.item() / target_rel_norm
        # target_rel_y_unit = target_location_rel_y.item() / target_rel_norm
        '''
        acceleration = self.vehicle.get_acceleration()
        a = math.sqrt(acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2)
        direction_list = [0., 0., 0., 0., 0., 0., 0.]
        if direction == -1:
            direction_list[0] = 1.
        else:
            direction_list[direction] = 1.

        measurements = np.array([
            current_point.location.x - center_.location.x,
            current_point.location.y - center_.location.y,
            current_point.rotation.yaw / 180.0,
            (current_point.rotation.pitch - center_.rotation.pitch) / 90.0,
            (current_point.rotation.yaw - center_.rotation.yaw) / 90.0,
            (current_point.rotation.roll - center_.rotation.roll) / 90.0,
            kmh / 10.0,
            a / 10.0])
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

    def speed_reward(self, kmh):
        return kmh / 45.0

    def get_distance(self, current_location, target_location):
        d_x = current_location.location.x
        d_y = current_location.location.y
        d_z = current_location.location.z
        player_location = np.array([d_x, d_y, d_z])
        goal_location = np.array([target_location.location.x,
                                  target_location.location.y,
                                  target_location.location.z])
        d = np.linalg.norm(player_location - goal_location)
        return d

    def get_yaw(self, current_point, right_point):
        c_yaw = current_point.rotation.yaw
        r_yaw = right_point.rotation.yaw
        cha_yaw = abs(c_yaw - r_yaw)
        cha_yaw = cha_yaw % 360
        if cha_yaw <= 1:
            return 10
        else:
            return - cha_yaw / 90 + 1
