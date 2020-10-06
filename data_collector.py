import threading
import subprocess
import os
import sys
import argparse
import pygame
import random
import numpy as np
import logging
import time

import carla
from carla import ColorConverter as cc
from carla import Transform
from matplotlib import cm
import open3d as o3d
# Note: add PythonAPI and PythonAPI/carla into python path

from examples.synchronous_mode import CarlaSyncMode
from examples.automatic_control import World, HUD, KeyboardControl, BehaviorAgent, BasicAgent, get_actor_display_name
from examples.automatic_control import CollisionSensor, LaneInvasionSensor, GnssSensor
from utils import vector3d_to_array, Timer
from bounding_box import create_kitti_datapoint
from dataexport import save_ref_files, save_image_data, save_kitti_data, save_lidar_data
from dataexport import save_groundplanes, save_calibration_matrices
from camera_utils import draw_2d_bounding_boxes, draw_3d_bounding_boxes

CLASSES_TO_LABEL = ["vehicle", "pedestrian"]


VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

class CameraManager(object):
    """
    A simplified camera manager class that spawns and keeps all required sensors for KITTI dataset
    """

    def __init__(self, parent_actor, hud, args):
        """Constructor method"""
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.index = 0

        attachment = carla.AttachmentType

        camera_rgb_attributes = {'image_size_x': str(args.width),
                                 'image_size_y': str(args.height)}
        camera_depth_attributes = {'image_size_x': str(args.width),
                                   'image_size_y': str(args.height)}
        lidar_ray_cast_attributes = {'channels': '64',                        # kitti paper: 64  repo: 40
                                     'range': str(args.lidar_range),          # kitti paper: 120
                                     'points_per_second': '1300000',          # kitti paper: ~1300000  repo: 720000
                                     'rotation_frequency': '10',              # repo/paper: 10
                                     'upper_fov': '7.0',                      # repo: 7  paper vert fov: 26.9°
                                     'lower_fov': '-16.0',                    # repo: -16  paper vert fov: 26.9°
                                     'atmosphere_attenuation_rate': '0.004',  # carla: 0.004
                                     'noise_stddev': '0.0',                   # carla: 0.0
                                     'dropoff_general_rate': '0.10',          # carla: 0.45
                                     'dropoff_zero_intensity': '0.4',         # carla: 0.4
                                     'dropoff_intensity_limit': '0.8'}        # carla: 0.8
        lidar_blickfeld_attributes = {'frame_mode': 'up',
                                      'scanlines': '50',
                                      'horizontal_fov_limit': '70',
                                      'vertical_fov_limit': '30.0',
                                      'mirror_frequency': '150.0',  # TODO what's this?
                                      'range': str(args.lidar_range)}

        self.default_sensor_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.default_sensor_attachment_type = attachment.Rigid

        # It seems there is a an issue with lidar transform that we should perform a rotation by -90 degree
        # for correct visualization and saving on file. This is solved in carla 0.9.10! Uncomment it for carla 0.9.9.
        # self.default_lidar_transform = carla.Transform(self.default_sensor_transform.location, carla.Rotation(yaw=-90))

        # Note that camera.rgb and camera.depth should have the same config for correct detection of occluded agents
        self.sensors = {'sensor.camera.rgb': {'name': 'Camera RGB',
                                              'attributes': camera_rgb_attributes,
                                              'transform': self.default_sensor_transform},
                        'sensor.camera.depth': {'name': 'Camera Depth (Gray Scale)',
                                                'attributes': camera_depth_attributes,
                                                'transform': self.default_sensor_transform}
                        }
        for lidar in args.lidars:
            if lidar == 'ray_cast':
                ray_cast_sensor_def = {'sensor.lidar.ray_cast': {'name': 'Ray-Cast',
                                                                 'attributes': lidar_ray_cast_attributes,
                                                                 'transform': self.default_sensor_transform}}
                self.sensors.update(ray_cast_sensor_def)
            elif lidar == 'blickfeld':
                blickfeld_sensor_def = {'sensor.lidar.blickfeld': {'name': 'Blickfeld',
                                                                   'attributes': lidar_blickfeld_attributes,
                                                                   'transform': self.default_sensor_transform}}
                self.sensors.update(blickfeld_sensor_def)

        self.setup_sensors(args)

    def get_intrinsic_matrix(self, camera):

        width = int(camera.attributes['image_size_x'])
        height = int(camera.attributes['image_size_y'])
        fov = float(camera.attributes['fov'])

        k = np.identity(3)
        k[0, 2] = width / 2.0
        k[1, 2] = height / 2.0
        k[0, 0] = k[1, 1] = width / (2.0 * np.tan(fov * np.pi / 360.0))

        return k

    def toggle_camera(self):
        """Activate a camera"""
        self.hud.notification('Only single camera transform is available!')

    def setup_sensors(self, args):
        # Spawns all sensors
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for sensor_key in self.sensors.keys():
            blp = bp_library.find(sensor_key)
            for key, val in self.sensors[sensor_key]['attributes'].items():
                blp.set_attribute(key, val)
            transform = self.sensors[sensor_key].get('transform', self.default_sensor_transform)
            attach_type = self.sensors[sensor_key].get('attach_type', self.default_sensor_attachment_type)
            sensor = self._parent.get_world().spawn_actor(blp,
                                                          transform,
                                                          attach_to=self._parent,
                                                          attachment_type=attach_type)
            self.sensors[sensor_key].update({'sensor': sensor})

            # Setup intrinsic matrix for camera rgb sensor
            if sensor_key == 'sensor.camera.rgb':
                camera_rgb_intrinsic = self.get_intrinsic_matrix(sensor)
                self.sensors[sensor_key].update({'calibration': camera_rgb_intrinsic})

            # Setup relative to camera matrix for lidar sensors
            elif 'lidar' in sensor_key:
                veh_cam_mat = self.sensors['sensor.camera.rgb']['transform'].get_inverse_matrix()
                lidar_veh_mat = self.sensors[sensor_key]['transform'].get_matrix()
                lidar_cam_mat = np.dot(veh_cam_mat, lidar_veh_mat)
                self.sensors[sensor_key].update({'lidar_cam_mat': lidar_cam_mat})

            if args.vis_lidar:
                # Setup open3d visualization for lidars
                if 'lidar' in sensor_key:
                    sensor_name = self.sensors[sensor_key]['name']
                    o3d_vis = o3d.visualization.Visualizer()
                    o3d_vis.create_window(
                        window_name=f'Carla {sensor_name} Lidar', width=960, height=540, left=480, top=270)
                    o3d_vis.get_render_option().background_color = [0.05, 0.05, 0.05]
                    o3d_vis.get_render_option().point_size = 1
                    o3d_vis.get_render_option().show_coordinate_frame = True
                    point_list = o3d.geometry.PointCloud()
                    self.sensors[sensor_key].update({'o3d_vis': o3d_vis, 'point_list': point_list, 'lidar_vis_frame': 0})

    def set_sensor(self, index, notify=True):
        """Set a sensor"""
        index = index % len(self.sensors.keys())
        if notify:
            key = self.sensors.keys()[index]
            self.hud.notification(self.sensors[key]['name'])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))


class World(World):
    """
    A simplified version of Carla World class where all necessary sensors are spawned initially and
    their transformations are fix (i.e., no camera toggling is available). The camera manager is thus simplified and
    is being rendered independently outside of this class.
    """
    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Set the seed if requested by user
        if args.seed is not None:
            random.seed(args.seed)

        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        logging.debug("Spawning the player")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            if args.autopilot:
                self.player.set_autopilot()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, args)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def render(self, display):
        """Destroy sensors"""
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroys all actors"""
        self.camera_manager.index = None
        for key, val in self.camera_manager.sensors.items():
            val['sensor'].destroy()
            val['sensor'] = None

    def destroy(self):
        print("Destroying actors...")
        actors = [val['sensor'] for val in self.camera_manager.sensors.values()]
        actors.extend([self.collision_sensor, self.lane_invasion_sensor, self.gnss_sensor, self.player])
        for actor in actors:
            if actor is not None and hasattr(actor, 'destroy'):
                actor.destroy()


class CarlaGame(object):

    def __init__(self, args):
        pygame.init()
        pygame.font.init()
        logging.debug('pygame started')

        self.tot_target_reached = 0
        self.num_min_waypoints = 21

        self._timer = Timer()
        self.reset_episode = True
        self.clock = pygame.time.Clock()
        self.world = None
        # To keep track of how far the car has driven since the last capture of data
        self._agent_location_on_last_capture = None
        self._frames_since_last_capture = 0
        # How many frames we have captured since reset
        self._captured_frames_since_restart = 0
        if args.save_data:
            self.captured_frame_no = self.current_captured_frame_num(args)

        self.display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)
        client.reload_world()
        self.hud = HUD(args.width, args.height)
        self.world = World(client.get_world(), self.hud, args)

        self.controller = KeyboardControl(self.world)
        if (args.autopilot == False):
            self.agent = BehaviorAgent(self.world.player)

            self.spawn_points = self.world.map.get_spawn_points()
            random.shuffle(self.spawn_points)

            if self.spawn_points[0].location != self.agent.vehicle.get_location():
                destination = self.spawn_points[0].location
            else:
                destination = self.spawn_points[1].location

            self.agent.set_destination(self.agent.vehicle.get_location(), destination, clean=True)

    def current_captured_frame_num(self, args):
        # Figures out which frame number we currently are on
        # This is run once, when we start the simulator in case we already have a dataset.
        # The user can then choose to overwrite or append to the dataset.
        label_path = os.path.join(args.phase_dir, 'label_2/')
        logging.debug('Path to label directory: {}'.format(label_path))
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        logging.info("Number of existing data files: {}".format(num_existing_data_files))
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(args.phase_dir))
        if answer.upper() == "O":
            logging.info(
                "Resetting frame number to 0 and overwriting existing")
            # Overwrite the data
            return 0
        logging.info("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files

    def _on_new_episode(self, args):
        logging.info('Starting a new episode...')

        self._timer = Timer()
        self.reset_episode = False
        self._is_on_reverse = False

        # Reset all tracking variables
        self._agent_location_on_last_capture = None
        self._frames_since_last_capture = 0
        self._captured_frames_since_restart = 0

        if (args.autopilot == False):
            self.agent.reroute(self.spawn_points)

    def _distance_since_last_recording(self):
        if self._agent_location_on_last_capture is None:
            return None
        cur_pos = vector3d_to_array(self.world.player.get_transform().location)
        last_pos = vector3d_to_array(self._agent_location_on_last_capture)

        def dist_func(x, y): return sum((x - y) ** 2)

        return dist_func(cur_pos, last_pos)

    def _save_datapoints(self, datapoints, cam_calibration, rgb_image, point_clouds, lidar_heights, lidar_cam_mats, args):
        # Determine whether to save files
        distance_driven = self._distance_since_last_recording()
        logging.debug("Distance driven since last recording: {}".format(distance_driven))
        has_driven_long_enough = distance_driven is None or distance_driven > args.distance_since_last_recording
        if (self._timer.step + 1) % args.steps_between_recordings == 0:
            if has_driven_long_enough and datapoints:
                self._update_agent_location()
                # Save screen, lidar and kitti training labels together with calibration and groundplane files
                self._save_training_files(datapoints, cam_calibration, point_clouds, rgb_image, lidar_heights, lidar_cam_mats, args)
                self.captured_frame_no += 1
                self._captured_frames_since_restart += 1
                self._frames_since_last_capture = 0
            else:
                logging.debug(
                    "Could save datapoint, but agent has not driven {} meters" +
                    " since last recording (Currently {} meters)".format(
                        args.distance_since_last_recording, distance_driven))
        else:
            self._frames_since_last_capture += 1
            logging.debug(
                "Could not save training data - no visible agents of selected classes in scene")

    def _render(self, display, sensor_data_dict, args):

        image = np.copy(sensor_data_dict['sensor.camera.rgb']['image'])
        image_depth = sensor_data_dict['sensor.camera.depth']['image']  # Gray-scale depth image for visualization
        depth_map = sensor_data_dict['sensor.camera.depth']['depth']

        # Retrieve and draw datapoints on rgb image
        datapoint_gen_time = time.time()
        image, datapoints, bounding_boxes, boxes_2d = self._generate_datapoints(image, depth_map, args)
        logging.info("datapoint generation time: ", (time.time() - datapoint_gen_time) * 1000.)
        # Display RGB Image
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        display.blit(surface, (0, 0))

        # TODO Display lidar point cloud on rgb image
        if args.vis_lidar:
            for key in sensor_data_dict.keys():
                if 'lidar' in key:
                    lidar_points = sensor_data_dict[key]['points']
                    lidar_intensity = lidar_points[:, -1]
                    lidar_point_list = self.world.camera_manager.sensors[key]['point_list']  # assume pass by ref
                    lidar_vis_frame = self.world.camera_manager.sensors[key]['lidar_vis_frame']
                    o3d_vis = self.world.camera_manager.sensors[key]['o3d_vis']
                    intensity_col = 1.0 - np.log(lidar_intensity) / np.log(np.exp(-0.004 * 100))
                    int_color = np.c_[
                        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
                        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
                        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

                    # We're negating the y to correclty visualize a world that matches
                    # what we see in Unreal since Open3D uses a right-handed coordinate system
                    lidar_raycast_points_o3d = np.copy(lidar_points)
                    lidar_raycast_points_o3d[:, 1] = -lidar_raycast_points_o3d[:, 1]
                    lidar_point_list.points = o3d.utility.Vector3dVector(lidar_raycast_points_o3d[:, :3])
                    lidar_point_list.colors = o3d.utility.Vector3dVector(int_color)

                    if lidar_vis_frame == 2:
                        o3d_vis.add_geometry(lidar_point_list)
                        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=2, origin=[0, 0, 0])
                        o3d_vis.add_geometry(mesh_frame)

                    o3d_vis.update_geometry(lidar_point_list)

                    o3d_vis.poll_events()
                    o3d_vis.update_renderer()
                    # # This can fix Open3D jittering issues:
                    # time.sleep(0.005)
                    lidar_vis_frame += 1
                    self.world.camera_manager.sensors[key]['lidar_vis_frame'] = lidar_vis_frame

        # Display 3D Bounding Boxes
        if args.vis_boxes3d:
            draw_3d_bounding_boxes(display, bounding_boxes)

        # Display 2D Bounding Boxes
        elif args.vis_boxes2d:
            draw_2d_bounding_boxes(display, boxes_2d)

        if args.vis_ry:
            myfont = pygame.font.SysFont('Comic Sans MS', 30)
            for datapoint in datapoints:
                ry = np.degrees(datapoint.rotation_y)
                textsurface = myfont.render('ry: {:.2f}'.format(np.round(ry)), False, (0, 0, 0))
                bbox_top_right = (datapoint.bbox[2], datapoint.bbox[1])
                display.blit(textsurface, bbox_top_right)

        return datapoints

    def _save_training_files(self, datapoints, cam_calibration, point_clouds, rgb_image, lidar_heights, lidar_cam_mats, args):
        logging.info("Attempting to save at timer step {}, frame no: {}".format(
            self._timer.step, self.captured_frame_no))

        for g_path, l_path, c_path, point_cloud, l_height, lidar_cam_mat in zip(args.groundplane_paths,
                                                                 args.lidar_paths,
                                                                 args.calibration_paths,
                                                                 point_clouds,
                                                                 lidar_heights,
                                                                 lidar_cam_mats):
            groundplane_fname = g_path.format(self.captured_frame_no)
            lidar_fname = l_path.format(self.captured_frame_no)
            calib_filename = c_path.format(self.captured_frame_no)
            save_groundplanes(groundplane_fname, self.world.player.get_transform(), l_height)
            save_lidar_data(lidar_fname, point_cloud)
            save_calibration_matrices(calib_filename, cam_calibration, lidar_cam_mat)

        kitti_fname = args.label_path.format(self.captured_frame_no)
        img_fname = args.image_path.format(self.captured_frame_no)
        save_ref_files(args.phase_dir, self.captured_frame_no)
        save_image_data(img_fname, rgb_image)
        save_kitti_data(kitti_fname, datapoints)

    def _update_agent_location(self):
        self._agent_location_on_last_capture = self.world.player.get_transform().location

    def _generate_datapoints(self, image, depth_map, args):
        """
        Returns a list of datapoints (labels and such) that are generated this frame
        together with the main image image
        """

        datapoints = []
        bounding_boxes = []
        boxes_2d = []
        image = image.copy()
        agents_list = []

        if 'pedestrian' in CLASSES_TO_LABEL:
            pedestrians_list = self.world.world.get_actors().filter('walker.pedestrian.*')
            agents_list.extend(pedestrians_list)
        if ('car' in CLASSES_TO_LABEL) or ('vehicle' in CLASSES_TO_LABEL):
            vehicles_list = self.world.world.get_actors().filter('vehicle.*')
            agents_list.extend(vehicles_list)

        # Stores all datapoints for the current frames
        for agent in agents_list:
            image, kitti_datapoint, bounding_box = create_kitti_datapoint(agent=agent,
                                                                          camera=self.world.camera_manager.sensors['sensor.camera.rgb']['sensor'],
                                                                          cam_calibration=self.world.camera_manager.sensors['sensor.camera.rgb']['calibration'],
                                                                          image=image,
                                                                          depth_map=depth_map,
                                                                          player_transform=self.world.player.get_transform(),
                                                                          max_render_depth=args.lidar_range)
            if kitti_datapoint:
                datapoints.append(kitti_datapoint)
                bounding_boxes.append(bounding_box)
                boxes_2d.append(kitti_datapoint.bbox)
        return image, datapoints, bounding_boxes, boxes_2d

    def _preprocess_sensor_data(self, sensor_data):
        processed_data = dict()

        for sensor_key, sensor_data in sensor_data.items():
            # Assume a same preprocessing step for both blickfeld and raycasting lidars
            if 'sensor.lidar' in sensor_key:
                lidar_range = float(self.world.camera_manager.sensors[sensor_key]['sensor'].attributes['range'])

                points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))
                points = np.copy(points)

                # TODO blickfeld should be checked against 0.9.10 carla version
                if 'blickfeld' in sensor_key:
                    # Swapping x and y axis in blickfeld lidar for correct visualization
                    # points[:, [0, 1]] = points[:, [1, 0]]

                    # To make coordinate system consistent with default ray_cast lidar we turn right-hand coordinate sys
                    # to left-hand for blickfeld
                    # points[:, 0] = -points[:, 0]

                    # Looks like that the blickfeld produces values for x-axis beyond lidar range.
                    # We thus cut the points based on the lidar range.
                    points[points[:, 1] > lidar_range] = lidar_range - 0.01

                lidar_data = np.array(points[:, :2])
                lidar_data *= min(self.hud.dim) / (2.0 * lidar_range)
                lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
                lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
                lidar_data = lidar_data.astype(np.int32)
                lidar_data = np.reshape(lidar_data, (-1, 2))
                lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
                lidar_img = np.zeros(lidar_img_size)
                lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

                processed_data.update({sensor_key: {'image': lidar_img, 'points': points}})
            else:
                color_converter = cc.Depth if 'depth' in sensor_key else cc.Raw
                sensor_data.convert(color_converter)
                array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
                array = array[:, :, :3]
                if 'depth' in sensor_key:
                    # Decoding depth: https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-depth-map
                    array = array.astype(np.float32)
                    normalized_depth = np.dot(array, [65536.0, 256.0, 1.0])
                    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
                    depth = normalized_depth * 1000
                    processed_data.update({sensor_key: {'image': array, 'depth': depth}})
                else:

                    array = array[:, :, ::-1]
                    processed_data.update({sensor_key: {'image': array}})

        return processed_data

    def game_loop(self, args):
        """ Main loop for agent"""

        sensors = [val['sensor'] for val in self.world.camera_manager.sensors.values()]
        # Create a synchronous mode context.
        with CarlaSyncMode(self.world.world, *sensors, fps=args.fps) as sync_mode:
            while True:
                if self.controller.parse_events():
                    return
                self._timer.tick()
                self.clock.tick()

                # Advance the simulation and wait for the data.
                # Assume that the sensor data have the same order as in self.world.camera_manager.sensors
                simulation_data = sync_mode.tick(timeout=2.0)
                assert len(simulation_data) == len(self.world.camera_manager.sensors.keys()) + 1  # +1 for snapshot data

                world_snapshot = simulation_data[0]  # maybe be used in future

                sensors_data_dict = dict()
                for i, key in enumerate(self.world.camera_manager.sensors.keys()):
                    sensors_data_dict.update({key: simulation_data[i+1]})

                # Reset the environment if the agent is stuck or can't find any agents or
                # if we have captured enough frames in this one
                is_stuck = self._frames_since_last_capture >= args.num_empty_frames_before_reset
                is_enough_datapoints = (self._captured_frames_since_restart + 1) % args.num_recordings_before_reset == 0

                if (is_stuck or is_enough_datapoints) and args.save_data:
                    if is_stuck:
                        logging.warning("The agent is either stuck or can't find any agents!")
                    if is_enough_datapoints:
                        logging.info("Enough datapoints captured. The episode is going to restart.")
                    self._on_new_episode(args)
                    # If we dont sleep, the client will continue to render
                    self.reset_episode = True
                    continue

                if (args.autopilot == False):
                    self.agent.update_information(self.world)

                # Tick HUD
                self.world.tick(self.clock)

                processed_sensor_data = self._preprocess_sensor_data(sensors_data_dict)

                # Rendering sensor images and creating KITTI datapoints for each frame
                # TODO makes sense to have on_render only dealing with rendering not datapoint generation.
                datapoints = self._render(self.display, processed_sensor_data, args)

                # Rendering HUD
                self.world.render(self.display)
                pygame.display.flip()

                if args.save_data:

                    point_clouds = []
                    lidar_heights = []
                    lidar_cam_mats = []
                    for key in processed_sensor_data.keys():
                        if 'lidar' in key:
                            point_cloud = processed_sensor_data[key]['points']
                            lidar_height = self.world.camera_manager.sensors[key]['transform'].location.z
                            lidar_cam_mat = self.world.camera_manager.sensors[key]['lidar_cam_mat']
                            point_clouds.append(point_cloud)
                            lidar_heights.append(lidar_height)
                            lidar_cam_mats.append(lidar_cam_mat)

                    rgb_image = processed_sensor_data['sensor.camera.rgb']['image']
                    self._save_datapoints(datapoints,
                                          self.world.camera_manager.sensors['sensor.camera.rgb']['calibration'],
                                          rgb_image,
                                          point_clouds,
                                          lidar_heights, lidar_cam_mats, args)

                if (args.autopilot == False):
                    # Set new destination when target has been reached
                    if len(self.agent.get_local_planner().waypoints_queue) < self.num_min_waypoints and args.loop:
                        self.agent.reroute(self.spawn_points)
                        self.tot_target_reached += 1
                        self.world.hud.notification("The target has been reached " +
                                               str(self.tot_target_reached) + " times.", seconds=4.0)

                    elif len(self.agent.get_local_planner().waypoints_queue) == 0 and not args.loop:
                        print("Target reached, mission accomplished...")
                        break

                    speed_limit = self.world.player.get_speed_limit()
                    self.agent.get_local_planner().set_speed(speed_limit)

                    control = self.agent.run_step()
                    self.world.player.apply_control(control)

def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1248x384',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--vis_lidar',
        action='store_true',
        dest='vis_lidar',
        help='Whether or not to visualize lidar point clouds.')
    argparser.add_argument(
        '--vis_boxes3d',
        action='store_true',
        dest='vis_boxes3d',
        help='Whether or not to visualize 3D bounding boxes for agents.')
    argparser.add_argument(
        '--vis_boxes2d',
        action='store_true',
        dest='vis_boxes2d',
        help='Whether or not to visualize 2D bounding boxes for agents.')
    argparser.add_argument(
        '--vis_ry',
        action='store_true',
        dest='vis_ry',
        help='Whether or not to visualize rotation ry of agents around Y-axis.')
    argparser.add_argument(
        '--steps_between_recordings',
        default=10,
        type=int,
        help='How many frames to wait between each capture of screen, bounding boxes and lidar.')
    argparser.add_argument(
        '--lidar_data_format',
        default='bin',
        type=str,
        help='Lidar can be saved in bin to comply to kitti, or the standard .ply format')
    argparser.add_argument(
        '--distance_since_last_recording',
        default=10.,
        type=float,
        help='How many meters the car must drive before a new capture is triggered.')
    argparser.add_argument(
        '--num_recordings_before_reset',
        default=20,
        type=int,
        help='How many datapoints to record before resetting the scene.')
    argparser.add_argument(
        '--num_empty_frames_before_reset',
        default=100,
        type=int,
        help='How many frames to render before resetting the environment. For example, the agent may be stuck')
    argparser.add_argument(
        '--lidars',
        nargs='+',
        help='List of lidar types used for visualization and data collection.' +
             'Available options are \"ray_cast\" and \"blickfeld\"',
        default=['ray_cast'])
    argparser.add_argument(
        '--lidar_range',
        default=100.,
        type=float,
        help='Max render depth in meters')
    argparser.add_argument(
        '--save_data',
        action='store_true',
        dest='save_data',
        help='Whether or not to save training data')
    argparser.add_argument(
        '--phase',
        default='training',
        type=str,
        help='Phase can be one of (training|val|test)')
    argparser.add_argument(
        '--output_dir',
        default='./data/object',
        type=str,
        help='Path to output directory')
    argparser.add_argument(
        '--fps',
        default=10,
        type=int,
        help='Simulation FPS')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='Enable default autopilot. If not, a behavior agent will be used')

    carla_root = os.getenv('CARLA_ROOT')
    assert carla_root is not None, 'Please set the CARLA_ROOT variable environment!'

    # TODO this is a workaround for spawning pedestrians and vehicles. In future should be replaced by
    #  scenario runner configs
    spawn_npc_path = os.path.join(carla_root, 'PythonAPI', 'examples', 'spawn_npc.py')

    def target(**kwargs):
        process = subprocess.Popen([spawn_npc_path, '-w 200'], **kwargs)
        process.communicate()

    thread = threading.Thread(target=target, kwargs={'stdout':subprocess.PIPE, 'shell':True})
    thread.start()

    args = argparser.parse_args()
    phase_dir = os.path.join(args.output_dir, args.phase)
    default_folders = ['calib', 'image_2', 'label_2', 'velodyne', 'planes']

    assert all([lidar in ['ray_cast', 'blickfeld'] for lidar in args.lidars])

    """ DATA SAVE PATHS """
    args.phase_dir = phase_dir

    args.label_path = os.path.join(phase_dir, 'label_2/{0:06}.txt')
    args.image_path = os.path.join(phase_dir, 'image_2/{0:06}.png')


    args.width, args.height = [int(x) for x in args.res.split('x')]

    if args.save_data:

        default_groundplane_path = os.path.join(phase_dir, 'planes/{0:06}.txt')
        default_lidar_path = os.path.join(phase_dir, 'velodyne/{0:06}.bin')
        default_calibration_path = os.path.join(phase_dir, 'calib/{0:06}.txt')

        groundplane_paths = [default_groundplane_path]
        lidar_paths = [default_lidar_path]
        calibration_paths = [default_calibration_path]

        if len(args.lidars) > 1:
            logging.warning('More than one lidar type is provided.' +
                            'The first lidar type gets default names for kitti dataset.' +
                            'Others will get default names with a postfix.')

            for lidar in args.lidars[1:]:
                default_folders.append(f'planes_{lidar}')
                default_folders.append(f'velodyne_{lidar}')
                default_folders.append(f'calib_{lidar}')
                groundplane_paths.append(os.path.join(phase_dir, f'planes_{lidar}', '{0:06}.txt'))
                lidar_paths.append(os.path.join(phase_dir, f'velodyne_{lidar}', '{0:06}.bin'))
                calibration_paths.append(os.path.join(phase_dir, f'calib_{lidar}', '{0:06}.txt'))

        for folder in default_folders:
            directory = os.path.join(phase_dir, folder)
            if not os.path.exists(directory):
                os.makedirs(directory)

        args.groundplane_paths = groundplane_paths
        args.lidar_paths = lidar_paths
        args.calibration_paths = calibration_paths

    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    carla_game = CarlaGame(args)
    try:

        carla_game.game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
