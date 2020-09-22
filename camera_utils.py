import numpy as np
from numpy.linalg import inv
import pygame

# TODO Get width and height from the args
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
BB_COLOR = (248, 64, 24)


def draw_2d_bounding_boxes(display, boxes_2d):
    for bbox_2d in boxes_2d:
        (min_x, min_y, max_x, max_y) = bbox_2d
        bbox_2d_width = (max_x - min_x)
        bbox_2d_height = (max_y - min_y)
        rect = pygame.Rect((int(min_x), int(min_y)), (int(bbox_2d_width), int(bbox_2d_height)))
        pygame.draw.rect(display, (255, 0, 255), rect, 1)


def calc_projected_2d_bbox(vertices_pos2d):
    """ Takes in all vertices in pixel projection and calculates min and max of all x and y coordinates.
        Returns left top, right bottom pixel coordinates for the 2d bounding box as a list of four values.
        Note that vertices_pos2d contains a list of (y_pos2d, x_pos2d) tuples, or None
    """
    x_coords = vertices_pos2d[:, 0]
    y_coords = vertices_pos2d[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    return [min_x, min_y, max_x, max_y]


def proj_to_camera(pos_vector, extrinsic_mat):
    # transform the points to camera
    transformed_3d_pos = np.dot(inv(extrinsic_mat), pos_vector)
    return transformed_3d_pos


def point_in_canvas(pos):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < WINDOW_HEIGHT) and (pos[1] >= 0) and (pos[1] < WINDOW_WIDTH):
        return True
    return False


# TODO Use Carla API or pygame for drawing a rectangle
def draw_rect(array, pos, size, color=(255, 0, 0)):
    """Draws a rect"""
    point_0 = (pos[0]-size/2, pos[1]-size/2)
    point_1 = (pos[0]+size/2, pos[1]+size/2)
    if point_in_canvas(point_0) and point_in_canvas(point_1):
        for i in range(size):
            for j in range(size):
                array[int(point_0[0]+i), int(point_0[1]+j)] = color


def point_is_occluded(point, vertex_depth, depth_map):
    """ Checks whether or not the four pixels directly around the given point has less depth than the given vertex depth
        If True, this means that the point is occluded.
    """
    y, x = map(int, point)
    from itertools import product
    neigbours = product((1, -1), repeat=2)
    is_occluded = []
    for dy, dx in neigbours:
        if point_in_canvas((dy+y, dx+x)):
            # If the depth map says the pixel is closer to the camera than the actual vertex
            if depth_map[y+dy, x+dx] < vertex_depth:
                is_occluded.append(True)
            else:
                is_occluded.append(False)
    # Only say point is occluded if all four neighbours are closer to camera than vertex
    return all(is_occluded)
