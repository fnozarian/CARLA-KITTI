import numpy as np
from numpy.linalg import inv
import pygame

# TODO Get width and height from the args
WINDOW_WIDTH = 1248
WINDOW_HEIGHT = 384
BB_COLOR = (248, 64, 24)


def draw_3d_bounding_boxes(display, bounding_boxes):
    """
    Draws bounding boxes on pygame display.
    """

    bb_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    bb_surface.set_colorkey((0, 0, 0))
    for bbox in bounding_boxes:
        points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
        # draw lines
        # base
        pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
        pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
        pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
        pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
        pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
        # top
        pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
        pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
        pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
        pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
        # base-top
        pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
        pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
        pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
        pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
    display.blit(bb_surface, (0, 0))

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


def crop_boxes_in_canvas(cam_bboxes):
    neg_x_inds = np.where(cam_bboxes[:, 0] < 0)[0]
    out_x_inds = np.where(cam_bboxes[:, 0] > WINDOW_WIDTH)[0]
    neg_y_inds = np.where(cam_bboxes[:, 1] < 0)[0]
    out_y_inds = np.where(cam_bboxes[:, 1] > WINDOW_HEIGHT)[0]
    cam_bboxes[neg_x_inds, 0] = 0
    cam_bboxes[out_x_inds, 0] = WINDOW_WIDTH
    cam_bboxes[neg_y_inds, 1] = 0
    cam_bboxes[out_y_inds, 1] = WINDOW_HEIGHT

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
