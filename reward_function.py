import math
import numpy as np

def reward_function(params):

    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    steering_angle = params['steering_angle']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    progress = params['progress']
    is_offtrack = params['is_offtrack']
    is_crashed = params['is_crashed']

    MAX_SPEED = 4.0 
    DIRECTION_THRESHOLD = 10.0 
    STEERING_THRESHOLD = 15.0 

    reward = 1.0

    noise = np.random.normal(0, 0.1)

    if is_offtrack or not all_wheels_on_track or is_crashed:
        return 1e-3 + noise

    distance_penalty = (distance_from_center / (track_width / 2)) ** 2
    reward += (1 - distance_penalty) + noise

    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]) * (180 / math.pi)
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    direction_reward = max(1.0 - (direction_diff / DIRECTION_THRESHOLD), 0)
    reward += direction_reward + noise

    if direction_diff < DIRECTION_THRESHOLD:
        speed_reward = speed / MAX_SPEED
    else:
        speed_reward = (MAX_SPEED - speed) / MAX_SPEED
    reward += speed_reward + noise

    steering_penalty = abs(steering_angle) / STEERING_THRESHOLD
    reward -= min(steering_penalty, 1.0) + noise

    reward += (progress / 100.0) + noise

    return max(reward, 0)