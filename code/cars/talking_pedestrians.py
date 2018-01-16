from enum import Enum
from matplotlib import animation
from util import *

import city
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


class PersonIntention(Enum):
    CROSS_ROAD = 0
    TALK_WITH_FRIEND = 1
    IDLE = 2


class CarIntention(Enum):
    GO = 0
    GO_CAREFUL = 1
    WAIT_FOR_PERSON = 2
    IDLE = 3


slow_walking_speed = m_per_sec(2)
walking_speed = m_per_sec(5)
running_speed = m_per_sec(8)
turn_speed = 90

average_car_speed = m_per_sec(50)
fast_car_speed = m_per_sec(70)
slow_car_speed = m_per_sec(20)

world_width = 100
world_length = 100
car_width = 1.7
car_length = 4.5
person_radius = 0.7
road_width = 8
pavement_width = person_radius * 4

car_1_start = np.array([world_width / 2 + road_width / 4, car_length / 2])
car_1_wait_for_person_waypoint = np.array([
    world_width / 2 + road_width / 4,
    world_length / 2 - 4 * person_radius - car_length / 2
])
car_1_go_waypoint = np.array([
    world_width / 2 + road_width / 4,
    world_length - car_length / 2
])
zebra_bottom_left_corner = np.array(
    [world_width / 2 - road_width / 2, world_length / 2 - 3 * person_radius]
)
zebra_top_right_corner = np.array(
    [world_width / 2 + road_width / 2, world_length / 2 + 3 * person_radius]
)
person_1_start = np.array([
    world_width / 2 - road_width / 2 - pavement_width / 2,
    world_length / 2 + 2 * person_radius
])
person_1_end = np.array([
    world_width / 2 + road_width / 2 + pavement_width / 2,
    world_length / 2 + 2 * person_radius
])
person_2_start = np.array([
    world_width / 2 - road_width / 2 - pavement_width / 2,
    world_length / 2 - 2 * person_radius
])
person_2_end = np.array([
    world_width / 2 + road_width / 2 + pavement_width / 2,
    world_length / 2 - 2 * person_radius
])

position_sensor_std = 0.01
angle_sensor_std = 40
probability_of_switching_intention = 0.01
zebra_tolerance = person_radius
max_expected_risk_tolerance = 0.1

num_timesteps = 50
dt = 0.25
memory_time = 3
prediction_time = 3
num_particles = 1000
time_stop_talking = 1


def person_1_policy(current_person_1, intention):
    if intention == PersonIntention.IDLE:
        return current_person_1.copy()
    else:
        if intention == PersonIntention.CROSS_ROAD:
            waypoint = person_1_end
            speed = walking_speed
            angle_target = 0
        elif intention == PersonIntention.TALK_WITH_FRIEND:
            waypoint = person_1_start
            speed = 0
            angle_target = -90

        new_person_1 = current_person_1.copy()
        if current_person_1.angle != angle_target:
            if np.linalg.norm((angle_target - current_person_1.angle) % 360) <= turn_speed * dt:
                d_angle = angle_target - current_person_1.angle
            else:
                d_angle = np.sign((angle_target - current_person_1.angle) % 360) * turn_speed * dt

            new_person_1.angle = (current_person_1.angle + d_angle) % 360
        else:
            if np.linalg.norm(waypoint - current_person_1.position) <= speed * dt:
                d_position = waypoint - current_person_1.position
            else:
                d_position = normalize_vector(waypoint - current_person_1.position) * speed * dt

            new_person_1.position = current_person_1.position + d_position

        return new_person_1


def intention_transition(current_intention):
    prob = np.zeros([len(PersonIntention)])
    for i, pi in enumerate(PersonIntention):
        if pi == current_intention:
            prob[i] = 1 - probability_of_switching_intention
        else:
            prob[i] = probability_of_switching_intention / (len(PersonIntention) - 1)

    return np.random.choice(PersonIntention, p=prob)


def get_person_1_posterior(person_1_history, initial_person, num_particles, num_samples=None):
    if num_samples is None:
        num_samples = num_particles

    num_timesteps = len(person_1_history)
    intention = np.zeros([num_particles, num_timesteps], dtype=object)
    person = np.zeros([num_particles, num_timesteps], dtype=object)
    log_weights = np.zeros(num_particles)

    for particle_idx in range(num_particles):
        intention[particle_idx, 0] = np.random.choice(PersonIntention)
        person[particle_idx, 0] = initial_person

        log_weights[particle_idx] += scipy.stats.multivariate_normal.logpdf(
            person_1_history[0].position,
            mean=person[particle_idx, 0].position,
            cov=np.eye(2) * position_sensor_std**2
        ) + scipy.stats.norm.logpdf(
            person_1_history[0].angle,
            loc=person[particle_idx, 0].angle,
            scale=angle_sensor_std
        )

        for time_idx in range(1, num_timesteps):
            intention[particle_idx, time_idx] = intention_transition(intention[particle_idx, time_idx - 1])
            person[particle_idx, time_idx] = person_1_policy(
                person[particle_idx, time_idx - 1], intention[particle_idx, time_idx - 1]
            )

            log_weights[particle_idx] += scipy.stats.multivariate_normal.logpdf(
                person_1_history[time_idx].position,
                mean=person[particle_idx, time_idx].position,
                cov=np.eye(2) * position_sensor_std**2
            ) + scipy.stats.norm.logpdf(
                person_1_history[time_idx].angle,
                loc=person[particle_idx, time_idx].angle,
                scale=angle_sensor_std
            )

    resampled_idx = np.random.choice(num_particles, size=num_samples, p=np.exp(lognormexp(log_weights)))

    return person[resampled_idx, :], intention[resampled_idx, :]


def get_person_1_prediction(person_1_history, initial_person, num_prediction_timesteps, num_particles, num_samples=None):
    person, intention = get_person_1_posterior(person_1_history, initial_person, num_particles, num_samples)
    num_samples = len(intention)
    num_timesteps = len(person_1_history)

    intention_prediction = []
    person_prediction = []
    for sample_idx in range(num_samples):
        intention_temp = []
        person_temp = []

        current_intention = intention[sample_idx][-1]
        current_person = person[sample_idx][-1]
        for prediction_time_idx in range(num_prediction_timesteps):
            current_person = person_1_policy(current_person, current_intention)
            person_temp.append(current_person)

            current_intention = intention_transition(current_intention)
            intention_temp.append(current_intention)

        person_prediction.append(person_temp)
        intention_prediction.append(intention_temp)

    return person_prediction, intention_prediction


def near_zebra(position):
    return (
        (position[0] >= zebra_bottom_left_corner[0] - zebra_tolerance) and
        (position[0] <= zebra_top_right_corner[0] + zebra_tolerance) and
        (position[1] >= zebra_bottom_left_corner[1] - zebra_tolerance) and
        (position[1] <= zebra_top_right_corner[1] + zebra_tolerance)
    )


def get_risk(person_position, car_position):
    return int(near_zebra(person_position) and near_zebra(car_position))


def get_expected_risk(person_1_samples, car_1_position):
    return np.mean([
        get_risk(person_1.position, car_1_position)
        for person_1 in person_1_samples
    ])


def get_max_expected_risk(person_1_prediction, car_1_prediction):
    num_prediction_timesteps = len(person_1_prediction[0])
    expected_risk = []
    for prediction_time_idx in range(num_prediction_timesteps):
        expected_risk.append(
            get_expected_risk(
                [
                    person_1_prediction_sample[prediction_time_idx]
                    for person_1_prediction_sample in person_1_prediction
                ],
                car_1_prediction[prediction_time_idx].position
            )
        )

    return np.max(expected_risk)


def get_car_1_go_prediction(current_car_1, num_prediction_timesteps):
    temp_world_history = [city.World([current_car_1.copy()], world_width, world_length)]
    car_1_go_prediction = []
    for prediction_time_idx in range(num_prediction_timesteps):
        car_1_go_prediction.append(
            car_1_policy(temp_world_history, CarIntention.GO, None, None, None)
        )
        temp_world_history[-1].world_objects_dict['car_1'] = car_1_go_prediction[-1]

    return car_1_go_prediction


def car_1_policy(observed_world_history, intention, memory_time, prediction_time, num_particles, debug=False):
    current_car_1 = observed_world_history[-1].world_objects_dict['car_1']
    if intention == CarIntention.IDLE:
        return current_car_1.copy()
    else:
        if intention == CarIntention.WAIT_FOR_PERSON:
            speed = slow_car_speed
            waypoint = car_1_wait_for_person_waypoint
            if np.linalg.norm(waypoint - current_car_1.position) <= speed * dt:
                d_position = waypoint - current_car_1.position
            else:
                d_position = normalize_vector(waypoint - current_car_1.position) * speed * dt

            next_car_1 = current_car_1.copy()
            next_car_1.position = current_car_1.position + d_position

            return next_car_1
        elif intention == CarIntention.GO:
            speed = average_car_speed
            waypoint = car_1_go_waypoint
            if np.linalg.norm(waypoint - current_car_1.position) <= speed * dt:
                d_position = waypoint - current_car_1.position
            else:
                d_position = normalize_vector(waypoint - current_car_1.position) * speed * dt

            next_car_1 = current_car_1.copy()
            next_car_1.position = current_car_1.position + d_position

            return next_car_1
        elif intention == CarIntention.GO_CAREFUL:
            num_memory_timesteps = max(min(int(np.ceil(memory_time / dt)), len(observed_world_history)), 1)
            num_prediction_timesteps = int(np.ceil(prediction_time / dt))

            remembered_person_1_history = [
                world.world_objects_dict['person_1']
                for world in observed_world_history[-num_memory_timesteps:]
            ]
            person_1_prediction, _ = get_person_1_prediction(
                remembered_person_1_history, remembered_person_1_history[0], num_prediction_timesteps, num_particles
            )
            car_1_prediction = get_car_1_go_prediction(current_car_1, num_prediction_timesteps)
            max_expected_risk = get_max_expected_risk(person_1_prediction, car_1_prediction)

            if max_expected_risk > max_expected_risk_tolerance:
                if debug:
                    return car_1_policy(observed_world_history, CarIntention.WAIT_FOR_PERSON, None, None, None), person_1_prediction, car_1_prediction, max_expected_risk
                else:
                    return car_1_policy(observed_world_history, CarIntention.WAIT_FOR_PERSON, None, None, None)
            else:
                if debug:
                    return car_1_policy(observed_world_history, CarIntention.GO, None, None, None), person_1_prediction, car_1_prediction, max_expected_risk
                else:
                    return car_1_policy(observed_world_history, CarIntention.GO, None, None, None)


def main():
    # inanimate objects
    road = city.Road('road_1', road_width, np.array([world_width / 2, 0]), np.array([world_width / 2, world_length]), color='lightgray')

    # agents
    car_1 = city.Car('car_1', car_1_start, car_width, car_length, angle=90, color='black')
    person_1 = city.Person('person_1', person_1_start, person_radius, angle=-90, color='black')
    person_2 = city.Person('person_2', person_2_start, person_radius, angle=90, color='black')

    # world
    true_worlds = [None] * num_timesteps
    true_worlds[0] = city.World([road, car_1, person_1, person_2], world_width, world_length)

    for time_idx in range(1, num_timesteps):
        print('time = {:.2f}'.format(time_idx * dt))
        person_1_intention = PersonIntention.TALK_WITH_FRIEND if time_idx * dt < time_stop_talking else PersonIntention.CROSS_ROAD
        # person_1_intention = PersonIntention.TALK_WITH_FRIEND if time_idx * dt < time_stop_talking else PersonIntention.TALK_WITH_FRIEND
        new_person_1 = person_1_policy(true_worlds[time_idx - 1].world_objects_dict['person_1'], person_1_intention)
        new_car_1 = car_1_policy(true_worlds[:time_idx], CarIntention.GO_CAREFUL, memory_time, prediction_time, num_particles)

        true_worlds[time_idx] = true_worlds[time_idx - 1].copy()
        true_worlds[time_idx].world_objects_dict['person_1'] = new_person_1
        true_worlds[time_idx].world_objects_dict['car_1'] = new_car_1


    # save
    dpi = 200
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    fig.set_dpi(dpi)

    def animate(frame):
        ax.cla()
        return true_worlds[frame].draw(ax).artists

    speedup_factor = 1
    anim = animation.FuncAnimation(fig, animate, frames=num_timesteps, interval=dt * 1000 / speedup_factor, blit=True)
    filename = 'cross.mp4'
    # filename = 'talk.mp4'
    anim.save(filename, dpi=dpi)
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
