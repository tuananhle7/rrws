from enum import Enum
from matplotlib import animation
from util import *

import city
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


slow_walking_speed = m_per_sec(2)
walking_speed = m_per_sec(5)
running_speed = m_per_sec(12)
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
    world_length / 2 - 6 * person_radius - car_length / 2
])
car_1_go_waypoint = np.array([
    world_width / 2 + road_width / 4,
    world_length - car_length / 2
])
car_2_start = np.array([world_width / 2 - road_width / 4, world_length - car_length / 2])
car_2_wait_for_person_waypoint = np.array([
    world_width / 2 - road_width / 4,
    world_length / 2 + 6 * person_radius + car_length / 2
])
car_2_go_waypoint = np.array([
    world_width / 2 - road_width / 4,
    car_length / 2
])

building_width = 10
building_length = 10
building_points = np.array([
    [world_width / 2 - road_width / 2 - pavement_width - building_width, world_length / 2 - pavement_width / 2 - building_length],
    [world_width / 2 - road_width / 2 - pavement_width, world_length / 2 - pavement_width / 2 - building_length],
    [world_width / 2 - road_width / 2 - pavement_width, world_length / 2 - pavement_width / 2],
    [world_width / 2 - road_width / 2 - pavement_width - building_width, world_length / 2 - pavement_width / 2],
])
zebra_bottom_left_corner = np.array([world_width / 2 - road_width / 2, world_length / 2 - 3 * person_radius])
zebra_top_right_corner = np.array([world_width / 2 + road_width / 2, world_length / 2 + 3 * person_radius])
person_1_start = np.array([
    world_width / 2 - road_width / 2 - pavement_width - building_width / 2,
    world_length / 2
])
person_1_end = np.array([
    world_width / 2 + road_width / 2 + pavement_width / 2,
    world_length / 2
])

position_sensor_std = 0.01
angle_sensor_std = 40
probability_of_switching_intention = 0.01
probability_of_car_switching_intention = 0.01
zebra_tolerance = person_radius
max_expected_risk_tolerance = 0.1

num_timesteps = 50
dt = 0.25
memory_time = 3
prediction_time = 3
num_particles = 10
car_2_memory_time = 3
car_2_prediction_time = 3
car_2_num_particles = 10
car_intention_prior = [0.099, 0.9, 0.0005, 0.0005]

time_idle = 0.75


class PersonIntention(Enum):
    CROSS_ROAD = 0
    IDLE = 1


class CarIntention(Enum):
    GO = 0
    GO_CAREFUL = 1
    WAIT_FOR_PERSON = 2
    IDLE = 3


def person_1_policy(current_person_1, intention):
    if intention == PersonIntention.IDLE:
        return current_person_1.copy()
    elif intention == PersonIntention.CROSS_ROAD:
        waypoint = person_1_end
        speed = running_speed

        new_person_1 = current_person_1.copy()
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


def get_person_1_posterior_for_car_2(person_1_history, initial_person, num_particles, num_samples=None):
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


def get_person_1_prediction_for_car_2(person_1_history, initial_person, num_prediction_timesteps, num_particles, num_samples=None):
    person, intention = get_person_1_posterior_for_car_2(person_1_history, initial_person, num_particles, num_samples)
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


def get_expected_risk(person_1_samples, car_2_position):
    return np.mean([
        0 if person_1 is None else get_risk(person_1.position, car_2_position)
        for person_1 in person_1_samples
    ])


def get_max_expected_risk(person_1_prediction, car_2_prediction):
    num_prediction_timesteps = len(person_1_prediction[0])
    expected_risk = []
    for prediction_time_idx in range(num_prediction_timesteps):
        expected_risk.append(
            get_expected_risk(
                [
                    person_1_prediction_sample[prediction_time_idx]
                    for person_1_prediction_sample in person_1_prediction
                ],
                car_2_prediction[prediction_time_idx].position
            )
        )

    return np.max(expected_risk)


def get_car_2_go_prediction(current_car_2, num_prediction_timesteps):
    temp_world_history = [city.World([current_car_2.copy()], world_width, world_length)]
    car_2_go_prediction = []
    for prediction_time_idx in range(num_prediction_timesteps):
        car_2_go_prediction.append(
            car_2_policy(temp_world_history, CarIntention.GO, None, None, None)
        )
        temp_world_history[-1].world_objects_dict['car_2'] = car_2_go_prediction[-1]

    return car_2_go_prediction


def car_2_policy(observed_world_history, intention, memory_time, prediction_time, num_particles, debug=False):
    current_car_2 = observed_world_history[-1].world_objects_dict['car_2']
    if intention == CarIntention.IDLE:
        return current_car_2.copy()
    else:
        if intention == CarIntention.WAIT_FOR_PERSON:
            speed = slow_car_speed
            waypoint = car_2_wait_for_person_waypoint
            if np.linalg.norm(waypoint - current_car_2.position) <= speed * dt:
                d_position = waypoint - current_car_2.position
            else:
                d_position = normalize_vector(waypoint - current_car_2.position) * speed * dt

            next_car_2 = current_car_2.copy()
            next_car_2.position = current_car_2.position + d_position

            return next_car_2
        elif intention == CarIntention.GO:
            speed = average_car_speed
            waypoint = car_2_go_waypoint
            if np.linalg.norm(waypoint - current_car_2.position) <= speed * dt:
                d_position = waypoint - current_car_2.position
            else:
                d_position = normalize_vector(waypoint - current_car_2.position) * speed * dt

            next_car_2 = current_car_2.copy()
            next_car_2.position = current_car_2.position + d_position

            return next_car_2
        elif intention == CarIntention.GO_CAREFUL:
            num_memory_timesteps = max(min(int(np.ceil(memory_time / dt)), len(observed_world_history)), 1)
            num_prediction_timesteps = int(np.ceil(prediction_time / dt))

            remembered_person_1_history = [
                world.world_objects_dict.get('person_1')
                for world in observed_world_history[-num_memory_timesteps:]
            ]

            if None in remembered_person_1_history:
                if debug:
                    return car_2_policy(observed_world_history, CarIntention.GO, None, None, None), person_1_prediction, car_2_prediction, max_expected_risk
                else:
                    return car_2_policy(observed_world_history, CarIntention.GO, None, None, None)
            else:
                person_1_prediction, _ = get_person_1_prediction_for_car_2(
                    remembered_person_1_history, remembered_person_1_history[0], num_prediction_timesteps, num_particles
                )
                car_2_prediction = get_car_2_go_prediction(current_car_2, num_prediction_timesteps)
                max_expected_risk = get_max_expected_risk(person_1_prediction, car_2_prediction)

                if max_expected_risk > max_expected_risk_tolerance:
                    if debug:
                        return car_2_policy(observed_world_history, CarIntention.WAIT_FOR_PERSON, None, None, None), person_1_prediction, car_2_prediction, max_expected_risk
                    else:
                        return car_2_policy(observed_world_history, CarIntention.WAIT_FOR_PERSON, None, None, None)
                else:
                    if debug:
                        return car_2_policy(observed_world_history, CarIntention.GO, None, None, None), person_1_prediction, car_2_prediction, max_expected_risk
                    else:
                        return car_2_policy(observed_world_history, CarIntention.GO, None, None, None)


def get_person_1_posterior_for_car_1(car_2_history, initial_car_2, initial_person_1,
                                     car_2_memory_time, car_2_prediction_time, car_2_num_particles,
                                     num_particles, num_samples=None):
    if num_samples is None:
        num_samples = num_particles

    num_timesteps = len(car_2_history)
    presence_of_person_1 = np.zeros([num_particles])
    person_1_intention = np.zeros([num_particles, num_timesteps], dtype=object)
    person_1 = np.zeros([num_particles, num_timesteps], dtype=object)
    car_2_intention = np.zeros([num_particles, num_timesteps], dtype=object)
    car_2 = np.zeros([num_particles, num_timesteps], dtype=object)
    log_weights = np.zeros(num_particles)

    for particle_idx in range(num_particles):
        presence_of_person_1[particle_idx] = np.random.choice(2)
        if presence_of_person_1[particle_idx] == 1:
            person_1_intention[particle_idx, 0] = np.random.choice(PersonIntention)
            person_1[particle_idx, 0] = initial_person_1
        else:
            person_1_intention[particle_idx, 0] = None
            person_1[particle_idx, 0] = None

        car_2_intention[particle_idx, 0] = np.random.choice(CarIntention, p=car_intention_prior)
        car_2[particle_idx, 0] = initial_car_2
        log_weights[particle_idx] += scipy.stats.multivariate_normal.logpdf(
            car_2_history[0].position,
            mean=car_2[particle_idx, 0].position,
            cov=np.eye(2) * position_sensor_std**2
        )

        for time_idx in range(1, num_timesteps):
            if presence_of_person_1[particle_idx] == 1:
                person_1_intention[particle_idx, time_idx] = intention_transition(person_1_intention[particle_idx, time_idx - 1])
                person_1[particle_idx, time_idx] = person_1_policy(
                    person_1[particle_idx, time_idx - 1], person_1_intention[particle_idx, time_idx - 1]
                )
                car_2_intention[particle_idx, time_idx] = car_intention_transition(
                    car_2_intention[particle_idx, time_idx - 1]
                )
                observed_world_history_for_car_2 = [
                    city.World([p, c], world_width, world_length)
                    for (p, c) in zip(person_1[particle_idx, :time_idx], car_2_history[:time_idx])
                ]
                car_2[particle_idx, time_idx] = car_2_policy(
                    observed_world_history_for_car_2, car_2_intention[particle_idx, time_idx - 1],
                    car_2_memory_time, car_2_prediction_time, car_2_num_particles
                )
            else:
                person_1_intention[particle_idx, 0] = None
                person_1[particle_idx, 0] = None
                car_2_intention[particle_idx, time_idx] = car_intention_transition(
                    car_2_intention[particle_idx, time_idx - 1]
                )
                observed_world_history_for_car_2 = [
                    city.World([c], world_width, world_length)
                    for c in car_2_history[:time_idx]
                ]
                car_2[particle_idx, time_idx] = car_2_policy(
                    observed_world_history_for_car_2, car_2_intention[particle_idx, time_idx - 1],
                    car_2_memory_time, car_2_prediction_time, car_2_num_particles
                )

            log_weights[particle_idx] += scipy.stats.multivariate_normal.logpdf(
                car_2_history[time_idx].position,
                mean=car_2[particle_idx, time_idx].position,
                cov=np.eye(2) * position_sensor_std**2
            )

    resampled_idx = np.random.choice(num_particles, size=num_samples, p=np.exp(lognormexp(log_weights)))

    return presence_of_person_1[resampled_idx], person_1_intention[resampled_idx, :], person_1[resampled_idx, :], car_2_intention[resampled_idx, :], car_2[resampled_idx, :]


def get_person_1_prediction_for_car_1(car_2_history, initial_car_2, initial_person_1, car_2_memory_time, car_2_prediction_time, car_2_num_particles, num_prediction_timesteps, num_particles, num_samples=None):
    presence_of_person_1, person_1_intention, person_1, car_2_intention, car_2 = get_person_1_posterior_for_car_1(car_2_history, initial_car_2, initial_person_1, car_2_memory_time, car_2_prediction_time, car_2_num_particles, num_particles, num_samples)
    num_samples = len(person_1_intention)
    num_timesteps = len(car_2_history)

    person_1_prediction = []
    person_1_intention_prediction = []
    for sample_idx in range(num_samples):
        person_1_temp = []
        person_1_intention_temp = []

        current_person_1 = person_1[sample_idx][-1]
        current_person_1_intention = person_1_intention[sample_idx][-1]
        for prediction_time_idx in range(num_prediction_timesteps):
            if presence_of_person_1[sample_idx] == 1:
                current_person_1 = person_1_policy(current_person_1, current_person_1_intention)
                current_person_1_intention = intention_transition(current_person_1_intention)
            else:
                current_person_1 = None
                current_person_1_intention = None

            person_1_temp.append(current_person_1)
            person_1_intention_temp.append(current_person_1_intention)

        person_1_prediction.append(person_1_temp)
        person_1_intention_prediction.append(person_1_intention_temp)

    return person_1_prediction, person_1_intention_prediction


def car_intention_transition(current_intention):
    prob = np.zeros([len(CarIntention)])
    for i, pi in enumerate(CarIntention):
        if pi == current_intention:
            prob[i] = 1 - probability_of_car_switching_intention
        else:
            prob[i] = probability_of_car_switching_intention / (len(CarIntention) - 1)

    return np.random.choice(CarIntention, p=prob)


def get_car_1_go_prediction(current_car_1, num_prediction_timesteps):
    temp_world_history = [city.World([current_car_1.copy()], world_width, world_length)]
    car_1_go_prediction = []
    for prediction_time_idx in range(num_prediction_timesteps):
        car_1_go_prediction.append(
            car_1_policy(
                temp_world_history, None, None, None,
                CarIntention.GO, None, None, None
            )
        )
        temp_world_history[-1].world_objects_dict['car_1'] = car_1_go_prediction[-1]

    return car_1_go_prediction


def get_person_1_heuristic_initial_position(num_elapsed_timesteps):
    elapsed_time = num_elapsed_timesteps * dt
    velocity = normalize_vector(person_1_end - person_1_start) * running_speed
    person_1_position = person_1_start + max(elapsed_time - time_idle, 0) * velocity
    person_1 = city.Person('person_1', person_1_position, person_radius, 0, color='black')
    return person_1


def car_1_policy(observed_world_history, car_2_memory_time, car_2_prediction_time, car_2_num_particles,
                 intention, memory_time, prediction_time, num_particles, debug=False):

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

            remembered_car_2_history = [
                world.world_objects_dict['car_2']
                for world in observed_world_history[-num_memory_timesteps:]
            ]

            initial_person_1 = get_person_1_heuristic_initial_position(
                len(observed_world_history) - num_memory_timesteps
            )
            person_1_prediction, _ = get_person_1_prediction_for_car_1(
                remembered_car_2_history, remembered_car_2_history[0], initial_person_1,
                car_2_memory_time, car_2_prediction_time, car_2_num_particles,
                num_prediction_timesteps, num_particles
            )

            car_1_prediction = get_car_1_go_prediction(current_car_1, num_prediction_timesteps)
            max_expected_risk = get_max_expected_risk(person_1_prediction, car_1_prediction)

            print('max_expected_risk = {}'.format(max_expected_risk))
            if max_expected_risk > max_expected_risk_tolerance:
                if debug:
                    return car_1_policy(
                        observed_world_history, None, None, None,
                        CarIntention.WAIT_FOR_PERSON, None, None, None
                    ), person_1_prediction, car_1_prediction, max_expected_risk
                else:
                    return car_1_policy(
                        observed_world_history, None, None, None,
                        CarIntention.WAIT_FOR_PERSON, None, None, None
                    )
            else:
                if debug:
                    return car_1_policy(
                        observed_world_history, None, None, None,
                        CarIntention.GO, None, None, None
                    ), person_1_prediction, car_1_prediction, max_expected_risk
                else:
                    return car_1_policy(
                        observed_world_history, None, None, None,
                        CarIntention.GO, None, None, None
                    )


def main():
    # inanimate objects
    road = city.Road('road_1', road_width, np.array([world_width / 2, 0]), np.array([world_width / 2, world_length]), color='lightgray')
    building_1 = city.Building('building_1', building_points, color='gray')

    # agents
    car_1 = city.Car('car_1', car_1_start, car_width, car_length, angle=90, color='black')
    car_2 = city.Car('car_2', car_2_start, car_width, car_length, angle=-90, color='gray')
    person_1 = city.Person('person_1', person_1_start, person_radius, angle=0, color='black')

    # world
    true_worlds = [None] * num_timesteps
    true_worlds[0] = city.World([road, building_1, car_1, car_2, person_1], world_width, world_length)

    for time_idx in range(1, num_timesteps):
        person_1_intention = PersonIntention.CROSS_ROAD if time_idx * dt > time_idle else PersonIntention.IDLE
        print('time = {}'.format(time_idx * dt))
        print('person_1_intention = {}'.format(person_1_intention))

        new_person_1 = person_1_policy(true_worlds[time_idx - 1].world_objects_dict['person_1'], person_1_intention)
        new_car_1 = car_1_policy(
            true_worlds[:time_idx], car_2_memory_time, car_2_prediction_time, car_2_num_particles,
            CarIntention.GO_CAREFUL, memory_time, prediction_time, num_particles
        )
        new_car_2 = car_2_policy(true_worlds[:time_idx], CarIntention.GO_CAREFUL, memory_time, prediction_time, num_particles)
        heuristic_initial_position = get_person_1_heuristic_initial_position(time_idx).position
    #     print('heuristic_initial_position = {}'.format(heuristic_initial_position))
    #     print('position = {}'.format(new_person_1.position))

        true_worlds[time_idx] = true_worlds[time_idx - 1].copy()
        true_worlds[time_idx].world_objects_dict['person_1'] = new_person_1
        true_worlds[time_idx].world_objects_dict['car_1'] = new_car_1
        true_worlds[time_idx].world_objects_dict['car_2'] = new_car_2

    # save
    save_dpi = 200
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    fig.set_dpi(save_dpi)

    def animate(frame):
        ax.cla()
        return true_worlds[frame].draw(ax).artists

    speedup_factor = 1
    anim = animation.FuncAnimation(fig, animate, frames=num_timesteps, interval=dt * 1000 / speedup_factor, blit=True)
    filename = 'two_cars.mp4'
    anim.save(filename, dpi=save_dpi)
    print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()
