import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np


class WorldObject:
    name = None

    def draw(self, ax, *args, **kwargs):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class World:
    def __init__(self, world_objects, width, length):
        self.world_objects_dict = {}
        for world_object in world_objects:
            self.world_objects_dict[world_object.name] = world_object
        self.width = width
        self.length = length

    def draw(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(5, 5)

        for name, world_object in self.world_objects_dict.items():
            if isinstance(world_object, Inanimate):
                ax = world_object.draw(ax)

        for name, world_object in self.world_objects_dict.items():
            if isinstance(world_object, Agent):
                ax = world_object.draw(ax)

        for name, world_object in self.world_objects_dict.items():
            if (not isinstance(world_object, Agent)) and (not isinstance(world_object, Inanimate)):
                ax = world_object.draw(ax)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        return ax

    def copy(self):
        return World([world_object.copy() for name, world_object in self.world_objects_dict.items()], self.width, self.length)

    def __repr__(self):
        return 'World(world_objects = {}, width = {}, length = {})'.format(
            [world_object for name, world_object in self.world_objects_dict.items()],
            self.width,
            self.length
        )

    __str__ = __repr__


class Agent(WorldObject):
    pass


class Person(Agent):
    def __init__(self, name, position, radius, angle, color):
        self.name = name
        self.position = position
        self.radius = radius
        self.angle = angle
        self.color = color

    def draw(self, ax):
        circle = plt.Circle(self.position, radius=self.radius, color=self.color)
        ax.add_patch(circle)

        angle_radians = np.radians(self.angle)
        triangle_position = (self.position[0] + 2 * self.radius * np.cos(angle_radians), self.position[1] + 2 * self.radius * np.sin(angle_radians))
        triangle = matplotlib.patches.RegularPolygon(triangle_position, 3, self.radius, orientation=angle_radians + np.pi / 6, color=self.color, fill=None)
        ax.add_patch(triangle)

        return ax

    def copy(self):
        return Person(self.name, self.position, self.radius, self.angle, self.color)

    def __repr__(self):
        return 'Person(name = {}, position = {}, radius = {}, angle = {}, color = {})'.format(
            self.name,
            self.position,
            self.radius,
            self.angle,
            self.color
        )

    __str__ = __repr__


class Car(Agent):
    def __init__(self, name, position, width, length, angle, color):
        self.name = name
        self.position = position
        self.width = width
        self.length = length
        self.angle = angle
        self.color = color

    def draw(self, ax):
        rectangle_transform = matplotlib.transforms.Affine2D().rotate_deg_around(self.position[0], self.position[1], self.angle)
        rectangle = plt.Rectangle((self.position[0] - self.length / 2, self.position[1] - self.width / 2), self.length, self.width, color=self.color, transform=rectangle_transform + ax.transData)
        ax.add_patch(rectangle)

        radius = self.width / 2
        triangle_transform = matplotlib.transforms.Affine2D().translate(self.length / 2 + radius, 0).rotate_deg_around(self.position[0], self.position[1], self.angle)
        triangle = matplotlib.patches.RegularPolygon(self.position, 3, radius, orientation=np.pi / 6, color='black', fill=None, transform=triangle_transform + ax.transData)
        ax.add_patch(triangle)

        return ax

    def copy(self):
        return Car(self.name, self.position, self.width, self.length, self.angle, self.color)

    def __repr__(self):
        return 'Car(name = {}, position = {}, width = {}, length = {}, angle = {}, color = {})'.format(
            self.name,
            self.position,
            self.width,
            self.length,
            self.angle,
            self.color
        )

    __str__ = __repr__


class Inanimate(WorldObject):
    pass


class Road(Inanimate):
    def __init__(self, name, width, start, end, color):
        self.name = name
        self.width = width
        self.start = start
        self.end = end
        self.length = np.linalg.norm(self.end - self.start)
        self.color = color

    def draw(self, ax):
        angle = np.arctan2(self.end[1] - self.start[1], self.end[0] - self.start[0])
        transform = matplotlib.transforms.Affine2D().rotate_around(self.start[0], self.start[1], angle) + ax.transData
        rectangle = plt.Rectangle((self.start[0], self.start[1] - self.width / 2), self.length, self.width, transform=transform, color=self.color)
        ax.add_patch(rectangle)


        return ax

    def copy(self):
        return Road(self.name, self.width, self.start, self.end, self.color)


class Building(Inanimate):
    def __init__(self, name, points, color):
        self.name = name
        self.points = points
        self.color = color

    def draw(self, ax):
        polygon = plt.Polygon(self.points, color=self.color)
        ax.add_patch(polygon)
        return ax

    def copy(self):
        return Building(self.name, self.points, self.color)
