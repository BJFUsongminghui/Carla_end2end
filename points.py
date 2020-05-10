import carla

actions = [[0., 0., 0.], [0., -0.25, 0], [0., 0.25, 0],
           [1., 0., 0], [1., -0.5, 0], [1., -0.25, 0], [1., 0.25, 0], [1., 0.5, 0],
           [0., 0., 1.0]]
# Town1
straight_points = [
    [carla.Transform(carla.Location(x=225, y=2.0, z=.431), carla.Rotation(pitch=0, yaw=0, roll=0)),
     carla.Transform(carla.Location(x=306.3, y=2.1, z=.431), carla.Rotation(pitch=0, yaw=0, roll=0))],
    [carla.Transform(carla.Location(x=392.5, y=36.3, z=.431), carla.Rotation(pitch=0, yaw=90, roll=0)),
     carla.Transform(carla.Location(x=392.5, y=86.0, z=.431), carla.Rotation(pitch=0, yaw=90, roll=0))],
    [carla.Transform(carla.Location(x=392.5, y=46.3, z=.431), carla.Rotation(pitch=0, yaw=90, roll=0)),
     carla.Transform(carla.Location(x=392.5, y=116.0, z=.431), carla.Rotation(pitch=0, yaw=90, roll=0))],
    [carla.Transform(carla.Location(x=392.5, y=106.0, z=.431), carla.Rotation(pitch=0, yaw=90, roll=0)),
     carla.Transform(carla.Location(x=392.5, y=206.0, z=.431), carla.Rotation(pitch=0, yaw=90, roll=0))],
    # [carla.Transform(carla.Location(x=356.2, y=326.2, z=1.8431), carla.Rotation(pitch=0, yaw=180, roll=0)),
    #  carla.Transform(carla.Location(x=280.0, y=326.2, z=1.8431), carla.Rotation(pitch=0, yaw=180, roll=0))],
    # [carla.Transform(carla.Location(x=107.9, y=326.2, z=1.8431), carla.Rotation(pitch=0, yaw=180, roll=0)),
    # carla.Transform(carla.Location(x=16.6, y=326.2, z=1.8431), carla.Rotation(pitch=0, yaw=180, roll=0))],
    [carla.Transform(carla.Location(x=1.8, y=306.2, z=.431), carla.Rotation(pitch=0, yaw=-90, roll=0)),
     carla.Transform(carla.Location(x=1.8, y=200.0, z=.431), carla.Rotation(pitch=0, yaw=-90, roll=0))]
]

right_points = [
    [carla.Transform(carla.Location(x=158.1, y=16, z=.8431), carla.Rotation(pitch=0, yaw=-90, roll=0)),
     carla.Transform(carla.Location(x=177, y=2., z=.8431), carla.Rotation(pitch=0, yaw=0, roll=0))],
    [carla.Transform(carla.Location(x=317.5, y=2., z=.8431), carla.Rotation(pitch=0, yaw=0, roll=0)),
     carla.Transform(carla.Location(x=335, y=17.6, z=.8431), carla.Rotation(pitch=0, yaw=90, roll=0))],
    [carla.Transform(carla.Location(x=334.5, y=39.3, z=.8431), carla.Rotation(pitch=0, yaw=90, roll=0)),
     carla.Transform(carla.Location(x=314.5, y=55.8, z=.8431), carla.Rotation(pitch=0, yaw=180, roll=0))]

]