import genesis as gs

# 1. Initialize Genesis with GPU backend
gs.init(backend=gs.gpu)

# 2. Create a scene with the visualizer enabled
scene = gs.Scene(show_viewer=True)

# 3. Add a floor
plane = scene.add_entity(gs.morphs.Plane())

# 4. Add a Franka Robot Arm (standard example asset)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

# 5. Build the simulation environment
scene.build()

# 6. Run the simulation loop
print("Simulation starting... Press 'ESC' in the viewer to close.")
for i in range(10000):
    scene.step()