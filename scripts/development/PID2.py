import pymunk
import pymunk.pygame_util


class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=1 / 60):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self.integral = 0
        self.previous_error = 0

    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error

        return output  # This is the force to apply


# Create a pymunk space
space = pymunk.Space()
space.gravity = (0, 0)

# Create a dynamic body
mass = 1.0
moment = pymunk.moment_for_circle(mass, 0, 10)  # Moment of inertia
body = pymunk.Body(mass, moment)
body.position = (100, 100)
shape = pymunk.Circle(body, 10)
space.add(body, shape)

dt = 1 / 60  # 60 Hz physics step
# Initialize PID controller
pid = PIDController(Kp=55, Ki=0, Kd=0, dt=dt)

# Simulation loop
target_velocity = 100  # Target velocity in x direction

for step in range(300):  # Simulate for 5 seconds
    current_velocity = body.velocity.x
    force_x = pid.compute(target_velocity, current_velocity)

    # Apply force to control velocity (assuming no y-axis movement)
    body.apply_force_at_world_point((force_x, 0), (0, 0))

    # Step pymunk simulation
    space.step(dt)

    print(
        f"Step {step}: Velocity = {body.velocity.x:.2f}, Applied Force = {force_x:.2f}")
