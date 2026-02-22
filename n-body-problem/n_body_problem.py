import click
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import csv
import ast
from tqdm import tqdm
import datetime


class Body:
    """
    Represents a celestial body in an n-body simulation. This class stores and manages the state of the body,
    including its mass, current position, velocity, and acceleration. These attributes are updated throughout the
    simulation to reflect the body's dynamics in response to gravitational forces.

    Attributes:
        mass (float): The mass of the body, which must be a positive number.
        position (numpy.ndarray): The current position of the body as a NumPy array of floats. This array will
                                  initially contain the starting coordinates and be updated as the simulation progresses.
                                  It can have either two or three elements, representing the coordinates in 2D or 3D space respectively.
        velocity (numpy.ndarray): The current velocity of the body, initially set to the starting velocity and updated
                                  continuously throughout the simulation. It is structured similarly to the position array.
        acceleration (numpy.ndarray): The current acceleration of the body, initially set to zero and updated each
                                      simulation step based on the gravitational forces exerted by other bodies.

    Methods:
        __init__(mass, initial_position, initial_velocity): Initializes a new Body instance with specified mass,
                                                            starting position, and starting velocity.
    """

    def __init__(self, mass, initial_position, initial_velocity):
        self.mass = mass
        self.position = np.array(initial_position, dtype=np.float64)
        self.velocity = np.array(initial_velocity, dtype=np.float64)
        self.acceleration = np.zeros_like(self.position)


class NBodySystem:
    """
    Represents the entire n-body simulation system, containing multiple celestial bodies interacting under gravitational forces.
    This class manages the simulation environment, including the dimensions of the space (2D or 3D), initializing celestial bodies,
    and storing their history throughout the simulation.

    Attributes:
        Constant (float): Gravitational constant used in the simulation, set to 6.67340e-11 m^3 kg^-1 s^-2.
        number_of_dimensions (int): The number of spatial dimensions for the simulation, either 2 or 3.
        bodies (list of Body): A list of Body instances representing the celestial bodies in the simulation.
        history (dict): A dictionary that logs the historical state of each body during the simulation. It is indexed by the body's
                        index in the `bodies` list and contains lists of positions and velocities at each time step.
    """

    Constant = 0.0000000000667340

    def __init__(self, initial_conditions: list, number_of_dimensions: int):
        """
        Initializes a new instance of the NBodySystem class with specified initial conditions for each body and the
        number of dimensions for the simulation.

        Parameters:
            initial_conditions (list of dict): A list of dictionaries, where each dictionary contains the 'mass', 'position', and 'velocity'
                                               of a body. These are used to initialize the bodies in the system.
            number_of_dimensions (int): The number of spatial dimensions for the simulation, which can be either 2 or 3.

        Raises:
            ValueError: If the number of dimensions is not 2 or 3, or if any body's position or velocity does not match the number of dimensions,
                        or if any body's mass is not positive.
        """
        if number_of_dimensions not in [2, 3]:
            raise ValueError(
                "Number of dimensions can be either 2 or 3, got number_of_dimensions:",
                number_of_dimensions,
            )
        self.number_of_dimensions = number_of_dimensions

        self.bodies = []
        for condition in initial_conditions:
            if (
                len(condition["position"]) != number_of_dimensions
                or len(condition["velocity"]) != number_of_dimensions
            ):
                raise ValueError(
                    "Position and velocity must match the number of dimensions"
                )
            if condition["mass"] <= 0:
                raise ValueError("Mass must be greater than 0")
            body = Body(condition["mass"], condition["position"], condition["velocity"])
            self.bodies.append(body)
        self.history = {}

    def simulate(self, final_time: float, time_increment: float, solver: str):
        """
        Conducts the simulation of the n-body system using the specified numerical solver over a set period of time.
        This method updates the positions, velocities, and accelerations of all bodies in the system based on
        gravitational interactions.

        Parameters:
            final_time (float): The total time, in simulation units, for which the simulation will run.
            time_increment (float): The time increment, in simulation units, between each step of the simulation.
            solver (str): Specifies the numerical method to use for the simulation. Options are 'euler' or 'leap_frog'.

        Description:
            - The 'euler' method updates the positions and velocities of the bodies using the straightforward Euler
              integration method, which is a first-order numerical procedure.
            - The 'leap_frog' method uses the Leapfrog integration technique, which is more accurate for simulations of
              orbital dynamics as it is symplectic, hence better at conserving the system's total energy over time.

        Updates:
            This method updates the `history` dictionary attribute of the NBodySystem class. Each body's mass,
            positions, and velocities are recorded at each time step throughout the simulation.

        Raises:
            ValueError: If the solver parameter is not one of the recognized options ('euler' or 'leap_frog').

        Notes:
            - The `history` dictionary is indexed by each body's index, with each entry containing 'mass', 'position',
              and 'velocity' lists that store the history of these properties through the time steps.
            - Use tqdm to visualize the progress of the simulation in a command-line interface, providing feedback on
              the progress.

        Example:
            simulate(100.0, 0.1, 'euler')
            This would simulate 100.0 units of time, updating every 0.1 time units using the Euler method.
        """
        number_of_steps = int(final_time / time_increment)
        self.history = {}
        for i, body in enumerate(self.bodies):
            self.history[i] = {
                "mass": body.mass,
                "position": [body.position.tolist()],
                "velocity": [body.velocity.tolist()],
            }

        if solver not in ["euler", "leap_frog"]:
            raise ValueError("solver must be either 'euler' or 'leap_frog'")

        if solver == "euler":
            # Use tqdm to create a progress bar for the simulation steps
            for step in tqdm(range(number_of_steps), desc="Simulating"):
                for body in self.bodies:
                    body.acceleration = np.zeros(self.number_of_dimensions)
                    for other_body in self.bodies:
                        if body == other_body:
                            continue
                        distance_vector = body.position - other_body.position
                        body.acceleration += (
                            -self.Constant
                            * other_body.mass
                            * distance_vector
                            / norm(distance_vector) ** 3
                        )

                for i, body in enumerate(self.bodies):
                    body.position += body.velocity * time_increment
                    body.velocity += body.acceleration * time_increment
                    self.history[i]["position"].append(body.position.tolist())
                    self.history[i]["velocity"].append(body.velocity.tolist())

        if solver == "leap_frog":

            for body in self.bodies:
                body.acceleration = np.zeros_like(body.position)
                for other_body in self.bodies:
                    if body == other_body:
                        continue
                    distance_vector = body.position - other_body.position
                    body.acceleration += (
                        -self.Constant
                        * other_body.mass
                        * distance_vector
                        / norm(distance_vector) ** 3
                    )

            for i, body in enumerate(self.bodies):
                body.velocity += body.acceleration * (time_increment / 2)
                self.history[i]["velocity"].append(body.velocity.tolist())

            # Use tqdm to create a progress bar for the simulation steps
            for step in tqdm(range(number_of_steps), desc="Simulating"):
                for i, body in enumerate(self.bodies):
                    body.position += body.velocity * time_increment
                    self.history[i]["position"].append(body.position.tolist())

                for body in self.bodies:
                    body.acceleration = np.zeros_like(body.position)
                    for other_body in self.bodies:
                        if body == other_body:
                            continue
                        distance_vector = body.position - other_body.position
                        body.acceleration += (
                            -self.Constant
                            * other_body.mass
                            * distance_vector
                            / norm(distance_vector) ** 3
                        )

                for i, body in enumerate(self.bodies):
                    body.velocity += body.acceleration * time_increment
                    self.history[i]["velocity"].append(body.velocity.tolist())

    def plot_trajectories_with_matplotlib(self, bodies_to_plot=None):
        """
        Plots the trajectories of selected bodies in the n-body simulation using Matplotlib.
        This method generates a plot of the trajectories of each body over the course of the simulation in 2D or 3D.

        Parameters:
            bodies_to_plot (list of int, optional): A list of indices representing the bodies whose trajectories
                                                    are to be plotted. If None, trajectories for all bodies are plotted.

        Output:
            A PNG file named with a timestamp indicating when the plot was generated. The file shows the trajectories
            of the bodies in space, marked with start and end positions.

        Notes:
            - The plot is saved to a PNG file in the current directory.
            - The start positions are marked in red, and the end positions are marked in green.
            - The dimensionality (2D or 3D) is determined by self.number_of_dimensions.

        Example:
            plot_trajectories_with_matplotlib()
            This will plot and save the trajectories of all bodies.

            plot_trajectories_with_matplotlib([0, 2])
            This will plot and save the trajectories of the bodies indexed 0 and 2.
        """
        print("Plotting Trajectories")

        if self.number_of_dimensions == 3:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
        else:
            plt.figure(figsize=(10, 10))

        if bodies_to_plot is None:
            bodies_to_plot = range(len(self.bodies))

        for i in bodies_to_plot:
            history = self.history[i]["position"]
            if self.number_of_dimensions == 3:
                x_coords = [position[0] for position in history]
                y_coords = [position[1] for position in history]
                z_coords = [position[2] for position in history]
                ax.plot3D(x_coords, y_coords, z_coords, label=f"Body {i} Trajectory")
                ax.scatter3D(
                    x_coords[0], y_coords[0], z_coords[0], color="red", zorder=5
                )
                ax.scatter3D(
                    x_coords[-1], y_coords[-1], z_coords[-1], color="green", zorder=5
                )
            else:
                x_coords = [position[0] for position in history]
                y_coords = [position[1] for position in history]
                plt.plot(x_coords, y_coords, label=f"Body {i} Trajectory")
                plt.scatter(x_coords[0], y_coords[0], color="red", zorder=5)
                plt.scatter(x_coords[-1], y_coords[-1], color="green", zorder=5)

        if self.number_of_dimensions == 3:
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_zlabel("Z Position")
            ax.legend()
            ax.grid(True)
            plt.title("Trajectories in the N-Body Simulation (3D)")
        else:
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            plt.title("Trajectories in the N-Body Simulation (2D)")

        # Save the figure to a file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_of_trajectories_{timestamp}.png"
        plt.savefig(filename)
        plt.close()  # Close the figure to free memory

    def plot_energies_with_matplotlib(self, solver: str, bodies_to_plot=None):
        """
        Plots the kinetic, potential, and total energies of the celestial bodies in the n-body simulation
        using Matplotlib. This method allows visualization of energy changes over the course of the simulation
        to help analyze the system's dynamics and conservation properties.

        Parameters:
            solver (str): The numerical solver used in the simulation ('euler' or 'leap_frog'). This affects how the
                          energies are calculated, particularly the kinetic energy which can depend on how velocities
                          are updated in the solver.
            bodies_to_plot (list of int, optional): A list of indices representing the bodies whose energies
                                                    are to be plotted. If None, energies for all bodies are plotted.

        Output:
            A PNG file named with a timestamp indicating when the energy plots were generated. The file contains
            separate subplots for each body showing their kinetic, potential, and total energies, as well as a system-wide
            energy plot to visualize overall energy conservation.

        Notes:
            - This method organizes multiple subplots into a grid that adjusts based on the number of bodies, ensuring
              each body's energy plot is visible.
            - Kinetic and potential energies are calculated from the bodies' motion and interactions. Total energy
              is the sum of kinetic and potential energies.
            - The figure is saved to a PNG file in the current directory and then closed to free up memory.

        Example:
            plot_energies_with_matplotlib('euler')
            This will plot and save the energies of all bodies using the Euler method.

            plot_energies_with_matplotlib('leap_frog', [0, 2])
            This will plot and save the energies of the bodies indexed 0 and 2 using the Leapfrog method.
        """
        if bodies_to_plot is None:
            bodies_to_plot = range(len(self.bodies))

        # Calculate the dimensions for a square-ish layout of subplots
        num_bodies = len(bodies_to_plot) + 1
        num_cols = int(np.ceil(np.sqrt(num_bodies)))
        num_rows = int(np.ceil(num_bodies / num_cols))

        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows)
        )
        # If there's only one subplot, axs might not be an array; make it one for consistency
        axs = np.array(axs).reshape(num_rows, num_cols)

        system_kinetic_energy = [0] * len(self.history[0]["position"])
        system_potential_energy = [0] * len(self.history[0]["position"])

        # Wrap bodies_to_plot with tqdm for progress indication
        for subplot_index, body_index in tqdm(
            enumerate(bodies_to_plot),
            total=len(bodies_to_plot),
            desc="Plotting Energies",
        ):
            body = self.bodies[body_index]

            if solver == "euler":
                kinetic_energy = [
                    (1 / 2) * body.mass * norm(np.array(v)) ** 2
                    for v in self.history[body_index]["velocity"]
                ]
            if solver == "leap_frog":
                kinetic_energy = [
                    (1 / 2)
                    * body.mass
                    * norm(self.history[body_index]["velocity"][0]) ** 2
                ]
                for i in range(len(self.history[body_index]["velocity"]) - 2):
                    velocity_before_halfstep = np.array(
                        self.history[body_index]["velocity"][i + 1]
                    )
                    velocity_after_halfstep = np.array(
                        self.history[body_index]["velocity"][i + 2]
                    )
                    kinetic_energy.append(
                        (1 / 2)
                        * body.mass
                        * norm((velocity_before_halfstep + velocity_after_halfstep) / 2)
                        ** 2
                    )
            potential_energy = []

            for step in range(len(self.history[body_index]["position"])):
                pot_energy_at_step = 0
                for j, other_body in enumerate(self.bodies):
                    if body_index == j:
                        continue
                    distance_vector = np.array(
                        self.history[body_index]["position"][step]
                    ) - np.array(self.history[j]["position"][step])
                    pot_energy_at_step += (
                        -self.Constant
                        * (body.mass * other_body.mass)
                        / norm(distance_vector)
                    )
                potential_energy.append(pot_energy_at_step)

                # Accumulate system-wide energies
                system_kinetic_energy[step] += kinetic_energy[step]
                system_potential_energy[step] += (
                    pot_energy_at_step / 2
                )  # Avoid double-counting

            total_energy = [k + p for k, p in zip(kinetic_energy, potential_energy)]
            time_steps = range(len(kinetic_energy))

            ax = axs[subplot_index // num_cols, subplot_index % num_cols]
            ax.plot(time_steps, kinetic_energy, label="Kinetic Energy")
            ax.plot(time_steps, potential_energy, label="Potential Energy")
            ax.plot(time_steps, total_energy, label="Total Energy")
            ax.set_title(f"Body {body_index} Energies")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Energy (Joules)")
            ax.legend()
            ax.grid(True)

        # System-wide energies plot on the next subplot
        system_total_energy = [
            k + p for k, p in zip(system_kinetic_energy, system_potential_energy)
        ]
        ax = axs[(subplot_index + 1) // num_cols, (subplot_index + 1) % num_cols]
        ax.plot(time_steps, system_kinetic_energy, label="System Kinetic Energy")
        ax.plot(time_steps, system_potential_energy, label="System Potential Energy")
        ax.plot(time_steps, system_total_energy, label="System Total Energy")
        ax.set_title("System-wide Energies")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Energy (Joules)")
        ax.legend()
        ax.grid(True)

        # Hide unused subplots if any
        for idx in range(subplot_index + 2, num_rows * num_cols):
            axs[idx // num_cols, idx % num_cols].axis("off")

        # Adjust layout for better spacing
        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_of_energies_{timestamp}.png"
        plt.savefig(filename)
        plt.close()  # Close the figure to free memory

    def generate_csv(self, time_increment):
        """
        Generates a CSV file containing the simulation data for each body at every timestep. This includes the mass,
        positions, and corresponding simulation timesteps, allowing for further analysis or record-keeping.

        Parameters:
            time_increment (float): The time increment used in the simulation, which determines the time scale for each
                                    timestep recorded in the CSV.

        Output:
            A CSV file named with a timestamp indicating when it was generated. This file contains rows corresponding to
            each timestep, with columns for each body's position and the mass for all bodies listed in the first row.

        Notes:
            - The CSV file is saved to the current directory with a timestamp in its filename to avoid overwriting
              previous files.
            - Each body's position is recorded per timestep. If a body does not have recorded data for a timestep (due
              to ending early in some types of simulations), 'None' is used as a placeholder for its position.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_data_{timestamp}.csv"

        # Determine the maximum number of timesteps
        max_timesteps = max(
            len(body_data["position"]) for body_data in self.history.values()
        )
        number_of_bodies = len(self.history)

        # Prepare the CSV file for writing
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write the first row with 'Masses'
            masses_row = ["Masses"] + [
                body_data["mass"] for body_data in self.history.values()
            ]
            writer.writerow(masses_row)

            # Prepare headers for position data
            position_headers = ["Timestep"]
            for body_index in range(number_of_bodies):
                # Depending on the number of dimensions, add x, y, (and z) positions for each body
                body_label = f"Body {body_index + 1}"
                if self.number_of_dimensions == 3:
                    position_headers.extend(
                        [f"{body_label} x", f"{body_label} y", f"{body_label} z"]
                    )
                else:
                    position_headers.extend([f"{body_label} x", f"{body_label} y"])
            writer.writerow(position_headers)

            # Write the position data for each timestep
            for timestep in range(max_timesteps):
                scaled_timestep = timestep * time_increment
                row = [scaled_timestep]
                for body_index in range(number_of_bodies):
                    body_data = self.history[body_index]
                    # If timestep exceeds available data, append None for each dimension
                    if timestep < len(body_data["position"]):
                        position = body_data["position"][timestep]
                    else:
                        position = [None] * self.number_of_dimensions
                    row.extend(position)
                writer.writerow(row)

        print("CSV file generated:", filename)


def read_initial_conditions_from_csv(csv_filepath):
    """
    Reads initial conditions for celestial bodies from a CSV file and returns them as a list of dictionaries.

    Parameters:
        csv_filepath (str): The path to the CSV file containing columns for 'mass', 'position', and 'velocity'.

    Returns:
        list of dict: Each dictionary contains 'mass', 'position', and 'velocity' for a celestial body.

    Each 'position' and 'velocity' should be stored in the CSV in a format evaluable to a Python tuple or list (e.g., "(1, 2, 3)").
    """
    initial_conditions = []
    with open(csv_filepath, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mass = float(row["mass"])
            position = ast.literal_eval(row["position"])
            velocity = ast.literal_eval(row["velocity"])
            initial_conditions.append(
                {"mass": mass, "position": position, "velocity": velocity}
            )
    return initial_conditions


def validate_solver(ctx, param, value):
    """
    Validates the selected solver method against allowed options.

    Parameters:
        value (str): The solver type to validate.

    Returns:
        str: Returns the solver if valid.

    Raises:
        click.BadParameter: If the solver is not 'euler' or 'leap_frog'.
    """
    if value not in ["euler", "leap_frog"]:
        raise click.BadParameter("solver must be either 'euler' or 'leap_frog'")
    return value


@click.command()
@click.argument("csv_filepath", type=click.Path(exists=True))
@click.option(
    "--dimensions", default=3, help="Number of dimensions (2 or 3).", type=int
)
@click.option(
    "--final_time", default=10000, help="Final time for simulation.", type=float
)
@click.option(
    "--time_increment",
    default=0.1,
    help="Time increment for each simulation step.",
    type=float,
)
@click.option(
    "--plot_trajectories",
    default=None,
    help="Optional comma-separated list of body indices to plot "
    "trajectories for. If omitted, plots all.",
    type=str,
)
@click.option(
    "--plot_energies",
    default=None,
    help="Optional comma-separated list of body indices to plot "
    "energies for. If omitted, plots all.",
    type=str,
)
@click.option(
    "--solver",
    default="leap_frog",
    callback=validate_solver,
    help="The solver to use for the simulation: 'euler' or 'leap_frog'. Default is 'leap_frog'.",
    type=str,
)
@click.option(
    "--generate-output-file",
    is_flag=True,
    help="Generates an output CSV file with data if set.",
)
def run_simulation(
    csv_filepath,
    dimensions,
    final_time,
    time_increment,
    plot_trajectories,
    plot_energies,
    solver,
    generate_output_file,
):
    """
    Command-line interface to run the n-body simulation with various configurable options.

    Arguments:
        csv_filepath (str): File path to the CSV with initial conditions for the bodies.
    Options:
        dimensions (int): Spatial dimensions of the simulation (2 or 3).
        final_time (float): Duration of the simulation in units.
        time_increment (float): Time step size for updating the simulation.
        plot_trajectories (str): Comma-separated indices of bodies to plot trajectories, plots all if omitted.
        plot_energies (str): Comma-separated indices of bodies to plot energies, plots all if omitted.
        solver (str): Numerical method for simulation ('euler' or 'leap_frog').
        generate_output_file (bool): If true, outputs a CSV file with the simulation data.

    The function initializes the simulation system, runs it, and optionally plots trajectories and energies of the
    bodies. It can also generate a CSV file with the detailed simulation data if requested.
    """
    initial_conditions = read_initial_conditions_from_csv(csv_filepath)
    system = NBodySystem(initial_conditions, dimensions)
    system.simulate(final_time, time_increment, solver)

    if plot_trajectories is not None:
        # Split the string by commas and convert each part to an integer
        bodies_to_plot = [int(x) for x in plot_trajectories.split(",")]
        system.plot_trajectories_with_matplotlib(bodies_to_plot)
    else:
        # When plot_trajectories is None or empty, plot all bodies
        system.plot_trajectories_with_matplotlib()

    if plot_energies is not None:
        # Split the string by commas and convert each part to an integer
        bodies_to_plot = [int(x) for x in plot_trajectories.split(",")]
        system.plot_energies_with_matplotlib(
            solver=solver, bodies_to_plot=bodies_to_plot
        )
    else:
        # When plot_energies is None or empty, plot all bodies
        system.plot_energies_with_matplotlib(solver=solver)

    if generate_output_file:
        system.generate_csv(time_increment)


if __name__ == "__main__":
    run_simulation()
