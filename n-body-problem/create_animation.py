import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import click
import tqdm


class NBodyAnimation:
    def __init__(
        self,
        csv_file_path,
        num_dimensions,
        duration,
        trail_length,
        with_projections=True,
    ):
        """
        Initializes the NBodyAnimation class which sets up the matplotlib figure, axes, and other attributes for animating the n-body simulation.

        Parameters:
        - csv_file_path (str): The path to the CSV file that contains the simulation data.
        - num_dimensions (int): The number of spatial dimensions of the simulation (2 or 3).
        - duration (float): The length of the animation in seconds.
        - trail_length (int): The number of past positions to visualize as a trail for each body, showing its recent path.
        - with_projections (bool): If True and the simulation is 3D, projections of the bodies onto the coordinate planes are also shown.

        This constructor also initializes the matplotlib plot elements needed for the animation, such as the points
        representing bodies and their motion trails. It sets appropriate plot dimensions and limits based on whether
        the simulation is 2D or 3D.

        Raises:
            ValueError: If the number of dimensions is not 2 or 3.
        """
        if num_dimensions not in [2, 3]:
            raise ValueError(
                "Number of dimensions can be either 2 or 3, got number_of_dimensions:",
                num_dimensions,
            )
        self.csv_file_path = csv_file_path
        self.num_dimensions = num_dimensions
        self.duration = duration
        self.trail_length = trail_length
        self.with_projections = (
            with_projections and num_dimensions == 3
        )  # Ensure projections are only possible in 3D
        # Setup the figure and axes with a 3D projection if required
        self.fig, self.ax = plt.subplots(
            figsize=(10, 10),
            subplot_kw={"projection": "3d"} if num_dimensions == 3 else {},
        )
        if num_dimensions == 3:
            self.ax.set_zlim(-10, 10)  # Set limits for Z-axis
            self.ax.set_zlabel("Z Coordinate")  # Label Z-axis
        else:
            # Recreate 2D figure and axis if not 3D
            self.fig, self.ax = plt.subplots(figsize=(10, 10))

        self.points = []  # Points will be initialized based on actual number of bodies
        self.trails = []  # Trails will be initialized based on actual number of bodies
        self.projections = []  # Only used if num_dimensions == 3
        self.max_pos = None  # To be determined based on data from CSV

    def load_data(self):
        """
        Loads and processes the simulation data from a CSV file, initializes plotting elements like points and trails, and sets up the visualization.

        This method reads the position data of bodies from a CSV file specified by the csv_file_path attribute. It initializes
        the color, points, and trails for each body based on the number of dimensions and bodies. It also handles projections if the
        simulation is in 3D and sets axis limits and labels accordingly.

        Raises:
            ValueError: If there's a mismatch in the number of position columns and expected dimensions times the number of bodies,
                        or if a data row does not match the expected number of dimensions per body.

        Side effects:
            Initializes matplotlib points, trails, and possibly projections on the axes based on the data read.
            Modifies internal state by setting up the positions array, colors, and plot elements like points and trails.
        """
        with open(self.csv_file_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Read the first line as headers
            masses = headers[1:]
            self.num_bodies = len(masses)
            position_headers = next(reader)  # Read the second line for position headers

            # Ensure that the number of position columns matches the expected number based on dimensions and body count
            if len(position_headers) - 1 != self.num_bodies * self.num_dimensions:
                raise ValueError(
                    "Mismatch in number of position columns and expected dimensions * bodies"
                )

            # Assign colors to each body using a colormap
            self.colors = plt.cm.viridis(
                np.linspace(0, 1, self.num_bodies)
            )  # Generate colors for each body

            # Initialize points for bodies in either 2D or 3D
            if self.num_dimensions == 3:
                self.points = [
                    self.ax.plot([], [], [], "o", color=self.colors[i])[0]
                    for i in range(self.num_bodies)
                ]
            else:
                self.points = [
                    plt.plot([], [], "o", color=self.colors[i])[0]
                    for i in range(self.num_bodies)
                ]

            # Initialize trails for each body
            self.trails = [
                [
                    self.ax.plot(
                        [], [], [], "-", alpha=i / self.trail_length, color="gray"
                    )[0]
                    for i in range(self.trail_length, 0, -1)
                ]
                for _ in range(self.num_bodies)
            ]

            # Initialize projections if required and in 3D
            if self.with_projections:
                for i in range(self.num_bodies):
                    projections = {
                        "xy": self.ax.plot(
                            [],
                            [],
                            "o",
                            markersize=3,
                            alpha=0.5,
                            zdir="z",
                            zs=-3,
                            color=self.colors[i],
                        )[0],
                        "xz": self.ax.plot(
                            [],
                            [],
                            "o",
                            markersize=3,
                            alpha=0.5,
                            zdir="y",
                            zs=3,
                            color=self.colors[i],
                        )[0],
                        "yz": self.ax.plot(
                            [],
                            [],
                            "o",
                            markersize=3,
                            alpha=0.5,
                            zdir="x",
                            zs=-3,
                            color=self.colors[i],
                        )[0],
                    }
                    self.projections.append(projections)

            # Read positions and times from the file
            timesteps = []
            positions = []
            for row in reader:
                if len(row) - 1 != self.num_bodies * self.num_dimensions:
                    raise ValueError(
                        "Data row does not match expected number of dimensions per body"
                    )
                timestep = float(row[0])
                timesteps.append(timestep)
                pos_data = list(map(float, row[1:]))
                positions.append(pos_data)

            # Reshape the positions data and resample it for the animation duration
            self.positions = np.array(positions).reshape(
                -1, self.num_bodies, self.num_dimensions
            )
            self.num_timesteps = len(timesteps)
            # Resample data to match the video duration and frame rate
            frames_needed = int(self.duration * 60)
            step_size = max(1, len(timesteps) // frames_needed)
            self.positions = self.positions[::step_size]

            # Set axis limits based on the maximum absolute position to ensure all data is visible
            self.max_pos = np.abs(self.positions).max()
            self.ax.set_xlim(-self.max_pos, self.max_pos)
            self.ax.set_ylim(-self.max_pos, self.max_pos)
            if self.num_dimensions == 3:
                self.ax.set_zlim(-self.max_pos, self.max_pos)

            self.ax.grid(True)
            self.ax.set_xlabel("X Coordinate")
            self.ax.set_ylabel("Y Coordinate")
            self.ax.set_title("N-Body Simulation")

            self.legend_labels = masses
            for i, point in enumerate(self.points):
                point.set_label(f"Body {i+1}")
            self.ax.legend(loc="upper right")

    def init(self):
        """
        Initialize the animation by clearing all the points, trails, and projections.

        This method is intended to be used as the initialization function for FuncAnimation. It clears the data
        of all plotted elements (points, trails, and projections) to ensure that the animation starts with a clean slate.

        Returns:
            list: A list of all the line objects (points and trail segments) that will be used in the animation.
        """
        # Clear the data for each point representing a body
        for point in self.points:
            point.set_data([], [])
            if self.num_dimensions == 3:
                point.set_3d_properties([])

        # Clear the data for each trail segment behind each body
        for trail in self.trails:
            for segment in trail:
                segment.set_data([], [])
                if self.num_dimensions == 3:
                    segment.set_3d_properties([])

        # Clear the data for all projections if they are requested
        if self.with_projections:
            for projection_set in self.projections:
                for projection in projection_set.values():
                    projection.set_data([], [])
                    if self.num_dimensions == 3:
                        projection.set_3d_properties([])

        # Collect all line objects to be updated by the animation function
        return (
            self.points
            + [segment for trail in self.trails for segment in trail]
            + (
                [
                    proj
                    for projection_set in self.projections
                    for proj in projection_set.values()
                ]
                if self.with_projections
                else []
            )
        )

    def update(self, frame):
        """
        Updates the positions of points, trails, and projections for a given frame of the animation.

        This method sets the current position of each body in the simulation and updates their trails and projections
        if applicable. It is called for each frame in the animation sequence to reflect the movement of bodies based
        on the simulation data.

        Parameters:
        - frame (int): The index of the current frame in the animation sequence.

        Returns:
            list: A list of all matplotlib artists that need to be redrawn for this frame, which includes points, trail segments,
                  and projection markers if 3D projections are enabled.
        """
        # Update the position of each point and its associated trail and projection
        for j, (point, trail, projection_set) in enumerate(
            zip(
                self.points,
                self.trails,
                self.projections if self.with_projections else [None] * self.num_bodies,
            )
        ):
            # Set new data for 2D or 3D points
            if self.num_dimensions == 3:
                x, y, z = self.positions[frame][j]
                point.set_data([x], [y])
                point.set_3d_properties([z])
            else:
                x, y = self.positions[frame][j][: self.num_dimensions]
                point.set_data([x], [y])

            # Update trail segments to show paths of motion
            for i, segment in enumerate(trail):
                if frame - i - 1 >= 0:
                    x_trail = self.positions[frame - i - 1 : frame - i + 1, j, 0]
                    y_trail = self.positions[frame - i - 1 : frame - i + 1, j, 1]
                    z_trail = (
                        self.positions[frame - i - 1 : frame - i + 1, j, 2]
                        if self.num_dimensions == 3
                        else [0] * 2
                    )
                    segment.set_data(x_trail, y_trail)
                    (
                        segment.set_3d_properties(z_trail)
                        if self.num_dimensions == 3
                        else segment.set_data(x_trail, y_trail)
                    )
                else:
                    segment.set_data([], [])
                    (
                        segment.set_3d_properties([])
                        if self.num_dimensions == 3
                        else segment.set_data([], [])
                    )

            # Update projections in 3D mode
            if self.with_projections:
                # Update projections
                projection_set["xy"].set_data([x], [y])
                projection_set["xy"].set_3d_properties([-self.max_pos])
                projection_set["xz"].set_data([x], [self.max_pos])
                projection_set["xz"].set_3d_properties([z])
                projection_set["yz"].set_data([-self.max_pos], [y])
                projection_set["yz"].set_3d_properties([z])

        # Collect all matplotlib artists for rendering in this frame
        return (
            self.points
            + [segment for trail in self.trails for segment in trail]
            + (
                [
                    proj
                    for projection_set in self.projections
                    for proj in projection_set.values()
                ]
                if self.with_projections
                else []
            )
        )

    def animate(self):
        """
        Creates and saves an animation based on the loaded simulation data.

        This method sets up and runs the animation using Matplotlib's FuncAnimation, leveraging the `update` method
        to animate the positions of the bodies frame by frame. It utilizes the `init` method to establish the initial
        state of the animation. The resulting animation is then saved to a file.

        The progress of the animation saving process is displayed using a tqdm progress bar, providing a visual cue
        of completion.

        Side Effects:
            - Saves an MP4 file of the animation to the local file system.
        """
        # Create the animation object, specifying the update and init functions, and enable blitting for optimized rendering
        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.positions),
            init_func=self.init,
            blit=True,
        )

        # Use tqdm progress bar to show animation saving progress
        with tqdm.tqdm(total=len(self.positions), desc="Saving animation") as pbar:

            def update_progress(current_frame, total_frames):
                pbar.update(1)

            # Save the animation to an MP4 file, using the progress callback to update the progress bar
            ani.save(
                "video_of_n_body_simulation.mp4",
                fps=60,
                progress_callback=update_progress,
                extra_args=["-vcodec", "libx264"],
            )


@click.command()
@click.argument("csv_file_path", type=click.Path(exists=True))
@click.option("--num_dimensions", default=3, type=int)
@click.option(
    "--duration", default=30, type=float, help="Duration of the animation in seconds."
)
@click.option(
    "--trail_length",
    default=100,
    type=int,
    help="Length of the trail behind each body.",
)
@click.option(
    "--with-projections",
    type=bool,
    default=True,
    help="Plot the projections on the walls in a 3D simulation.",
)
def create_animation(
    csv_file_path, num_dimensions, duration, trail_length, with_projections
):
    """
    Creates and saves an animation for an n-body gravitational simulation from provided CSV data.

    This function initializes an animation of orbiting bodies based on their positional data over time, as
    extracted from a CSV file. The animation can be either 2D or 3D depending on the specified number of dimensions.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the simulation data.
    - num_dimensions (int): Number of dimensions (2 or 3) of the simulation.
    - duration (float): Total duration of the animation in seconds.
    - trail_length (int): Number of frames to show the trail behind each body to depict motion.
    - with_projections (bool): Whether to include projections on the coordinate planes in a 3D simulation.

    The CSV file should contain time-stepped positions of each body in the simulation. The first row should list the
    masses of the bodies, followed by position data rows that detail the bodies' coordinates at each time step.
    """
    animation = NBodyAnimation(
        csv_file_path, num_dimensions, duration, trail_length, with_projections
    )
    animation.load_data()  # Load the data from CSV and prepare the animation setup
    animation.animate()  # Start the animation process and save the output as a video file


if __name__ == "__main__":
    create_animation()
