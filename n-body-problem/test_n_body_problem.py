import pytest
import numpy as np
from n_body_problem import Body, NBodySystem, read_initial_conditions_from_csv
from numpy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt
import datetime

matplotlib.use("Agg")


def test_body_initialization():
    mass = 5.0
    position = [1.0, 2.0, 3.0]
    velocity = [0.0, 0.0, 0.0]
    body = Body(mass, position, velocity)

    assert body.mass == mass
    assert np.array_equal(body.position, np.array(position, dtype=np.float64))
    assert np.array_equal(body.velocity, np.array(velocity, dtype=np.float64))
    assert np.array_equal(body.acceleration, np.zeros_like(body.position))


def test_nbodysystem_initialization_valid():
    initial_conditions = [
        {"mass": 5.0, "position": [1.0, 2.0, 3.0], "velocity": [0.0, 0.0, 0.0]},
        {"mass": 3.0, "position": [4.0, 5.0, 6.0], "velocity": [0.1, 0.2, 0.3]},
    ]
    system = NBodySystem(initial_conditions, 3)

    assert system.number_of_dimensions == 3
    assert len(system.bodies) == 2
    for i, body in enumerate(system.bodies):
        assert body.mass == initial_conditions[i]["mass"]
        assert np.array_equal(
            body.position, np.array(initial_conditions[i]["position"], dtype=np.float64)
        )
        assert np.array_equal(
            body.velocity, np.array(initial_conditions[i]["velocity"], dtype=np.float64)
        )


def test_nbodysystem_initialization_invalid_dimensions():
    initial_conditions = [{"mass": 5.0, "position": [1.0, 2.0], "velocity": [0.0, 0.0]}]
    with pytest.raises(ValueError):
        NBodySystem(initial_conditions, 4)


def test_nbodysystem_initialization_invalid_mass():
    initial_conditions = [
        {"mass": -1.0, "position": [1.0, 2.0, 3.0], "velocity": [0.0, 0.0, 0.0]}
    ]
    with pytest.raises(ValueError):
        NBodySystem(initial_conditions, 3)


def test_simulate_with_euler():
    initial_conditions = [
        {"mass": 5.0, "position": [1.0, 0.0, 0.0], "velocity": [0.0, 0.1, 0.0]},
        {"mass": 5.0, "position": [-1.0, 0.0, 0.0], "velocity": [0.0, -0.1, 0.0]},
    ]
    system = NBodySystem(initial_conditions, 3)
    system.simulate(final_time=10.0, time_increment=0.1, solver="euler")

    # Check that history has the correct length
    assert len(system.history) == 2
    assert len(system.history[0]["position"]) == int(10.0 / 0.1) + 1

    # Check conservation of momentum
    initial_total_momentum = np.sum(
        [body.mass * body.velocity for body in system.bodies], axis=0
    )
    final_total_momentum = np.sum(
        [
            body.mass * np.array(system.history[i]["velocity"][-1])
            for i, body in enumerate(system.bodies)
        ],
        axis=0,
    )
    np.testing.assert_allclose(initial_total_momentum, final_total_momentum, rtol=1e-5)

    # Check conservation of energy
    initial_total_energy = 0.0
    for i, body in enumerate(system.bodies):
        kinetic_energy = 0.5 * body.mass * norm(body.velocity) ** 2
        potential_energy = 0.0
        for j, other_body in enumerate(system.bodies):
            if i != j:
                distance_vector = body.position - other_body.position
                potential_energy += (
                    -system.Constant
                    * body.mass
                    * other_body.mass
                    / norm(distance_vector)
                )
        initial_total_energy += kinetic_energy + 0.5 * potential_energy

    final_total_energy = 0.0
    for i, body in enumerate(system.bodies):
        final_position = np.array(system.history[i]["position"][-1])
        final_velocity = np.array(system.history[i]["velocity"][-1])
        kinetic_energy = 0.5 * body.mass * norm(final_velocity) ** 2
        potential_energy = 0.0
        for j, other_body in enumerate(system.bodies):
            if i != j:
                final_other_position = np.array(system.history[j]["position"][-1])
                distance_vector = final_position - final_other_position
                potential_energy += (
                    -system.Constant
                    * body.mass
                    * other_body.mass
                    / norm(distance_vector)
                )
        final_total_energy += kinetic_energy + 0.5 * potential_energy

    np.testing.assert_allclose(initial_total_energy, final_total_energy, rtol=1e-5)


def test_simulate_with_leap_frog():
    initial_conditions = [
        {"mass": 5.0, "position": [1.0, 0.0, 0.0], "velocity": [0.0, 0.1, 0.0]},
        {"mass": 5.0, "position": [-1.0, 0.0, 0.0], "velocity": [0.0, -0.1, 0.0]},
    ]
    system = NBodySystem(initial_conditions, 3)
    system.simulate(final_time=10.0, time_increment=0.1, solver="leap_frog")

    # Check that history has the correct length
    assert len(system.history) == 2
    assert len(system.history[0]["position"]) == int(10.0 / 0.1) + 1

    # Check conservation of momentum
    initial_total_momentum = np.sum(
        [body.mass * body.velocity for body in system.bodies], axis=0
    )
    final_total_momentum = np.sum(
        [
            body.mass * np.array(system.history[i]["velocity"][-1])
            for i, body in enumerate(system.bodies)
        ],
        axis=0,
    )
    np.testing.assert_allclose(initial_total_momentum, final_total_momentum, rtol=1e-5)

    # Check conservation of energy
    initial_total_energy = 0.0
    for i, body in enumerate(system.bodies):
        kinetic_energy = 0.5 * body.mass * norm(body.velocity) ** 2
        potential_energy = 0.0
        for j, other_body in enumerate(system.bodies):
            if i != j:
                distance_vector = body.position - other_body.position
                potential_energy += (
                    -system.Constant
                    * body.mass
                    * other_body.mass
                    / norm(distance_vector)
                )
        initial_total_energy += kinetic_energy + 0.5 * potential_energy

    final_total_energy = 0.0
    for i, body in enumerate(system.bodies):
        final_position = np.array(system.history[i]["position"][-1])
        final_velocity = np.array(system.history[i]["velocity"][-1])
        kinetic_energy = 0.5 * body.mass * norm(final_velocity) ** 2
        potential_energy = 0.0
        for j, other_body in enumerate(system.bodies):
            if i != j:
                final_other_position = np.array(system.history[j]["position"][-1])
                distance_vector = final_position - final_other_position
                potential_energy += (
                    -system.Constant
                    * body.mass
                    * other_body.mass
                    / norm(distance_vector)
                )
        final_total_energy += kinetic_energy + 0.5 * potential_energy

    np.testing.assert_allclose(initial_total_energy, final_total_energy, rtol=1e-5)


@pytest.fixture
def nbody_system():
    initial_conditions = [
        {"mass": 5.0, "position": [1.0, 0.0, 0.0], "velocity": [0.0, 0.1, 0.0]},
        {"mass": 5.0, "position": [-1.0, 0.0, 0.0], "velocity": [0.0, -0.1, 0.0]},
    ]
    system = NBodySystem(initial_conditions, 3)
    system.simulate(final_time=1.0, time_increment=0.1, solver="euler")
    return system


@pytest.fixture
def mock_object(mocker):
    # Creating an instance of NBodySystem with mock initial conditions
    mock_obj = NBodySystem(initial_conditions=[], number_of_dimensions=2)
    mock_obj.bodies = [
        Body(1, [0, 0], [0, 0]),
        Body(1, [1, 1], [0, 0]),
        Body(1, [2, 2], [0, 0]),
    ]
    mock_obj.history = {
        0: {
            "position": [(0, 0), (1, 1), (2, 2)],
            "velocity": [(0, 0), (1, 1), (2, 2)],
            "mass": 1,
        },
        1: {
            "position": [(0, 0), (0.5, 0.5), (1, 1)],
            "velocity": [(0, 0), (0.5, 0.5), (1, 1)],
            "mass": 1,
        },
        2: {
            "position": [(0, 0), (1, 0), (1, 1)],
            "velocity": [(0, 0), (1, 0), (1, 1)],
            "mass": 1,
        },
    }
    return mock_obj


def test_plot_trajectories_with_matplotlib(mocker, mock_object):
    # Mock the current time to have a fixed timestamp for the filename
    mock_datetime = mocker.patch("datetime.datetime")
    mock_datetime.now.return_value = datetime.datetime(2023, 5, 23, 12, 0, 0)
    mock_datetime.now().strftime.return_value = "20230523_120000"

    # Spy on matplotlib functions and mock savefig to prevent file creation
    figure_spy = mocker.spy(plt, "figure")
    savefig_spy = mocker.patch("matplotlib.pyplot.savefig", return_value=None)

    # Call the method to be tested
    mock_object.plot_trajectories_with_matplotlib()

    # Assertions to ensure the plot is created
    figure_spy.assert_called_once_with(figsize=(10, 10))
    savefig_spy.assert_called_once_with("plot_of_trajectories_20230523_120000.png")


def test_plot_specific_energies_bodies(mocker, mock_object):
    # Mock savefig to prevent file creation and spy on subplots
    savefig_spy = mocker.patch("matplotlib.pyplot.savefig", return_value=None)
    subplots_spy = mocker.spy(plt, "subplots")

    # Call the method with specific bodies to plot
    mock_object.plot_energies_with_matplotlib("euler", [0, 2])

    # Assertions to ensure the plot is created
    subplots_spy.assert_called_once()
    savefig_spy.assert_called_once()


def test_generate_csv(mocker, mock_object):
    # Mock the current time to have a fixed timestamp for the filename
    mock_datetime = mocker.patch("datetime.datetime")
    mock_datetime.now.return_value = datetime.datetime(2023, 5, 23, 12, 0, 0)
    mock_datetime.now().strftime.return_value = "20230523_120000"

    # Mock open to use an in-memory file
    mock_open = mocker.patch("builtins.open", mocker.mock_open())

    # Call the method to be tested
    mock_object.generate_csv(1.0)

    # Get the file handle
    file_handle = mock_open()

    # Extract actual write calls
    actual_write_calls = file_handle.write.call_args_list

    # Expected write calls
    expected_write_calls = [
        mocker.call("Masses,1,1,1\r\n"),
        mocker.call(
            "Timestep,Body 1 x,Body 1 y,Body 2 x,Body 2 y,Body 3 x,Body 3 y\r\n"
        ),
        mocker.call("0.0,0,0,0,0,0,0\r\n"),
        mocker.call("1.0,1,1,0.5,0.5,1,0\r\n"),
        mocker.call("2.0,2,2,1,1,1,1\r\n"),
    ]

    assert actual_write_calls == expected_write_calls


def test_read_initial_conditions_from_csv(mocker):
    # Mock open to use an in-memory file with sample CSV content
    csv_content = """mass,position,velocity
1.0,"(0, 0, 0)","(1, 0, 0)"
2.0,"(1, 1, 1)","(0, 1, 0)"
"""

    mock_open = mocker.patch("builtins.open", mocker.mock_open(read_data=csv_content))

    # Call the method to be tested
    result = read_initial_conditions_from_csv("dummy_path.csv")

    # Expected result
    expected_result = [
        {"mass": 1.0, "position": (0, 0, 0), "velocity": (1, 0, 0)},
        {"mass": 2.0, "position": (1, 1, 1), "velocity": (0, 1, 0)},
    ]

    assert result == expected_result


def test_invalid_solver():
    initial_conditions = [
        {"mass": 5.0, "position": [1.0, 0.0, 0.0], "velocity": [0.0, 0.1, 0.0]},
    ]
    system = NBodySystem(initial_conditions, 3)
    with pytest.raises(ValueError):
        system.simulate(final_time=10.0, time_increment=0.1, solver="invalid_solver")
