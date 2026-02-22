import pytest
import numpy as np
import matplotlib.pyplot as plt
from create_animation import NBodyAnimation

# Mocking the plt.show() to prevent actual rendering in tests
plt.show = lambda: None
plt.switch_backend("Agg")


def test_nbodyanimation_initialization():
    animation = NBodyAnimation(
        csv_file_path="dummy.csv",
        num_dimensions=3,
        duration=30,
        trail_length=100,
        with_projections=True,
    )
    assert animation.csv_file_path == "dummy.csv"
    assert animation.num_dimensions == 3
    assert animation.duration == 30
    assert animation.trail_length == 100
    assert animation.with_projections is True
    assert animation.fig is not None
    assert animation.ax is not None


# Test for loading data from a CSV file
def test_load_data(mocker):
    mock_csv_content = """Masses,1,1,1
Timestep,Body 1 x,Body 1 y,Body 1 z,Body 2 x,Body 2 y,Body 2 z,Body 3 x,Body 3 y,Body 3 z
0.0,0,0,0,1,1,1,2,2,2
1.0,0.1,0.1,0.1,1.1,1.1,1.1,2.1,2.1,2.1
2.0,0.2,0.2,0.2,1.2,1.2,1.2,2.2,2.2,2.2
"""
    mock_open = mocker.patch(
        "builtins.open", mocker.mock_open(read_data=mock_csv_content)
    )

    animation = NBodyAnimation(
        csv_file_path="dummy.csv",
        num_dimensions=3,
        duration=30,
        trail_length=100,
        with_projections=True,
    )

    animation.load_data()

    assert animation.num_bodies == 3
    assert animation.positions.shape == (3, 3, 3)
    assert animation.max_pos == 2.2
    assert len(animation.points) == 3
    assert len(animation.trails) == 3
    if animation.with_projections:
        assert len(animation.projections) == 3


# Test for initialization of animation elements
def test_init(mocker):
    animation = NBodyAnimation(
        csv_file_path="dummy.csv",
        num_dimensions=3,
        duration=30,
        trail_length=100,
        with_projections=True,
    )

    # Mock data for the test
    animation.num_bodies = 3
    animation.positions = np.zeros((10, 3, 3))
    animation.max_pos = 1.0
    animation.load_data = mocker.Mock()
    animation.load_data()

    init_elements = animation.init()
    assert len(init_elements) == len(animation.points) + sum(
        len(trail) for trail in animation.trails
    ) + (len(animation.projections) * 3 if animation.with_projections else 0)


# Test for updating animation frames
def test_update(mocker):
    animation = NBodyAnimation(
        csv_file_path="dummy.csv",
        num_dimensions=3,
        duration=30,
        trail_length=100,
        with_projections=True,
    )

    # Mock data for the test
    animation.num_bodies = 3
    animation.positions = np.array(
        [
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            [[0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [2.1, 2.1, 2.1]],
            [[0.2, 0.2, 0.2], [1.2, 1.2, 1.2], [2.2, 2.2, 2.2]],
        ]
    )
    animation.load_data = mocker.Mock()
    animation.load_data()

    updated_elements = animation.update(1)

    assert len(updated_elements) == len(animation.points) + sum(
        len(trail) for trail in animation.trails
    ) + (len(animation.projections) * 3 if animation.with_projections else 0)
