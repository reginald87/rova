import math

import pytest

from apps.geography.services.geography_service import (
    InvalidCoordinateError,
    validate_coordinate,
)

pytestmark = pytest.mark.django_db


def test_valid_coordinate():
    point = validate_coordinate(6.5244, 3.3792)
    assert point.srid == 4326
    assert point.x == 3.3792  # longitude first in geometry order
    assert point.y == 6.5244


def test_boundary_coordinates_accepted():
    assert validate_coordinate(90.0, 180.0).srid == 4326
    assert validate_coordinate(-90.0, -180.0).srid == 4326
    assert validate_coordinate(0.0, 0.0).srid == 4326


@pytest.mark.parametrize("latitude", [91.0, -91.0, 90.5, -90.5, 200.0])
def test_invalid_latitude_rejected(latitude):
    with pytest.raises(InvalidCoordinateError):
        validate_coordinate(latitude, 3.0)


@pytest.mark.parametrize("longitude", [181.0, -181.0, 300.0, -300.0])
def test_invalid_longitude_rejected(longitude):
    with pytest.raises(InvalidCoordinateError):
        validate_coordinate(6.5, longitude)


def test_non_finite_values_rejected():
    with pytest.raises(InvalidCoordinateError):
        validate_coordinate(math.nan, 3.0)
    with pytest.raises(InvalidCoordinateError):
        validate_coordinate(6.5, math.inf)
    with pytest.raises(InvalidCoordinateError):
        validate_coordinate(float("inf"), 3.0)


def test_missing_coordinates_rejected() -> None:
    with pytest.raises(InvalidCoordinateError):
        validate_coordinate(None, 3.0)  # type: ignore[arg-type]
    with pytest.raises(InvalidCoordinateError):
        validate_coordinate(6.5, None)  # type: ignore[arg-type]
    with pytest.raises(InvalidCoordinateError):
        validate_coordinate("6.5", 3.0)  # type: ignore[arg-type]


def test_coordinates_never_clamped():
    with pytest.raises(InvalidCoordinateError):
        validate_coordinate(95.0, 3.0)
