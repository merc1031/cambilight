import numpy as np
import pytest

import cambilight.cambilight


@pytest.fixture(autouse=True)
def mock_lifxlan(mocker):
    return mocker.patch('lifxlan.LifxLAN')


@pytest.fixture(autouse=True)
def mock_camera(mocker, data):
    patched = mocker.patch('cv2.VideoCapture')
    patched.return_value.read.side_effect = data
    return patched


@pytest.fixture(autouse=True)
def mock_sleep(mocker):
    patched = mocker.patch('time.sleep')
    return patched


@pytest.fixture
def ncols():
    return 640


@pytest.fixture
def nrows():
    return 480


@pytest.fixture
def data(ncols, nrows, mk_pixel, ret):
    cols = range(ncols)
    rows = range(nrows)
    return (ret, np.array(
        [[mk_pixel(x, y) for y in cols] for x in rows]
    ))


@pytest.fixture
def mk_pixel():
    return lambda x, y: [255, 255, 255]


@pytest.fixture
def ret():
    return True


@pytest.fixture()
def basic_config():
    return {
        'top_left_factor': {
            'w': 0.29125,
            'h': 0.13889
        },
        'top_right_factor': {
            'w': 0.66667,
            'h': 0.13889
        },
        'bottom_left_factor': {
            'w': 0.31100,
            'h': 0.50926
        },
        'bottom_right_factor': {
            'w': 0.65104,
            'h': 0.50000
        },
        'num_zones': 50,
        'max_band_size': 0.115,
        'camera': {
            'w': 640,
            'h': 480
        },
        'debug': False,
        'no_affine': False,
        'no_crop': False,
        'log_time': False,
        'test_file': None,
    }


def describe_Cambilight():
    def calls_read(basic_config, mock_lifxlan, mock_camera, mock_sleep):
        cam = cambilight.cambilight.Cambilight(basic_config)
        cam.stop = True

        cam.inner_main()

        mock_lifxlan.return_value.get_device_by_name.assert_called_once_with('TV Bias')
        mock_lifxlan.return_value.get_device_by_name.return_value.set_power.assert_called_once_with(True)
        mock_camera.return_value.read.assert_called_once_with()
        mock_sleep.assert_called_once_with(.1)
        mock_camera.return_value.release.assert_called_once_with()
