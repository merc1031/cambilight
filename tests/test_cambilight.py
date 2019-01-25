import cv2
import numpy as np
import pytest

import cambilight.cambilight


def point_in_triangle(p, p0, p1, p2):
    def area(pp0, pp1, pp2):
        return ((pp0[0] * (pp1[1] - pp2[1]) + pp1[0] * (pp2[1] - pp0[1]) + pp2[0] * (pp0[1] - pp1[1]) ) / 2.0 )

    area_outer = area(p0, p1, p2)

    area_0 = area(p, p1, p2)
    area_1 = area(p0, p, p2)
    area_2 = area(p0, p1, p)
    if any(map(lambda x: x < 0, [area_0, area_1, area_2])):
        return False
    if any(map(lambda x: x == 0, [area_0, area_1, area_2])):
        return True

    return area_outer == (area_0 + area_1 + area_2)


@pytest.fixture(autouse=True)
def mock_observer(mocker):
    return mocker.patch('watchdog.observers.Observer')


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
        [[mk_pixel(x, y) for x in cols] for y in rows]
    ))


@pytest.fixture
def mk_pixel():
    return lambda x, y: [255, 255, 255]


@pytest.fixture
def ret():
    return True


@pytest.fixture()
def basic_config(nrows, ncols):
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
        'width': ncols,
        'height': nrows,
        "lights": [
            {
                "name": "TV Bias",
                "geometries": [
                    {
                        "screen_space": {
                            "top_left": {
                                "x": 0.0,
                                "y": 0.0
                            },
                            "bottom_right": {
                                "x": 1.0,
                                "y": 0.23
                            }
                        },
                        "orientation": "LtR",
                        "zone_order_reversed": True,
                        "zones": {
                            "start": 0,
                            "end": 50
                        },
                        "resolution": 50
                    }
                ]
            }
        ]
    }


def describe_Cambilight():
    def calls_external(basic_config, mock_lifxlan, mock_camera, mock_sleep):
        cam = cambilight.cambilight.Cambilight(basic_config, 'config.json')
        cam.stop = True

        cam.inner_main()

        mock_lifxlan.return_value.get_device_by_name.assert_called_once_with('TV Bias')
        mock_lifxlan.return_value.get_device_by_name.return_value.set_power.assert_called_once_with(True)
        mock_camera.return_value.read.assert_called_once_with()
        # mock_sleep.assert_called_once_with(.1)
        mock_camera.return_value.release.assert_called_once_with()

    def describe_cv_hsv_to_lifx_hsbk():
        @pytest.fixture
        def ncols():
            return 10

        @pytest.fixture
        def nrows():
            return 10

        @pytest.fixture
        def mk_pixel(ncols, nrows):
            def m(x, y):
                return [x * 18, 255, 0]
            return m

        def conversion(ncols, nrows, data):
            stride = (18 * (1 / 180) * 256 * (256 + 1))
            exp = np.array([
                [[int(y * stride), 65535, 0, 3500] for y in range(ncols)]
                for _ in range(nrows)])

            res = cambilight.cambilight.cv_hsv_to_lifx_hsbk(data[1])
            assert np.allclose(res, exp)

    def describe_remove_bars():
        def context_has_black_bars():
            @pytest.fixture
            def ncols():
                return 10

            @pytest.fixture
            def nrows():
                return 10

            @pytest.fixture
            def mk_pixel(ncols, nrows):
                def m(x, y):
                    if y < int(nrows * .20) or y >= int(nrows * .80):
                        return [0, 0, 0]
                    else:
                        return [255, 255, 255]
                return m

            def remove_bars(basic_config, mock_lifxlan, mock_camera, mock_sleep, data):
                res = cambilight.cambilight.ghetto_crop(data[1], basic_config)
                safe_zone = cambilight.cambilight.safe_zone(basic_config)
                s, e = (min(2, safe_zone[0]), max(8, safe_zone[1]))

                assert res.shape == data[1][s:e].shape
                assert np.allclose(res, data[1][s:e])

        def context_no_black_bars():
            def no_remove_bars(basic_config, mock_lifxlan, mock_camera, mock_sleep, data):
                res = cambilight.cambilight.ghetto_crop(data[1], basic_config)
                assert res.shape == data[1].shape
                assert np.allclose(res, data[1])

        def context_has_artifacts():
            @pytest.fixture
            def ncols():
                return 10

            @pytest.fixture
            def nrows():
                return 10

            @pytest.fixture
            def mk_pixel(ncols, nrows):
                def m(x, y):
                    if (y < int(nrows * .20) or y >= int(nrows * .80))\
                            and (x < int(ncols * .25) or x > int(ncols * .75)):
                        return [0, 0, 0]
                    else:
                        return [255, 255, 255]
                return m

            def remove_bars(basic_config, mock_lifxlan, mock_camera, mock_sleep, data):
                res = cambilight.cambilight.ghetto_crop(data[1], basic_config)
                safe_zone = cambilight.cambilight.safe_zone(basic_config)
                s, e = (min(2, safe_zone[0]), max(8, safe_zone[1]))

                assert res.shape == data[1][s:e].shape
                assert np.allclose(res, data[1][s:e])

        def context_dark_image_has_artifacts():
            @pytest.fixture
            def ncols():
                return 10

            @pytest.fixture
            def nrows():
                return 10

            @pytest.fixture
            def mk_pixel(ncols, nrows):
                def m(x, y):
                    if (y < int(nrows * .20) or y >= int(nrows * .80))\
                            and (x < int(ncols * .25) or x > int(ncols * .75)):
                        return [0, 0, 0]
                    else:
                        if y == 5:
                            return [0, 0, 0]
                        else:
                            return [255, 255, 255]
                return m

            def remove_bars(basic_config, mock_lifxlan, mock_camera, mock_sleep, data):
                res = cambilight.cambilight.ghetto_crop(data[1], basic_config)
                safe_zone = cambilight.cambilight.safe_zone(basic_config)
                s, e = (min(2, safe_zone[0]), max(8, safe_zone[1]))

                assert res.shape == data[1][s:e].shape
                assert np.allclose(res, data[1][s:e])

    def describe_affine():
        @pytest.fixture
        def ncols():
            return 100

        @pytest.fixture
        def nrows():
            return 100

        @pytest.fixture
        def mk_pixel(ncols, nrows, affine_factors, colors):
            def m(x, y):
                (tl, tr, bl, br) = cambilight.cambilight.to_corners(affine_factors)

                if point_in_triangle([x, y], tl, tr, bl) or point_in_triangle([x, y], bl, tr, br):
                    return colors(x, y)
                else:
                    return [0, 0, 0]
            return m

        @pytest.fixture
        def affine_factors(basic_config):
            return {**basic_config,
                    **{
                'top_left_factor': {
                    'w': .3,
                    'h': .3,
                },
                'top_right_factor': {
                    'w': .7,
                    'h': .2,
                },
                'bottom_left_factor': {
                    'w': .4,
                    'h': .9,
                },
                'bottom_right_factor': {
                    'w': .6,
                    'h': .8,
                },
            }}

        @pytest.fixture
        def colors(affine_factors):
            (tl, tr, bl, br) = cambilight.cambilight.to_corners(affine_factors)
            def from_coord(x, y):
                return {
                    tuple(tl): [255, 0, 0],
                    tuple(tr): [255, 255, 0],
                    tuple(bl): [0, 255, 0],
                    tuple(br): [0, 255, 255],
                }.get((x, y), [255, 255, 255])
            return from_coord

        def remove_bars(affine_factors, mock_lifxlan, mock_camera, mock_sleep, data, colors):
            res = cambilight.cambilight.ghetto_affine(data[1].astype(np.uint8), affine_factors)

            (tl, tr, bl, br) = cambilight.cambilight.to_corners(affine_factors)
            (tlc, trc, blc, brc) = tuple(map(lambda c: colors(c[0], c[1]), [tl, tr, bl, br])) 

            assert res.shape == data[1].shape
            assert np.array_equal(res[0,0], tlc)
            assert np.array_equal(res[0,affine_factors['height'] - 1], trc)
            assert np.array_equal(res[affine_factors['width'] - 1,0], blc)
            assert np.array_equal(res[affine_factors['width'] - 1,affine_factors['height'] - 1], brc)
