import json
import os.path
import time
import signal
import traceback

import click
import cv2
import lifxlan
import numpy as np
import watchdog.observers
import watchdog.events


class MyHandler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self, target, file):
        super().__init__(patterns=[file])
        self.target = target

    def on_modified(self, event):
        print(f'event type: {event.event_type}  path : {event.src_path}')


def print_d(context, *args, **kwargs):
    if context['debug']:
        print(*args, **kwargs)


def debug_frame(frame, channel, context, unformatted_path=None):
    if context['debug']:
        if frame.shape != (context['height'], context['width'], 3):
            frame = cv2.resize(frame, (context['width'], context['height']), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        context[channel].write(frame)
        if unformatted_path is not None:
            cv2.imwrite(unformatted_path.format(channel=channel), frame)


def timed(t, fn, *args, **kwargs):
    start_time = time.time()
    v = fn(*args, **kwargs)
    if kwargs.get('context', {}).get('log_time', False):
        print(t, time.time() - start_time)
    return v


def timed_drop(t, fn, *args, **kwargs):
    start_time = time.time()
    log_time = kwargs.pop('context', {}).get('log_time', False)
    v = fn(*args, **kwargs)
    if log_time:
        print(t, time.time() - start_time)
    return v


def to_corners(context):
    width = context['width']
    height = context['height']
    tl_factor = context['top_left_factor']
    tr_factor = context['top_right_factor']
    bl_factor = context['bottom_left_factor']
    br_factor = context['bottom_right_factor']

    tl = [int(width * tl_factor['w']),     int(height * tl_factor['h'])]
    tr = [int(width * tr_factor['w']),    int(height * tr_factor['h'])]
    bl = [int(width * bl_factor['w']),  int(height * bl_factor['h'])]
    br = [int(width * br_factor['w']), int(height * br_factor['h'])]
    return (tl, tr, bl, br)


def ghetto_affine(img, context):
    if context['no_affine']:
        return img

    width = context['width']
    height = context['height']

    (tl, tr, bl, br) = to_corners(context)

    source_points = np.array([tl, tr, bl, br], dtype=np.float32)
    target_points = np.array(
        [
            [0, 0],
            [int(width - 1), 0],
            [0, int(height - 1)],
            [int(width - 1), int(height - 1)]
        ],
        dtype=np.float32)

    transform = cv2.getPerspectiveTransform(source_points, target_points)

    demo = cv2.warpPerspective(img, transform, (int(width), int(height)))

    debug_frame(demo, 'affine', context, unformatted_path='/tmp1/cam-test-{channel}.png')

    return demo


def ghetto_masks(img_src, context):
    '''
    Context defines multiple lights
    Each light defines some number of geometries
    Each Geometry defines
        The screen space (x,y) -> (x1, y1) rectangle it occupies
        The orientation LtR, RtL, TtB, BtT
        The range of zones n -> m of the light we will occupy
        The resolution mapping of this leg to zones (2 means we only send 2 colors to the n->m zones range)
    '''
    shrinks = {}
    for l in context['lights']:
        img = np.copy(img_src)
        shrinks[l['name']] = []
        for g in l['geometries']:
            save = {}
            num_zones = g['resolution']

            start_y = int(context['height'] * g['screen_space']['top_left']['y'])
            end_y = int(context['height'] * (
                g['screen_space']['bottom_right']['y'] - g['screen_space']['top_left']['y']))
            geom_height = end_y - start_y
            reg_height = geom_height

            start_x = int(context['width'] * g['screen_space']['top_left']['x'])
            end_x = int(context['width'] * (
                g['screen_space']['bottom_right']['x'] - g['screen_space']['top_left']['x']))
            geom_width = end_x - start_x

            reg_width = int(geom_width / num_zones)

            if context['debug']:
                print(reg_height, reg_width)

            extra = geom_width - (reg_width * num_zones)
            first = int(extra / 2)
            last = int(extra / 2)

            for k in range(num_zones):
                mask = np.zeros((context['height'], context['width'], 1), np.uint8)

                tl_ = (start_x + first + (reg_width * k), start_y)
                br_ = (start_x + first + (reg_width * (k + 1)), reg_height)

                if k == 0:
                    mask[start_y:reg_height, start_x:start_x + first] = 1
                    tl_ = (start_x, start_y)
                    br_ = (first + reg_width, reg_height)
                if k == num_zones - 1:
                    mask[
                        start_y:reg_height,
                        (first + start_x + (reg_width * k)):(first + last + start_x + (reg_width * (k + 1)))] = 1
                    br_ = (first + last + (reg_width * (k + 1)), reg_height)
                mask[
                    start_y:reg_height,
                    first + start_x + (reg_width * k): first + start_x + (reg_width * (k + 1))] = 1

                val_m = cv2.mean(img, mask=mask)

                cv2.rectangle(img, tl_, br_, color=val_m, thickness=-1)

            debug_frame(img, 'masked', context, unformatted_path='/tmp1/cam-test-{channel}.png')

            center_one = start_x + (((first + reg_width) + (reg_width // 2)))
            start = center_one - reg_width

            end_gap = context['width'] - end_x
            center_last = end_gap - (last + reg_width + (reg_width // 2))
            end = center_last + reg_width

            print_d(context, 'width', context['width'])
            print_d(context, 'zones', context['num_zones'])
            criteria = np.r_[
                start,
                center_one:
                center_last:
                reg_width,
                end
            ]
            shrink = np.copy(img[5, criteria])
            save['geom'] = g.copy()
            save['shrink'] = shrink
            shrinks[l['name']].append(save)

    return shrinks


def ghetto_crop(img, context):
    if context['no_crop']:
        return img

    # Only check outside to avoid subtitles
    no_subs_l = int(.20 * context['width'])
    no_subs_r = int(.80 * context['width'])
    mask = np.zeros(context['width'], np.uint8)
    mask[np.r_[0:no_subs_l, no_subs_r:context['width']]] = 1

    # Now defend from eliminating dark scenes
    start_of_safe, end_of_safe = safe_zone(context)
    safe_mask = np.zeros(context['height'], np.uint8)
    safe_mask[start_of_safe:end_of_safe] = 1

    criteria = img[:, :, 2]

    removed = img[np.logical_or(
        ~(np.average(criteria, weights=mask, axis=1) <= 20),
        safe_mask
    )
    ]

    if context['debug']:
        print(removed.shape)
        debug_frame(removed, 'bars', context, unformatted_path='/tmp1/cam-test-{channel}.png')
    return removed


def safe_zone(context):
    start_of_safe = int(context['height'] * context['max_band_size'])
    end_of_safe = int((1 - context['max_band_size']) * context['height'])
    return (start_of_safe, end_of_safe)


@click.command()
@click.option('--test-file', default=None, type=str)
@click.option('--no-affine/--affine', default=False)
@click.option('--no-crop/--crop', default=False)
@click.option('--debug/--no-debug', default=False)
@click.option('--lifx-debug/--no-lifx-debug', default=False)
@click.option('--log-time/--no-log-time', default=False)
@click.option('--config', type=click.Path())
def main(test_file, no_affine, no_crop, debug, lifx_debug, log_time, config):
    with open(config) as fp:
        factors = json.load(fp)

    context = {
        **factors,
        **{
            'debug': debug,
            'no_affine': no_affine,
            'no_crop': no_crop,
            'log_time': log_time,
            'test_file': test_file,
            'lifx_debug': lifx_debug,
        }
    }

    return Cambilight(context, config).inner_main()


def cv_hsv_to_lifx_hsbk(hsv):
    """
    OpenCV HSV
    H            S           V
    0-179        0-255       0-255

    LIFX HSBK
    H            S           B          K
    0-65535      0-65535     0-65535    2500-9000
    """

    new_shape = hsv.shape[:-1] + (hsv.shape[-1] + 1,)
    hsbk = np.empty(new_shape, dtype=np.uint16)
    hsbk[..., :3] = hsv
    hsbk[..., 3] = 3500

    return (hsbk * [(1 / 180) * 256 * (256 + 1), 256 + 1, 256 + 1, 1]).astype(dtype=np.uint16)


class Cambilight:
    def __init__(self, context, file):
        self.context = context
        self.stop = False
        self.json_file = file

    def inner_main(self):

        print(os.path.dirname(self.json_file))
        event_handler = MyHandler(self, self.json_file)
        observer = watchdog.observers.Observer()
        observer.schedule(event_handler, path=os.path.dirname(self.json_file), recursive=False)
        observer.start()

        def handler(signum, frame):
            self.stop = True

        signal.signal(signal.SIGINT, handler)

        print('Starting capture')

        if self.context['test_file']:
            class IM:
                def read(self):
                    return True, cv2.imread(self.context['test_file'])

                def release(self):
                    pass

            cam = IM()

            self.context['width'] = 1920
            self.context['height'] = 1080
            self.context['width_float'] = 1920.0
            self.context['height_float'] = 1080.0
        else:
            cam = cv2.VideoCapture(0)   # 0 -> index of camera
            print_d(self.context, 'asking camera to set', self.context['camera'])
            cam.set(3, self.context['camera']['w'])
            cam.set(4, self.context['camera']['h'])
            cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            self.context['width'] = int(cam.get(3))
            self.context['height'] = int(cam.get(4))
            self.context['width_float'] = cam.get(3)
            self.context['height_float'] = cam.get(4)
            print_d(self.context, 'camera context', self.context['width'])
            print_d(self.context, 'camera context', self.context['height'])

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        if self.context['debug']:
            print_d(self.context, 'fourcc', fourcc)

            self.context['original'] = cv2.VideoWriter(
                '/tmp1/cam-test-original.avi',
                fourcc,
                30,
                (self.context['width'], self.context['height'])
            )
            self.context['affine'] = cv2.VideoWriter(
                '/tmp1/cam-test-affine.avi',
                fourcc,
                30,
                (self.context['width'], self.context['height'])
            )
            self.context['masked'] = cv2.VideoWriter(
                '/tmp1/cam-test-masked.avi',
                fourcc,
                30,
                (self.context['width'], self.context['height'])
            )
            self.context['bars'] = cv2.VideoWriter(
                '/tmp1/cam-test-bars.avi',
                fourcc,
                30,
                (self.context['width'], self.context['height'])
            )
            self.context['final'] = cv2.VideoWriter(
                '/tmp1/cam-test-final.avi',
                fourcc,
                30,
                (self.context['width'], self.context['height'])
            )

        lan = lifxlan.LifxLAN(26, verbose=self.context.get('lifx_debug', False))
        for l in self.context['lights']:
            bias = lan.get_device_by_name(l['name'])
            bias.set_power(True)
            if self.context['debug']:
                color_zones = bias.get_color_zones()

                print_d(self.context, 'bias light', bias)
                print_d(self.context, 'bias light zones', color_zones)
                print_d(self.context, 'bias light zone count', len(color_zones))
            l['instance'] = bias
            l['color_zones'] = color_zones

        last_time = 16
        while True:
            try:
                start_time = time.time()

                read, img = cam.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                if not read:
                    break

                if self.context['debug']:
                    debug_frame(img, 'original', self.context, unformatted_path='/tmp1/cam-test-{channel}.png')

                demo = timed('ghetto_affine', ghetto_affine, img, context=self.context)

                removed = timed('ghetto_crop', ghetto_crop, demo, context=self.context)

                all_shrinks = timed('ghetto_masks', ghetto_masks, removed, context=self.context)

                for l in self.context['lights']:

                    shrinks = all_shrinks[l['name']]
                    for i, shrink in enumerate(shrinks):
                        app = False
                        if i == len(shrinks) - 1:
                            app = True

                        lifx_hsv = cv_hsv_to_lifx_hsbk(shrink['shrink'])
                        if shrink['geom']['zone_order_reversed']:
                            lifx_hsv = np.flip(lifx_hsv, axis=0)

                        print_d(self.context, 'hsv', lifx_hsv)

                        actual_num_zones = shrink['geom']['zones']['end'] - shrink['geom']['zones']['start']
                        gap = actual_num_zones // shrink['geom']['resolution']
                        try:
                            for (i, color) in enumerate(lifx_hsv):
                                timed_drop(
                                    'set_zone_colors',
                                    l['instance'].set_zone_color,
                                    i,
                                    i + gap,
                                    color,
                                    duration=last_time,
                                    rapid=True,
                                    apply=app,
                                    context=self.context)
                        except Exception as e:
                            print(e)
                            traceback.print_exc()

                        if self.context['debug']:
                            final = cv2.resize(
                                np.expand_dims(shrink, axis=0),
                                (self.context['width'], self.context['height']),
                                interpolation=cv2.INTER_NEAREST
                            )
                            debug_frame(final, 'final', self.context, unformatted_path='/tmp1/cam-test-{channel}.png')

                last_time = time.time() - start_time
                if self.context['log_time']:
                    print('Full', last_time)
            except Exception as e:
                print(e)
                traceback.print_exc()

            if self.stop:
                break

        observer.stop()
        observer.join()
        cam.release()

        if self.context['debug']:
            self.context['final'].release()
            self.context['original'].release()
            self.context['affine'].release()
            self.context['masked'].release()
            self.context['bars'].release()
