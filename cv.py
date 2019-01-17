import json
import time
import signal

import click
import cv2
import lifxlan
import numpy as np
import colorsys


def timed(t, fn, *args, **kwargs):
    start_time = time.time()
    v = fn(*args, **kwargs)
    print(t, time.time() - start_time)
    return v


def debug_frame(frame, channel, context):
    if context['debug']:
        if frame.shape != (context['height'], context['width'], 3):
            frame = cv2.resize(frame, (context['width'], context['height']), interpolation=cv2.INTER_CUBIC)
        context[channel].write(frame)

def ghetto_affine(img, context):
    if context['no_affine']:
        return img

    width = context['width']
    height = context['height']
    tl_factor = context['top_left_factor']
    tr_factor = context['top_right_factor']
    bl_factor = context['bottom_left_factor']
    br_factor = context['bottom_right_factor']

    tl = [int(width * context['top_left_factor']['w']),     int(height * context['top_left_factor']['h'])]
    tr = [int(width * context['top_right_factor']['w']),    int(height * context['top_right_factor']['h'])]
    bl = [int(width * context['bottom_left_factor']['w']),  int(height * context['bottom_left_factor']['h'])]
    br = [int(width * context['bottom_right_factor']['w']), int(height * context['bottom_right_factor']['h'])]

    source_points = np.array([tl, tr, bl, br], dtype=np.float32)
    target_points = np.array(
        [
            [0, 0],
            [int(width - 1), 0],
            [0, int(height -1)],
            [int(width - 1), int(height - 1)]
        ],
        dtype=np.float32)

    transform = cv2.getPerspectiveTransform(source_points, target_points)
    demo = cv2.warpPerspective(img, transform, (int(width), int(height)))
    debug_frame(demo, 'out_a', context)

    return demo


def ghetto_masks(img, context):
    num_zones = context['num_zones']
    max_band_size = context['max_band_size']


    reg_height = int(context['height'] * (2 * max_band_size))
    reg_width = int(context['width'] / num_zones)

    if context['debug']:
        print(reg_height, reg_width)

    extra = context['width'] - (reg_width * num_zones)
    first = int(extra / 2)
    last = int(extra / 2)

    for k in range(num_zones):
        mask = np.zeros((context['height'], context['width'], 1), np.uint8)

        tl_ = (first + (reg_width * k), 0)
        br_ = (first + (reg_width * (k + 1)), reg_height)

        if k == 0:
            for i in range(first):
                for j in range(reg_height):
                    mask[j][i] = 1
            tl_ = (0, 0)
            br_ = (first + reg_width, reg_height)

        mask[0:reg_height, first + (reg_width * k): first + (reg_width * k + reg_height)] = 1
        # for i in range(reg_width):
        #     for j in range(reg_height):
        #         mask[j][first + (reg_width * k + i)] = 1

        if k == num_zones - 1:
            for i in range(last):
                for j in range(reg_height):
                    mask[j][first + (reg_width * (k + 1) + i)] = 1
            br_ = (first + last + (reg_width * (k + 1)), reg_height)

        val_m = cv2.mean(img, mask=mask)

        cv2.rectangle(img, tl_, br_, color=val_m, thickness=-1)

    debug_frame(img, 'out_m', context)

    return img


def ghetto_crop(img, context):
    if context['no_crop']:
        return img

    no_subs_l = int(.20 * context['width'])
    no_subs_r = int(.80 * context['width'])
    mask = np.zeros(context['width'], np.uint8)
    mask[np.r_[0:no_subs_l, no_subs_r:context['width']]] = 1
    removed = img[~(np.average(img, weights=mask, axis=1) <= [20, 20, 20]).all(axis=1)]
    if context['debug']:
        print(removed.shape)
        debug_frame(removed, 'out_b', context)
    return removed


@click.command()
@click.option('--test-file', default=None, type=str)
@click.option('--no-affine/--affine', default=False)
@click.option('--no-crop/--crop', default=False)
@click.option('--debug/--no-debug', default=False)
@click.option('--log-time/--no-log-time', default=False)
@click.option('--config', type=click.Path())
def main(test_file, no_affine, no_crop, debug, log_time, config):
    with open(config) as fp:
        factors = json.load(fp)

    context = {
        **factors,
        **{
            'debug': debug,
            'no_affine': no_affine,
            'no_crop': no_crop,
            'log_time': log_time,
        }
    }

    stop = {'stop': False}

    def handler(signum, frame):
        stop['stop'] = True

    signal.signal(signal.SIGINT, handler)

    print('Starting capture')

    if test_file:
        class IM:
            def read(self):
                return True, cv2.imread(test_file)
            def release(self):
                pass

        cam = IM()

        context['width'] = 1920
        context['height'] = 1080
        context['width_float'] = 1920.0
        context['height_float'] = 1080.0
    else:
        cam = cv2.VideoCapture(0)   # 0 -> index of camera
        cam.set(3, context['camera']['w'])
        cam.set(4, context['camera']['h'])
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        context['width'] = int(cam.get(3))
        context['height'] = int(cam.get(4))
        context['width_float'] = cam.get(3)
        context['height_float'] = cam.get(4)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    if context['debug']:
        print(context['width'])
        print(context['height'])
        print(fourcc)

        context['out_o'] = cv2.VideoWriter('/tmp1/cam-test-orig.avi', fourcc, 30, (context['width'], context['height']))
        context['out_a'] = cv2.VideoWriter('/tmp1/cam-test-affine.avi', fourcc, 30, (context['width'], context['height']))
        context['out_m'] = cv2.VideoWriter('/tmp1/cam-test-masked.avi', fourcc, 30, (context['width'], context['height']))
        context['out_b'] = cv2.VideoWriter('/tmp1/cam-test-bars.avi', fourcc, 30, (context['width'], context['height']))
        context['out_g'] = cv2.VideoWriter('/tmp1/cam-test-grey.avi', fourcc, 30, (context['width'], context['height']))
        context['out_t'] = cv2.VideoWriter('/tmp1/cam-test-thresh.avi', fourcc, 30, (context['width'], context['height']))
        context['out_c'] = cv2.VideoWriter('/tmp1/cam-test-countour.avi', fourcc, 30, (context['width'], context['height']))
        context['out_co'] = cv2.VideoWriter('/tmp1/cam-test-countour-overlay.avi', fourcc, 30, (context['width'], context['height']))
        context['out'] = cv2.VideoWriter('/tmp1/cam-test.avi', fourcc, 30, (context['width'], context['height']))

    lan = lifxlan.LifxLAN(26)
    bias = lan.get_device_by_name('TV Bias')
    bias.set_power(True)

    if context['debug']:
        print(bias)

    while True:
        try:
            start_time = time.time()

            read, img = cam.read()

            if not read:
                break

            if context['debug']:
                cv2.imwrite("/tmp1/cam-test.png", img)
                debug_frame(img, 'out_o', context)

            demo = timed('ghetto_affine', ghetto_affine, img, context)

            removed = timed('ghetto_crop', ghetto_crop, demo, context)

            removed = timed('ghetto_masks', ghetto_masks, removed, context)

            # shrink = cv2.resize(removed, (context['num_zones'], 1), interpolation=cv2.INTER_NEAREST)
            # shrink = cv2.resize(removed.row(0), (context['num_zones'], 1), interpolation=cv2.INTER_NEAREST)
            shrink = removed[5, 10::(context['width']//context['num_zones'] + 1)]
            if context['debug']:
                print(shrink)
                print(shrink.shape)

            hsv = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255) for [b, g, r] in shrink.tolist()]

            lifx_hsv = [[h*257*255, s*257*255, v*257*255, 3500] for [h, s, v] in hsv]

            if context['debug']:
                print(lifx_hsv)

            try:
                timed('set_zone_colors', bias.set_zone_colors, lifx_hsv, duration=200, rapid=True)
            except Exception as e:
                print(e)

            if context['debug']:
                final = cv2.resize(np.expand_dims(shrink, axis=0), (context['width'], context['height']), interpolation=cv2.INTER_NEAREST)
                debug_frame(final, 'out', context)

            if context['log_time']:
                print(time.time() - start_time)
        except Exception as e:
            print(e)

        if stop['stop']:
            break
        time.sleep(.1)

    cam.release()

    if context['debug']:
        context['out'].release()
        context['out_o'].release()
        context['out_a'].release()
        context['out_m'].release()
        context['out_b'].release()
        context['out_g'].release()
        context['out_t'].release()
        context['out_c'].release()
        context['out_co'].release()


if __name__ == '__main__':
    main()
