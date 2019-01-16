import time
import signal

import click
import cv2
import lifxlan
import numpy as np
import colorsys

@click.command()
@click.option('--test-file', default=None, type=str)
@click.option('--no-affine/--affine', default=False)
@click.option('--no-crop/--crop', default=False)
@click.option('--debug/--no-debug', default=False)
def main(test_file, no_affine, no_crop, debug):
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

        w = 1920
        h = 1080
    else:
        cam = cv2.VideoCapture(0)   # 0 -> index of camera
        cam.set(3, 1920)
        cam.set(4, 1080)
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        w = cam.get(3)
        h = cam.get(4)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    if debug:
        print(w)
        print(h)
        print(fourcc)

        out_o = cv2.VideoWriter('/tmp1/cam-test-orig.avi', fourcc, 30, (int(w), int(h)))
        out_a = cv2.VideoWriter('/tmp1/cam-test-affine.avi', fourcc, 30, (int(w), int(h)))
        out_m = cv2.VideoWriter('/tmp1/cam-test-masked.avi', fourcc, 30, (int(w), int(h)))
        out_b = cv2.VideoWriter('/tmp1/cam-test-bars.avi', fourcc, 30, (int(w), int(h)))
        out_g = cv2.VideoWriter('/tmp1/cam-test-grey.avi', fourcc, 30, (int(w), int(h)))
        out_t = cv2.VideoWriter('/tmp1/cam-test-thresh.avi', fourcc, 30, (int(w), int(h)))
        out_c = cv2.VideoWriter('/tmp1/cam-test-countour.avi', fourcc, 30, (int(w), int(h)))
        out_co = cv2.VideoWriter('/tmp1/cam-test-countour-overlay.avi', fourcc, 30, (int(w), int(h)))
        out = cv2.VideoWriter('/tmp1/cam-test.avi', fourcc, 30, (int(w), int(h)))

    lan = lifxlan.LifxLAN(26)
    bias = lan.get_device_by_name('TV Bias')
    bias.set_power(True)

    if debug:
        print(bias)

    while True:
        read, img = cam.read()
        if not read:
            break
        if debug:
            cv2.imwrite("/tmp1/cam-test.png", img)

        # Perspective transform
        # .571 -> 365 / 640
        # .782 -> 500 / 640
        # .160 -> 77 / 480
        # .187 -> 90 / 480
        # .417 -> 200 / 480
        # .647 -> 310 / 480

            out_o.write(img)


        if not no_affine:
            # tl = [int(w * .545), int(h * .168)]
            # tr = [int(w * .710), int(h * .185)]
            # bl = [int(w * .550), int(h * .445)]
            # br = [int(w * .705), int(h * .650)]
            tl = [int(w * .29125), int(h * .13889)]
            tr = [int(w * .66667), int(h * .13889)]
            bl = [int(w * .31100), int(h * .50926)]
            br = [int(w * .65104), int(h * .50000)]
            source_points = np.array([tl, tr, bl, br], dtype=np.float32)
            target_points = np.array([[0, 0], [int(w - 1), 0], [0, int(h -1)], [int(w - 1), int(h - 1)]], dtype=np.float32)

            transform = cv2.getPerspectiveTransform(source_points, target_points)
            demo = cv2.warpPerspective(img, transform, (int(w), int(h)))
            if debug:
                out_a.write(demo)
        else:
            demo = img

        # black removal
        # if False:
        #     gray = cv2.cvtColor(demo, cv2.COLOR_BGR2GRAY)
        #     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        #     cx,cy,cw,ch = cv2.boundingRect(thresh)
        #     crop = demo[cy:cy+ch,cx:cx+cw]
        #     removed = cv2.resize(crop, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
        #     out_b.write(removed)
        # else:
        #     removed = demo

        if not no_crop:
            # mask = demo>10
            # print(type(mask))
            # print(type(mask.any(1)))
            # removed = demo[np.ix_(mask.any(1), mask.any(0))]

            # gray = cv2.cvtColor(demo, cv2.COLOR_BGR2GRAY)
            # out_g.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            # _, thresh = cv2.threshold(gray, 13.0, 255, cv2.THRESH_BINARY)
            # out_t.write(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))
            # contours_im, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # out_c.write(cv2.cvtColor(contours_im, cv2.COLOR_GRAY2BGR))
            # contours = sorted(contours, key=lambda contour: len(contour), reverse=True)
            # roi = cv2.boundingRect(contours[0])
            # out_co.write(cv2.drawContours(demo.copy(), [contours[0]], 0, (0, 255, 0), 3))
            # print(roi)
            # removed = demo[roi[1]:roi[3], roi[0]:roi[2]]
            # removed = cv2.resize(removed, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
            # out_b.write(removed)

            removed = demo[~(np.average(demo, axis=1) <= [20, 20, 20]).all(axis=1)]
            if debug:
                print(removed.shape)
        else:
            removed = demo


        num_zones = 50
        max_band_size = .115


        reg_height = int(h * (2 * max_band_size))
        reg_width = int(w / num_zones)

        if debug:
            print(reg_height, reg_width)

        extra = w - (reg_width * num_zones)
        first = int(extra / 2)
        last = int(extra / 2)

        for k in range(num_zones):
            mask = np.zeros((int(h), int(w), 1), np.uint8)

            tl_ = (first + (reg_width * k), 0)
            br_ = (first + (reg_width * (k + 1)), reg_height)

            if k == 0:
                for i in range(first):
                    for j in range(reg_height):
                        mask[j][i] = 1
                tl_ = (0, 0)
                br_ = (first + reg_width, reg_height)
            for i in range(reg_width):
                for j in range(reg_height):
                    mask[j][first + (reg_width * k + i)] = 1
            if k == num_zones - 1:
                for i in range(last):
                    for j in range(reg_height):
                        mask[j][first + (reg_width * (k + 1) + i)] = 1
                br_ = (first + last + (reg_width * (k + 1)), reg_height)

            val_m = cv2.mean(removed, mask=mask)

            cv2.rectangle(removed, tl_, br_, color=val_m, thickness=-1)

        if debug:
            out_m.write(cv2.resize(removed, (int(w), int(h)), interpolation=cv2.INTER_CUBIC))

        shrink = cv2.resize(removed, (num_zones, 1), interpolation=cv2.INTER_NEAREST)
        # hsv = cv2.cvtColor(shrink, cv2.COLOR_BGR2HSV)

        # lifx_hsv = [[h*257, s*257, v*257, 3500] for [h, s, v] in hsv[0].tolist()]
        # lifx_hsv = [[h*257, s*257, v*257, 3500] for [h, s, v] in hsv[0].tolist()]
        hsv = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255) for [b, g, r] in shrink[0].tolist()]

        lifx_hsv = [[h*257*255, s*257*255, v*257*255, 3500] for [h, s, v] in hsv]

        if debug:
            print(lifx_hsv)
        try:
            bias.set_zone_colors(lifx_hsv, duration=500)
        except:
            pass

        if debug:
            final = cv2.cvtColor(cv2.resize(shrink, (int(w), int(h)), interpolation=cv2.INTER_NEAREST), cv2.COLOR_HSV2BGR)


        # print('Warped')
        # print(demo.shape)



        # contours, hierarchy, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[0]
        # cx,cy,cw,ch = cv2.boundingRect(cnt)
        # crop = demo[cy:cy+ch,cx:cx+cw]

        # print('Cropped')
        # print(crop.shape)
        # final = cv2.resize(crop, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
        # print('Resized')
        # print(final.shape)

            out.write(final)

        if stop['stop']:
            break

    cam.release()
    if debug:
        out.release()
        out_o.release()
        out_a.release()
        out_m.release()
        out_b.release()
        out_g.release()
        out_t.release()
        out_c.release()
        out_co.release()

if __name__ == '__main__':
    main()
