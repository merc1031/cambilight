import time
import signal

import cv2
import lifxlan
import numpy as np

stop = {'stop': False}

def handler(signum, frame):
    stop['stop'] = True

signal.signal(signal.SIGINT, handler)
if __name__ == '__main__':
    print('Starting capture')
    cam = cv2.VideoCapture(0)   # 0 -> index of camera
    cam.set(3, 1920)
    cam.set(4, 1080)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    w = cam.get(3)
    h = cam.get(4)
    print(w)
    print(h)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    print(fourcc)
    out_o = cv2.VideoWriter('/tmp1/cam-test-orig.avi', fourcc, 30, (int(w), int(h)))
    out = cv2.VideoWriter('/tmp1/cam-test.avi', fourcc, 30, (int(w), int(h)))
    lan = lifxlan.LifxLAN(26)
    bias = lan.get_device_by_name('TV Bias')
    bias.set_power(True)
    print(bias)
    while True:
        read, img = cam.read()
        if not read:
            break

        # Perspective transform
        # .571 -> 365 / 640
        # .782 -> 500 / 640
        # .160 -> 77 / 480
        # .187 -> 90 / 480
        # .417 -> 200 / 480
        # .647 -> 310 / 480

        tl = [int(w * .545), int(h * .168)]
        tr = [int(w * .710), int(h * .185)]
        bl = [int(w * .550), int(h * .445)]
        br = [int(w * .705), int(h * .650)]
        source_points = np.array([tl, tr, bl, br], dtype=np.float32)
        target_points = np.array([[0, 0], [int(w - 1), 0], [0, int(h -1)], [int(w - 1), int(h - 1)]], dtype=np.float32)

        transform = cv2.getPerspectiveTransform(source_points, target_points)
        demo = cv2.warpPerspective(img, transform, (int(w), int(h)))

        # black removal
        gray = cv2.cvtColor(demo, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        cx,cy,cw,ch = cv2.boundingRect(thresh)
        crop = demo[cy:cy+ch,cx:cx+cw]
        removed = cv2.resize(crop, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)

        out_o.write(removed)

        num_zones = 50
        max_band_size = .115


        reg_height = int(h * (2 * max_band_size))
        reg_width = int(w / num_zones)

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

        shrink = cv2.resize(removed, (num_zones, 1), interpolation=cv2.INTER_NEAREST)
        # hsv = cv2.cvtColor(shrink, cv2.COLOR_BGR2HSV)

        # lifx_hsv = [[h*257, s*257, v*257, 3500] for [h, s, v] in hsv[0].tolist()]
        # lifx_hsv = [[h*257, s*257, v*257, 3500] for [h, s, v] in hsv[0].tolist()]
        import colorsys
        hsv = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255) for [b, g, r] in shrink[0].tolist()]

        lifx_hsv = [[h*257*255, s*257*255, v*257*255, 3500] for [h, s, v] in hsv]

        print(lifx_hsv)
        bias.set_zone_colors(lifx_hsv, duration=500)
        time.sleep(.300)

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

        #cv2.imwrite("/tmp1/cam-test.png", img)
        out.write(final)

        if stop['stop']:
            break

cam.release()
out.release()

