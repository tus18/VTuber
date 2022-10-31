import cv2
import numpy as np

from ..backend.image import resize_image, convert_color_space, show_image
from ..backend.image import BGR2RGB

import dlib
from imutils import face_utils
from scipy.spatial import distance
import numpy as np
from PIL import Image

class Camera(object):
    """Camera abstract class.
    By default this camera uses the openCV functionality.
    It can be inherited to overwrite methods in case another camera API exists.
    """
    def __init__(self, device_id=0, name='Camera', intrinsics=None,
                 distortion=None):
        # TODO load parameters from camera name. Use ``load`` method.
        self.device_id = device_id
        self.name = name
        self.intrinsics = intrinsics
        self.distortion = None
        self._camera = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def intrinsics(self):
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, value):
        if value is None:
            value = np.zeros((4))
        self._intrinsics = value

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, distortion):
        self._distortion = distortion

    def start(self):
        """ Starts capturing device

        # Returns
            Camera object.
        """
        print("カメラスタート")
        self._camera = cv2.VideoCapture(self.device_id)
        if self._camera is None or not self._camera.isOpened():
            raise ValueError('Unable to open device', self.device_id)
        return self._camera

    def stop(self):
        """ Stops capturing device.
        """
        return self._camera.release()

    def read(self):
        """Reads camera input and returns a frame.

        # Returns
            Image array.
        """
        frame = self._camera.read()[1]
        return frame

    def is_open(self):
        """Checks if camera is open.

        # Returns
            Boolean
        """
        return self._camera.isOpened()

    def calibrate(self):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError

    def intrinsics_from_HFOV(self, HFOV=70, image_shape=None):
        """Computes camera intrinsics using horizontal field of view (HFOV).

        # Arguments
            HFOV: Angle in degrees of horizontal field of view.
            image_shape: List of two floats [height, width].

        # Returns
            camera intrinsics array (3, 3).

        # Notes:

                       \           /      ^
                        \         /       |
                         \ lens  /        | w/2
        horizontal field  \     / alpha/2 |
        of view (alpha)____\( )/_________ |      image
                           /( )\          |      plane
                          /     <-- f --> |
                         /       \        |
                        /         \       |
                       /           \      v

                    Pinhole camera model

        From the image above we know that: tan(alpha/2) = w/2f
        -> f = w/2 * (1/tan(alpha/2))

        alpha in webcams and phones is often between 50 and 70 degrees.
        -> 0.7 w <= f <= w
        """
        if image_shape is None:
            self.start()
            height, width = self.read().shape[0:2]
            self.stop()
        else:
            height, width = image_shape[:2]

        focal_length = (width / 2) * (1 / np.tan(np.deg2rad(HFOV / 2.0)))
        intrinsics = np.array([[focal_length, 0, width / 2.0],
                               [0, focal_length, height / 2.0],
                               [0, 0, 1.0]])
        self.intrinsics = intrinsics

    def take_photo(self):
        """Starts camera, reads buffer and returns an image.
        """
        self.start()
        image = self.read()
        # all pipelines start with RGB
        image = convert_color_space(image, BGR2RGB)
        self.stop()
        return image


class VideoPlayer(object):
    """Performs visualization inferences in a real-time video.

    # Properties
        image_size: List of two integers. Output size of the displayed image.
        pipeline: Function. Should take RGB image as input and it should
            output a dictionary with key 'image' containing a visualization
            of the inferences. Built-in pipelines can be found in
            ``paz/processing/pipelines``.

    # Methods
        run()
        record()
    """

    def __init__(self, image_size, pipeline, camera, topic='image'):
        self.image_size = image_size
        self.pipeline = pipeline
        self.camera = camera
        self.topic = topic

        #追記
        face_detector = dlib.get_frontal_face_detector()
        self.face_parts_detector = dlib.shape_predictor("D:\ドキュメント\openCV_program\emotion\shape_predictor_68_face_landmarks.dat")
        self.face_cascade = cv2.CascadeClassifier('D:\ドキュメント\openCV_program\emotion\haarcascade_frontalface_default.xml')
        img1="D:\ドキュメント\openCV_program\emotion\main.png"
        img2="D:\ドキュメント\openCV_program\emotion\close.png"
        img3="D:\ドキュメント\openCV_program\emotion\happy.png"
        img4="D:\ドキュメント\openCV_program\emotion\ang.png"
        self.cv_img1 = cv2.imread(img1,cv2.IMREAD_UNCHANGED)
        self.cv_img2 = cv2.imread(img2,cv2.IMREAD_UNCHANGED)
        self.cv_img3 = cv2.imread(img3,cv2.IMREAD_UNCHANGED)
        self.cv_img4 = cv2.imread(img4,cv2.IMREAD_UNCHANGED)
        self.x=0
        self.y=0
        self.w=0
        self.h=0

        #追記
    def overlayImage(self,src, overlay, location, size):
        overlay_height, overlay_width = overlay.shape[:2]

        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        pil_src = Image.fromarray(src)
        pil_src = pil_src.convert('RGBA')

        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
        pil_overlay = Image.fromarray(overlay)
        pil_overlay = pil_overlay.convert('RGBA')
        #顏の大きさに合わせてリサイズ
        pil_overlay = pil_overlay.resize(size)

        # 画像を合成
        pil_tmp = Image.new('RGBA', pil_src.size, (255, 255, 255, 0))
        pil_tmp.paste(pil_overlay, location, pil_overlay)
        result_image = Image.alpha_composite(pil_src, pil_tmp)

        # OpenCV形式に変換
        return cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)

        #追記
    def calc_ear(self,eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        eye_ear = (A + B) / (2.0 * C)
        return eye_ear

        #追記
    def face_landmark_find(self,img):
        eye = 10

        img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(img_gry, 1)

        for face in faces:
            landmark = self.face_parts_detector(img_gry, face)
            landmark = face_utils.shape_to_np(landmark)

            left_eye_ear = self.calc_ear(landmark[42:48])
            right_eye_ear = self.calc_ear(landmark[36:42])
            eye = (left_eye_ear + right_eye_ear) / 2.0

        return img,eye

    def step(self):
        """ Runs the pipeline process once

        # Returns
            Inferences from ``pipeline``.
        """
        if self.camera.is_open() is False:
            raise ValueError('Camera has not started. Call ``start`` method.')

        frame = self.camera.read()
        if frame is None:
            print('Frame: None')
            return None
        # all pipelines start with an RGB image
        frame = convert_color_space(frame, BGR2RGB)
        return self.pipeline(frame)

    def run(self):
        """Opens camera and starts continuous inference using ``pipeline``,
        until the user presses ``q`` inside the opened window.
        """
        print("ビデオスタート")
        self.camera.start()
        while True:
            output = self.step()
            if output is None:
                continue
            image = resize_image(output[self.topic], tuple(self.image_size))

            #追記
            rgb,eye = self.face_landmark_find(image)
            gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
            face = self.face_cascade.detectMultiScale(gray,1.3,5)
            print(eye)
            if face != ():
                (self.x,self.y,self.w,self.h) = face[0]
                self.x=self.x-1
                self.w=self.w+2
                self.h=self.h+2                
            if self.w != 0:
                if eye < 0.2:
                    rgb = self.overlayImage(rgb, self.cv_img2, (self.x, self.y), (self.w, self.h))
                elif eye >= 0.2:
                    rgb = self.overlayImage(rgb, self.cv_img1, (self.x, self.y), (self.w, self.h))

            show_image(rgb, 'inference', wait=False)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.camera.stop()
        cv2.destroyAllWindows()

    def record(self, name='video.avi', fps=20, fourCC='XVID'):
        """Opens camera and records continuous inference using ``pipeline``.

        # Arguments
            name: String. Video name. Must include the postfix .avi.
            fps: Int. Frames per second.
            fourCC: String. Indicates the four character code of the video.
            e.g. XVID, MJPG, X264.
        """
        self.camera.start()
        fourCC = cv2.VideoWriter_fourcc(*fourCC)
        writer = cv2.VideoWriter(name, fourCC, fps, self.image_size)
        while True:
            output = self.step()
            if output is None:
                continue
            image = resize_image(output['image'], tuple(self.image_size))
            show_image(image, 'inference', wait=False)
            writer.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.stop()
        writer.release()
        cv2.destroyAllWindows()

    def record_from_file(self, video_file_path, name='video.avi',
                         fps=20, fourCC='XVID'):
        """Load video and records continuous inference using ``pipeline``.

        # Arguments
            video_file_path: String. Path to the video file.
            name: String. Output video name. Must include the postfix .avi.
            fps: Int. Frames per second.
            fourCC: String. Indicates the four character code of the video.
            e.g. XVID, MJPG, X264.
        """

        fourCC = cv2.VideoWriter_fourcc(*fourCC)
        writer = cv2.VideoWriter(name, fourCC, fps, self.image_size)

        video = cv2.VideoCapture(video_file_path)
        if (video.isOpened() is False):
            print("Error opening video  file")

        while video.isOpened():
            is_frame_received, frame = video.read()
            if not is_frame_received:
                print("Frame not received. Exiting ...")
                break
            if is_frame_received is True:
                output = self.pipeline(frame)
                if output is None:
                    continue
                image = resize_image(output['image'], tuple(self.image_size))
                show_image(image, 'inference', wait=False)
                writer.write(image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        writer.release()
        cv2.destroyAllWindows()
