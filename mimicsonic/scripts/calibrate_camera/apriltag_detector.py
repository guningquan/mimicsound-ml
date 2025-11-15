# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2
import pupil_apriltags as apriltag

@dataclass
class TagDetection:
    """单个 AprilTag 的检测与位姿结果"""
    id: int
    corners: np.ndarray        # (4, 2) 像素坐标，顺序: TL, TR, BR, BL
    pose_R: np.ndarray         # (3, 3) 旋转矩阵，表示相机坐标系下 tag 的朝向
    pose_t: np.ndarray         # (3, 1) 平移向量，表示相机坐标系下 tag 的位置（米）


class AprilTagDetector_my:
    """
    使用 OpenCV ArUco 模块检测 AprilTag,并估计位姿。
    需要 opencv-contrib-python >= 4.7.0
    """
    _FAMILY_MAP = {
        # OpenCV 里的 AprilTag 字典名
        "tag16h5":  cv2.aruco.DICT_APRILTAG_16h5,
        "tag25h9":  cv2.aruco.DICT_APRILTAG_25h9,
        "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
        # 兼容常见写法
        "TAG16H5":  cv2.aruco.DICT_APRILTAG_16h5,
        "TAG25H9":  cv2.aruco.DICT_APRILTAG_25h9,
        "TAG36H10": cv2.aruco.DICT_APRILTAG_36h10,
        "TAG36H11": cv2.aruco.DICT_APRILTAG_36h11,
    }

    def __init__(self, quad_decimate: float = 1.0, tag_family: str = "tag36h11"):
        """
        quad_decimate: 下采样比例，>1 表示先缩小再检测以提速（类似 apriltag 的 decimate）。
        tag_family:    AprilTag 字典名，例如 'tag36h11'
        """
        if tag_family not in self._FAMILY_MAP:
            raise ValueError(f"Unsupported tag_family: {tag_family}. "
                             f"Supported: {list(self._FAMILY_MAP.keys())}")

        self.decimate = max(1.0, float(quad_decimate))
        self.dict = cv2.aruco.getPredefinedDictionary(self._FAMILY_MAP[tag_family])

        # 检测参数
        params = cv2.aruco.DetectorParameters()
        # 下面参数可按需微调
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        self.detector = cv2.aruco.ArucoDetector(self.dict, params)

        # 缓存最近一次检测的可视化图像
        self._last_vis: Optional[np.ndarray] = None

    @staticmethod
    def _camera_matrix_from_intrinsics(intrinsics: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        intrinsics: {"fx":..., "fy":..., "cx":..., "cy":...}
        返回 K(3x3), dist(5x1 零畸变，若你有畸变可在此替换)
        """
        fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        dist = np.zeros((5, 1), dtype=np.float64)  # 如已知畸变，可替换为真实 D
        return K, dist

    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        """
        OpenCV aruco 返回 corners 形状 (1, 4, 2)，顺序通常为 TL, TR, BR, BL。
        这里做显式转换为 (4,2) 并确保顺序一致。
        """
        c = corners.reshape(4, 2).astype(np.float64)
        return c  # aruco 已按 TL,TR,BR,BL 输出

    @staticmethod
    def _object_points_for_square(tag_size: float) -> np.ndarray:
        """
        返回 tag 的四个角在 tag 坐标系下的 3D 点（Z=0），单位米。
        对应像素角点顺序: TL, TR, BR, BL
        """
        s = tag_size / 2.0
        objp = np.array([
            [-s,  s, 0.0],  # TL
            [  s,  s, 0.0],  # TR
            [  s, -s, 0.0],  # BR
            [-s, -s, 0.0],  # BL
        ], dtype=np.float64)
        return objp

    def detect(self, img_bgr: np.ndarray, intrinsics: dict, tag_size: float) -> List[TagDetection]:
        """
        输入:
            img_bgr:   BGR 图像 (H,W,3), uint8
            intrinsics: {"fx","fy","cx","cy"}
            tag_size:  tag 实物边长 (米)
        输出:
            List[TagDetection]
        """
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("img_bgr must be a BGR image of shape (H,W,3)")

        H, W = img_bgr.shape[:2]
        scale = 1.0 / self.decimate if self.decimate > 1.0 else 1.0

        # 灰度与可选下采样
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            new_size = (int(W * scale), int(H * scale))
            gray_small = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        else:
            gray_small = gray

        # 检测
        corners_list, ids, _ = self.detector.detectMarkers(gray_small)

        detections: List[TagDetection] = []
        if ids is None or len(ids) == 0:
            self._last_vis = img_bgr.copy()
            return detections

        # 还原角点到原图尺度
        if scale < 1.0:
            corners_list = [c / scale for c in corners_list]

        # 相机内参
        K, dist = self._camera_matrix_from_intrinsics(intrinsics)
        objp = self._object_points_for_square(tag_size)

        vis = img_bgr.copy()

        for i, c in enumerate(corners_list):
            tag_id = int(ids[i][0])
            corners = self._order_corners(c)  # (4,2)

            # PnP 估计位姿（也可用 aruco.estimatePoseSingleMarkers，但这里直接用 solvePnP 更通用）
            ok, rvec, tvec = cv2.solvePnP(
                objp, corners,
                K, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE  # 对平面正方形很稳
            )
            if not ok:
                # 回退到 EPnP
                ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_EPNP)
                if not ok:
                    continue

            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)

            # 可视化：边框与坐标轴
            cv2.polylines(
                vis,
                [corners.astype(np.int32)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )
            cv2.putText(
                vis, f"id:{tag_id}",
                tuple(corners[0].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )
            try:
                # 画坐标轴（长度=tag_size*0.5）
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec, tag_size * 0.5)
            except Exception:
                pass

            detections.append(TagDetection(
                id=tag_id,
                corners=corners,
                pose_R=R.astype(np.float64),
                pose_t=t.astype(np.float64),
            ))

        self._last_vis = vis
        return detections

    def vis_tag(self, img_bgr: Optional[np.ndarray] = None) -> np.ndarray:
        """
        返回最近一次 detect 的可视化图像。
        也支持传入新的图像（将直接返回该图像的拷贝，不做绘制）。
        """
        if img_bgr is not None:
            return img_bgr.copy()
        if self._last_vis is None:
            raise RuntimeError("No visualization available. Call detect() first.")
        return self._last_vis.copy()
        
class AprilTagDetector:
    def __init__(self, families="tag36h11"):

        self.detector = apriltag.Detector(
            # families=families, quad_decimate=3.0, decode_sharpening=0.25
            families=families, quad_decimate=1.0, decode_sharpening=0.25
        )
        self.results = []

    def detect(self, img: np.ndarray, intrinsics: dict, tag_size: int):
        img = img.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
            tag_size=tag_size,
        )
        return self.results

    def __len__(self):
        return len(self.results)

    def vis_tag(self, img: np.ndarray):
        img = img.astype(np.uint8)
        for r in self.results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            # draw the bounding box of the AprilTag detection
            cv2.line(img, ptA, ptB, (0, 255, 0), 2)
            cv2.line(img, ptB, ptC, (0, 255, 0), 2)
            cv2.line(img, ptC, ptD, (0, 255, 0), 2)
            cv2.line(img, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(
                img,
                f"{tagFamily}-{r.tag_id}-{cX}-{cY}",
                (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return img

    def tags_centroid(self):
        """Return a dictionary, keys being tag id and values being positions of the centroid"""
        centroid_dict = {}
        for r in self.results:
            centroid_dict[r.tag_id] = [int(r.center[0]), int(r.center[1])]
        return centroid_dict