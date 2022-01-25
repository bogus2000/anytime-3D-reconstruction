import numpy as np
import cv2
import time

kitti_proj_mat = np.array(
[[7.21537720e+02, 0.00000000e+00, 6.09559326e+02, 4.48572807e+01],
 [0.00000000e+00, 7.21537720e+02, 1.72854004e+02, 2.16379106e-01],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 2.74588400e-03],
 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)

kitti_proj_mat_inv = np.linalg.inv(kitti_proj_mat)

offset = np.zeros((64,64,64,3))
for i in range(64):
    for j in range(64):
        for k in range(64):
            offset[i,j,k,:] = i,j,k


def matmul3x3(a, b):
    c00 = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0]
    c01 = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1] + a[0, 2] * b[2, 1]
    c02 = a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2] + a[0, 2] * b[2, 2]

    c10 = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0]
    c11 = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1] + a[1, 2] * b[2, 1]
    c12 = a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2] + a[1, 2] * b[2, 2]

    c20 = a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0]
    c21 = a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1] + a[2, 2] * b[2, 1]
    c22 = a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2] + a[2, 2] * b[2, 2]

    return np.array([[c00, c01, c02],
                     [c10, c11, c12],
                     [c20, c21, c22]])


def matmul4x4(a, b):
    c00 = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0] + a[0, 3] * b[3, 0]
    c01 = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1] + a[0, 2] * b[2, 1] + a[0, 3] * b[3, 1]
    c02 = a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2] + a[0, 2] * b[2, 2] + a[0, 3] * b[3, 2]
    c03 = a[0, 0] * b[0, 3] + a[0, 1] * b[1, 3] + a[0, 2] * b[2, 3] + a[0, 3] * b[3, 3]

    c10 = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0] + a[1, 3] * b[3, 0]
    c11 = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1] + a[1, 2] * b[2, 1] + a[1, 3] * b[3, 1]
    c12 = a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2] + a[1, 2] * b[2, 2] + a[1, 3] * b[3, 2]
    c13 = a[1, 0] * b[0, 3] + a[1, 1] * b[1, 3] + a[1, 2] * b[2, 3] + a[1, 3] * b[3, 3]

    c20 = a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0] + a[2, 3] * b[3, 0]
    c21 = a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1] + a[2, 2] * b[2, 1] + a[2, 3] * b[3, 1]
    c22 = a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2] + a[2, 2] * b[2, 2] + a[2, 3] * b[3, 2]
    c23 = a[2, 0] * b[0, 3] + a[2, 1] * b[1, 3] + a[2, 2] * b[2, 3] + a[2, 3] * b[3, 3]

    c30 = a[3, 0] * b[0, 0] + a[3, 1] * b[1, 0] + a[3, 2] * b[2, 0] + a[3, 3] * b[3, 0]
    c31 = a[3, 0] * b[0, 1] + a[3, 1] * b[1, 1] + a[3, 2] * b[2, 1] + a[3, 3] * b[3, 1]
    c32 = a[3, 0] * b[0, 2] + a[3, 1] * b[1, 2] + a[3, 2] * b[2, 2] + a[3, 3] * b[3, 2]
    c33 = a[3, 0] * b[0, 3] + a[3, 1] * b[1, 3] + a[3, 2] * b[2, 3] + a[3, 3] * b[3, 3]

    return np.array([[c00, c01, c02, c03],
                     [c10, c11, c12, c13],
                     [c20, c21, c22, c23],
                     [c30, c31, c32, c33]])

def matmul3x1(a,b):
    c0 = a[0,0] * b[0] + a[0,1] * b[1] + a[0,2] * b[2]
    c1 = a[1,0] * b[0] + a[1,1] * b[1] + a[1,2] * b[2]
    c2 = a[2,0] * b[0] + a[2,1] * b[1] + a[2,2] * b[2]
    return np.array([c0,c1,c2])

def matmul4x1(a, b):
    c0 = a[0,0] * b[0] + a[0,1] * b[1] + a[0,2] * b[2] + a[0,3] * b[3]
    c1 = a[1,0] * b[0] + a[1,1] * b[1] + a[1,2] * b[2] + a[1,3] * b[3]
    c2 = a[2,0] * b[0] + a[2,1] * b[1] + a[2,2] * b[2] + a[2,3] * b[3]
    c3 = a[3,0] * b[0] + a[3,1] * b[1] + a[3,2] * b[2] + a[3,3] * b[3]
    return np.array([c0,c1,c2,c3])


def getTranslation(proj_mat, R, bbox2D, bbox3D):
    x_min, y_min, x_max, y_max = bbox2D
    w, h, l = bbox3D
    dx, dy, dz = w / 2., l / 2., h / 2.
    iou_max = -1.
    trans_final = np.zeros(4)
    xmin_set_list = [[[-dx, -dy, -dz], [-dx, -dy, dz]], [[-dx, dy, -dz], [-dx, dy, dz]]]
    xmax_set_list = [[[dx, dy, -dz], [dx, dy, dz]], [[dx, -dy, dz], [dx, -dy, -dz]]]
    ymin_set_list = [[[-dx, -dy, dz], [dx, -dy, dz]], [[-dx, dy, dz], [dx, dy, dz]]]
    ymax_set_list = [[[-dx, dy, -dz], [dx, dy, -dz]], [[-dx, -dy, -dz], [dx, -dy, -dz]]]
    for xmin_set, xmax_set in zip(xmin_set_list + xmax_set_list, xmax_set_list + xmin_set_list):
        for ymin_set, ymax_set in zip(ymin_set_list, ymax_set_list):
            # for xmin
            for d_xmin in xmin_set:
                A0 = np.concatenate([np.identity(3), np.reshape(matmul3x1(R, d_xmin), (3, 1))], axis=-1)
                A0 = np.concatenate([A0, np.reshape([0, 0, 0, 1], (1, 4))], axis=0)
                A0 = matmul4x4(proj_mat, A0)
                B0 = A0
                A0 = A0[0, :] - x_min * A0[2, :]
                # for ymin
                for d_ymin in ymin_set:
                    A1 = np.concatenate([np.identity(3), np.reshape(matmul3x1(R, d_ymin), (3, 1))], axis=-1)
                    A1 = np.concatenate([A1, np.reshape([0, 0, 0, 1], (1, 4))], axis=0)
                    A1 = matmul4x4(proj_mat, A1)
                    B1 = A1
                    A1 = A1[1, :] - y_min * A1[2, :]
                    # for xmax
                    for d_xmax in xmax_set:
                        A2 = np.concatenate([np.identity(3), np.reshape(matmul3x1(R, d_xmax), (3, 1))], axis=-1)
                        A2 = np.concatenate([A2, np.reshape([0, 0, 0, 1], (1, 4))], axis=0)
                        A2 = matmul4x4(proj_mat, A2)
                        B2 = A2
                        A2 = A2[0, :] - x_max * A2[2, :]
                        # for ymax
                        for d_ymax in ymax_set:
                            A3 = np.concatenate([np.identity(3), np.reshape(matmul3x1(R, d_ymax), (3, 1))], axis=-1)
                            A3 = np.concatenate([A3, np.reshape([0, 0, 0, 1], (1, 4))], axis=0)
                            A3 = matmul4x4(proj_mat, A3)
                            B3 = A3
                            A3 = A3[1, :] - y_max * A3[2, :]
                            A = np.stack([A0, A1, A2, A3], axis=0)
                            U, S, VH = np.linalg.svd(A, full_matrices=True)
                            translation = VH[-1, :]

                            if translation[-1] * translation[-2] > 0:
                                translation = translation / translation[-1]
                                x_min_pred = matmul4x1(B0, translation)
                                x_min_pred = (x_min_pred[:2] / x_min_pred[2])[0]
                                y_min_pred = matmul4x1(B1, translation)
                                y_min_pred = (y_min_pred[:2] / y_min_pred[2])[1]
                                x_max_pred = matmul4x1(B2, translation)
                                x_max_pred = (x_max_pred[:2] / x_max_pred[2])[0]
                                y_max_pred = matmul4x1(B3, translation)
                                y_max_pred = (y_max_pred[:2] / y_max_pred[2])[1]

                                # if y_min<y_min_pred and y_max>y_max_pred:
                                if x_min_pred < x_max_pred and y_min_pred < y_max_pred:
                                    bbox2D_pred_area = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
                                    bbox2D_gt_area = (x_max - x_min) * (y_max - y_min)
                                    x_min_inter, x_max_inter = np.max((x_min_pred, x_min)), np.min((x_max_pred, x_max))
                                    y_min_inter, y_max_inter = np.max((y_min_pred, y_min)), np.min((y_max_pred, y_max))
                                    inter_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
                                    iou = inter_area / (bbox2D_pred_area + bbox2D_gt_area - inter_area)
                                    if iou < 1. and iou_max < iou:
                                        iou_max = iou
                                        trans_final = translation

    return np.reshape(trans_final[:-1], (3, 1))

def getRay(P_inv, pixel):
    px, py = pixel
    pz = 1.0
    p_point = np.array([px,py,pz,1.])
    ray = matmul4x1(P_inv, p_point)[:3]
    if ray[-1]<0:
        print('neg z', ray)
    return ray/np.linalg.norm(ray)

def getRayRotation(ray):
    ray = ray/np.linalg.norm(ray)
    # assume ray followed rotx and roty in order
    rx,ry,rz = ray
    cy = np.sqrt(ry*ry+rz*rz)
    cx = rz/np.sqrt(ry*ry+rz*rz)
    sx = -ry/np.sqrt(ry*ry+rz*rz)
    sy = rx
    R = np.array([[cy, 0., sy],
                 [sx*sy, cx, -sx*cy],
                 [-cx*sy, sx, cx*cy]])
    return R


def objRescaleTransform(objPoints, h, w, l, R):
    t = time.time()
    dim = 64
    objPoints = objPoints.reshape(dim, dim, dim).astype('float')
    mask = objPoints > 0.5
    points = offset[mask]

    t = time.time()
    points = np.array(points).astype('float')
    points = points - np.min(points, axis=0)
    scale = np.max([h, w, l]) / np.max(points)
    points = points * scale
    points = points - np.max(points, axis=0) / 2.0

    t = time.time()
    points = np.transpose(np.concatenate([points, np.ones((len(points), 1))], axis=-1), [1, 0])
    rotPoints = np.transpose(np.matmul(R, points), [1, 0])[:, :3]
    return np.array(rotPoints)


def get3DbboxProjection(projmat, R, t, w, h, l):
    a = np.zeros((2, 2, 2, 2))
    dx, dy, dz = -w / 2., -l / 2., -h / 2.
    for i in range(2):
        dx = -1. * dx
        for j in range(2):
            dy = -1. * dy
            for k in range(2):
                dz = -1. * dz
                x = matmul3x1(R, np.array([dx, dy, dz])) + np.reshape(t, (3,))
                x = np.array([x[0], x[1], x[2], 1.])
                x_proj = matmul4x1(projmat, x)
                x_proj = x_proj[:2] / x_proj[2]
                a[i, j, k, :] = x_proj
    return a


def draw2Dbbox(image, bbox2d, color=(0, 255, 0), thickness=2):
    p0 = (bbox2d[0], bbox2d[1])
    p1 = (bbox2d[2], bbox2d[3])
    cv2.rectangle(image, p0, p1, color=color, thickness=thickness)

def draw3Dbbox(image, proj_bbox3d, color=(255, 0, 255), thickness=2):
    proj_bbox3d = proj_bbox3d.astype('int32')
    # for cube
    cv2.line(image, tuple(proj_bbox3d[0, 0, 0, :]), tuple(proj_bbox3d[0, 1, 0, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 0, 0, :]), tuple(proj_bbox3d[1, 0, 0, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 0, :]), tuple(proj_bbox3d[1, 1, 0, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[1, 0, 0, :]), tuple(proj_bbox3d[1, 1, 0, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 0, :]), tuple(proj_bbox3d[0, 1, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 0, 0, :]), tuple(proj_bbox3d[0, 0, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[1, 1, 0, :]), tuple(proj_bbox3d[1, 1, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 1, :]), tuple(proj_bbox3d[1, 1, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[1, 0, 0, :]), tuple(proj_bbox3d[1, 0, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 1, :]), tuple(proj_bbox3d[0, 0, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[1, 0, 1, :]), tuple(proj_bbox3d[1, 1, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 0, 1, :]), tuple(proj_bbox3d[1, 0, 1, :]), color=color, thickness=thickness)

    # X of forward and backward
    color = (255, 0, 0)
    thickness = 1
    cv2.line(image, tuple(proj_bbox3d[0, 0, 0, :]), tuple(proj_bbox3d[1, 0, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 0, 1, :]), tuple(proj_bbox3d[1, 0, 0, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 0, :]), tuple(proj_bbox3d[1, 1, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 1, :]), tuple(proj_bbox3d[1, 1, 0, :]), color=color, thickness=thickness)

def getObjectInRealWorld(
        normalized_bbox2D_list, bbox3D_list, sin_list, cos_list, shape_3D_list,
        image_size,
        proj_mat=kitti_proj_mat, proj_mat_inv=kitti_proj_mat_inv):

    image_col, image_row = image_size
    objsPose, objsBbox3DSize, objsPoints = [], [], []
    objsBbox2D, objsBbox3DProj = [], []

    for bbox2D, bbox3D, sin, cos, shape_3D in zip(normalized_bbox2D_list, bbox3D_list, sin_list, cos_list, shape_3D_list):
        b2x1,b2y1,b2x2,b2y2, _obj_prob = bbox2D
        # avoid too close obj
        if b2x1 > 1e-1 and b2x2 < 1. - 1e-1 and b2y2 < 1. - 1e-1:
            b2x1 = b2x1 * image_col
            b2y1 = b2y1 * image_row
            b2x2 = b2x2 * image_col
            b2y2 = b2y2 * image_row
            b3w,b3h,b3l = bbox3D
            sinA,sinE,sinI = sin
            cosA,cosE,cosI = cos

            # elevation angle is wired. Manually enforcing
            beta = -5.0 / 180.0 * np.pi
            sinE_t = sinE*np.cos(beta) - cosE*np.sin(beta)
            cosE_t = cosE*np.cos(beta) + sinE*np.sin(beta)
            sinE = sinE_t
            cosE = cosE_t

            # ======================== get rotation of obj
            # 1. RA*RE*RI
            r11, r12, r13 = -sinA * sinE * sinI + cosA * cosI, -sinA * cosE, sinA * sinE * cosI + sinI * cosA
            r21, r22, r23 = sinA * cosI + sinE * sinI * cosA, cosA * cosE, sinA * sinI - sinE * cosA * cosI
            r31, r32, r33 = -sinI * cosE, sinE, cosE * cosI

            # pascal->kitti : 90rot for x axis
            R = np.array([[r11, r12, r13],
                          [-r31, -r32, -r33],
                          [r21, r22, r23]])

            # apply ray orientation
            px, py = (b2x2 + b2x1) / 2., (b2y2 + b2y1) / 2.
            ray = getRay(proj_mat_inv, (px, py))
            R_ray = getRayRotation(ray)
            R = matmul3x3(R_ray, R)

            # ======================== get translation of obj
            X = getTranslation(proj_mat, R, (b2x1, b2y1, b2x2, b2y2), (b3w, b3h, b3l))

            # ======================== append R,T of obj
            objPose = np.concatenate([np.concatenate([R, X], axis=-1), np.reshape([0, 0, 0, 1], (1, 4))], axis=0)

            # ======================== transform 3d shape of obj according to R,T
            objPoints = objRescaleTransform(shape_3D, b3h, b3w, b3l, objPose)

            # ====================== bbox3D projection on image
            proj_bbox3D = get3DbboxProjection(proj_mat, R, X, b3h, b3w, b3l)

            # if translation != trivial solution (zero-vector) or too close to image plane, append
            if X[2] > 1e-1:
                objsPose.append(objPose)
                objsPoints.append(objPoints)
                objsBbox2D.append([int(b2x1), int(b2y1), int(b2x2), int(b2y2)])
                objsBbox3DProj.append(proj_bbox3D)
                objsBbox3DSize.append([b3h, b3l, b3w])

    objsPose = np.array(objsPose)
    objsBbox3DSize = np.array(objsBbox3DSize)
    objsPoints = np.array(objsPoints)
    objsBbox2D = np.array(objsBbox2D)
    objsBbox3DProj = np.array(objsBbox3DProj)

    return objsPose, objsBbox3DSize, objsPoints, objsBbox2D, objsBbox3DProj









