# -*- coding: utf-8 -*-
"""
將 COFW 官方的 COFW_train.mat / COFW_test.mat 轉換為 STAR
(https://github.com/ZhenglinZhou/STAR) 訓練所需的 tsv 標註格式 + 圖片目錄，
並提供與原始 mat 數據（及任意參考 tsv）的一致性校驗。

STAR 的 lib/dataset/alignmentDataset.py 中 __getitem__ 對 tsv 每行的解析為：
    col0: image_path          例: ./rawImages/train/0.png  (代碼會去掉 './rawImages/' 前綴再拼接 image_dir)
    col1: landmarks_5pts      10 個浮點數 "x,y,..."        reshape(5, 2)，用於對齊
    col2: landmarks_target    58 個浮點數 (29點*2, x,y 交錯)
    col3: scale               人臉尺度 = max(bbox_w, bbox_h) / 200.0
    col4: center_w            bbox 中心 x = xmin + w/2
    col5: center_h            bbox 中心 y = ymin + h/2
    col6: tags                29 個 0/1 遮擋標記

COFW mat (v7.3, h5py 可讀) 內部格式（已由參考 tsv 全量驗證）：
    IsTr / IsT       : 圖像 cell 引用，圖像存為 MATLAB HxWxC (RGB)，h5py 讀出為 (C,W,H)，需 transpose
    bboxesTr / bboxesT: (4, N) => (xmin, ymin, width, height)
    phisTr / phisT   : (87, N) => 前29=x坐標, 中29=y坐標, 後29=遮擋標記(0/1)
    5點取 29 點中的索引 [16,17,20,22,23]（雙眼、鼻尖、雙嘴角），與 STAR/ADNet 的
    COFW train.tsv/test.tsv 全部 1852 行逐一精確一致。
"""

import os
import sys
import argparse
import hashlib

import h5py
import numpy as np
import pandas as pd
import cv2

# 5 點對齊關鍵點在 COFW 29 點中的索引（0-based）: 左眼、右眼、鼻尖、左嘴角、右嘴角
FIVE_PTS_IDX = [16, 17, 20, 22, 23]
NUM_LANDMARKS = 29


# ----------------------------------------------------------------------
# 1. 從 mat 中解析單個樣本
# ----------------------------------------------------------------------
def load_mat_sample(mat, index, img_token, bbox_token, phis_token):
    """返回 (image_rgb[H,W,C uint8], bbox[xmin,ymin,w,h], phi[87])"""
    img_ref = mat[img_token][0, index]          # 圖像引用
    img = mat[img_ref][:]                       # (C, W, H) uint8 (MATLAB HxWxC RGB)
    img = np.transpose(img)                     # -> (H, W, C) RGB

    bbox = np.array(mat[bbox_token][:, index], dtype=np.float64).ravel()   # (xmin, ymin, w, h)
    phi = np.array(mat[phis_token][:, index], dtype=np.float64).ravel()    # (87,)

    assert phi.shape[0] == NUM_LANDMARKS * 3, \
        f"phis 維度異常: {phi.shape[0]}, 期望 87 (29*3: x, y, 遮擋標記)"
    return img, bbox, phi


def phi_to_landmarks(phi):
    """87 維 phi -> (29, 2) 交錯坐標 (x,y)"""
    xs, ys = phi[:NUM_LANDMARKS], phi[NUM_LANDMARKS:NUM_LANDMARKS * 2]
    return np.stack([xs, ys], axis=1)


def phi_to_tags(phi):
    """87 維 phi -> (29,) 遮擋標記"""
    return phi[NUM_LANDMARKS * 2:]


def bbox_to_scale_center(bbox):
    """(xmin,ymin,w,h) -> (scale, center_w, center_h)，與 STAR/HRNet 約定一致"""
    xmin, ymin, bw, bh = bbox
    scale = max(bw, bh) / 200.0
    center_w = xmin + bw / 2.0
    center_h = ymin + bh / 2.0
    return scale, center_w, center_h


def fmt_float(v):
    return repr(float(v))


def fmt_array(arr):
    return ",".join(fmt_float(v) for v in np.asarray(arr).ravel())


# ----------------------------------------------------------------------
# 2. mat -> STAR 格式（圖片 + tsv）
# ----------------------------------------------------------------------
def mat_to_star(mat_file, img_token, bbox_token, phis_token,
                out_img_dir, out_tsv, img_rel_prefix="./rawImages"):
    """
    圖片保存到 out_img_dir/{0.jpg形式編號}.png，tsv 的 col0 寫
    '{img_rel_prefix}/{split_name}/{i}.png'，與 STAR alignmentDataset.py 中
    image_path.replace('./rawImages/', '') 的處理兼容。
    """
    os.makedirs(out_img_dir, exist_ok=True)
    split_name = os.path.basename(out_img_dir.rstrip("/\\"))  # train / test

    mat = h5py.File(mat_file, "r")
    total_num = mat[img_token].shape[1]
    print(f"[mat_to_star] {mat_file}: 共 {total_num} 張")

    n_written = 0
    with open(out_tsv, "w") as f:
        for i in range(total_num):
            img_rgb, bbox, phi = load_mat_sample(mat, i, img_token, bbox_token, phis_token)
            lmk = phi_to_landmarks(phi)                       # (29,2)
            five = lmk[FIVE_PTS_IDX]                          # (5,2)
            tags = phi_to_tags(phi)                           # (29,)
            scale, center_w, center_h = bbox_to_scale_center(bbox)

            # mat 中是 RGB，cv2.imwrite 需要 BGR
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_path = os.path.join(out_img_dir, f"{i}.png")
            cv2.imwrite(img_path, img_bgr)

            rel_path = f"{img_rel_prefix}/{split_name}/{i}.png"
            row = "\t".join([
                rel_path,
                fmt_array(five),            # col1: 5pts
                fmt_array(lmk),             # col2: 29 個關鍵點
                fmt_float(scale),           # col3
                fmt_float(center_w),        # col4
                fmt_float(center_h),        # col5
                ",".join(str(int(t)) for t in tags),  # col6: tags
            ])
            f.write(row + "\n")
            n_written += 1

    mat.close()
    print(f"[mat_to_star] 寫入圖片 {n_written} 張至 {out_img_dir}, tsv 至 {out_tsv}")


# ----------------------------------------------------------------------
# 3. 校驗: tsv（生成的或參考的） vs 原始 mat
# ----------------------------------------------------------------------
def _md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def validate_tsv_vs_mat(mat_file, img_token, bbox_token, phis_token,
                        tsv_file, image_dir, check_image_content=True):
    """
    逐行檢查 tsv 與 mat 是否一致：
      - 行數
      - col0 路徑形式、圖片文件存在、（可選）像素內容與 mat 中圖像完全一致
      - col1 5pts   == lmk[[16,17,20,22,23]]
      - col2 關鍵點 == mat phi 前58維（交錯）
      - col3 scale  == max(w,h)/200
      - col4/5 center == bbox 中心
      - col6 tags   == phi 後29維
    """
    mat = h5py.File(mat_file, "r")
    total_num = mat[img_token].shape[1]

    df = pd.read_csv(tsv_file, sep="\t", header=None)
    n = len(df)

    errors = []
    warn = []
    if n != total_num:
        errors.append(f"[全局] tsv 行數 {n} != mat 樣本數 {total_num}")

    max_check = n
    for i in range(max_check):
        try:
            img_rgb, bbox, phi = load_mat_sample(mat, i, img_token, bbox_token, phis_token)
            lmk = phi_to_landmarks(phi)
            five = lmk[FIVE_PTS_IDX]
            tags = phi_to_tags(phi)
            scale, center_w, center_h = bbox_to_scale_center(bbox)

            row = df.iloc[i]

            # col0: 路徑
            path_raw = str(row[0]).replace("\\", "/")
            if not path_raw.endswith(f"/{i}.png") and not path_raw.endswith(f"/{i}.jpg"):
                errors.append(f"[行{i}] 圖像名不匹配: {path_raw}")
            path_in_tsv = path_raw.replace("./rawImages/", "")
            disk_path = os.path.join(image_dir, path_in_tsv)
            if not os.path.exists(disk_path):
                errors.append(f"[行{i}] 圖片不存在: {disk_path}")
            elif check_image_content:
                img_disk = cv2.imread(disk_path, cv2.IMREAD_COLOR)
                if img_disk is None:
                    errors.append(f"[行{i}] 圖片讀取失敗: {disk_path}")
                else:
                    img_disk_rgb = cv2.cvtColor(img_disk, cv2.COLOR_BGR2RGB)
                    if img_disk_rgb.shape != img_rgb.shape or not np.array_equal(img_disk_rgb, img_rgb):
                        errors.append(f"[行{i}] 圖片內容與 mat 不一致 (md5: {_md5_file(disk_path)})")

            # col1: 5pts
            five_tsv = np.array(list(map(float, str(row[1]).split(",")))).reshape(5, 2)
            if not np.allclose(five_tsv, five, atol=1e-4):
                errors.append(f"[行{i}] 5pts 不一致")

            # col2: 關鍵點
            lmk_tsv = np.array(list(map(float, str(row[2]).split(",")))).reshape(NUM_LANDMARKS, 2)
            if not np.allclose(lmk_tsv, lmk, atol=1e-4):
                errors.append(f"[行{i}] landmarks 不一致")

            # col3/4/5: scale, center
            if abs(float(row[3]) - scale) > 1e-4:
                errors.append(f"[行{i}] scale 不一致: tsv={row[3]} mat={scale}")
            if abs(float(row[4]) - center_w) > 1e-4 or abs(float(row[5]) - center_h) > 1e-4:
                errors.append(f"[行{i}] center 不一致: tsv=({row[4]},{row[5]}) mat=({center_w},{center_h})")

            # col6: tags
            tags_tsv = np.array(list(map(lambda x: int(float(x)), str(row[6]).split(","))))
            if tags_tsv.shape[0] != NUM_LANDMARKS or not np.array_equal(tags_tsv, tags.astype(int)):
                errors.append(f"[行{i}] tags 不一致")

        except Exception as e:
            errors.append(f"[行{i}] 校驗異常: {e}")

    mat.close()
    print(f"[validate] {tsv_file}: 檢查 {min(n, total_num)} 行, 發現 {len(errors)} 處不一致, {len(warn)} 條警告")
    for e in errors[:50]:
        print("   ", e)
    if len(errors) > 50:
        print(f"    ... 其餘 {len(errors) - 50} 條省略")
    return len(errors) == 0


# ----------------------------------------------------------------------
# 4. 校驗: 生成 tsv vs 參考 tsv（逐欄位數值比較）
# ----------------------------------------------------------------------
def validate_tsv_vs_tsv(gen_tsv, ref_tsv, tol=1e-3):
    g, r = pd.read_csv(gen_tsv, sep="\t", header=None), pd.read_csv(ref_tsv, sep="\t", header=None)
    if len(g) != len(r):
        print(f"[tsv-vs-tsv] 行數不一致: {len(g)} vs {len(r)}")
        return False
    bad = 0
    for i in range(len(g)):
        # 路徑（只比較圖像編號是否一致）
        gi = os.path.splitext(os.path.basename(str(g.iloc[i, 0]).replace("\\", "/")))[0]
        ri = os.path.splitext(os.path.basename(str(r.iloc[i, 0]).replace("\\", "/")))[0]
        if gi != ri:
            print(f"[行{i}] 圖像名不一致: {g.iloc[i,0]} vs {r.iloc[i,0]}"); bad += 1; continue
        for col in [1, 2]:
            a = np.array(list(map(float, str(g.iloc[i, col]).split(","))))
            b = np.array(list(map(float, str(r.iloc[i, col]).split(","))))
            if a.shape != b.shape or not np.allclose(a, b, atol=tol):
                print(f"[行{i}] col{col} 數值不一致"); bad += 1
        for col in [3, 4, 5]:
            if abs(float(g.iloc[i, col]) - float(r.iloc[i, col])) > tol:
                print(f"[行{i}] col{col} 不一致: {g.iloc[i,col]} vs {r.iloc[i,col]}"); bad += 1
        tg = [int(float(x)) for x in str(g.iloc[i, 6]).split(",")]
        tr = [int(float(x)) for x in str(r.iloc[i, 6]).split(",")]
        if tg != tr:
            print(f"[行{i}] tags 不一致"); bad += 1
    print(f"[tsv-vs-tsv] {gen_tsv} vs {ref_tsv}: 檢查 {len(g)} 行, {bad} 處不一致")
    return bad == 0


# ----------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="E:/DataSet/face_data/COFW", help="mat 文件所在目錄")
    ap.add_argument("--image_dir", default ="E:/DataSet/face_data/COFW",
                    help="STAR 的 image_dir (其下應有 COFW/train, COFW/test)")
    ap.add_argument("--annot_dir", default="E:/DataSet/face_data/COFW",
                    help="STAR 的 annot_dir (其下應有 COFW/train.tsv, COFW/test.tsv)")
    ap.add_argument("--skip_convert", action="store_true", help="只做校驗不轉換")
    ap.add_argument("--ref_train_tsv", default="", help="可選: 參考 train.tsv (如 STAR 官方 metadata)")
    ap.add_argument("--ref_test_tsv", default="", help="可選: 參考 test.tsv")
    args = ap.parse_args()

    tasks = [
        ("COFW_train.mat", "IsTr", "bboxesTr", "phisTr", "train"),
        ("COFW_test.mat", "IsT", "bboxesT", "phisT", "test"),
    ]

    all_ok = True
    for mat_name, img_t, bbox_t, phis_t, split in tasks:
        mat_path = os.path.join(args.data_root, mat_name)
        out_img_dir = os.path.join(args.image_dir, "COFW", split)
        out_tsv = os.path.join(args.annot_dir, "COFW", f"{split}.tsv")
        os.makedirs(os.path.dirname(out_tsv), exist_ok=True)

        if not args.skip_convert:
            mat_to_star(mat_path, img_t, bbox_t, phis_t, out_img_dir, out_tsv)

        # (a) tsv + 圖片  vs  原始 mat
        ok = validate_tsv_vs_mat(mat_path, img_t, bbox_t, phis_t,
                                 out_tsv, os.path.join(args.image_dir, "COFW"))
        all_ok &= ok

        # (b) 可選: 與參考 tsv 逐欄位比較
        ref = args.ref_train_tsv if split == "train" else args.ref_test_tsv
        if ref:
            all_ok &= validate_tsv_vs_tsv(out_tsv, ref)

    print("\n全部校驗通過 ✅" if all_ok else "\n存在不一致項，請檢查上方輸出 ❌")
    sys.exit(0 if all_ok else 1)


# The images converted by this script are ready to use directly,
# with their formats fully aligned with the train.tsv and test.tsv provided by the star author.
# The only thing left untested is whether these two converted tsv files can run properly.

if __name__ == "__main__":
    main()
