@echo off
title ArcFace Training Pipeline
echo ==========================================
echo Starting ArcFace Training Pipeline
echo Backbone: IResNet-50
echo Dataset: .rec format in dataset/ directory
echo ==========================================

:: Kiểm tra sự tồn tại của thư mục dataset
if not exist "dataset" (
    echo [ERROR] Thu muc 'dataset' khong ton tai!
    echo Vui long tao thu muc 'dataset' va dat file .rec vao trong.
    pause
    exit /b
)

:: Chạy lệnh huấn luyện
python main.py --mode train --dataset dataset

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Da co loi xay ra trong qua trình huan luyen.
) else (
    echo [SUCCESS] Huan luyen hoan tat!
)

pause
