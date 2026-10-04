// ─── TẬP DỮ LIỆU HUẤN LUYỆN THỰC TẾ (TỪ FINETUNE_METRICS.CSV CỦA FINETUNED_BEST.ONNX) ──────────────────────
const trainingLogs = {
    epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    loss: [12.7378, 12.9860, 13.6725, 15.4910, 18.4784, 19.4600, 18.2300, 17.3367, 16.6293, 16.0973, 15.6625, 15.2994, 14.9840, 14.7321, 14.5260, 14.3348, 14.1957, 14.0552, 13.9490, 13.8683, 13.7999, 13.7573, 13.7209, 13.6873, 13.6649],
    lfwAcc: [0.9840, 0.9843, 0.9838, 0.9842, 0.9845, 0.9853, 0.9843, 0.9847, 0.9830, 0.9828, 0.9832, 0.9832, 0.9818, 0.9822, 0.9833, 0.9830, 0.9818, 0.9823, 0.9815, 0.9823, 0.9823, 0.9815, 0.9827, 0.9825, 0.9822],
    agedbAcc: [0.8682, 0.8687, 0.8740, 0.8705, 0.8678, 0.8723, 0.8705, 0.8708, 0.8690, 0.8632, 0.8633, 0.8603, 0.8612, 0.8582, 0.8595, 0.8623, 0.8623, 0.8620, 0.8602, 0.8588, 0.8588, 0.8583, 0.8602, 0.8590, 0.8590],
    cfpAcc: [0.8813, 0.8820, 0.8816, 0.8831, 0.8834, 0.8830, 0.8831, 0.8840, 0.8843, 0.8830, 0.8831, 0.8837, 0.8824, 0.8823, 0.8816, 0.8831, 0.8814, 0.8800, 0.8800, 0.8790, 0.8791, 0.8791, 0.8804, 0.8801, 0.8787]
};

// ─── THỐNG KÊ SINH TRẮC HỌC (BIÊN PHÂN PHỐI KHOẢNG CÁCH) ──────────────
// Được cập nhật từ chỉ số thực tế trên tập LFW của mô hình finetuned_best.onnx:
// - Positive (cặp cùng người): Mean = 1.00 (tương đương Cosine 0.498), Std = 0.28
// - Negative (cặp khác người): Mean = 1.95 (tương đương Cosine 0.024), Std = 0.15
const stats = {
    posMean: 1.00,
    posStd: 0.28,
    negMean: 1.95,
    negStd: 0.15
};

// Hàm xấp xỉ lỗi Gaussian (erf) để tính CDF
function erf(x) {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    const sign = x < 0 ? -1 : 1;
    const t = 1.0 / (1.0 + p * Math.abs(x));
    const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
}

// Hàm phân phối lũy tích (CDF) của phân phối chuẩn
function normalCDF(x, mean, std) {
    const z = (x - mean) / (std * Math.sqrt(2));
    return 0.5 * (1 + erf(z));
}

// Hàm mật độ xác suất (PDF) của phân phối chuẩn
function normalPDF(x, mean, std) {
    return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
}

// ─── ĐIỀU PHỐI TAB TRANG WEB ─────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', function(e) {
        e.preventDefault();
        
        // Tắt webcam nếu chuyển đi từ tab webcam
        const targetTab = this.getAttribute('data-tab');
        if (targetTab !== 'webcam-realtime') {
            stopWebcam();
        }
        
        // Cập nhật class Active trong menu sidebar
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        this.classList.add('active');
        
        // Chuyển đổi hiển thị Panel nội dung
        document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
        document.getElementById(`tab-${targetTab}`).classList.add('active');
        
        // Thay đổi tiêu đề trang
        const pageTitle = document.getElementById('page-title');
        if (targetTab === 'dashboard') pageTitle.textContent = "Hệ thống Nhận diện Khuôn mặt ArcFace";
        else if (targetTab === 'live-demo') pageTitle.textContent = "Thử nghiệm Xác thực Khuôn mặt 1:1";
        else if (targetTab === 'webcam-realtime') pageTitle.textContent = "Nhận diện Trực tiếp (Webcam Real-time)";
        else if (targetTab === 'training-process') pageTitle.textContent = "Quy trình Huấn luyện & Tối ưu hóa ArcFace";
        else if (targetTab === 'robustness-evaluation') pageTitle.textContent = "Đánh giá Độ bền bỉ & Chống Giả mạo (Robustness)";
        else if (targetTab === 'sota-techniques') pageTitle.textContent = "Các Kỹ Thuật Huấn Luyện SOTA Của Mô Hình";
    });
});

// ─── KHỞI TẠO BIỂU ĐỒ (CHART.JS) ─────────────────────────────────────
let trainingChart, distChart;

function initCharts() {
    // 1. Biểu đồ lịch sử huấn luyện (Training History Chart)
    const ctxTrain = document.getElementById('trainingChart').getContext('2d');
    trainingChart = new Chart(ctxTrain, {
        type: 'line',
        data: {
            labels: trainingLogs.epochs,
            datasets: [
                {
                    label: 'Train Loss (Y1)',
                    data: trainingLogs.loss,
                    borderColor: '#EE4C2C',
                    backgroundColor: 'rgba(238, 76, 44, 0.05)',
                    borderWidth: 2,
                    yAxisID: 'y',
                    tension: 0.2
                },
                {
                    label: 'LFW Validation Acc (Y2)',
                    data: trainingLogs.lfwAcc,
                    borderColor: '#00f2fe',
                    backgroundColor: 'transparent',
                    borderWidth: 2.5,
                    borderDash: [2, 2],
                    yAxisID: 'y1',
                    tension: 0.2
                },
                {
                    label: 'AgeDB-30 Acc (Y2)',
                    data: trainingLogs.agedbAcc,
                    borderColor: '#b19ffb',
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    yAxisID: 'y1',
                    tension: 0.2
                },
                {
                    label: 'CFP-FP Acc (Y2)',
                    data: trainingLogs.cfpAcc,
                    borderColor: '#38f9d7',
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    yAxisID: 'y1',
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#9CA3AF', font: { family: 'Inter', size: 11 } }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Epoch', color: '#9CA3AF' },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#9CA3AF' }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'ArcFace Loss', color: '#EE4C2C' },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#9CA3AF' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Accuracy', color: '#00f2fe' },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#9CA3AF' },
                    min: 0.4,
                    max: 1.0
                }
            }
        }
    });

    // 2. Biểu đồ Phân phối khoảng cách (Distribution Chart)
    const ctxDist = document.getElementById('distanceDistributionChart').getContext('2d');
    const distLabels = [];
    const posDataset = [];
    const negDataset = [];
    
    for (let x = 0.2; x <= 2.2; x += 0.05) {
        distLabels.push(x.toFixed(2));
        posDataset.push(normalPDF(x, stats.posMean, stats.posStd));
        negDataset.push(normalPDF(x, stats.negMean, stats.negStd));
    }

    distChart = new Chart(ctxDist, {
        type: 'line',
        data: {
            labels: distLabels,
            datasets: [
                {
                    label: 'Cùng danh tính (Intra-class)',
                    data: posDataset,
                    borderColor: '#38f9d7',
                    backgroundColor: 'rgba(56, 249, 215, 0.1)',
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2
                },
                {
                    label: 'Khác danh tính (Inter-class)',
                    data: negDataset,
                    borderColor: '#ff9a9e',
                    backgroundColor: 'rgba(255, 154, 158, 0.1)',
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#9CA3AF', font: { family: 'Inter' } }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Khoảng cách Euclidean L2', color: '#9CA3AF' },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#9CA3AF' }
                },
                y: {
                    display: false
                }
            }
        }
    });
}



// ─── LIVE DEMO 1:1 LOGIC ────────────────────────────────────────────

// Trạng thái Demo
const state = {
    mode: 'onnx', // Mặc định luôn là ONNX
    onnxLoaded: false,
    session: null,
    faceDetector: null, // Google MediaPipe Face Detector for auto-crop
    faceBoxA: null, // Toạ độ khuôn mặt A {x, y, w, h}
    faceBoxB: null, // Toạ độ khuôn mặt B {x, y, w, h}
    imageA: null, // Image Element hoặc File
    imageB: null,
    lastDistance: null,
    lastCosine: null
};

// Cấu hình Nút (Sẽ được gán động sau khi tải trang con)
let imgA = null;
let imgB = null;
let placeholderA = null;
let placeholderB = null;

let dropzoneA = null;
let dropzoneB = null;
let fileInputA = null;
let fileInputB = null;

let btnVerify = null;
let resultsDisplay = null;

function initLiveDemoDOMElements() {
    imgA = document.getElementById('img-preview-A');
    imgB = document.getElementById('img-preview-B');
    placeholderA = document.getElementById('placeholder-A');
    placeholderB = document.getElementById('placeholder-B');

    dropzoneA = document.getElementById('dropzone-A');
    dropzoneB = document.getElementById('dropzone-B');
    fileInputA = document.getElementById('file-input-A');
    fileInputB = document.getElementById('file-input-B');

    btnVerify = document.getElementById('btn-verify');
    resultsDisplay = document.getElementById('results-display');
    
    if (btnVerify) {
        btnVerify.addEventListener('click', runVerification);
    }
    
    // Đăng ký sự kiện thay đổi mức bảo mật để cập nhật ngay kết quả
    const securitySelect = document.getElementById('security-level');
    if (securitySelect) {
        securitySelect.addEventListener('change', function() {
            if (state.lastDistance !== null) {
                const threshold = parseFloat(this.value);
                const isMatch = state.lastDistance <= threshold;
                showResults(state.lastDistance, state.lastCosine, isMatch, threshold, state.lastNormA, state.lastNormB);
            }
        });
    }
}

// Xử lý nạp ảnh tự do ở chế độ ONNX
function enableCustomUpload() {
    if (!dropzoneA || !dropzoneB || !fileInputA || !fileInputB) return;
    
    dropzoneA.addEventListener('click', triggerInputA);
    dropzoneB.addEventListener('click', triggerInputB);
    
    dropzoneA.addEventListener('dragover', handleDragOver);
    dropzoneB.addEventListener('dragover', handleDragOver);
    
    dropzoneA.addEventListener('dragleave', handleDragLeave);
    dropzoneB.addEventListener('dragleave', handleDragLeave);
    
    dropzoneA.addEventListener('drop', handleDropA);
    dropzoneB.addEventListener('drop', handleDropB);
    
    fileInputA.addEventListener('change', handleFileA);
    fileInputB.addEventListener('change', handleFileB);
}

function triggerInputA() { if (fileInputA) fileInputA.click(); }
function triggerInputB() { if (fileInputB) fileInputB.click(); }

function handleDragOver(e) {
    e.preventDefault();
    this.classList.add('dragover');
}

function handleDragLeave() {
    this.classList.remove('dragover');
}

function handleDropA(e) {
    e.preventDefault();
    this.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        processFile(e.dataTransfer.files[0], 'A');
    }
}

function handleDropB(e) {
    e.preventDefault();
    this.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        processFile(e.dataTransfer.files[0], 'B');
    }
}

function handleFileA(e) {
    if (e.target.files.length > 0) {
        processFile(e.target.files[0], 'A');
    }
}

function handleFileB(e) {
    if (e.target.files.length > 0) {
        processFile(e.target.files[0], 'B');
    }
}

function processFile(file, side) {
    const reader = new FileReader();
    reader.onload = function(event) {
        const img = (side === 'A') ? imgA : imgB;
        const placeholder = (side === 'A') ? placeholderA : placeholderB;
        
        img.onload = async function() {
            if (btnVerify) btnVerify.setAttribute('disabled', 'true');
            // Chạy phát hiện khuôn mặt trực tiếp ngay khi ảnh tải xong
            await detectAndDrawFace(side);
            checkVerifyButtonStatus();
        };
        
        img.src = event.target.result;
        img.style.display = 'block';
        placeholder.style.display = 'none';
        
        if (side === 'A') state.imageA = img;
        else state.imageB = img;
    };
    reader.readAsDataURL(file);
}

function checkVerifyButtonStatus() {
    if (!btnVerify) return;
    if (state.imageA && state.imageB && state.onnxLoaded) {
        btnVerify.removeAttribute('disabled');
    } else {
        btnVerify.setAttribute('disabled', 'true');
    }
}

function clearImages() {
    if (imgA) {
        imgA.src = "";
        imgA.style.display = 'none';
    }
    if (placeholderA) placeholderA.style.display = 'flex';
    
    if (imgB) {
        imgB.src = "";
        imgB.style.display = 'none';
    }
    if (placeholderB) placeholderB.style.display = 'flex';
    
    state.imageA = null;
    state.imageB = null;
    state.faceBoxA = null;
    state.faceBoxB = null;
    state.keypointsA = null;
    state.keypointsB = null;
    state.lastDistance = null;
    state.lastCosine = null;
    
    // Xoá overlay canvas
    clearOverlay('A');
    clearOverlay('B');
    
    if (btnVerify) btnVerify.setAttribute('disabled', 'true');
    if (resultsDisplay) resultsDisplay.style.display = 'none';
}

function clearOverlay(side) {
    const canvas = document.getElementById(`overlay-${side}`);
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

// ─── TỰ ĐỘNG DÒ TÌM VÀ VẼ OVERLAY KHUÔN MẶT (MEDIAPIPE FACE DETECTOR) ──
async function detectAndDrawFace(side) {
    const img = (side === 'A') ? imgA : imgB;
    const canvas = document.getElementById(`overlay-${side}`);
    if (!canvas || !img) return;
    const ctx = canvas.getContext('2d');
    
    if (!state.faceDetector) {
        console.log("MediaPipe Face Detector is not loaded yet.");
        return;
    }
    
    // Đảm bảo ảnh đã tải xong hoàn toàn
    if (!img.complete || img.naturalWidth === 0) {
        return;
    }
    
    // Đồng bộ độ phân giải nội bộ của canvas khớp 100% với kích thước tự nhiên của ảnh
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    try {
        console.log(`Running MediaPipe Face Detection on side ${side} with natural dimensions ${img.naturalWidth}x${img.naturalHeight}...`);
        
        const predictions = await state.faceDetector.estimateFaces(img);
        
        if (predictions.length > 0) {
            const pred = predictions[0];
            const box = pred.box;
            
            // Lưu lại toạ độ gốc trên ảnh tự nhiên để phục vụ tiền xử lý crop ảnh
            const faceBox = {
                x: box.xMin,
                y: box.yMin,
                w: box.width,
                h: box.height
            };
            
            if (side === 'A') {
                state.faceBoxA = faceBox;
                state.keypointsA = pred.keypoints;
            } else {
                state.faceBoxB = faceBox;
                state.keypointsB = pred.keypoints;
            }
            
            // Độ dày nét vẽ tỷ lệ theo độ phân giải ảnh tự nhiên
            const baseScale = img.naturalWidth;
            const strokeWidth = Math.max(3, Math.round(baseScale / 120));
            const blurSize = Math.max(4, Math.round(baseScale / 90));
            const dotRadius = Math.max(3, Math.round(baseScale / 200));
            
            // Vẽ bounding box màu xanh Neon
            ctx.strokeStyle = '#38f9d7';
            ctx.lineWidth = strokeWidth;
            ctx.shadowBlur = blurSize;
            ctx.shadowColor = '#38f9d7';
            ctx.strokeRect(box.xMin, box.yMin, box.width, box.height);
            
            // Vẽ 6 điểm mốc quan trọng (mắt, mũi, miệng, tai) màu xanh dương Neon
            ctx.fillStyle = '#00f2fe';
            ctx.shadowColor = '#00f2fe';
            if (pred.keypoints && pred.keypoints.length > 0) {
                pred.keypoints.forEach(kp => {
                    ctx.beginPath();
                    ctx.arc(kp.x, kp.y, dotRadius, 0, 2 * Math.PI);
                    ctx.fill();
                });
            }
            
            console.log(`Successfully drawn face box for side ${side}:`, faceBox);
        } else {
            if (side === 'A') {
                state.faceBoxA = null;
                state.keypointsA = null;
            } else {
                state.faceBoxB = null;
                state.keypointsB = null;
            }
            console.log(`No face detected on side ${side}.`);
        }
    } catch (error) {
        console.error(`Error during face detection on side ${side}:`, error);
    }
}

// ─── TỰ ĐỘNG NẠP MÔ HÌNH ARCFACE ONNX ───────────────────────────────
async function loadArcFaceModel() {
    const onnxStatusText = document.getElementById('onnx-status-text');
    const onnxIndicator = document.getElementById('onnx-indicator');
    
    onnxStatusText.textContent = 'Đang tự động nạp mô hình ArcFace (WASM)...';
    onnxIndicator.className = "status-indicator yellow";
    
    try {
        const modelPath = 'models/finetuned_best.onnx';
        const options = { executionProviders: ['wasm'] };
        
        state.session = await ort.InferenceSession.create(modelPath, options);
        state.onnxLoaded = true;
        
        onnxStatusText.textContent = 'Đã nạp thành công!';
        onnxIndicator.className = "status-indicator green";
        
        checkVerifyButtonStatus();
        console.log("ArcFace ONNX model loaded successfully!");
    } catch (error) {
        console.error("ONNX Load error:", error);
        onnxStatusText.textContent = 'Nạp thất bại! Đảm bảo server đang chạy.';
        onnxIndicator.className = "status-indicator red";
        onnxIndicator.style.backgroundColor = "red";
        onnxIndicator.style.boxShadow = "0 0 8px red";
    }
}

// ─── SO KHỚP XÁC MINH (RUN VERIFY) ──────────────────────────────────
async function runVerification() {
    if (!resultsDisplay || !dropzoneA || !dropzoneB || !btnVerify) return;

    // Ẩn bảng kết quả cũ, bật hiệu ứng quét quét Laser
    resultsDisplay.style.display = 'none';
    dropzoneA.classList.add('scanning');
    dropzoneB.classList.add('scanning');
    btnVerify.setAttribute('disabled', 'true');
    
    setTimeout(async () => {
        let distance = 0;
        let cosineSim = 0;
        let isMatch = false;
        
        // Chế độ ONNX thật: Trích xuất embedding bằng ONNX Runtime Web!
        try {
            const tensorA = await preprocessImageToTensor(state.imageA);
            const tensorB = await preprocessImageToTensor(state.imageB);
            
            const feedsA = { [state.session.inputNames[0]]: tensorA };
            const feedsB = { [state.session.inputNames[0]]: tensorB };
            
            const outputA = await state.session.run(feedsA);
            const outputB = await state.session.run(feedsB);
            
            const embA = outputA[state.session.outputNames[0]].data; // Float32Array
            const embB = outputB[state.session.outputNames[0]].data;
            
            // L2 Normalization của 2 vector đầu ra (512-D)
            const normA = normalizeL2(embA);
            const normB = normalizeL2(embB);
            
            // Tính toán khoảng cách và độ tương đồng
            distance = calculateSquaredL2Distance(normA, normB);
            cosineSim = calculateCosineSimilarity(normA, normB);
            
            // Lưu lại kết quả vào state để phục vụ thay đổi dropdown tức thời
            state.lastDistance = distance;
            state.lastCosine = cosineSim;
            state.lastNormA = normA;
            state.lastNormB = normB;
        } catch (err) {
            console.error("Inference Error:", err);
            alert("Đã xảy ra lỗi khi chạy ONNX Web. Kiểm tra định dạng đầu vào hoặc console log.");
            dropzoneA.classList.remove('scanning');
            dropzoneB.classList.remove('scanning');
            btnVerify.removeAttribute('disabled');
            return;
        }
        
        // Lấy ngưỡng từ dropdown bảo mật (Mặc định là 1.30 tương đương Cosine 0.65)
        const securitySelect = document.getElementById('security-level');
        const threshold = securitySelect ? parseFloat(securitySelect.value) : 1.20;
        isMatch = distance <= threshold;
        
        // Tắt hiệu ứng Laser quét
        dropzoneA.classList.remove('scanning');
        dropzoneB.classList.remove('scanning');
        btnVerify.removeAttribute('disabled');
        
        // Cập nhật giao diện kết quả
        showResults(distance, cosineSim, isMatch, threshold, state.lastNormA, state.lastNormB);
    }, 2000); // Trì hoãn 2 giây mô phỏng quét cho đẹp mắt
}

// Tiền xử lý ảnh HTML5 Canvas -> Tensor Float32 [1, 3, 112, 112]
// NẾU CÓ TOẠ ĐỘ KHUÔN MẶT -> TỰ ĐỘNG CẮT SÁT MẶT (CROP & ALIGN)
async function preprocessImageToTensor(imgElement, faceBoxCustom = null, keypointsCustom = null) {
    const canvas = document.getElementById('canvas-preprocessing');
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, 112, 112);
    
    const side = (imgElement === imgA) ? 'A' : 'B';
    
    const toggleAutoCrop = document.getElementById('toggle-auto-crop');
    const autoCropEnabled = toggleAutoCrop ? toggleAutoCrop.checked : true;
    
    const faceBox = autoCropEnabled ? (faceBoxCustom !== null ? faceBoxCustom : ((side === 'A') ? state.faceBoxA : state.faceBoxB)) : null;
    const keypoints = autoCropEnabled ? (keypointsCustom !== null ? keypointsCustom : ((side === 'A') ? state.keypointsA : state.keypointsB)) : null;
    
    const naturalWidth = imgElement.naturalWidth || imgElement.videoWidth || imgElement.width;
    const naturalHeight = imgElement.naturalHeight || imgElement.videoHeight || imgElement.height;
    
    let processed = false;
    
    // 1. Thử căn chỉnh bằng 2 mắt sử dụng Similarity Transform (nhất quán với PyTorch training)
    if (keypoints && keypoints.length >= 2) {
        // Đảm bảo sắp xếp đúng mắt trái (bên trái ảnh) và mắt phải (bên phải ảnh) theo trục X
        let eyeLeft = keypoints[0];
        let eyeRight = keypoints[1];
        if (eyeLeft.x > eyeRight.x) {
            eyeLeft = keypoints[1];
            eyeRight = keypoints[0];
        }
        
        if (eyeLeft && eyeRight) {
            const dx = eyeRight.x - eyeLeft.x;
            const dy = eyeRight.y - eyeLeft.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist > 5) { // Đảm bảo khoảng cách mắt hợp lý
                // Tọa độ đích cho mắt trong ảnh 112x112 theo chuẩn ArcFace:
                // Mắt trái (trong ảnh): (38.2946, 51.6963)
                // Mắt phải (trong ảnh): (73.5318, 51.5014)
                const targetEyeDist = 35.2372;
                const targetLeftX = 38.2946;
                const targetLeftY = 51.6963;
                
                // Tính toán ma trận biến đổi Similarity Transform
                const a = (targetEyeDist * dx) / (dist * dist);
                const b = (-targetEyeDist * dy) / (dist * dist);
                
                const tx = targetLeftX - (a * eyeLeft.x - b * eyeLeft.y);
                const ty = targetLeftY - (b * eyeLeft.x + a * eyeLeft.y);
                
                // Áp dụng biến đổi ma trận để xoay, dịch chuyển và co giãn ảnh gốc
                ctx.setTransform(a, b, -b, a, tx, ty);
                ctx.drawImage(imgElement, 0, 0);
                ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset ma trận về mặc định
                
                const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                console.log(`Aligned face using eyes similarity transform on side ${side}: angle = ${angle.toFixed(1)} deg, eye distance = ${dist.toFixed(1)}px`);
                processed = true;
            }
        }
    }
    
    // 2. Nếu không căn chỉnh được bằng mắt (hoặc không tìm thấy keypoints), fallback về crop hộp bao vuông
    if (!processed && faceBox) {
        // Tăng lề bao quanh hộp khuôn mặt khoảng 10% để thống nhất với cấu hình trên app di động (PassNote)
        const marginX = faceBox.w * 0.10;
        const marginY = faceBox.h * 0.10;
        
        let cropX = faceBox.x - marginX;
        let cropY = faceBox.y - marginY;
        let cropW = faceBox.w + 2 * marginX;
        let cropH = faceBox.h + 2 * marginY;
        
        // Biến đổi vùng cắt thành hình vuông để tránh làm méo mặt khi resize
        const size = Math.max(cropW, cropH);
        const centerX = cropX + cropW / 2;
        const centerY = cropY + cropH / 2;
        
        cropX = Math.max(0, centerX - size / 2);
        cropY = Math.max(0, centerY - size / 2);
        cropW = Math.min(naturalWidth - cropX, size);
        cropH = Math.min(naturalHeight - cropY, size);
        
        // Vẽ vùng đã cắt sát mặt lên canvas chuẩn 112x112
        ctx.drawImage(imgElement, cropX, cropY, cropW, cropH, 0, 0, 112, 112);
        console.log(`Fallback to square crop on side ${side}: x=${cropX.toFixed(1)}, y=${cropY.toFixed(1)}, w=${cropW.toFixed(1)}`);
        processed = true;
    }
    
    // 3. Nếu hoàn toàn không có mặt, co dãn toàn bộ ảnh
    if (!processed) {
        ctx.drawImage(imgElement, 0, 0, 112, 112);
        console.log(`No face box or keypoints detected. Fallback to full image resize on side ${side}`);
    }
    
    // Vẽ lên các canvas hiển thị đầu vào AI (A) và (B) (chỉ áp dụng cho ảnh tĩnh trong Live Demo)
    if (!(imgElement instanceof HTMLVideoElement)) {
        const previewCanvas = document.getElementById(`canvas-preview-${side}`);
        if (previewCanvas) {
            const previewCtx = previewCanvas.getContext('2d');
            previewCtx.clearRect(0, 0, 112, 112);
            previewCtx.drawImage(canvas, 0, 0);
        }
    }
    
    const imgData = ctx.getImageData(0, 0, 112, 112);
    const data = imgData.data; // RGBA byte array
    
    const floatData = new Float32Array(3 * 112 * 112);
    
    // Chuẩn hóa ảnh: (x - 127.5) / 128.0 để khớp chính xác với cấu hình huấn luyện ArcFace PyTorch
    // Sắp xếp dạng [1, 3, 112, 112] (Planar RGB)
    for (let i = 0; i < 112 * 112; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        
        floatData[i] = (r - 127.5) / 128.0;
        floatData[112 * 112 + i] = (g - 127.5) / 128.0;
        floatData[2 * 112 * 112 + i] = (b - 127.5) / 128.0;
    }
    
    return new ort.Tensor('float32', floatData, [1, 3, 112, 112]);
}

// Chuẩn hóa L2 cho một mảng vector đặc trưng
function normalizeL2(vec) {
    let sumSquares = 0;
    for (let i = 0; i < vec.length; i++) {
        sumSquares += vec[i] * vec[i];
    }
    const norm = Math.sqrt(Math.max(sumSquares, 1e-9));
    const normalized = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
        normalized[i] = vec[i] / norm;
    }
    return normalized;
}

// Tính khoảng cách Euclidean bình phương (L2^2)
function calculateSquaredL2Distance(vecA, vecB) {
    let diffSum = 0;
    for (let i = 0; i < vecA.length; i++) {
        const diff = vecA[i] - vecB[i];
        diffSum += diff * diff;
    }
    return diffSum;
}

// Tính độ tương đồng Cosine
function calculateCosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
    }
    return dotProduct; // Vì đã chuẩn hóa L2 nên dotProduct chính là Cosine similarity
}

function showResults(dist, cosine, isMatch, threshold, normA = null, normB = null) {
    const matchBadge = document.getElementById('match-badge');
    const resultText = document.getElementById('result-text');
    
    const distVal = document.getElementById('dist-result-val');
    const distProgress = document.getElementById('dist-progress');
    const distNote = document.getElementById('dist-note');
    
    const cosVal = document.getElementById('cosine-result-val');
    const cosProgress = document.getElementById('cosine-progress');
    
    if (matchBadge) {
        if (isMatch) {
            matchBadge.className = "result-badge match";
            matchBadge.textContent = "TRÙNG KHỚP";
        } else {
            matchBadge.className = "result-badge no-match";
            matchBadge.textContent = "KHÁC NHAU";
        }
    }

    if (resultText) {
        if (isMatch) {
            resultText.innerHTML = "Xác nhận: <span style='color:#38f9d7; font-weight:bold;'>CÙNG một người!</span>";
        } else {
            resultText.innerHTML = "Xác nhận: <span style='color:#ef4444; font-weight:bold;'>KHÁC biệt danh tính!</span>";
        }
    }
    
    // Cập nhật số liệu hiển thị
    if (distVal) distVal.textContent = dist.toFixed(3);
    
    // Tính phần trăm hiển thị cho L2 distance (tầm từ 0 đến 2.0)
    const distPercent = Math.min((dist / 2.0) * 100, 100);
    if (distProgress) distProgress.style.width = `${distPercent}%`;
    if (distNote) distNote.textContent = `Ngưỡng: ≤ ${threshold.toFixed(2)} (Thực tế: ${dist <= threshold ? 'Đạt' : 'Vượt quá'})`;
    
    if (cosVal) cosVal.textContent = cosine.toFixed(3);
    // Chuẩn hóa Cosine từ [-1, 1] sang [0%, 100%]
    const cosPercent = ((cosine + 1) / 2) * 100;
    if (cosProgress) cosProgress.style.width = `${cosPercent}%`;
    
    if (normA && normB) {
        renderVectorStrip('vector-strip-a', normA);
        renderVectorStrip('vector-strip-b', normB);
    }
    
    if (resultsDisplay) resultsDisplay.style.display = 'flex';
}

// Hàm nạp động nội dung Live Demo từ file html ngoài
async function loadLiveDemoHTML() {
    try {
        const response = await fetch('live_demo.html');
        if (!response.ok) throw new Error("Không thể nạp file live_demo.html");
        const html = await response.text();
        const container = document.getElementById('tab-live-demo');
        if (container) {
            container.innerHTML = html;
            console.log("Đã nạp thành công Live Demo!");
            if (window.MathJax) {
                MathJax.typesetPromise([container]).catch(err => console.error("MathJax error (Live Demo):", err));
            }
        }
    } catch (error) {
        console.error("Lỗi khi load Live Demo:", error);
        const container = document.getElementById('tab-live-demo');
        if (container) {
            container.innerHTML = `<div class="banner warning-banner" style="margin: 2rem;">Lỗi khi tải Live Demo: ${error.message}</div>`;
        }
    }
}

// Hàm nạp động nội dung Webcam Real-time từ file html ngoài
async function loadWebcamRealtimeHTML() {
    try {
        const response = await fetch('webcam_realtime.html');
        if (!response.ok) throw new Error("Không thể nạp file webcam_realtime.html");
        const html = await response.text();
        const container = document.getElementById('tab-webcam-realtime');
        if (container) {
            container.innerHTML = html;
            console.log("Đã nạp thành công Webcam Real-time!");
        }
    } catch (error) {
        console.error("Lỗi khi load Webcam Real-time:", error);
    }
}

// Hàm nạp động nội dung Quy trình train từ file html ngoài
async function loadTrainingProcessHTML() {
    try {
        const response = await fetch('training_process.html');
        if (!response.ok) throw new Error("Không thể nạp file training_process.html");
        const html = await response.text();
        const container = document.getElementById('tab-training-process');
        if (container) {
            container.innerHTML = html;
            console.log("Đã nạp thành công Quy trình train!");
            if (window.MathJax) {
                MathJax.typesetPromise([container]).catch(err => console.error("MathJax error:", err));
            }
        }
    } catch (error) {
        console.error("Lỗi khi load Quy trình train:", error);
        const container = document.getElementById('tab-training-process');
        if (container) {
            container.innerHTML = `<div class="banner warning-banner" style="margin: 2rem;">Lỗi khi tải quy trình train: ${error.message}</div>`;
        }
    }
}

// Hàm nạp động nội dung Đánh giá Robustness từ file html ngoài
async function loadRobustnessEvaluationHTML() {
    try {
        const response = await fetch('robustness_evaluation.html');
        if (!response.ok) throw new Error("Không thể nạp file robustness_evaluation.html");
        const html = await response.text();
        const container = document.getElementById('tab-robustness-evaluation');
        if (container) {
            container.innerHTML = html;
            console.log("Đã nạp thành công Đánh giá Robustness!");
            if (window.MathJax) {
                MathJax.typesetPromise([container]).catch(err => console.error("MathJax error (Robustness):", err));
            }
        }
    } catch (error) {
        console.error("Lỗi khi load Đánh giá Robustness:", error);
        const container = document.getElementById('tab-robustness-evaluation');
        if (container) {
            container.innerHTML = `<div class="banner warning-banner" style="margin: 2rem;">Lỗi khi tải đánh giá robustness: ${error.message}</div>`;
        }
    }
}

// ─── ĐỌC DỮ LIỆU VÀ TƯƠNG TÁC TỔNG HỢP 15 KỸ THUẬT SOTA ───────────────────────
const sotaData = {
    iresnet: {
        title: "IResNet-50 Backbone",
        category: "Kiến trúc & Hàm Loss",
        desc: "Mạng xương sống IResNet-50 (Improved ResNet) là mạng nơ-ron tích chập sâu cải tiến được tối ưu hóa riêng cho việc trích chọn đặc trưng khuôn mặt (biometric embedding).",
        significance: "Giúp luồng gradient chảy mượt mà ở các tầng cực sâu, giữ lại các đặc trưng âm quan trọng thông qua PReLU và ngăn mất mát thông tin hình học của các điểm mốc trên mặt nhờ loại bỏ hoàn toàn tầng Max Pooling.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Backbone</strong></td><td>iresnet50</td><td>Độ sâu 50 tầng tích chập cải tiến</td></tr>
                <tr><td><strong>Dropout</strong></td><td>0.4</td><td>Tỷ lệ ngắt kết nối để chống overfitting</td></tr>
                <tr><td><strong>Activation</strong></td><td>PReLU</td><td>Học thêm hệ số cho phần giá trị âm</td></tr>
            </table>
        `,
        math: "Công thức PReLU: $$f(x) = \\max(0, x) + a \\cdot \\min(0, x)$$ trong đó $a$ là tham số có thể học được để giữ lại đặc trưng âm thay vì gán bằng 0 như ReLU thường.",
        code: `class IResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes) # Phép kích hoạt PReLU học tham số
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes)`
    },
    arcface: {
        title: "ArcFace Loss (Additive Angular Margin)",
        category: "Kiến trúc & Hàm Loss",
        desc: "Hàm mất mát ArcFace đưa thêm một biên độ phạt góc m trực tiếp vào góc θ giữa đặc trưng ảnh và trọng số lớp đúng, nén chặt đặc trưng cùng nhóm trên mặt cầu đơn vị.",
        significance: "Siết chặt khoảng cách giữa các bức ảnh của cùng một người (intra-class) và đẩy rộng khoảng cách giữa những người khác nhau (inter-class), tạo ranh giới bảo mật cực rộng.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Margin (m)</strong></td><td>0.4 (Train) / 0.5 (Fine-tune)</td><td>Biên độ phạt góc (khoảng 23 - 28 độ)</td></tr>
                <tr><td><strong>Scale (s)</strong></td><td>64.0</td><td>Bán kính mặt cầu đơn vị khuếch đại logits để tối ưu hóa tốt hơn</td></tr>
            </table>
        `,
        math: "Công thức ArcFace: $$L = -\\log \\frac{e^{s \\cdot \\cos(\\theta_{y_i} + m)}}{e^{s \\cdot \\cos(\\theta_{y_i} + m)} + \\sum_{j \\neq y_i} e^{s \\cdot \\cos\\theta_j}}$$",
        code: `class ArcMarginProduct(nn.Module):
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m # cos(theta + m)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s`
    },
    onnx: {
        title: "Export mô hình ONNX",
        category: "Kiến trúc & Hàm Loss",
        desc: "Trọng số huấn luyện tốt nhất (.pth) được tối ưu hóa cấu trúc đồ thị tính toán và xuất sang định dạng ONNX (.onnx) để chạy suy luận trực tiếp trên trình duyệt client.",
        significance: "Cho phép chạy nhận diện thời gian thực trên web bằng công nghệ ONNX Runtime Web (WASM) với độ trễ dưới 20ms mà không cần GPU hay Server tính toán hộ.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>File xuất</strong></td><td>finetuned_best.onnx</td><td>Model ONNX được tích hợp vào thư mục Web</td></tr>
                <tr><td><strong>Runtime Providers</strong></td><td>['wasm']</td><td>Chạy trên WebAssembly đa luồng trên trình duyệt web</td></tr>
            </table>
        `,
        math: "Đo lường độ trễ trích xuất đặc trưng trên CPU Client (Web): $$\\Delta t \\le 20\\text{ ms}$$ nhờ tập lệnh SIMD đa luồng.",
        code: `# export_onnx.py
import torch
model = get_backbone('iresnet50')
model.load_state_dict(torch.load('backbone_best.pth'))
dummy_input = torch.randn(1, 3, 112, 112)
torch.onnx.export(model, dummy_input, 'finetuned_best.onnx', 
                  input_names=['input'], output_names=['output'],
                  opset_version=12)`
    },
    recordio: {
        title: "Pure Python RecordIO Reader",
        category: "Data Pipeline",
        desc: "Bộ đọc nhị phân RecordIO viết hoàn toàn bằng Python sử dụng tệp chỉ mục (.idx) để seek vị trí byte (Offset) và hàm struct unpack để giải mã header sinh trắc học.",
        significance: "Thay thế hoàn toàn thư viện MXNet khổng lồ và lỗi thời, cho phép chạy mã nguồn huấn luyện dễ dàng trên Windows, Linux mà không gặp lỗi biên dịch thư viện C++.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Header định dạng</strong></td><td>=fQQi</td><td>Kích thước cứng 24 bytes (không bị padding tự động của HĐH)</td></tr>
                <tr><td><strong>Cách căn lề</strong></td><td>4-byte alignment</td><td>padding = (4 - (length % 4)) % 4</td></tr>
            </table>
        `,
        math: "Công thức tính số byte đệm: $$\\text{padding} = (4 - (L \\bmod 4)) \\bmod 4$$ để nhảy con trỏ tệp đến bản ghi tiếp theo một cách chuẩn xác.",
        code: `# recordio.py
magic, length_flant = struct.unpack('<II', header)
length = length_flant & 0x00ffffff
data = self.f.read(length)
padding = (4 - (length % 4)) % 4
if padding > 0:
    self.f.read(padding)`
    },
    flip: {
        title: "Random Horizontal Flip (Lật ảnh ngẫu nhiên)",
        category: "Data Pipeline",
        desc: "Trong quá trình nạp dữ liệu train, mỗi bức ảnh khuôn mặt có 50% cơ hội bị lật ngược theo chiều ngang (trái sang phải) trước khi đưa vào mạng tích chập.",
        significance: "Kỹ thuật tăng cường dữ liệu cơ bản giúp mô hình không bị học vẹt vị trí nốt ruồi, hướng nghiêng mặt của tập huấn luyện, nâng cao độ tổng quát hóa.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Xác suất lật</strong></td><td>0.5 (50%)</td><td>transforms.RandomHorizontalFlip()</td></tr>
            </table>
        `,
        math: "Phép biến đổi tọa độ ảnh: $$(x, y) \\to (W - 1 - x, y)$$ để tạo ra biến thể khuôn mặt đối xứng.",
        code: `# dataset.py
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # Tự động lật ảnh ngẫu nhiên
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])`
    },
    tta: {
        title: "Test-Time Augmentation (TTA)",
        category: "Data Pipeline",
        desc: "Kỹ thuật trích xuất vector đặc trưng của cả ảnh gốc và ảnh lật ngang trong lúc kiểm thử (validation), sau đó cộng hai vector lại và chuẩn hóa L2.",
        significance: "Giúp vector đặc trưng ổn định hơn trước các thay đổi góc chụp trái/phải, triệt tiêu nhiễu nhạy cảm hướng và tăng thêm 0.5% - 1% Accuracy.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Kích thước ảnh</strong></td><td>112x112</td><td>Được lật ngang qua trục width (dim 3)</td></tr>
            </table>
        `,
        math: "Công thức gộp vector đặc trưng: $$\\vec{f}_{TTA} = \\text{Normalize}(\\vec{f}_{original} + \\vec{f}_{flipped})$$",
        code: `# eval.py - Trong perform_val
batch_flip = torch.flip(batch, [3]) # Lật ngang ảnh
embed = model(batch)
embed_flip = model(batch_flip)
combined = embed + embed_flip
combined = torch.nn.functional.normalize(combined)`
    },
    kfold: {
        title: "10-Fold Cross Validation",
        category: "Data Pipeline",
        desc: "Chia tập kiểm thử (ví dụ 6.000 cặp ảnh LFW) làm 10 phần chéo. Dùng 9 phần để dò tìm ngưỡng phân loại tốt nhất và đo độ chính xác trên phần còn lại.",
        significance: "Đảm bảo kết quả đo đạc độ chính xác khách quan nhất, không bị hiện tượng 'ăn may' trên tập dữ liệu nhỏ và phản ánh đúng hiệu năng thực tế.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Số Splits (K)</strong></td><td>10 Folds</td><td>Được cấu hình bằng sklearn.model_selection.KFold</td></tr>
            </table>
        `,
        math: "Độ chính xác trung bình: $$\\text{Accuracy}_{avg} = \\frac{1}{10} \\sum_{i=1}^{10} \\text{Accuracy}_i$$",
        code: `# eval.py - Tính ROC qua KFold
k_fold = KFold(n_splits=nrof_folds, shuffle=False)
for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
    # Tìm ngưỡng tốt nhất trên tập train_set
    for threshold_idx, threshold in enumerate(thresholds):
        acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set]...)
    best_thresholds[fold_idx] = thresholds[np.argmax(acc_train)]`
    },
    'lr-warmup': {
        title: "Linear Learning Rate Warmup",
        category: "Chiến lược Huấn luyện",
        desc: "Tăng dần tốc độ học từ một giá trị siêu nhỏ (1e-4) lên tốc độ học cơ bản (0.03) trong các epoch đầu tiên của quá trình huấn luyện.",
        significance: "Giúp ổn định các trọng số khởi tạo ngẫu nhiên và các giá trị trung bình động của lớp Batch Normalization, tránh làm bùng nổ gradient ở đầu buổi train.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Thời gian warmup</strong></td><td>3 Epochs</td><td>Config.warmup_epochs = 3</td></tr>
                <tr><td><strong>LR khởi đầu</strong></td><td>1e-4</td><td>Config.warmup_init_lr = 1e-4</td></tr>
                <tr><td><strong>LR cơ bản</strong></td><td>0.03</td><td>Config.base_lr = 0.03</td></tr>
            </table>
        `,
        math: "Công thức tăng LR tuyến tính: $$\\eta_t = \\eta_{init} + (\\eta_{base} - \\eta_{init}) \\cdot \\frac{t}{T_{warmup}}$$",
        code: `# train.py - Tuyến tính tăng LR
if global_step <= warmup_steps:
    alpha = global_step / max(1, warmup_steps)
    cur_lr = Config.warmup_init_lr + (Config.base_lr - Config.warmup_init_lr) * alpha
    for pg in optimizer.param_groups:
        pg['lr'] = cur_lr`
    },
    'margin-warmup': {
        title: "Margin Warmup",
        category: "Chiến lược Huấn luyện",
        desc: "Tăng biên độ phạt góc m của ArcFace một cách tuyến tính từ 0 đến giá trị cực đại m_max (0.4 hoặc 0.5) trong những epoch đầu tiên.",
        significance: "Hình phạt góc m lớn của ArcFace là một ràng buộc khó. Tăng dần m giúp mô hình chuyển dịch mượt mà từ Softmax thường sang ArcFace, tránh lỗi nổ gradient và NaN Loss.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Thời gian warmup</strong></td><td>5 Epochs</td><td>Config.margin_warmup_epochs = 5</td></tr>
                <tr><td><strong>Margin tối đa (m)</strong></td><td>0.4 (Train) / 0.5 (Fine-tune)</td><td>Giá trị biên độ đích</td></tr>
            </table>
        `,
        math: "Công thức tăng margin tuyến tính: $$m_t = m_{max} \\cdot \\frac{t}{T_{margin\\_warmup}}$$",
        code: `# train.py - Tuyến tính tăng Margin
if global_step <= margin_warmup_steps:
    alpha_m = global_step / max(1, margin_warmup_steps)
    cur_m = Config.m * alpha_m
    header.update_margin(cur_m)
else:
    header.update_margin(Config.m)`
    },
    cosine: {
        title: "Cosine Annealing LR Decay",
        category: "Chiến lược Huấn luyện",
        desc: "Phương pháp điều chỉnh tốc độ học giảm dần từ giá trị cực đại về sát cực tiểu (1e-5) theo hình đồ thị đường cong Cosine.",
        significance: "Giúp tốc độ học giảm rất mượt mà ở các epoch cuối, cho phép mô hình thực hiện các bước tinh chỉnh cực nhỏ để hội tụ sâu vào điểm tối ưu thực tế.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>LR tối thiểu</strong></td><td>1e-5</td><td>eta_min = 1e-5</td></tr>
                <tr><td><strong>Thời kỳ giảm</strong></td><td>Epochs - Warmup Epochs</td><td>T_max = 22 epochs</td></tr>
            </table>
        `,
        math: "Công thức giảm LR Cosine: $$\\eta_t = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})\\left(1 + \\cos\\left(\\frac{T_{cur}}{T_{max}}\\pi\\right)\\right)$$",
        code: `# train.py
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=(Config.epochs - Config.warmup_epochs), 
    eta_min=1e-5
)`
    },
    ema: {
        title: "EMA (Exponential Moving Average)",
        category: "Chiến lược Huấn luyện",
        desc: "Duy trì song song một bộ trọng số trung bình động lũy thừa được cập nhật siêu chậm dựa trên trọng số đang train của mô hình chính.",
        significance: "Lọc bỏ hoàn toàn các dao động răng cưa trong quá trình train, hướng mô hình hội tụ vào các vùng cực tiểu bằng phẳng (Flat Minima) giúp nâng cao khả năng tổng quát hóa thực tế.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Hệ số mượt (decay)</strong></td><td>0.999</td><td>Giữ lại 99.9% giá trị lịch sử</td></tr>
            </table>
        `,
        math: "Công thức EMA: $$\\theta_{EMA}^{(t)} = \\beta \\theta_{EMA}^{(t-1)} + (1 - \\beta) \\theta^{(t)}$$ với hệ số $\\beta = 0.999$.",
        code: `# ema.py - Cập nhật trọng số mượt
def update(self):
    with torch.no_grad():
        for name, param in self.model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)`
    },
    'lr-group': {
        title: "Phân rã nhóm tham số & LR riêng biệt (Fine-tuning)",
        category: "Chiến lược Huấn luyện",
        desc: "Chia mô hình thành các nhóm tham số khác nhau: backbone (IResNet) và classifier head (ArcFace) để cấu hình tốc độ học và giảm trọng số (weight decay) riêng lẻ.",
        significance: "Khi fine-tune, ta áp dụng hệ số nhân LR = 0.1 cho backbone để bảo vệ tri thức đã học và LR = 1.0 cho ArcFace head để học nhanh các danh tính mới.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>LR Backbone</strong></td><td>10x nhỏ hơn (0.0003)</td><td>Tránh phá hủy đặc trưng backbone đã tốt</td></tr>
                <tr><td><strong>LR ArcFace Head</strong></td><td>Chuẩn (0.003)</td><td>Học nhanh lớp phân loại mới</td></tr>
            </table>
        `,
        math: "Tốc độ học nhóm: $$\\eta_{backbone} = 0.1 \\cdot \\eta_{base}, \\quad \\eta_{head} = 1.0 \\cdot \\eta_{base}$$",
        code: `# train_finetune.py
groups.append({'params': backbone_decay, 'weight_decay': weight_decay, 'lr_mult': 0.1})
groups.append({'params': header_decay, 'weight_decay': weight_decay, 'lr_mult': 1.0})
# Optimizer nạp nhóm tham số
optimizer = torch.optim.SGD(groups, lr=Config.base_lr)`
    },
    amp: {
        title: "AMP FP16 (Automatic Mixed Precision)",
        category: "Hiệu năng & Phần cứng",
        desc: "Kỹ thuật tự động tính toán các phép tích chập và nhân ma trận nặng ở kiểu 16-bit float (FP16), đồng thời giữ trọng số cập nhật ở kiểu 32-bit (FP32).",
        significance: "Tiết kiệm 50% dung lượng VRAM GPU và tăng tốc độ huấn luyện gấp 2-3 lần nhờ tối ưu hóa lõi phần cứng Tensor Cores của NVIDIA.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Kiểu dữ liệu</strong></td><td>FP16 & FP32</td><td>Độ chính xác hỗn hợp</td></tr>
                <tr><td><strong>Scaler</strong></td><td>GradScaler</td><td>Phóng to gradient chống lỗi underflow</td></tr>
            </table>
        `,
        math: "Hạn chế lỗi Underflow bằng cách nhân phóng đại Loss: $$L_{scaled} = L \\cdot S_{scale}$$ trước khi backward.",
        code: `# train.py - Khối AMP PyTorch
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    features = backbone(imgs)
    thetas   = header(features, labels)
    loss     = criterion(thetas, labels)
scaler.scale(loss).backward()`
    },
    accumulation: {
        title: "Gradient Accumulation (Tích lũy Gradient)",
        category: "Hiệu năng & Phần cứng",
        desc: "Cơ chế gom cộng dồn đạo hàm qua nhiều batch nhỏ trước khi thực hiện bước cập nhật trọng số và xóa bộ nhớ đệm đạo hàm.",
        significance: "Giả lập một Batch Size cực lớn (ví dụ 256) trên một chiếc GPU hạn chế về VRAM chỉ đủ chạy Batch Size thực tế bằng 64, tránh hoàn toàn lỗi Out of Memory.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Batch Size thực</strong></td><td>64</td><td>Chạy vừa vặn trong 6GB VRAM</td></tr>
                <tr><td><strong>Steps tích lũy</strong></td><td>4</td><td>grad_accumulate_steps = 4</td></tr>
                <tr><td><strong>Batch Size ảo</strong></td><td>256</td><td>64 x 4 = 256 ảnh được mô phỏng</td></tr>
            </table>
        `,
        math: "Đạo hàm tích lũy: $$\\nabla \\theta = \\sum_{i=1}^{N_{steps}} \\nabla \\theta_i$$ trước khi chạy `optimizer.step()`.",
        code: `# train_finetune.py
loss_scaled = loss / Config.grad_accumulate_steps
scaler.scale(loss_scaled).backward()
if (i + 1) % Config.grad_accumulate_steps == 0:
    scaler.step(optimizer)
    optimizer.zero_grad()`
    },
    safeguards: {
        title: "NaN Safeguards & Gradient Clipping",
        category: "Hiệu năng & Phần cứng",
        desc: "Các biện pháp bảo vệ số thực như kẹp giá trị cosine, giới hạn số chia dưới căn thức khi tính toán ArcFace và khống chế độ dài vector gradient tối đa.",
        significance: "Ngăn chặn hoàn toàn lỗi sụp đổ mô hình (NaN Loss / Exploding Gradients) gây ra do lỗi tính toán lượng giác ngược hoặc đạo hàm vô hạn của hàm căn khi chạy FP16.",
        config: `
            <table class="sota-table">
                <tr><th>Thông số</th><th>Giá trị</th><th>Mô tả</th></tr>
                <tr><td><strong>Max Grad Norm</strong></td><td>5.0</td><td>Giới hạn chuẩn đạo hàm tối đa</td></tr>
                <tr><td><strong>Epsilon clamping</strong></td><td>1e-7</td><td>Giới hạn số thực tối thiểu</td></tr>
            </table>
        `,
        math: "Công thức cắt đạo hàm: $$\\vec{g} \\leftarrow \\vec{g} \\cdot \\min\\left(1, \\frac{\\text{max\\_norm}}{\\|\\vec{g}\\|}\\right)$$",
        code: `# model.py & train.py
cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7) # Tránh lỗi arccos
sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(min=1e-7)) # Tránh chia cho 0
# Cắt đạo hàm
torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)`
    }
};

function setupSotaInteractiveListeners() {
    console.log("Setting up SOTA Interactive Explorer Listeners...");
    const items = document.querySelectorAll('.sota-item');
    const container = document.getElementById('sota-detail-container');
    
    if (items.length === 0 || !container) {
        console.warn("Could not find SOTA elements. Retrying in 100ms...");
        setTimeout(setupSotaInteractiveListeners, 100);
        return;
    }
    
    items.forEach(item => {
        item.addEventListener('click', function() {
            // Đổi active class
            items.forEach(i => i.classList.remove('active'));
            this.classList.add('active');
            
            const techKey = this.getAttribute('data-tech');
            const data = sotaData[techKey];
            
            if (data) {
                // Hiển thị nội dung chi tiết
                container.className = "sota-detail-panel has-content";
                container.innerHTML = `
                    <div class="sota-detail-header">
                        <div class="badge-group">
                            <span class="badge" style="color:var(--color-blue); border-color:rgba(0, 242, 254, 0.2); background:rgba(0, 242, 254, 0.05); font-weight:bold;">${data.category}</span>
                            <span class="badge">SOTA Technique</span>
                        </div>
                        <h2>${data.title}</h2>
                    </div>
                    
                    <div class="sota-detail-section">
                        <h3>Mô tả công nghệ</h3>
                        <p>${data.desc}</p>
                    </div>
                    
                    <div class="sota-detail-section">
                        <h3>Ý nghĩa khoa học (Chống học vẹt &amp; Tăng tốc)</h3>
                        <p>${data.significance}</p>
                    </div>
                    
                    <div class="sota-detail-section">
                        <h3>Cấu hình thực tế trong Dự án</h3>
                        ${data.config}
                    </div>
                    
                    <div class="sota-detail-section">
                        <h3>Công thức Toán học hình học</h3>
                        <p style="background:rgba(0,0,0,0.2); padding:1rem; border-radius:8px; font-family:serif; font-size:1.05rem; border-left:3px solid var(--color-green);">${data.math}</p>
                    </div>
                    
                    <div class="sota-detail-section">
                        <h3>Đoạn mã nguồn PyTorch thực tế</h3>
                        <div class="sota-code-box">
                            <pre><code>${escapeHtml(data.code)}</code></pre>
                        </div>
                    </div>
                `;
                
                // Kích hoạt MathJax để vẽ công thức toán
                if (window.MathJax) {
                    MathJax.typesetPromise([container]).catch(err => console.error("MathJax inline render error:", err));
                }
            }
        });
    });
}

function escapeHtml(text) {
    return text
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

// Hàm nạp động nội dung Kỹ thuật SOTA và kích hoạt tương tác
async function loadSotaTechniquesHTML() {
    try {
        const response = await fetch('sota_techniques.html');
        if (!response.ok) throw new Error("Không thể nạp file sota_techniques.html");
        const html = await response.text();
        const container = document.getElementById('tab-sota-techniques');
        if (container) {
            container.innerHTML = html;
            console.log("Đã nạp thành công Trang Kỹ thuật SOTA!");
            setupSotaInteractiveListeners();
        }
    } catch (error) {
        console.error("Lỗi khi load Trang Kỹ thuật SOTA:", error);
        const container = document.getElementById('tab-sota-techniques');
        if (container) {
            container.innerHTML = `<div class="banner warning-banner" style="margin: 2rem;">Lỗi khi tải trang Kỹ thuật SOTA: ${error.message}</div>`;
        }
    }
}

// ─── RUN INITIALIZATION ─────────────────────────────────────────────
window.onload = async function() {
    // Tải động các trang con trước khi khởi tạo DOM và mô hình
    await loadLiveDemoHTML();
    await loadWebcamRealtimeHTML();
    await loadTrainingProcessHTML();
    await loadRobustnessEvaluationHTML();
    await loadSotaTechniquesHTML();
    
    // Khởi tạo các phần tử DOM sau khi HTML động đã được inject
    initLiveDemoDOMElements();
    
    initCharts();
    enableCustomUpload(); // Kích hoạt kéo thả upload ảnh
    setupWebcamListeners(); // Kích hoạt webcam real-time
    

    
    // 1. Khởi tạo và nạp bộ dò mặt MediaPipe Face Detector tự động
    const blazefaceStatusText = document.getElementById('blazeface-status-text');
    const blazefaceIndicator = document.getElementById('blazeface-indicator');
    
    try {
        console.log("Loading Google MediaPipe Face Detector...");
        const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
        const detectorConfig = {
            runtime: 'mediapipe',
            solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection'
        };
        state.faceDetector = await faceDetection.createDetector(model, detectorConfig);
        
        if (blazefaceStatusText) blazefaceStatusText.textContent = "Đã nạp bộ dò mặt MediaPipe!";
        if (blazefaceIndicator) blazefaceIndicator.className = "status-indicator green";
        console.log("MediaPipe Face Detector loaded successfully!");
    } catch (error) {
        console.error("Failed to load MediaPipe Face Detector:", error);
        if (blazefaceStatusText) blazefaceStatusText.textContent = "Nạp bộ dò mặt lỗi!";
        if (blazefaceIndicator) {
            blazefaceIndicator.className = "status-indicator red";
            blazefaceIndicator.style.backgroundColor = "red";
        }
    }
    
    // 2. Tự động nạp mô hình ArcFace ONNX
    await loadArcFaceModel();
};

// ─── CẢI TIẾN: VECTOR EMBEDDING VISUALIZER ───────────────────────────
function renderVectorStrip(containerId, embedding) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    
    const numDims = 80; // Hiển thị 80 chiều đầu
    for (let i = 0; i < numDims; i++) {
        const val = embedding[i];
        const box = document.createElement('div');
        box.className = 'vector-box';
        
        let alpha = Math.min(1.0, Math.abs(val) / 0.12);
        let color = 'rgba(107, 114, 128, 0.1)';
        
        if (val > 0) {
            color = `rgba(0, 242, 254, ${alpha})`; // Cyan neon
        } else if (val < 0) {
            color = `rgba(236, 72, 153, ${alpha})`; // Pink neon
        }
        
        box.style.backgroundColor = color;
        box.title = `Dim ${i}: ${val.toFixed(4)}`;
        container.appendChild(box);
    }
}

// ─── CẢI TIẾN: WEBCAM REAL-TIME IDENTIFICATION ───────────────────────
let webcamStream = null;
let webcamInterval = null;
let isWebcamOn = false;
let webcamFpsArray = [];
let webcamLastTime = performance.now();
let webcamRegisterEmbedding = null;
let webcamRegisterThumb = null;
let registeredGallery = [];

// Trạng thái mới cho nâng cấp webcam
let isSoundEnabled = false;
let audioCtx = null;
let isProcessingWebcam = false;
let lastTracks = [];


const livenessState = {
    enabled: false,
    challenge: 'NONE', // 'NONE', 'LOOK_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT', 'VERIFIED'
    yawHistory: [],
    pitchHistory: [],
    yawProgress: { straight: false, left: false, right: false },
    verified: false,
    lastStatusText: ''
};

const accessLog = [];
let lastLogName = '';
let lastLogTime = 0;
const LOG_DEBOUNCE_MS = 8000;

function setupWebcamListeners() {
    const btnToggle = document.getElementById('btn-toggle-webcam');
    const btnSnap = document.getElementById('btn-webcam-snap');
    const btnRegister = document.getElementById('btn-webcam-register');
    
    if (btnToggle) btnToggle.addEventListener('click', toggleWebcam);
    if (btnSnap) btnSnap.addEventListener('click', snapWebcamFace);
    if (btnRegister) btnRegister.addEventListener('click', registerWebcamFace);
    
    const regNameInput = document.getElementById('webcam-reg-name');
    if (regNameInput) {
        regNameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') registerWebcamFace();
        });
    }

    // Các listener mới cho nâng cấp
    loadGalleryFromLocalStorage();
    
    const btnExport = document.getElementById('btn-export-db');
    const btnImport = document.getElementById('btn-import-db');
    const importFileInput = document.getElementById('webcam-import-file');
    
    if (btnExport) btnExport.addEventListener('click', exportGalleryJSON);
    if (btnImport) btnImport.addEventListener('click', () => importFileInput && importFileInput.click());
    if (importFileInput) importFileInput.addEventListener('change', importGalleryJSON);
    
    const btnSound = document.getElementById('btn-toggle-sound');
    if (btnSound) btnSound.addEventListener('click', toggleSound);
    
    const toggleLiveness = document.getElementById('toggle-liveness-check');
    if (toggleLiveness) {
        toggleLiveness.addEventListener('change', function() {
            resetLivenessState();
        });
    }
}


async function toggleWebcam() {
    if (isWebcamOn) {
        stopWebcam();
    } else {
        await startWebcam();
    }
}

async function startWebcam() {
    const video = document.getElementById('webcam-stream-video');
    const canvas = document.getElementById('webcam-stream-canvas');
    const statusOverlay = document.getElementById('webcam-status-overlay');
    const statusMsg = document.getElementById('webcam-status-message');
    const btnToggleText = document.getElementById('btn-toggle-webcam-text');
    
    if (!state.onnxLoaded || !state.faceDetector) {
        alert("Mô hình nhận dạng hoặc Bộ dò mặt chưa nạp xong! Vui lòng đợi trong giây lát.");
        return;
    }
    
    statusMsg.textContent = "Đang kết nối camera...";
    
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720, facingMode: 'user' },
            audio: false
        });
        
        video.srcObject = webcamStream;
        isWebcamOn = true;
        
        if (btnToggleText) btnToggleText.textContent = "Tắt Camera";
        document.getElementById('btn-toggle-webcam').style.background = "linear-gradient(135deg, #ef4444 0%, #ff3b70 100%)";
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            if (statusOverlay) {
                statusOverlay.classList.add('hidden');
                statusOverlay.style.display = 'none';
            }
            document.getElementById('btn-webcam-snap').removeAttribute('disabled');
            
            // Khởi tạo liveness
            resetLivenessState();
            playAudioTone('startup');
            
            runWebcamLoop();
        };
    } catch (err) {
        console.error(err);
        if (statusMsg) statusMsg.textContent = "Không truy cập được camera. Hãy cấp quyền webcam.";
        alert("Không mở được camera. Hãy chắc chắn bạn cấp quyền camera cho trình duyệt.");
    }
}

function stopWebcam() {
    const video = document.getElementById('webcam-stream-video');
    const canvas = document.getElementById('webcam-stream-canvas');
    const statusOverlay = document.getElementById('webcam-status-overlay');
    const statusMsg = document.getElementById('webcam-status-message');
    const btnToggleText = document.getElementById('btn-toggle-webcam-text');
    
    // Ẩn liveness overlays khi camera tắt
    const livenessOverlay = document.getElementById('webcam-liveness-status');
    const livenessInstr = document.getElementById('webcam-liveness-instruction');
    if (livenessOverlay) livenessOverlay.style.display = 'none';
    if (livenessInstr) livenessInstr.style.display = 'none';
    
    if (webcamInterval) {
        clearInterval(webcamInterval);
        webcamInterval = null;
    }
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    isWebcamOn = false;
    if (video) video.srcObject = null;
    
    if (btnToggleText) btnToggleText.textContent = "Khởi động Camera";
    const btnToggle = document.getElementById('btn-toggle-webcam');
    if (btnToggle) btnToggle.style.background = "var(--primary-gradient)";
    
    if (statusOverlay) {
        statusOverlay.classList.remove('hidden');
        statusOverlay.style.display = 'flex';
        if (statusMsg) statusMsg.textContent = "Đang chờ khởi động camera...";
    }
    
    const btnSnap = document.getElementById('btn-webcam-snap');
    const btnReg = document.getElementById('btn-webcam-register');
    if (btnSnap) btnSnap.setAttribute('disabled', 'true');
    if (btnReg) btnReg.setAttribute('disabled', 'true');
    
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    const fpsVal = document.getElementById('webcam-fps');
    const latVal = document.getElementById('webcam-latency');
    if (fpsVal) fpsVal.textContent = '--';
    if (latVal) latVal.textContent = '-- ms';
}


function runWebcamLoop() {
    const video = document.getElementById('webcam-stream-video');
    const canvas = document.getElementById('webcam-stream-canvas');
    const ctx = canvas.getContext('2d');
    
    // Tạo canvas offline để chuyển đổi tensor từ video đúng kích thước tự nhiên
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    
    webcamInterval = setInterval(async () => {
        if (isProcessingWebcam) return; // Đang bận xử lý khung hình trước, bỏ qua khung này
        if (!isWebcamOn || video.paused || video.ended) return;
        
        isProcessingWebcam = true;
        const startTime = performance.now();
        
        try {
            tempCanvas.width = video.videoWidth;
            tempCanvas.height = video.videoHeight;
            tempCtx.drawImage(video, 0, 0);
            
            let predictions = [];
            
            try {
                predictions = await state.faceDetector.estimateFaces(tempCanvas);
            } catch (e) {
                console.error("Face detector loop error:", e);
            }
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // ─── LOGIC LIVENESS CHALLENGE (Áp dụng cho mặt chính đầu tiên phát hiện được) ───
            if (livenessState.enabled && predictions.length > 0) {
                const pred = predictions[0];
                if (pred.keypoints && pred.keypoints.length >= 4) {
                    // Sắp xếp mắt trái/phải theo trục X
                    let eyeLeft = pred.keypoints[0];
                    let eyeRight = pred.keypoints[1];
                    if (eyeLeft.x > eyeRight.x) {
                        eyeLeft = pred.keypoints[1];
                        eyeRight = pred.keypoints[0];
                    }
                    const nose = pred.keypoints[2];
                    
                    const eyeDist = Math.max(5, eyeRight.x - eyeLeft.x);
                    const yawRatio = (nose.x - eyeLeft.x) / eyeDist;
                    
                    // Cập nhật máy trạng thái Liveness
                    const instrEl = document.getElementById('webcam-liveness-instruction');
                    const statusEl = document.getElementById('webcam-liveness-status');
                    const statusTextEl = document.getElementById('webcam-liveness-status-text');
                    
                    if (!livenessState.verified) {
                        if (livenessState.challenge === 'LOOK_STRAIGHT') {
                            if (yawRatio >= 0.38 && yawRatio <= 0.62) {
                                livenessState.yawProgress.straight = true;
                                livenessState.challenge = 'TURN_LEFT';
                                playAudioTone('confirm'); // Beep nhẹ
                            }
                        } else if (livenessState.challenge === 'TURN_LEFT') {
                            if (yawRatio > 0.72) { // Quay trái (mũi lệch sát mắt phải ở X)
                                livenessState.yawProgress.left = true;
                                livenessState.challenge = 'TURN_RIGHT';
                                playAudioTone('confirm'); // Beep nhẹ
                            }
                        } else if (livenessState.challenge === 'TURN_RIGHT') {
                            if (yawRatio < 0.28) { // Quay phải (mũi lệch sát mắt trái ở X)
                                livenessState.yawProgress.right = true;
                                livenessState.verified = true;
                                livenessState.challenge = 'VERIFIED';
                                playAudioTone('confirm'); // Chime hoàn thành
                            }
                        }
                    }
                    
                    // Hiển thị giao diện chỉ dẫn Liveness
                    if (statusEl) statusEl.style.display = 'flex';
                    if (livenessState.verified) {
                        if (instrEl) instrEl.style.display = 'none';
                        if (statusEl) statusEl.className = 'liveness-indicator-overlay verified';
                        if (statusTextEl) statusTextEl.textContent = 'Liveness: VERIFIED (Hợp lệ)';
                    } else {
                        if (instrEl) {
                            instrEl.style.display = 'block';
                            if (livenessState.challenge === 'LOOK_STRAIGHT') {
                                instrEl.textContent = 'HÃY NHÌN THẲNG';
                                if (statusTextEl) statusTextEl.textContent = 'Liveness: LOOK STRAIGHT (Bước 1/3)';
                            } else if (livenessState.challenge === 'TURN_LEFT') {
                                instrEl.textContent = 'HÃY XOAY TRÁI (QUAY MẶT SANG PHẢI MÀN HÌNH)';
                                if (statusTextEl) statusTextEl.textContent = 'Liveness: TURN LEFT (Bước 2/3)';
                            } else if (livenessState.challenge === 'TURN_RIGHT') {
                                instrEl.textContent = 'HÃY QUAY PHẢI (NHÌN SANG TRÁI MÀN HÌNH)';
                                if (statusTextEl) statusTextEl.textContent = 'Liveness: TURN RIGHT (Bước 3/3)';
                            }
                        }
                    }
                }
            } else if (predictions.length === 0) {
                // Reset liveness challenge nếu người dùng rời khỏi khung hình
                if (livenessState.enabled && !livenessState.verified) {
                    livenessState.challenge = 'LOOK_STRAIGHT';
                    livenessState.yawProgress = { straight: false, left: false, right: false };
                    const instrEl = document.getElementById('webcam-liveness-instruction');
                    const statusTextEl = document.getElementById('webcam-liveness-status-text');
                    if (instrEl) {
                        instrEl.style.display = 'block';
                        instrEl.textContent = 'HÃY NHÌN THẲNG';
                    }
                    if (statusTextEl) statusTextEl.textContent = 'Liveness: LOOK STRAIGHT (Bước 1/3)';
                }
            }
            
            let currentTracks = [];
            const nowMs = Date.now();
            
            if (predictions.length > 0) {
                for (let i = 0; i < predictions.length; i++) {
                    const pred = predictions[i];
                    const box = {
                        x: pred.box.xMin,
                        y: pred.box.yMin,
                        width: pred.box.width,
                        height: pred.box.height
                    };
                    
                    const cx = box.x + box.width / 2;
                    const cy = box.y + box.height / 2;
                    
                    // Tìm kiếm và so sánh tọa độ tâm với các khuôn mặt cũ
                    let matchedTrack = null;
                    let minDist = box.width * 0.6;
                    
                    for (let t = 0; t < lastTracks.length; t++) {
                        const track = lastTracks[t];
                        const dx = cx - track.cx;
                        const dy = cy - track.cy;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < minDist) {
                            minDist = dist;
                            matchedTrack = track;
                        }
                    }
                    
                    let color = '#ef4444'; // Đỏ mặc định cho Stranger
                    let nameLabel = 'Người lạ (Unknown)';
                    let bestScore = 0.0;
                    let bestIndex = -1;
                    
                    let shouldRunONNX = true;
                    if (matchedTrack && (nowMs - matchedTrack.lastInferenceTime < 600)) {
                        shouldRunONNX = false;
                    }
                    
                    if (!shouldRunONNX && matchedTrack) {
                        // Tái sử dụng thông tin nhận dạng cũ để vẽ nhanh
                        nameLabel = matchedTrack.nameLabel;
                        bestScore = matchedTrack.bestScore;
                        color = matchedTrack.color;
                        bestIndex = matchedTrack.bestIndex;
                        
                        // Cập nhật vị trí mới
                        matchedTrack.cx = cx;
                        matchedTrack.cy = cy;
                        currentTracks.push(matchedTrack);
                    } else {
                        // Chạy suy diễn mô hình AI (ONNX) - Throttled định kỳ
                        try {
                            const faceBoxNorm = { x: box.x, y: box.y, w: box.width, h: box.height };
                            const inputTensor = await preprocessImageToTensor(video, faceBoxNorm, pred.keypoints);
                            
                            const feeds = { [state.session.inputNames[0]]: inputTensor };
                            const output = await state.session.run(feeds);
                            const emb = output[state.session.outputNames[0]].data;
                            const normEmb = normalizeL2(emb);
                            
                            if (registeredGallery.length > 0) {
                                let bestSim = -1.0;
                                for (let g = 0; g < registeredGallery.length; g++) {
                                    const score = calculateCosineSimilarity(normEmb, registeredGallery[g].embedding);
                                    if (score > bestSim) {
                                        bestSim = score;
                                        bestIndex = g;
                                    }
                                }
                                
                                const securitySelect = document.getElementById('security-level');
                                const l2Threshold = securitySelect ? parseFloat(securitySelect.value) : 1.20;
                                const cosThreshold = (2.0 - l2Threshold) / 2.0;
                                
                                bestScore = bestSim;
                                
                                if (bestSim >= cosThreshold) {
                                    if (livenessState.enabled && !livenessState.verified) {
                                        nameLabel = `${registeredGallery[bestIndex].name} (Chờ Liveness)`;
                                        color = '#f59e0b';
                                    } else {
                                        nameLabel = registeredGallery[bestIndex].name;
                                        color = '#38f9d7';
                                    }
                                } else {
                                    if (livenessState.enabled && !livenessState.verified) {
                                        nameLabel = 'Unknown (Chờ Liveness)';
                                        color = '#f59e0b';
                                    } else {
                                        nameLabel = 'Người lạ (Unknown)';
                                        color = '#ef4444';
                                    }
                                }
                            }
                        } catch (err) {
                            console.error("ONNX inference loop error:", err);
                        }
                        
                        // Cập nhật/Thêm mới vào hàng theo dõi
                        if (matchedTrack) {
                            matchedTrack.cx = cx;
                            matchedTrack.cy = cy;
                            matchedTrack.nameLabel = nameLabel;
                            matchedTrack.bestScore = bestScore;
                            matchedTrack.color = color;
                            matchedTrack.bestIndex = bestIndex;
                            matchedTrack.lastInferenceTime = nowMs;
                            currentTracks.push(matchedTrack);
                        } else {
                            const newTrack = {
                                cx: cx,
                                cy: cy,
                                nameLabel: nameLabel,
                                bestScore: bestScore,
                                color: color,
                                bestIndex: bestIndex,
                                lastInferenceTime: nowMs
                            };
                            currentTracks.push(newTrack);
                        }
                    }
                    
                    // ─── LOGIC GHI NHẬT KÝ ACCESS CONTROL (Đã debounce) ───
                    const livenessOK = !livenessState.enabled || livenessState.verified;
                    const isMatch = bestIndex !== -1 && bestScore > 0.1 && nameLabel === registeredGallery[bestIndex].name;
                    const finalStatus = (isMatch && livenessOK) ? 'GRANTED' : 'DENIED';
                    
                    let logDisplayName = 'Người lạ (Unknown)';
                    if (bestIndex !== -1 && bestScore > 0.1) {
                        logDisplayName = registeredGallery[bestIndex].name;
                    }
                    
                    const isDifferentPerson = (logDisplayName !== lastLogName);
                    const isTimeout = (nowMs - lastLogTime > LOG_DEBOUNCE_MS);
                    
                    if ((isDifferentPerson || isTimeout) && (registeredGallery.length > 0 || logDisplayName === 'Người lạ (Unknown)')) {
                        lastLogName = logDisplayName;
                        lastLogTime = nowMs;
                        
                        const faceThumb = cropFaceToDataURL(video, box);
                        addAccessLog(logDisplayName, bestScore, finalStatus, faceThumb, livenessOK);
                        
                        const container = document.querySelector('.webcam-container-wrapper');
                        if (finalStatus === 'GRANTED') {
                            playAudioTone('granted');
                            if (container) {
                                container.style.boxShadow = '0 0 25px rgba(56, 249, 215, 0.6)';
                                setTimeout(() => container.style.boxShadow = 'none', 1000);
                            }
                        } else {
                            playAudioTone('denied');
                            if (container) {
                                container.style.boxShadow = '0 0 25px rgba(239, 68, 68, 0.6)';
                                setTimeout(() => container.style.boxShadow = 'none', 1000);
                            }
                        }
                    }
                    
                    // ─── VẼ CÁC HIỆU ỨNG CYBER HUD LÊN CANVAS ───
                    const x = box.x;
                    const y = box.y;
                    const w = box.width;
                    const h = box.height;
                    
                    ctx.strokeStyle = color;
                    ctx.lineWidth = Math.max(3, Math.round(video.videoWidth / 200));
                    ctx.shadowBlur = 10;
                    ctx.shadowColor = color;
                    
                    const cornerLength = Math.min(w, h) * 0.20;
                    
                    // Vẽ Corner Brackets
                    ctx.beginPath();
                    ctx.moveTo(x + cornerLength, y);
                    ctx.lineTo(x, y);
                    ctx.lineTo(x, y + cornerLength);
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.moveTo(x + w - cornerLength, y);
                    ctx.lineTo(x + w, y);
                    ctx.lineTo(x + w, y + cornerLength);
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.moveTo(x, y + h - cornerLength);
                    ctx.lineTo(x, y + h);
                    ctx.lineTo(x + cornerLength, y + h);
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.moveTo(x + w - cornerLength, y + h);
                    ctx.lineTo(x + w, y + h);
                    ctx.lineTo(x + w, y + h - cornerLength);
                    ctx.stroke();
                    

                    
                    // Vẽ vòng ngắm Reticle tròn nét đứt bao quanh mặt
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
                    ctx.lineWidth = 1;
                    ctx.setLineDash([4, 4]);
                    ctx.beginPath();
                    ctx.arc(x + w / 2, y + h / 2, Math.min(w, h) * 0.45, 0, 2 * Math.PI);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    
                    // Vẽ 6 landmarks
                    ctx.fillStyle = '#00f2fe';
                    ctx.shadowBlur = 4;
                    ctx.shadowColor = '#00f2fe';
                    const dotRadius = Math.max(2, Math.round(video.videoWidth / 250));
                    if (pred.keypoints && pred.keypoints.length > 0) {
                        pred.keypoints.forEach(kp => {
                            ctx.beginPath();
                            ctx.arc(kp.x, kp.y, dotRadius, 0, 2 * Math.PI);
                            ctx.fill();
                        });
                    }
                    
                    // Vẽ nhãn văn bản của danh tính
                    ctx.shadowBlur = 0;
                    ctx.font = 'bold 12px Inter, sans-serif';
                    const textStr = bestScore > 0.0 ? `${nameLabel} (${(bestScore * 100).toFixed(0)}%)` : nameLabel;
                    const textWidth = ctx.measureText(textStr).width;
                    
                    ctx.fillStyle = 'rgba(11, 15, 25, 0.85)';
                    ctx.fillRect(x, y - 24, textWidth + 16, 24);
                    
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1;
                    ctx.strokeRect(x, y - 24, textWidth + 16, 24);
                    
                    ctx.fillStyle = color;
                    ctx.fillText(textStr, x + 8, y - 8);
                }
            }
            
            // Cập nhật bộ nhớ theo dõi cho khung hình tiếp theo
            lastTracks = currentTracks;
            
            // Tính tổng độ trễ cho cả khung hình
            const latency = performance.now() - startTime;
            const latVal = document.getElementById('webcam-latency');
            if (latVal) latVal.textContent = `${latency.toFixed(1)} ms`;
            
            // Tính FPS
            const now = performance.now();
            const fps = 1000 / (now - webcamLastTime);
            webcamLastTime = now;
            webcamFpsArray.push(fps);
            if (webcamFpsArray.length > 10) webcamFpsArray.shift();
            const avgFps = webcamFpsArray.reduce((a, b) => a + b, 0) / webcamFpsArray.length;
            const fpsVal = document.getElementById('webcam-fps');
            if (fpsVal) fpsVal.textContent = avgFps.toFixed(1);
            
        } finally {
            isProcessingWebcam = false; // Luôn giải phóng lock
        }
    }, 30); // Giảm xuống 30ms (hướng tới 30 FPS)
}


async function snapWebcamFace() {
    if (!isWebcamOn) return;
    
    // Kiểm tra liveness trước khi chụp đăng ký nếu bật chế độ liveness
    if (livenessState.enabled && !livenessState.verified) {
        alert("Vui lòng hoàn thành Xác thực Liveness (Nhìn thẳng, Xoay Trái, Xoay Phải) trước khi chụp ảnh đăng ký!");
        return;
    }
    
    const video = document.getElementById('webcam-stream-video');
    
    // Tạm dừng 1 chút để dò mặt và chụp
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);
    
    let predictions = [];
    try {
        predictions = await state.faceDetector.estimateFaces(tempCanvas);
    } catch (e) {
        console.error("Estimate faces error:", e);
    }
    
    if (predictions.length > 0) {
        const pred = predictions[0];
        const box = {
            x: pred.box.xMin,
            y: pred.box.yMin,
            width: pred.box.width,
            height: pred.box.height
        };
        
        try {
            // 1. Tạo thumbnail 48x48 cho danh sách CSDL bằng hàm helper crop
            webcamRegisterThumb = cropFaceToDataURL(video, box);
            
            // 2. Trích xuất embedding bằng ONNX
            const faceBoxNorm = { x: box.x, y: box.y, w: box.width, h: box.height };
            const inputTensor = await preprocessImageToTensor(video, faceBoxNorm, pred.keypoints);
            
            const feeds = { [state.session.inputNames[0]]: inputTensor };
            const output = await state.session.run(feeds);
            const emb = output[state.session.outputNames[0]].data;
            
            webcamRegisterEmbedding = normalizeL2(emb);
            
            document.getElementById('btn-webcam-register').removeAttribute('disabled');
            document.getElementById('webcam-reg-name').focus();
            
            alert("Đã phân tích khuôn mặt! Vui lòng nhập Họ tên và nhấn nút Đăng ký.");
        } catch (err) {
            console.error(err);
            alert("Lỗi khi trích chọn đặc trưng ảnh chụp.");
        }
    } else {
        alert("Không tìm thấy khuôn mặt nào trước camera để chụp! Hãy ghé sát camera hơn.");
    }
}

function registerWebcamFace() {
    const nameInput = document.getElementById('webcam-reg-name');
    const name = nameInput.value.trim();
    
    if (!name) {
        alert("Vui lòng điền Họ tên thành viên!");
        return;
    }
    if (!webcamRegisterEmbedding) {
        alert("Hãy Chụp ảnh khuôn mặt trước!");
        return;
    }
    
    // Thêm vào database tạm thời, chuyển đổi sang Array thường để dễ JSON hóa
    registeredGallery.push({
        name: name,
        embedding: Array.isArray(webcamRegisterEmbedding) ? webcamRegisterEmbedding : Array.from(webcamRegisterEmbedding),
        thumb: webcamRegisterThumb
    });
    
    // Lưu vào LocalStorage
    saveGalleryToLocalStorage();
    
    // Cập nhật giao diện
    updateWebcamGalleryUI();
    
    // Phát nhạc xác nhận thành công
    playAudioTone('confirm');
    
    // Reset form đăng ký
    nameInput.value = '';
    webcamRegisterEmbedding = null;
    webcamRegisterThumb = null;
    document.getElementById('btn-webcam-register').setAttribute('disabled', 'true');
    
    alert(`Đã đăng ký thành viên "${name}" thành công vào Gallery!`);
}

function updateWebcamGalleryUI() {
    const container = document.getElementById('webcam-gallery-list');
    const countSpan = document.getElementById('webcam-gallery-count');
    
    if (!countSpan || !container) return;
    
    countSpan.textContent = registeredGallery.length;
    
    if (registeredGallery.length === 0) {
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: var(--text-muted); font-size: 0.8rem;">
                Chưa có khuôn mặt nào được đăng ký
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    registeredGallery.forEach((item, index) => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'gallery-item';
        itemDiv.innerHTML = `
            <div class="gallery-item-left">
                <img class="gallery-thumb" src="${item.thumb}" alt="Avatar">
                <span class="gallery-name">${item.name}</span>
            </div>
            <button class="btn-delete" data-index="${index}" title="Xóa">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px; height:16px;"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            </button>
        `;
        
        itemDiv.querySelector('.btn-delete').addEventListener('click', () => {
            registeredGallery.splice(index, 1);
            saveGalleryToLocalStorage();
            updateWebcamGalleryUI();
        });
        
        container.appendChild(itemDiv);
    });
}

// ─── CÁC HÀM TRỢ GIÚP (HELPERS) CHO NÂNG CẤP WEBCAM ───────────────────

function cropFaceToDataURL(video, box) {
    try {
        const thumbCanvas = document.createElement('canvas');
        thumbCanvas.width = 48;
        thumbCanvas.height = 48;
        const thumbCtx = thumbCanvas.getContext('2d');
        
        const videoW = video.videoWidth || video.width || 640;
        const videoH = video.videoHeight || video.height || 480;
        
        const sx = Math.max(0, Math.min(videoW - 10, box.x));
        const sy = Math.max(0, Math.min(videoH - 10, box.y));
        const sw = Math.max(10, Math.min(videoW - sx, box.width || box.w));
        const sh = Math.max(10, Math.min(videoH - sy, box.height || box.h));
        
        thumbCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 48, 48);
        return thumbCanvas.toDataURL('image/jpeg');
    } catch (e) {
        console.error("Error cropping face thumb:", e);
        return '';
    }
}

function resetLivenessState() {
    const toggleLiveness = document.getElementById('toggle-liveness-check');
    const overlay = document.getElementById('webcam-liveness-status');
    const instr = document.getElementById('webcam-liveness-instruction');
    
    livenessState.enabled = toggleLiveness ? toggleLiveness.checked : false;
    livenessState.verified = !livenessState.enabled; // Nếu tắt liveness thì mặc định coi như đã verified
    livenessState.challenge = livenessState.enabled ? 'LOOK_STRAIGHT' : 'NONE';
    livenessState.yawProgress = { straight: false, left: false, right: false };
    livenessState.yawHistory = [];
    livenessState.pitchHistory = [];
    
    if (livenessState.enabled) {
        if (overlay) {
            overlay.className = "liveness-indicator-overlay pending";
            overlay.style.display = "flex";
            const statusText = document.getElementById('webcam-liveness-status-text');
            if (statusText) statusText.textContent = "Liveness: LOOK STRAIGHT (Bước 1/3)";
        }
        if (instr) {
            instr.style.display = "block";
            instr.textContent = "HÃY NHÌN THẲNG";
        }
    } else {
        if (overlay) overlay.style.display = "none";
        if (instr) instr.style.display = "none";
    }
}

function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
}

function playAudioTone(type) {
    return; // Âm thanh đã bị loại bỏ theo yêu cầu của người dùng
}

function toggleSound() {
    isSoundEnabled = !isSoundEnabled;
    const btn = document.getElementById('btn-toggle-sound');
    const path = document.getElementById('sound-wave-path');
    
    if (isSoundEnabled) {
        if (btn) btn.classList.add('active');
        if (path) path.style.display = 'block';
        playAudioTone('startup');
    } else {
        if (btn) btn.classList.remove('active');
        if (path) path.style.display = 'none';
    }
}

function saveGalleryToLocalStorage() {
    try {
        localStorage.setItem('arcnet_gallery', JSON.stringify(registeredGallery));
        console.log("Saved gallery to localStorage. Count:", registeredGallery.length);
    } catch (e) {
        console.error("Failed to save gallery to localStorage:", e);
    }
}

function loadGalleryFromLocalStorage() {
    try {
        const data = localStorage.getItem('arcnet_gallery');
        if (data) {
            const rawList = JSON.parse(data);
            registeredGallery = rawList.map(item => ({
                name: item.name,
                embedding: new Float32Array(item.embedding),
                thumb: item.thumb
            }));
            updateWebcamGalleryUI();
            console.log("Loaded gallery from localStorage. Count:", registeredGallery.length);
        }
    } catch (e) {
        console.error("Failed to load gallery from localStorage:", e);
    }
}

function exportGalleryJSON() {
    if (registeredGallery.length === 0) {
        alert("Cơ sở dữ liệu rỗng! Không có gì để xuất.");
        return;
    }
    try {
        const exportList = registeredGallery.map(item => ({
            name: item.name,
            embedding: Array.isArray(item.embedding) ? item.embedding : Array.from(item.embedding),
            thumb: item.thumb
        }));
        
        const jsonStr = JSON.stringify(exportList, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `arcface_gallery_db_${new Date().toISOString().slice(0,10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (e) {
        console.error("Export error:", e);
        alert("Lỗi khi xuất CSDL.");
    }
}

function importGalleryJSON(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const list = JSON.parse(e.target.result);
            if (!Array.isArray(list)) throw new Error("Định dạng tệp không hợp lệ (phải là Array)");
            
            let count = 0;
            list.forEach(item => {
                if (item.name && item.embedding && item.thumb) {
                    registeredGallery.push({
                        name: item.name,
                        embedding: new Float32Array(item.embedding),
                        thumb: item.thumb
                    });
                    count++;
                }
            });
            
            if (count > 0) {
                saveGalleryToLocalStorage();
                updateWebcamGalleryUI();
                playAudioTone('confirm');
                alert(`Đã nhập thành công ${count} khuôn mặt vào cơ sở dữ liệu!`);
            } else {
                alert("Không tìm thấy khuôn mặt hợp lệ trong tệp.");
            }
        } catch (err) {
            console.error("Import error:", err);
            alert("Lỗi khi nhập tệp: " + err.message);
        }
        event.target.value = ''; // Reset file input
    };
    reader.readAsText(file);
}

function addAccessLog(name, score, status, faceThumb, livenessPassed) {
    const timeStr = new Date().toTimeString().split(' ')[0];
    const logItem = {
        time: timeStr,
        name: name,
        score: score,
        status: status, // 'GRANTED' hoặc 'DENIED'
        thumb: faceThumb,
        liveness: livenessPassed
    };
    
    // Thêm vào đầu danh sách
    accessLog.unshift(logItem);
    if (accessLog.length > 5) {
        accessLog.pop();
    }
    
    updateAccessLogUI();
}

function updateAccessLogUI() {
    const container = document.getElementById('webcam-access-log-list');
    if (!container) return;
    
    if (accessLog.length === 0) {
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: var(--text-muted); font-size: 0.8rem;">
                Chưa có sự kiện check-in nào được ghi nhận
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    accessLog.forEach(log => {
        const itemDiv = document.createElement('div');
        const isGranted = log.status === 'GRANTED';
        itemDiv.className = `log-item ${isGranted ? 'granted' : 'denied'}`;
        
        const scoreStr = log.score > 0 ? `Similarity: ${(log.score * 100).toFixed(0)}%` : `Similarity: --`;
        const livenessBadge = log.liveness ? 
            `<span style="color: var(--color-green); font-size: 0.7rem; font-weight: 600;">(Liveness OK)</span>` : 
            `<span style="color: #ef4444; font-size: 0.7rem; font-weight: 600;">(Liveness Failed)</span>`;
            
        itemDiv.innerHTML = `
            <div class="log-meta">
                <span class="log-time">${log.time}</span>
                <img class="log-face-thumb" src="${log.thumb}" alt="Face">
                <div class="log-info">
                    <span class="log-name">${log.name}</span>
                    <span class="log-score">${scoreStr} ${livenessBadge}</span>
                </div>
            </div>
            <span class="log-badge ${isGranted ? 'granted' : 'denied'}">${isGranted ? 'Access Granted' : 'Access Denied'}</span>
        `;
        container.appendChild(itemDiv);
    });
}


