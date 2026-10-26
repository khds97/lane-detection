#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

constexpr int W = 640, H = 480;
constexpr int GRAY_THRESHOLD = 150;
constexpr int SOBEL_THRESHOLD = 50;
constexpr int WINDOWS = 16, MARGIN = 10, MIN_PIXELS = 100;
constexpr double EMA_ALPHA = 0.3;

struct Poly {
    double a = 0, b = 0, c = 0;
    bool valid = false;
    double x(double y) const { return a*y*y + b*y + c; }
};

struct Perspective {
    Mat forward, inverse;
    vector<Point2f> src, dst;
};
void drawSrcPoints(Mat& frame, const Perspective& p)
{
    //src 또는 dst로 바꾸기
    //src 또는 dst 좌표 4개를 연결할 색상
    Scalar red(0, 0, 255);
    Scalar yellow(0, 255, 255);

    for (int i = 0; i < 4; ++i)
    {
        Point center(
            cvRound(p.src[i].x),
            cvRound(p.src[i].y)
        );

        // 빨간 원 표시
        circle(frame, center, 7, red, FILLED);

        // 점 번호와 좌표 표시
        string label = to_string(i) + ": (" +
            to_string(center.x) + ", " +
            to_string(center.y) + ")";

        putText(
            frame,
            label,
            Point(center.x + 10, center.y - 10),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            yellow,
            2
        );

        // 다음 점과 연결
        Point next(
            cvRound(p.src[(i + 1) % 4].x),
            cvRound(p.src[(i + 1) % 4].y)
        );

        line(frame, center, next, red, 2);
    }
}

void drawDstPoints(Mat& frame, const Perspective& p)
{
    //src 또는 dst로 바꾸기
    //src 또는 dst 좌표 4개를 연결할 색상
    Scalar red(0, 0, 255);
    Scalar yellow(0, 255, 255);

    for (int i = 0; i < 4; ++i)
    {
        Point center(
            cvRound(p.dst[i].x),
            cvRound(p.dst[i].y)
        );

        // 빨간 원 표시
        circle(frame, center, 7, red, FILLED);

        // 점 번호와 좌표 표시
        string label = to_string(i) + ": (" +
            to_string(center.x) + ", " +
            to_string(center.y) + ")";

        putText(
            frame,
            label,
            Point(center.x + 10, center.y - 10),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            yellow,
            2
        );

        // 다음 점과 연결
        Point next(
            cvRound(p.dst[(i + 1) % 4].x),
            cvRound(p.dst[(i + 1) % 4].y)
        );

        line(frame, center, next, red, 2);
    }
}

Perspective makePerspective(Size size) {
    float w = static_cast<float>(size.width);
    float h = static_cast<float>(size.height);
    Perspective p;
    // STARTING POINT ONLY: tune these four points to the actual camera view.
    p.src = { {w*.10f,h*.85f}, {w*.42f,h*.60f},
              {w*.58f,h*.60f}, {w*.90f,h*.85f} };
    p.dst = { {w*.35f,h}, {w*.30f,0},
              {w*.70f,0}, {w*.65f,h} };
    p.forward = getPerspectiveTransform(p.src, p.dst);
    p.inverse = getPerspectiveTransform(p.dst, p.src);
    return p;
}

Mat applyBevROI(const Mat& bev)
{
    Mat mask = Mat::zeros(bev.size(), CV_8UC1);

    const int w = bev.cols;
    const int h = bev.rows;

    vector<Point> roi = {
        Point(cvRound(w*.35f), h),
        Point(cvRound(w*.35), 0),
        Point(cvRound(w*.65f), 0),
        Point(cvRound(w*.65f), h)
    };

    fillConvexPoly(mask, roi, Scalar(255));

    Mat masked = Mat::zeros(bev.size(), bev.type());
    bev.copyTo(masked, mask);

    return masked;
}

Mat laneMask(const Mat& bev, Mat& gray, Mat& gradient)
{
    // 1. BEV 영상을 grayscale로 변환
    cvtColor(bev, gray, COLOR_BGR2GRAY);
    // 2. 밝기 기준으로 차선 후보 검출
    Mat bright;
    threshold(
        gray,
        bright,
        GRAY_THRESHOLD,
        255,
        THRESH_BINARY
    );

    // 3. Sobel X 연산으로 수직 방향 경계 검출
Mat sx, absSx, normalized;

Sobel(gray, sx, CV_32F, 1, 0, 3);
convertScaleAbs(sx, absSx);

double minVal = 0.0;
double maxVal = 0.0;
minMaxLoc(absSx, &minVal, &maxVal);

gradient = Mat::zeros(gray.size(), CV_8UC1);

if (maxVal > 0)
{
    double scale = 255.0 / maxVal;

    cout << "Before max = " << maxVal << '\n';
    cout << "Normalization scale = " << scale << '\n';

    // 정규화
    absSx.convertTo(
        normalized,
        CV_8U,
        scale
    );

    // 정규화 전후 픽셀값 비교
    cout << "\nBefore -> After Normalization\n";
    cout << "-----------------------------\n";

    for (int y = 100; y <= 400; y += 75)
    {
        for (int x = 100; x <= 500; x += 100)
        {
            int before = absSx.at<uchar>(y, x);
            int after = normalized.at<uchar>(y, x);

            cout << "(" << x << "," << y << ") "
                 << before << " -> " << after << '\n';
        }
    }

    // 정규화된 영상에 임계값 적용
    threshold(
        normalized,
        gradient,
        SOBEL_THRESHOLD,
        255,
        THRESH_BINARY
    );
}
	// 최종 gradient 마스크
	imshow("3. After Threshold", gradient);

    // 4. 밝기 후보와 Sobel 후보 결합
    Mat combined;
    bitwise_or(bright, gradient, combined);

    // 5. 작은 흰색 노이즈 제거
    Mat kernelOpen = getStructuringElement(
        MORPH_RECT,
        Size(3, 3)
    );

    morphologyEx(
        combined,
        combined,
        MORPH_OPEN,
        kernelOpen
    );

    // 6. 끊어진 후보 영역의 작은 틈 메우기
    Mat kernelClose = getStructuringElement(
        MORPH_RECT,
        Size(3, 3)
    );

    morphologyEx(
        combined,
        combined,
        MORPH_CLOSE,
        kernelClose
    );

    return combined;
}

int histogramPeak(const Mat& binary, int x0, int x1) {
    vector<int> hist(x1-x0, 0);
    for (int y = binary.rows/3; y < binary.rows; ++y) {
        const uchar* row = binary.ptr<uchar>(y);
        for (int x=x0; x<x1; ++x)
            if (row[x]) hist[x-x0]++;
    }
    auto it = max_element(hist.begin(), hist.end());
    if (it == hist.end() || *it == 0) return -1;
    return x0 + static_cast<int>(distance(hist.begin(), it));
}

struct Pixels { vector<Point> left, right; };

Pixels collectPixels(
    const Mat& binary,
    Mat& debug,
    const optional<Poly>& prevLeft,
    const optional<Poly>& prevRight)
{
    Pixels out;
    debug = Mat::zeros(binary.size(), CV_8UC3);

    const int width = binary.cols;
    const int height = binary.rows;
    const int mid = width / 2;
    const int winH = max(1, height / WINDOWS);

    // 첫 프레임 또는 이전 차선이 유효하지 않을 때만 히스토그램으로 시작점을 찾는다.
    int leftBase = -1;
    int rightBase = -1;

    if (!prevLeft || !prevLeft->valid) {
        leftBase = histogramPeak(binary, 0, mid);
    }
    if (!prevRight || !prevRight->valid) {
        rightBase = histogramPeak(binary, mid, width);
    }

    // 이전 프레임의 곡선이 있으면 각 윈도우를 그 곡선 주변에 배치한다.
    // 이전 곡선이 없을 때는 기존 히스토그램 기반 sliding-window 방식을 사용한다.
    for (int win = 0; win < WINDOWS; ++win) {
        const int y1 = height - win * winH;
        const int y0 = max(0, height - (win + 1) * winH);
        const int yCenter = (y0 + min(y1, height) - 1) / 2;

        int ll = 0, lr = 0, rl = 0, rr = 0;

        if (prevLeft && prevLeft->valid) {
            const int expectedX = cvRound(prevLeft->x(yCenter));
            ll = max(0, expectedX - MARGIN);
            lr = min(width, expectedX + MARGIN + 1);
        } else if (leftBase >= 0) {
            ll = max(0, leftBase - MARGIN);
            lr = min(mid, leftBase + MARGIN + 1);
        }

        if (prevRight && prevRight->valid) {
            const int expectedX = cvRound(prevRight->x(yCenter));
            rl = max(mid, expectedX - MARGIN);
            rr = min(width, expectedX + MARGIN + 1);
        } else if (rightBase >= 0) {
            rl = max(mid, rightBase - MARGIN);
            rr = min(width, rightBase + MARGIN + 1);
        }

        // 검출 가능한 범위가 아니면 해당 윈도우는 건너뛴다.
        if (ll < lr) {
            rectangle(debug, Point(ll, y0), Point(lr - 1, min(y1, height) - 1),
                      Scalar(255, 0, 0), 1);
        }
        if (rl < rr) {
            rectangle(debug, Point(rl, y0), Point(rr - 1, min(y1, height) - 1),
                      Scalar(0, 0, 255), 1);
        }

        vector<Point> lp, rp;
        for (int y = y0; y < min(y1, height); ++y) {
            const uchar* row = binary.ptr<uchar>(y);

            if (ll < lr) {
                for (int x = ll; x < lr; ++x) {
                    if (row[x]) lp.emplace_back(x, y);
                }
            }
            if (rl < rr) {
                for (int x = rl; x < rr; ++x) {
                    if (row[x]) rp.emplace_back(x, y);
                }
            }
        }

        out.left.insert(out.left.end(), lp.begin(), lp.end());
        out.right.insert(out.right.end(), rp.begin(), rp.end());

        // 첫 프레임의 histogram 기반 탐색에서만 중심을 갱신한다.
        // 이전 곡선이 있는 경우 윈도우는 이전 곡선 주변에 고정해 순간적인 잡음의
        // 영향으로 탐색 중심이 연쇄적으로 이동하는 것을 방지한다.
        if (!(prevLeft && prevLeft->valid) &&
            static_cast<int>(lp.size()) >= 20) {
            long long sum = 0;
            for (const auto& q : lp) sum += q.x;
            leftBase = static_cast<int>(sum / lp.size());
        }

        if (!(prevRight && prevRight->valid) &&
            static_cast<int>(rp.size()) >= 20) {
            long long sum = 0;
            for (const auto& q : rp) sum += q.x;
            rightBase = static_cast<int>(sum / rp.size());
        }
    }

    return out;
}

optional<Poly> fitQuadratic(const vector<Point>& pts) {
    if (pts.size() < 3) return nullopt;

    // 피팅에 사용할 최대 픽셀 수
    constexpr size_t MAX_POINTS = 1500;

    size_t step = max(
        size_t(1),
        (pts.size() + MAX_POINTS - 1) / MAX_POINTS
    );

    vector<Point> sampled;
    sampled.reserve(min(pts.size(), MAX_POINTS));

    for (size_t i = 0; i < pts.size(); i += step) {
        sampled.push_back(pts[i]);
    }

    if (sampled.size() < 3) return nullopt;

    Mat A(static_cast<int>(sampled.size()), 3, CV_64F);
    Mat B(static_cast<int>(sampled.size()), 1, CV_64F);

    for (size_t i = 0; i < sampled.size(); ++i) {
        double y = sampled[i].y;
        double x = sampled[i].x;

        A.at<double>(static_cast<int>(i), 0) = y * y;
        A.at<double>(static_cast<int>(i), 1) = y;
        A.at<double>(static_cast<int>(i), 2) = 1.0;

        B.at<double>(static_cast<int>(i), 0) = x;
    }

    Mat coef;
    if (!solve(A, B, coef, DECOMP_SVD)) {
        return nullopt;
    }

    Poly p{
        coef.at<double>(0),
        coef.at<double>(1),
        coef.at<double>(2),
        true
    };

    if (!isfinite(p.x(H - 1)) ||
        p.x(H - 1) < -W ||
        p.x(H - 1) > 2 * W) {
        return nullopt;
    }

    return p;
}

Poly smooth(const Poly& now, const optional<Poly>& prev) {
    if (!prev || !prev->valid) return now;
    return {
        EMA_ALPHA*now.a+(1-EMA_ALPHA)*prev->a,
        EMA_ALPHA*now.b+(1-EMA_ALPHA)*prev->b,
        EMA_ALPHA*now.c+(1-EMA_ALPHA)*prev->c,
        true
    };
}

void drawPoly(Mat& img, const Poly& p, Scalar color, int thickness=3) {
    if (!p.valid) return;
    vector<Point> curve;
    for (int y=img.rows-1; y>=0; y-=3) {
        int x=cvRound(p.x(y));
        if (x>=0 && x<img.cols) curve.emplace_back(x,y);
    }
    if (curve.size()>=2) polylines(img,curve,false,color,thickness,LINE_AA);
}

int main(int argc, char** argv) {
    ///if (argc < 2) {
        ///cerr << "Usage: " << argv[0] << " <video_path> [--show-steps]\\n";
        ///return 1;
    ///}
    ///string path=argv[1];
    ///bool showSteps=(argc>=3 && string(argv[2])=="--show-steps");
    
    const string path = "/home/ubuntu/lane_detection/challenge.mp4";
    const bool showSteps = true;  // 중간 처리 영상 표시 여부
      
    VideoCapture cap(path);
    if (!cap.isOpened()) {
        cerr << "Could not open video: " << path << "\\n";
        return 1;
    }
    
    Mat frame;

    Perspective perspective=makePerspective(Size(W,H));
    
    optional<Poly> prevLeft, prevRight;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        resize(frame,frame,Size(W,H));

        // Camera perspective -> bird's-eye view (BEV).
        Mat bev;
        warpPerspective(frame,bev,perspective.forward,Size(W,H));
        
        Mat bevROI = applyBevROI(bev);
        Mat gray, gradient;
        Mat binary=laneMask(bevROI,gray,gradient);

        Mat windows;
        Pixels pixels = collectPixels(binary, windows, prevLeft, prevRight);

        optional<Poly> leftFit, rightFit;
        if (static_cast<int>(pixels.left.size()) >= MIN_PIXELS)
            leftFit=fitQuadratic(pixels.left);
        if (static_cast<int>(pixels.right.size()) >= MIN_PIXELS)
            rightFit=fitQuadratic(pixels.right);

        if (leftFit) { *leftFit=smooth(*leftFit,prevLeft); prevLeft=*leftFit; }
        if (rightFit) { *rightFit=smooth(*rightFit,prevRight); prevRight=*rightFit; }

        Poly left=leftFit.value_or(Poly{});
        Poly right=rightFit.value_or(Poly{});

        //Mat bevCurves;

	//if (showSteps) {
  	  //bevCurves = bev.clone();
 	  //drawPoly(bevCurves, left, Scalar(0,255,0));
	  //drawPoly(bevCurves, right, Scalar(0,255,255));
	//}

        // Draw BEV curves into a transparent layer, then inverse-transform
        // that layer back onto the original camera perspective.
        Mat layer=Mat::zeros(bev.size(),CV_8UC3);
        drawPoly(layer,left,Scalar(0,255,0),4);
        drawPoly(layer,right,Scalar(0,255,255),4);
        if (left.valid && right.valid) {
            vector<Point> area;
            for (int y=H-1;y>=0;y-=3) {
                int x=cvRound(left.x(y));
                if (x>=0 && x<W) area.emplace_back(x,y);
            }
            for (int y=0;y<H;y+=3) {
                int x=cvRound(right.x(y));
                if (x>=0 && x<W) area.emplace_back(x,y);
            }
            if (area.size()>=6) {
                vector<vector<Point>> polys{area};
                fillPoly(layer,polys,Scalar(0,80,0));
                drawPoly(layer,left,Scalar(0,255,0),4);
                drawPoly(layer,right,Scalar(0,255,255),4);
            }
        }

        Mat projected;
        warpPerspective(layer,projected,perspective.inverse,frame.size());

        Mat output=frame.clone();
        addWeighted(output,1.0,projected,0.8,0.0,output);

        string status="Lane pair not detected";
        if (left.valid && right.valid) {
            double y=H*0.90;
            double center=(left.x(y)+right.x(y))*0.5;
            double offset=center-W*0.5;
            status="Lane center offset: "+to_string(static_cast<int>(round(offset)))+
                   " px (positive = right)";
        }
        putText(output,status,Point(12,28),FONT_HERSHEY_SIMPLEX,0.55,
                (left.valid && right.valid)?Scalar(0,255,0):Scalar(0,0,255),
                2,LINE_AA);
        imshow("BEV Polynomial Lane Detection",output);

        if (showSteps) {
        
            //src좌표 표시용
            Mat srcView = frame.clone();
            drawSrcPoints(srcView, perspective);
            //imshow("SRC Points - Original", srcView);
            
            Mat dstView = bev.clone();
            drawDstPoints(dstView, perspective);
            //imshow("DST Points - Original", dstView);
            
            //imshow("1 - BEV",bev);
            
            //imshow("2 - Grayscale",gray);
            //imshow("3 - Binary candidates",binary);
            //imshow("4 - Sliding windows",windows);
            //imshow("5 - BEV curves",bevCurves);
            //imshow("6 - Sobel X",gradient);
        }
        int key=waitKey(33)&0xFF;
        if (key==27 || key=='q') break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
