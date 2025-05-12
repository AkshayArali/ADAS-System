// Akshay Arali

#include "lane_lines.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

std::vector<cv::Vec4i> laneDetection(Mat& frame) {

    // 1) Preprocess
    Mat gray; 
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5,5), 0);
    
    Mat edges;
    Canny(gray, edges, 50, 150);

    // 2) Region of interest (trapezoid)
    int h = frame.rows, w = frame.cols;
    Mat mask = Mat::zeros(edges.size(), edges.type());
    Point pts[4] = {
        Point(0, h),
        Point(w, h),
        Point(int(w*0.6), int(h*0.6)),
        Point(int(w*0.4), int(h*0.6))
    };
    vector<Point> roiPts(pts, pts+4);
    fillPoly(mask, vector<vector<Point>>{roiPts}, Scalar(255));
    Mat roiEdges;
    bitwise_and(edges, mask, roiEdges);

    // 3) Hough lines
    vector<Vec4i> lines;
    HoughLinesP(roiEdges, lines, 1, CV_PI/180, 50, 50, 10);

    // 4) Separate left/right by slope
    vector<Vec4i> leftL, rightL;
    #pragma omp parallel
    {
        vector<Vec4i> localLeft, localRight;
        #pragma omp for nowait
        for (int i = 0; i < lines.size(); ++i) {
            Vec4i l = lines[i];
            float dx = float(l[2] - l[0]), dy = float(l[3] - l[1]);
            float slope = dy / (dx + 1e-5f);
            if (fabs(slope) < 0.5f) continue;
            if (slope < 0 && l[0] < w / 2)
                localLeft.push_back(l);
            else if (slope > 0 && l[0] > w / 2)
                localRight.push_back(l);
        }
        #pragma omp critical
        {
            leftL.insert(leftL.end(), localLeft.begin(), localLeft.end());
            rightL.insert(rightL.end(), localRight.begin(), localRight.end());
        }
    }

    // 5) Fit one line each side (or default)
    Point2f lb, lt, rb, rt;
    auto fitSide = [&](vector<Vec4i>& side, Point2f& bot, Point2f& top, float yTopFrac) {
        if (!side.empty()) {
            vector<Point2f> pts;

            #pragma omp parallel for
            for (int i = 0; i < side.size(); ++i) {
                #pragma omp critical
                {
                    pts.emplace_back(side[i][0], side[i][1]);
                    pts.emplace_back(side[i][2], side[i][3]);
                }
            }
            
            Vec4f f; 
            fitLine(pts, f, DIST_L2, 0, 0.01, 0.01);
            float vx=f[0], vy=f[1], x0=f[2], y0=f[3];
            bot = Point2f(x0 + vx*(h - y0)/vy, float(h));
            top = Point2f(x0 + vx*(h*yTopFrac - y0)/vy, h*yTopFrac);
        } else {
            // fallback to ROI corners
            bot = side.size()==0 && &bot==&lb ? Point2f(0,h) : Point2f(w,h);
            top = side.size()==0 && &bot==&lb ? Point2f(w*0.4f,h*0.6f) : Point2f(w*0.6f,h*0.6f);
        }
    };

    fitSide(leftL,  lb, lt, 0.6f);
    fitSide(rightL, rb, rt, 0.6f);

    // 6) Temporal smoothing
    static Point2f p_lb, p_lt, p_rb, p_rt;
    auto blend = [&](Point2f in, Point2f& prev)->Point2f {
        if (prev==Point2f()) prev = in;
        prev = prev*0.8f + in*0.2f;
        return prev;
    };
    Point2f s_lb = blend(lb, p_lb);
    Point2f s_lt = blend(lt, p_lt);
    Point2f s_rb = blend(rb, p_rb);
    Point2f s_rt = blend(rt, p_rt);
    
    std::vector<cv::Vec4i> points;
    points.reserve(2);
    
    points.emplace_back(cvRound(s_lb.x), cvRound(s_lb.y),
                        cvRound(s_lt.x), cvRound(s_lt.y));
    
    points.emplace_back(cvRound(s_rb.x), cvRound(s_rb.y),
                        cvRound(s_rt.x), cvRound(s_rt.y));
    
    return points;
}
